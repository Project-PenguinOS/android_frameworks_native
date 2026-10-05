/*
 * Copyright 2021 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define ATRACE_TAG ATRACE_TAG_GRAPHICS
#include "BlurFilter.h"
#include <SkBlendMode.h>
#include <SkM44.h>
#include <SkCanvas.h>
#include <SkPaint.h>
#include <SkRRect.h>
#include <SkRuntimeEffect.h>
#include <SkSize.h>
#include <SkString.h>
#include <SkSurface.h>
#include <SkTileMode.h>
#include <common/ThreadStateCrashLogger.h>
#include <common/trace.h>
#include <log/log.h>
#include <sys/system_properties.h>

#include <algorithm>
#include <cstring>

#include "RuntimeEffectManager.h"

namespace android {
namespace renderengine {
namespace skia {

const SkString kEffectSource_BlurFilter_MixEffect(R"(
    uniform shader blurredInput;
    uniform shader originalInput;
    uniform float mixFactor;

    half4 main(float2 xy) {
        return half4(mix(originalInput.eval(xy), blurredInput.eval(xy), mixFactor)).rgb1;
    }
)");

// PenguinOS Liquid Glass. A rounded blur region is drawn as a slab of glass: towards its rim the
// glass curves like a lens and bends what's behind it inwards, the colours behind it are lifted,
// and its rim catches the light, brightest along the top left.
const SkString kEffectSource_LiquidGlassEffect(R"(
    uniform shader blurredInput;
    uniform shader originalInput;
    uniform float clarity;
    uniform float4 bounds;
    uniform float radius;
    uniform float band;
    uniform float refraction;
    uniform float rim;

    float roundRect(float2 p, float2 c, float2 h, float r) {
        float2 q = abs(p - c) - (h - r);
        return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    }

    half4 main(float2 xy) {
        float2 c = (bounds.xy + bounds.zw) * 0.5;
        float2 h = (bounds.zw - bounds.xy) * 0.5;
        float r = min(radius, min(h.x, h.y));
        float d = roundRect(xy, c, h, r);
        float e = 0.75;
        float2 g = float2(roundRect(xy + float2(e, 0.0), c, h, r) - roundRect(xy - float2(e, 0.0), c, h, r),
                          roundRect(xy + float2(0.0, e), c, h, r) - roundRect(xy - float2(0.0, e), c, h, r));
        float2 n = g / max(length(g), 0.0001);

        // 0 across the flat middle, 1 at the very edge.
        float t = clamp(1.0 + d / band, 0.0, 1.0);
        float bend = refraction * t * t * t;
        float2 src = xy - n * bend;
        // Frosted across the middle so what sits on the glass stays legible, clearer towards the
        // rim, where the bent picture of what's behind shows through.
        float clear = clarity + (1.0 - clarity) * 0.55 * t * t;
        half3 rgb = mix(blurredInput.eval(src).rgb, originalInput.eval(src).rgb, half(clear));

        half l = dot(rgb, half3(0.2126, 0.7152, 0.0722));
        rgb = mix(half3(l), rgb, 1.25);
        rgb = rgb * 1.04 + 0.02;

        float facing = dot(n, normalize(float2(-0.6, -1.0)));
        float edge = pow(t, 6.0);
        rgb += half(rim * edge * (0.55 + 0.45 * facing));
        rgb += half(rim * 0.35 * edge * max(-facing, 0.0));
        return half4(clamp(rgb, 0.0, 1.0), 1.0);
    }
)");

static constexpr const char* kLiquidGlassProperty = "persist.sys.penguin.liquid_glass";

// Whether Penguin Laboratory's Liquid Glass is on, re-read only when the property changes.
static bool liquidGlassEnabled() {
    static uint32_t areaSerial = UINT32_MAX;
    static const prop_info* info = nullptr;
    static uint32_t serial = UINT32_MAX;
    static bool enabled = false;
    if (info == nullptr) {
        const uint32_t current = __system_property_area_serial();
        if (current == areaSerial) return false;
        areaSerial = current;
        info = __system_property_find(kLiquidGlassProperty);
        if (info == nullptr) return false;
    }
    const uint32_t current = __system_property_serial(info);
    if (current != serial) {
        serial = current;
        __system_property_read_callback(
                info,
                [](void* cookie, const char*, const char* value, uint32_t) {
                    *static_cast<bool*>(cookie) =
                            strcmp(value, "1") == 0 || strcmp(value, "true") == 0;
                },
                &enabled);
    }
    return enabled;
}

static SkMatrix getShaderTransform(const SkCanvas* canvas, const SkRect& blurRect,
                                   const float scale, const float zoomScale) {
    // 1. Apply the blur shader matrix, which scales up the blurred surface to its real size
    auto matrix = SkMatrix::Scale(scale, scale);
    // 2. Since the blurred surface has the size of the layer, we align it with the
    // top left corner of the layer position.
    matrix.postConcat(SkMatrix::Translate(blurRect.fLeft, blurRect.fTop));
    // 3. Apply the "zoom" effect as an extra scale + translate around the center of the blur.
    if (zoomScale != 1.0f) {
        matrix.postScale(zoomScale, zoomScale);
        matrix.postTranslate(
                blurRect.width() * (1 - zoomScale) / 2.0f,
                blurRect.height() * (1 - zoomScale) / 2.0f);
    }
    // 4. Finally, apply the inverse canvas matrix. The snapshot made in the BlurFilter is in the
    // original surface orientation. The inverse matrix has to be applied to align the blur
    // surface with the current orientation/position of the canvas.
    SkMatrix drawInverse;
    if (canvas != nullptr && canvas->getTotalMatrix().invert(&drawInverse)) {
        matrix.postConcat(drawInverse);
    }
    return matrix;
}

BlurFilter::BlurFilter(RuntimeEffectManager& effectManager, const float maxCrossFadeRadius)
      : mMaxCrossFadeRadius(maxCrossFadeRadius),
        mMixEffect(effectManager.mKnownEffects[kBlurFilter_MixEffect]),
        mLiquidGlassEffect(effectManager.mKnownEffects[kLiquidGlassEffect]) {}

float BlurFilter::getMaxCrossFadeRadius() const {
    return mMaxCrossFadeRadius;
}

void BlurFilter::drawBlurRegion(SkCanvas* canvas, const SkRRect& effectRegion,
                                const uint32_t blurRadius, const float zoomScale,
                                const float blurAlpha,
                                const SkRect& blurRect, sk_sp<SkImage> blurredImage,
                                sk_sp<SkImage> input) {
    SFTRACE_CALL();

    SkPaint paint;
    paint.setAlphaf(blurAlpha);

    auto blurMatrix = getShaderTransform(canvas, blurRect, kInverseInputScale, zoomScale);

    SkSamplingOptions linearSampling(SkFilterMode::kLinear, SkMipmapMode::kNone);
    const auto blurShader = blurredImage->makeShader(SkTileMode::kMirror, SkTileMode::kMirror,
                                                     linearSampling, &blurMatrix);

    if (!effectRegion.isRect() && mLiquidGlassEffect != nullptr && liquidGlassEnabled()) {
        const SkRect& rect = effectRegion.rect();
        const float radius = effectRegion.radii(SkRRect::kUpperLeft_Corner).fX;
        const float shortSide = std::min(rect.width(), rect.height());
        // The curved rim is as deep as the corner, but never past the middle of the glass.
        const float band = std::clamp(radius, 6.0f, shortSide * 0.5f);
        SkRuntimeShaderBuilder glass(mLiquidGlassEffect);
        glass.child("blurredInput") = blurShader;
        SkMatrix inputMatrix;
        if (!canvas->getTotalMatrix().invert(&inputMatrix)) {
            ALOGE("matrix was unable to be inverted");
        }
        glass.child("originalInput") =
                input != nullptr
                        ? input->makeShader(SkTileMode::kMirror, SkTileMode::kMirror,
                                            linearSampling, inputMatrix)
                        : blurShader;
        glass.uniform("clarity") = 0.0f;
        glass.uniform("bounds") = SkV4{rect.fLeft, rect.fTop, rect.fRight, rect.fBottom};
        glass.uniform("radius") = radius;
        glass.uniform("band") = band;
        glass.uniform("refraction") = band * 0.7f;
        glass.uniform("rim") = 0.32f;
        paint.setShader(glass.makeShader());
        paint.setAntiAlias(true);
        canvas->drawRRect(effectRegion, paint);
        return;
    }

    if (blurRadius < mMaxCrossFadeRadius) {
        LOG_THREAD_STATE_AND_CRASH_IF(!input);

        // For sampling Skia's API expects the inverse of what logically seems appropriate. In this
        // case you might expect the matrix to simply be the canvas matrix.
        SkMatrix inputMatrix;
        if (!canvas->getTotalMatrix().invert(&inputMatrix)) {
            ALOGE("matrix was unable to be inverted");
        }
        if (zoomScale != 1.0f) {
            inputMatrix.preTranslate(
                    blurRect.width() * (1 - zoomScale) / 2.0f,
                    blurRect.height() * (1 - zoomScale) / 2.0f);
            inputMatrix.preScale(zoomScale, zoomScale);
        }

        SkRuntimeShaderBuilder blurBuilder(mMixEffect);
        blurBuilder.child("blurredInput") = blurShader;
        blurBuilder.child("originalInput") =
                input->makeShader(SkTileMode::kMirror, SkTileMode::kMirror, linearSampling,
                                  inputMatrix);

        blurBuilder.uniform("mixFactor") = blurRadius / mMaxCrossFadeRadius;

        paint.setShader(blurBuilder.makeShader());
    } else {
        paint.setShader(blurShader);
    }

    if (effectRegion.isRect()) {
        if (blurAlpha == 1.0f) {
            paint.setBlendMode(SkBlendMode::kSrc);
        }
        canvas->drawRect(effectRegion.rect(), paint);
    } else {
        paint.setAntiAlias(true);
        canvas->drawRRect(effectRegion, paint);
    }
}

} // namespace skia
} // namespace renderengine
} // namespace android

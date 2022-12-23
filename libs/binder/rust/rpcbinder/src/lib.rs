/*
 * Copyright (C) 2022 The Android Open Source Project
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

//! API for RPC Binder services.

mod client;
mod server;

pub use client::{
    get_preconnected_rpc_interface, get_preconnected_rpc_service, get_unix_domain_rpc_interface,
    get_unix_domain_rpc_service, get_vsock_rpc_interface, get_vsock_rpc_service,
};
pub use server::{
    run_init_unix_domain_rpc_server, run_vsock_rpc_server, run_vsock_rpc_server_with_factory,
};

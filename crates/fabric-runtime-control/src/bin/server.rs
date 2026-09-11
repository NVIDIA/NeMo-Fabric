// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resident Unix-socket server for a Fabric runtime.

use std::path::PathBuf;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let first = args.next();
    let socket = match first.as_deref().and_then(|value| value.to_str()) {
        None | Some("serve") => args
            .next()
            .map(PathBuf::from)
            .unwrap_or_else(nemo_fabric_runtime_control::default_socket_path),
        Some(_) => PathBuf::from(first.expect("first argument")),
    };
    if args.next().is_some() {
        eprintln!("usage: fabric-runtime-server [serve] [socket]");
        std::process::exit(2);
    }
    if let Err(error) = nemo_fabric_runtime_control::serve(&socket) {
        eprintln!("Fabric runtime server failed: {error}");
        std::process::exit(1);
    }
}

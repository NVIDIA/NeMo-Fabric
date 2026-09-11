// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed command-line client for the Fabric runtime server.

use std::path::PathBuf;

use nemo_fabric_runtime_control::RuntimeControlOperation;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let command = args.next().and_then(|value| value.into_string().ok());
    if command.as_deref() == Some("collect-artifact") {
        if args.next().is_some() {
            eprintln!("usage: fabric-runtime-ctl collect-artifact");
            std::process::exit(2);
        }
        if let Err(error) = nemo_fabric_runtime_control::export_artifact(
            std::io::stdin().lock(),
            std::io::stdout().lock(),
        ) {
            eprintln!("Fabric runtime artifact export failed: {error}");
            std::process::exit(1);
        }
        return;
    }
    let operation = command
        .and_then(|value| value.parse::<RuntimeControlOperation>().ok())
        .unwrap_or_else(|| {
            eprintln!(
                "usage: fabric-runtime-ctl <start|invoke|stop> [socket]\n       fabric-runtime-ctl collect-artifact"
            );
            std::process::exit(2);
        });
    let socket = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(nemo_fabric_runtime_control::default_socket_path);
    if args.next().is_some() {
        eprintln!("usage: fabric-runtime-ctl <start|invoke|stop> [socket]");
        std::process::exit(2);
    }
    if let Err(error) = nemo_fabric_runtime_control::control(
        &socket,
        operation,
        std::io::stdin().lock(),
        std::io::stdout().lock(),
    ) {
        eprintln!("Fabric runtime control failed: {error}");
        std::process::exit(1);
    }
}

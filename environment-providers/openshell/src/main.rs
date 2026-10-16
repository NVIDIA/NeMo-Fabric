// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI entry point for the optional OpenShell environment provider.
//!
//! Accepts only `serve --stdio` and delegates the private provider protocol
//! loop to the library; the host Fabric SDK starts this process on demand.

use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args != ["serve", "--stdio"] {
        eprintln!("usage: fabric-environment-openshell serve --stdio");
        return ExitCode::from(2);
    }
    match nemo_fabric_openshell_provider::serve_stdio().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

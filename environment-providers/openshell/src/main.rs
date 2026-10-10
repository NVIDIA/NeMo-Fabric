// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fabric's optional OpenShell environment-provider process.

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

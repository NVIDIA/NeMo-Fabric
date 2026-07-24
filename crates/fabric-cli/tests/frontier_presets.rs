// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI regression coverage for presets that require an NVIDIA Frontier endpoint.

use std::process::Command;

const FRONTIER_BASE_URL_ENV: &str = "NVIDIA_FRONTIER_BASE_URL";

#[test]
fn frontier_presets_require_an_explicit_endpoint() {
    for preset in ["claude", "codex"] {
        let missing = Command::new(env!("CARGO_BIN_EXE_nemo-fabric"))
            .args(["plan", "--preset", preset])
            .env_remove(FRONTIER_BASE_URL_ENV)
            .output()
            .expect("run CLI without Frontier endpoint");
        assert!(!missing.status.success(), "{preset} unexpectedly planned");
        assert!(
            String::from_utf8_lossy(&missing.stderr).contains(FRONTIER_BASE_URL_ENV),
            "{preset} did not identify the missing endpoint"
        );

        let configured = Command::new(env!("CARGO_BIN_EXE_nemo-fabric"))
            .args(["plan", "--preset", preset])
            .env(FRONTIER_BASE_URL_ENV, "https://frontier.example/v1")
            .output()
            .expect("run CLI with Frontier endpoint");
        assert!(
            configured.status.success(),
            "{preset} failed with an explicit endpoint: {}",
            String::from_utf8_lossy(&configured.stderr)
        );
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI regression coverage for presets that require an NVIDIA Frontier endpoint.

use std::process::Command;

const NVIDIA_FRONTIER_BASE_URL_ENV: &str = "NVIDIA_FRONTIER_BASE_URL";

#[test]
fn frontier_presets_require_an_explicit_endpoint() {
    for preset in ["claude", "codex"] {
        let missing = Command::new(env!("CARGO_BIN_EXE_nemo-fabric"))
            .args(["plan", "--preset", preset])
            .env_remove("NVIDIA_API_KEY")
            .env_remove(NVIDIA_FRONTIER_BASE_URL_ENV)
            .output()
            .expect("run CLI without Frontier endpoint");
        assert!(!missing.status.success(), "{preset} unexpectedly planned");
        assert!(missing.stdout.is_empty(), "{preset} emitted a partial plan");
        assert!(
            String::from_utf8_lossy(&missing.stderr).contains(NVIDIA_FRONTIER_BASE_URL_ENV),
            "{preset} did not identify the missing endpoint"
        );

        let configured = Command::new(env!("CARGO_BIN_EXE_nemo-fabric"))
            .args(["plan", "--preset", preset])
            .env_remove("NVIDIA_API_KEY")
            .env(NVIDIA_FRONTIER_BASE_URL_ENV, "https://frontier.example/v1")
            .output()
            .expect("run CLI with Frontier endpoint");
        assert!(
            configured.status.success(),
            "{preset} failed with an explicit endpoint: {}",
            String::from_utf8_lossy(&configured.stderr)
        );
        let plan: serde_json::Value =
            serde_json::from_slice(&configured.stdout).expect("parse CLI plan");
        assert_eq!(
            plan["config"]["models"]["default"]["settings"]["base_url"].as_str(),
            Some("https://frontier.example/v1"),
            "{preset} did not preserve the Frontier endpoint"
        );
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Recording-root resolution, available before a runtime exists.

use crate::config::RunPlan;
use std::path::PathBuf;

/// The parent of request-scoped run journals. Requires recording to be enabled.
pub fn run_record_folder(plan: &RunPlan) -> Option<PathBuf> {
    if !plan.config.runtime.record_enabled() {
        return None;
    }
    plan.config
        .runtime
        .artifacts
        .as_ref()
        .map(|path| {
            if path.is_absolute() {
                path.clone()
            } else {
                plan.base_dir.join(path)
            }
        })
        .or_else(|| {
            plan.environment_plan
                .as_ref()
                .and_then(|environment| environment.artifacts.clone())
        })
        .map(|root| root.join(".fabric"))
}

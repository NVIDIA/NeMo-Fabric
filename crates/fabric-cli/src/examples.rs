// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Maintained runnable examples composed from presets and shared assets.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nemo_fabric_core::{FabricConfig, MetadataConfig, SkillConfig};

use crate::assets::{EmbeddedFile, StagedAssets};
use crate::presets::{self, Preset};

pub(crate) const CODE_REVIEW_WORKSPACE: &str =
    include_str!("../assets/examples/code-review/repo/calculator.py");
pub(crate) const CODE_REVIEW_SKILL: &str =
    include_str!("../assets/examples/code-review/skills/code-review.md");

const CODE_REVIEW_FILES: &[EmbeddedFile] = &[
    EmbeddedFile {
        path: "repo/calculator.py",
        contents: CODE_REVIEW_WORKSPACE,
    },
    EmbeddedFile {
        path: "skills/code-review.md",
        contents: CODE_REVIEW_SKILL,
    },
];
const CODE_REVIEW_VARIANTS: &[&str] = &["scripted", "hermes", "claude", "codex", "deepagents"];

/// Metadata and variant catalog for one maintained CLI example.
#[derive(Debug, Clone, Copy)]
pub struct Example {
    /// Stable command-line name.
    pub name: &'static str,
    /// Short experimentation-oriented description.
    pub description: &'static str,
    /// Variant selected when `--variant` is omitted.
    pub default_variant: &'static str,
    /// Complete supported variant names.
    pub variants: &'static [&'static str],
}

impl Example {
    fn preset(self, variant: Option<&str>) -> Result<Preset, String> {
        let variant = variant.unwrap_or(self.default_variant);
        if !self.variants.contains(&variant) {
            return Err(format!(
                "unknown variant {variant:?} for example {:?}; available: {}",
                self.name,
                self.variants.join(", ")
            ));
        }
        presets::find(variant).ok_or_else(|| {
            format!(
                "example {:?} references missing preset {variant:?}",
                self.name
            )
        })
    }

    /// Construct one complete example config without staging its assets.
    pub fn config(self, variant: Option<&str>) -> Result<FabricConfig, String> {
        self.preset(variant).map(code_review_config)
    }

    /// Construct and stage one complete example variant.
    pub fn select(self, variant: Option<&str>) -> Result<SelectedExample, String> {
        let preset = self.preset(variant)?;
        let files = self.embedded_files(preset);
        let assets = StagedAssets::create(&files)
            .map_err(|error| format!("failed to stage example {:?}: {error}", self.name))?;
        Ok(SelectedExample {
            example: self,
            variant: preset,
            config: code_review_config(preset),
            assets,
        })
    }

    pub(crate) fn embedded_files(self, preset: Preset) -> Vec<EmbeddedFile> {
        let mut files = Vec::from(preset.embedded_files());
        files.extend_from_slice(CODE_REVIEW_FILES);
        files
    }
}

/// Selected example variant and its staged installation-safe assets.
#[derive(Debug)]
pub struct SelectedExample {
    /// Example metadata.
    pub example: Example,
    /// Preset reused by this complete example variant.
    pub variant: Preset,
    /// Complete typed example configuration.
    pub config: FabricConfig,
    assets: StagedAssets,
}

impl SelectedExample {
    /// Return the base directory for resolving this example.
    pub fn base_dir(&self) -> &Path {
        self.assets.path()
    }
}

/// Return the maintained example catalog.
pub fn all() -> &'static [Example] {
    &EXAMPLES
}

/// Find an example by its stable CLI name.
pub fn find(name: &str) -> Option<Example> {
    EXAMPLES
        .iter()
        .copied()
        .find(|example| example.name == name)
}

const EXAMPLES: [Example; 1] = [Example {
    name: "code-review",
    description: "Review a small Python workspace using a maintained skill.",
    default_variant: "scripted",
    variants: CODE_REVIEW_VARIANTS,
}];

fn code_review_config(preset: Preset) -> FabricConfig {
    let mut config = preset.config();
    config.metadata = MetadataConfig {
        name: "code-review-agent".to_string(),
        description: Some("Maintained example for reviewing a small Python workspace.".to_string()),
        extensions: BTreeMap::new(),
    };
    let environment = config
        .environment
        .as_mut()
        .expect("CLI presets always define an execution environment");
    environment.workspace = Some(PathBuf::from("repo"));
    config.skills = Some(SkillConfig {
        paths: vec![PathBuf::from("skills/code-review.md")],
        extensions: BTreeMap::new(),
    });
    config
}

#[cfg(test)]
mod tests {
    use nemo_fabric_core::{ResolveContext, RunRequest, resolve_run_plan_from_config, run_plan};

    use super::*;

    #[test]
    fn variants_reuse_the_preset_catalog() {
        let example = find("code-review").expect("code review example");
        for variant in example.variants {
            assert!(presets::find(variant).is_some(), "missing preset {variant}");
        }
    }

    #[test]
    fn code_review_stages_one_shared_asset_tree_and_runs() {
        let selected = find("code-review")
            .expect("code review example")
            .select(Some("scripted"))
            .expect("select example");
        assert!(selected.base_dir().join("repo/calculator.py").is_file());
        assert!(selected.base_dir().join("skills/code-review.md").is_file());
        let plan = resolve_run_plan_from_config(
            selected.config.clone(),
            ResolveContext::new(selected.base_dir()),
        )
        .expect("plan example");
        assert_eq!(plan.agent_name, "code-review-agent");
        assert_eq!(
            plan.environment_plan
                .as_ref()
                .expect("environment")
                .workspace
                .as_ref(),
            Some(&selected.base_dir().join("repo"))
        );
        let result =
            run_plan(&plan, RunRequest::text("review")).expect("run deterministic example variant");
        assert_eq!(result.output["response"], "review");
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Complete typed preset configurations maintained by the CLI.

use std::collections::BTreeMap;

use nemo_fabric_core::{
    ControlLocation, EnvironmentConfig, EnvironmentOwnership, FabricConfig, HarnessConfig,
    MetadataConfig, ModelConfig, ResolutionStrategy, RuntimeConfig,
};
use serde_json::{Map, Value, json};

use crate::assets::{EmbeddedFile, StagedAssets};

const SCRIPTED_DESCRIPTOR: &str = include_str!("../assets/adapters/scripted/fabric-adapter.json");
const SCRIPTED_RUNNER: &str = include_str!("../assets/adapters/scripted/run.py");
const HERMES_DESCRIPTOR: &str = include_str!("../assets/adapters/hermes/fabric-adapter.json");
const CLAUDE_DESCRIPTOR: &str = include_str!("../assets/adapters/claude/fabric-adapter.json");
const CODEX_DESCRIPTOR: &str = include_str!("../assets/adapters/codex/fabric-adapter.json");
const DEEPAGENTS_DESCRIPTOR: &str =
    include_str!("../assets/adapters/deepagents/fabric-adapter.json");
const NVIDIA_API_CATALOG_BASE_URL: &str = "https://integrate.api.nvidia.com/v1";

const SCRIPTED_ASSETS: &[EmbeddedFile] = &[
    EmbeddedFile {
        path: "adapters/scripted/fabric-adapter.json",
        contents: SCRIPTED_DESCRIPTOR,
    },
    EmbeddedFile {
        path: "adapters/scripted/run.py",
        contents: SCRIPTED_RUNNER,
    },
];
const HERMES_ASSETS: &[EmbeddedFile] = &[EmbeddedFile {
    path: "adapters/hermes/fabric-adapter.json",
    contents: HERMES_DESCRIPTOR,
}];
const CLAUDE_ASSETS: &[EmbeddedFile] = &[EmbeddedFile {
    path: "adapters/claude/fabric-adapter.json",
    contents: CLAUDE_DESCRIPTOR,
}];
const CODEX_ASSETS: &[EmbeddedFile] = &[EmbeddedFile {
    path: "adapters/codex/fabric-adapter.json",
    contents: CODEX_DESCRIPTOR,
}];
const DEEPAGENTS_ASSETS: &[EmbeddedFile] = &[EmbeddedFile {
    path: "adapters/deepagents/fabric-adapter.json",
    contents: DEEPAGENTS_DESCRIPTOR,
}];

/// Metadata and constructor for one complete CLI preset.
#[derive(Debug, Clone, Copy)]
pub struct Preset {
    /// Stable command-line name.
    pub name: &'static str,
    /// Short experimentation-oriented description.
    pub description: &'static str,
    /// Environment variables required by the harness/model combination.
    pub required_env: &'static [&'static str],
    build: fn() -> FabricConfig,
    assets: &'static [EmbeddedFile],
}

impl Preset {
    /// Construct a fresh complete typed config.
    pub fn config(self) -> FabricConfig {
        (self.build)()
    }

    /// Stage the adapter assets required to plan or run this preset.
    pub fn stage(self) -> std::io::Result<SelectedPreset> {
        Ok(SelectedPreset {
            preset: self,
            config: self.config(),
            assets: StagedAssets::create(self.assets)?,
        })
    }

    pub(crate) fn embedded_files(self) -> &'static [EmbeddedFile] {
        self.assets
    }
}

/// Selected preset and the staged base directory that must outlive its run.
#[derive(Debug)]
pub struct SelectedPreset {
    /// Preset metadata.
    pub preset: Preset,
    /// Complete typed configuration.
    pub config: FabricConfig,
    assets: StagedAssets,
}

impl SelectedPreset {
    /// Return the base directory for resolving this preset.
    pub fn base_dir(&self) -> &std::path::Path {
        self.assets.path()
    }
}

/// Return the maintained preset catalog.
pub fn all() -> &'static [Preset] {
    &PRESETS
}

/// Find a preset by its stable CLI name.
pub fn find(name: &str) -> Option<Preset> {
    PRESETS.iter().copied().find(|preset| preset.name == name)
}

const PRESETS: [Preset; 5] = [
    Preset {
        name: "scripted",
        description: "Credential-free deterministic smoke preset.",
        required_env: &[],
        build: scripted,
        assets: SCRIPTED_ASSETS,
    },
    Preset {
        name: "hermes",
        description: "Hermes Agent with an NVIDIA-hosted model.",
        required_env: &["NVIDIA_API_KEY"],
        build: hermes,
        assets: HERMES_ASSETS,
    },
    Preset {
        name: "claude",
        description: "Claude Code with an NVIDIA-hosted Claude model.",
        required_env: &["NVIDIA_API_KEY", "NVIDIA_FRONTIER_BASE_URL"],
        build: claude,
        assets: CLAUDE_ASSETS,
    },
    Preset {
        name: "codex",
        description: "Codex with an NVIDIA-hosted OpenAI model.",
        required_env: &["NVIDIA_API_KEY", "NVIDIA_FRONTIER_BASE_URL"],
        build: codex,
        assets: CODEX_ASSETS,
    },
    Preset {
        name: "deepagents",
        description: "LangChain Deep Agents with an NVIDIA-hosted model.",
        required_env: &["NVIDIA_API_KEY"],
        build: deepagents,
        assets: DEEPAGENTS_ASSETS,
    },
];

fn scripted() -> FabricConfig {
    config(
        "scripted-agent",
        "Credential-free NeMo Fabric CLI smoke preset.",
        "nvidia.fabric.scripted",
        None,
        Map::new(),
    )
}

fn hermes() -> FabricConfig {
    config(
        "hermes-agent",
        "NeMo Fabric Hermes CLI preset.",
        "nvidia.fabric.hermes",
        Some(model(
            "nvidia",
            "nvidia/nemotron-3-nano-30b-a3b",
            Some("NVIDIA_API_KEY"),
            Some(NVIDIA_API_CATALOG_BASE_URL),
        )),
        Map::new(),
    )
}

fn claude() -> FabricConfig {
    config(
        "claude-agent",
        "NeMo Fabric Claude CLI preset.",
        "nvidia.fabric.claude",
        Some(model(
            "nvidia",
            "aws/anthropic/claude-opus-4-5",
            Some("NVIDIA_API_KEY"),
            std::env::var("NVIDIA_FRONTIER_BASE_URL").ok().as_deref(),
        )),
        Map::from_iter([("permission_mode".to_string(), json!("dontAsk"))]),
    )
}

fn codex() -> FabricConfig {
    config(
        "codex-agent",
        "NeMo Fabric Codex CLI preset.",
        "nvidia.fabric.codex",
        Some(model(
            "nvidia",
            "azure/openai/gpt-5.4",
            Some("NVIDIA_API_KEY"),
            std::env::var("NVIDIA_FRONTIER_BASE_URL").ok().as_deref(),
        )),
        Map::from_iter([
            ("sandbox".to_string(), json!("workspace-write")),
            (
                "config_overrides".to_string(),
                json!({
                    "features.apps": false,
                    "features.multi_agent": false,
                    "features.plugins": false,
                    "web_search": "disabled",
                }),
            ),
        ]),
    )
}

fn deepagents() -> FabricConfig {
    config(
        "deepagents-agent",
        "NeMo Fabric Deep Agents CLI preset.",
        "nvidia.fabric.langchain.deepagents",
        Some(model(
            "nvidia",
            "nvidia/nemotron-3-nano-30b-a3b",
            Some("NVIDIA_API_KEY"),
            Some(NVIDIA_API_CATALOG_BASE_URL),
        )),
        Map::new(),
    )
}

fn config(
    name: &str,
    description: &str,
    adapter_id: &str,
    default_model: Option<ModelConfig>,
    settings: Map<String, Value>,
) -> FabricConfig {
    FabricConfig {
        schema_version: "fabric.agent/v1alpha1".to_string(),
        metadata: MetadataConfig {
            name: name.to_string(),
            description: Some(description.to_string()),
            extensions: BTreeMap::new(),
        },
        harness: HarnessConfig {
            adapter_id: adapter_id.to_string(),
            resolution: Some(ResolutionStrategy::Preinstalled),
            settings,
            extensions: BTreeMap::new(),
        },
        models: default_model
            .map(|model| BTreeMap::from_iter([("default".to_string(), model)]))
            .unwrap_or_default(),
        runtime: RuntimeConfig {
            input_schema: "text".to_string(),
            output_schema: "message".to_string(),
            artifacts: None,
            extensions: BTreeMap::new(),
        },
        environment: Some(EnvironmentConfig {
            provider: "local".to_string(),
            control_location: ControlLocation::InEnvControl,
            ownership: EnvironmentOwnership::FabricOwned,
            workspace: None,
            artifacts: None,
            connection: Map::new(),
            metadata: Map::new(),
            settings: Map::new(),
            extensions: BTreeMap::new(),
        }),
        tools: None,
        skills: None,
        mcp: None,
        telemetry: None,
        relay: None,
        extensions: BTreeMap::new(),
    }
}

fn model(
    provider: &str,
    name: &str,
    api_key_env: Option<&str>,
    base_url: Option<&str>,
) -> ModelConfig {
    ModelConfig {
        provider: provider.to_string(),
        model: name.to_string(),
        temperature: None,
        api_key_env: api_key_env.map(str::to_string),
        settings: base_url
            .map(|value| Map::from_iter([("base_url".to_string(), json!(value))]))
            .unwrap_or_default(),
        extensions: BTreeMap::new(),
    }
}

#[cfg(test)]
mod tests {
    use nemo_fabric_core::{ResolveContext, resolve_run_plan_from_config};

    use super::*;

    #[test]
    fn catalog_names_are_unique_and_configs_are_complete() {
        let mut names = std::collections::BTreeSet::new();
        for preset in all() {
            assert!(names.insert(preset.name));
            let config = preset.config();
            assert_eq!(config.metadata.name, format!("{}-agent", preset.name));
            assert!(!config.harness.adapter_id.is_empty());
        }
    }

    #[test]
    fn scripted_preset_plans_from_staged_assets() {
        let selected = find("scripted")
            .expect("scripted preset")
            .stage()
            .expect("stage preset");
        let plan = resolve_run_plan_from_config(
            selected.config.clone(),
            ResolveContext::new(selected.base_dir()),
        )
        .expect("plan scripted preset");

        assert_eq!(plan.agent_name, "scripted-agent");
        assert_eq!(
            plan.adapter_descriptor
                .expect("adapter")
                .descriptor
                .adapter_id,
            "nvidia.fabric.scripted"
        );
    }

    #[test]
    fn adapter_presets_use_nvidia_credentials() {
        for name in ["hermes", "claude", "codex", "deepagents"] {
            let preset = find(name).expect("adapter preset");
            let config = preset.config();
            let model = config.models.get("default").expect("default model");

            assert_eq!(model.provider, "nvidia");
            assert_eq!(model.api_key_env.as_deref(), Some("NVIDIA_API_KEY"));
            assert!(preset.required_env.contains(&"NVIDIA_API_KEY"));
        }

        for name in ["hermes", "deepagents"] {
            let config = find(name).expect("catalog preset").config();
            let model = config.models.get("default").expect("default model");
            assert_eq!(
                model.settings.get("base_url").and_then(Value::as_str),
                Some(NVIDIA_API_CATALOG_BASE_URL)
            );
        }

        for name in ["claude", "codex"] {
            assert!(
                find(name)
                    .expect("frontier preset")
                    .required_env
                    .contains(&"NVIDIA_FRONTIER_BASE_URL")
            );
        }
    }

    #[test]
    fn packaged_descriptors_match_repository_adapters() {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let repository_adapters = manifest_dir.join("../../adapters");
        if !repository_adapters.is_dir() {
            return;
        }

        for (name, embedded) in [
            ("hermes", HERMES_DESCRIPTOR),
            ("claude", CLAUDE_DESCRIPTOR),
            ("codex", CODEX_DESCRIPTOR),
            ("deepagents", DEEPAGENTS_DESCRIPTOR),
        ] {
            let canonical =
                std::fs::read_to_string(repository_adapters.join(name).join("fabric-adapter.json"))
                    .expect("read repository adapter descriptor");
            assert_eq!(embedded, canonical, "{name} descriptor drifted");
        }
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Editable applications generated from the maintained example catalog.

use std::fs;
use std::path::{Path, PathBuf};

use clap::ValueEnum;
use nemo_fabric_core::{FabricConfig, ModelConfig};
use serde_json::Value;

use crate::examples::Example;
use crate::presets;

const PYTHON_MAIN: &str = include_str!("../templates/python/main.py.tmpl");
const PYTHON_PROJECT: &str = include_str!("../templates/python/pyproject.toml.tmpl");
const RUST_MAIN: &str = include_str!("../templates/rust/main.rs.tmpl");
const RUST_PROJECT: &str = include_str!("../templates/rust/Cargo.toml.tmpl");
const README: &str = include_str!("../templates/README.md.tmpl");

#[derive(Debug)]
struct ScaffoldFile {
    path: String,
    contents: String,
}

/// Language API used by a generated editable application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Language {
    /// Generate a Python SDK application.
    Python,
    /// Generate a direct Rust core application.
    Rust,
}

impl Language {
    /// Return the command-line spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::Rust => "rust",
        }
    }
}

/// Generate an ordinary editable application and return its destination.
pub fn init(
    example: Example,
    variant: Option<&str>,
    language: Language,
    destination: impl AsRef<Path>,
) -> Result<PathBuf, String> {
    let destination = destination.as_ref();
    if destination.exists() {
        return Err(format!(
            "destination already exists: {}",
            destination.display()
        ));
    }
    let variant = variant.unwrap_or(example.default_variant);
    let preset =
        presets::find(variant).ok_or_else(|| format!("unknown preset variant {variant:?}"))?;
    let config = example.config(Some(variant))?;
    let mut files = example
        .embedded_files(preset)
        .into_iter()
        .map(|file| ScaffoldFile {
            path: file.path.to_string(),
            contents: file.contents.to_string(),
        })
        .collect::<Vec<_>>();
    files.extend(language_files(language, &config, example.name));
    files.push(ScaffoldFile {
        path: "README.md".to_string(),
        contents: README
            .replace("{{EXAMPLE}}", example.name)
            .replace("{{VARIANT}}", variant)
            .replace("{{LANGUAGE}}", language.as_str()),
    });
    write_files(destination, &files)?;
    Ok(destination.to_path_buf())
}

fn language_files(language: Language, config: &FabricConfig, example: &str) -> Vec<ScaffoldFile> {
    let (project_path, project, main_path, main) = match language {
        Language::Python => (
            "pyproject.toml",
            PYTHON_PROJECT.replace("{{PACKAGE}}", &package_name(example)),
            "main.py",
            render_python(config),
        ),
        Language::Rust => (
            "Cargo.toml",
            RUST_PROJECT.replace("{{PACKAGE}}", &package_name(example)),
            "src/main.rs",
            render_rust(config),
        ),
    };
    vec![
        ScaffoldFile {
            path: project_path.to_string(),
            contents: project,
        },
        ScaffoldFile {
            path: main_path.to_string(),
            contents: main,
        },
    ]
}

fn write_files(destination: &Path, files: &[ScaffoldFile]) -> Result<(), String> {
    fs::create_dir(destination).map_err(|error| {
        format!(
            "failed to create destination {}: {error}",
            destination.display()
        )
    })?;
    for file in files {
        let path = destination.join(&file.path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        }
        fs::write(&path, &file.contents)
            .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    }
    Ok(())
}

fn render_python(config: &FabricConfig) -> String {
    PYTHON_MAIN
        .replace("{{AGENT_NAME}}", &python_string(&config.metadata.name))
        .replace(
            "{{DESCRIPTION}}",
            &python_string(config.metadata.description.as_deref().unwrap_or("")),
        )
        .replace("{{ADAPTER_ID}}", &python_string(&config.harness.adapter_id))
        .replace(
            "{{HARNESS_SETTINGS}}",
            &python_value(&Value::Object(config.harness.settings.clone())),
        )
        .replace("{{MODELS}}", &python_models(config.models.get("default")))
}

fn python_models(model: Option<&ModelConfig>) -> String {
    let Some(model) = model else {
        return "{}".to_string();
    };
    format!(
        "{{\"default\": ModelConfig(provider={}, model={}, api_key_env={})}}",
        python_string(&model.provider),
        python_string(&model.model),
        model
            .api_key_env
            .as_deref()
            .map(python_string)
            .unwrap_or_else(|| "None".to_string())
    )
}

fn python_value(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => python_string(value),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!("{}: {}", python_string(key), python_value(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn python_string(value: &str) -> String {
    serde_json::to_string(value).expect("strings serialize")
}

fn render_rust(config: &FabricConfig) -> String {
    RUST_MAIN
        .replace("{{AGENT_NAME}}", &rust_string(&config.metadata.name))
        .replace(
            "{{DESCRIPTION}}",
            &rust_string(config.metadata.description.as_deref().unwrap_or("")),
        )
        .replace("{{ADAPTER_ID}}", &rust_string(&config.harness.adapter_id))
        .replace(
            "{{HARNESS_SETTINGS}}",
            &rust_settings(&config.harness.settings),
        )
        .replace("{{MODELS}}", &rust_models(config.models.get("default")))
}

fn rust_settings(settings: &serde_json::Map<String, Value>) -> String {
    if settings.is_empty() {
        return "Map::new()".to_string();
    }
    format!(
        "Map::from_iter([{}])",
        settings
            .iter()
            .map(|(key, value)| format!("({}.to_string(), json!({}))", rust_string(key), value))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn rust_models(model: Option<&ModelConfig>) -> String {
    let Some(model) = model else {
        return "BTreeMap::new()".to_string();
    };
    let api_key = model
        .api_key_env
        .as_deref()
        .map(|value| format!("Some({}.to_string())", rust_string(value)))
        .unwrap_or_else(|| "None".to_string());
    format!(
        "BTreeMap::from_iter([(\"default\".to_string(), ModelConfig {{ provider: {}.to_string(), model: {}.to_string(), temperature: None, api_key_env: {api_key}, settings: Map::new(), extensions: BTreeMap::new() }})])",
        rust_string(&model.provider),
        rust_string(&model.model),
    )
}

fn rust_string(value: &str) -> String {
    format!("{value:?}")
}

fn package_name(example: &str) -> String {
    example.replace('_', "-")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples;

    fn destination(test: &str, language: Language) -> PathBuf {
        std::env::temp_dir().join(format!(
            "nemo-fabric-scaffold-test-{}-{test}-{}",
            std::process::id(),
            language.as_str()
        ))
    }

    #[test]
    fn generates_both_languages_from_the_same_example_assets_and_config() {
        let example = examples::find("code-review").expect("example");
        for language in [Language::Python, Language::Rust] {
            let destination = destination("generate", language);
            let _ = fs::remove_dir_all(&destination);
            init(example, Some("hermes"), language, &destination).expect("generate scaffold");
            assert!(destination.join("repo/calculator.py").is_file());
            assert!(destination.join("skills/code-review.md").is_file());
            assert!(
                destination
                    .join("adapters/hermes/fabric-adapter.json")
                    .is_file()
            );
            assert_eq!(
                fs::read_to_string(destination.join("repo/calculator.py")).expect("read workspace"),
                crate::examples::CODE_REVIEW_WORKSPACE
            );
            let launcher = match language {
                Language::Python => destination.join("main.py"),
                Language::Rust => destination.join("src/main.rs"),
            };
            let source = fs::read_to_string(launcher).expect("read launcher");
            assert!(source.contains("nvidia.fabric.hermes"));
            assert!(source.contains("nvidia/nvidia/Nemotron-3-Nano-30B-A3B"));
            fs::remove_dir_all(destination).expect("remove scaffold");
        }
    }

    #[test]
    fn refuses_to_overwrite_a_destination() {
        let destination = destination("overwrite", Language::Python);
        let _ = fs::remove_dir_all(&destination);
        fs::create_dir(&destination).expect("create destination");
        let error = init(
            examples::find("code-review").expect("example"),
            None,
            Language::Python,
            &destination,
        )
        .expect_err("must refuse overwrite");
        assert!(error.contains("already exists"));
        fs::remove_dir_all(destination).expect("remove destination");
    }
}

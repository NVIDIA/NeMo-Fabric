// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NeMo Fabric experimentation CLI.

use clap::error::ErrorKind;
use clap::{ArgGroup, Args, Parser, Subcommand};
use nemo_fabric_core::{
    ResolveContext, RunRequest, RunStatus, doctor_plan, resolve_run_plan_from_config, run_plan,
};

use crate::examples;
use crate::presets;
use crate::scaffold::{self, Language};

#[derive(Debug, Parser)]
#[command(name = "nemo-fabric")]
#[command(about = "Experiment with NeMo Fabric presets and maintained examples")]
#[command(arg_required_else_help = true)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Discover complete built-in presets.
    Preset {
        #[command(subcommand)]
        command: PresetCommand,
    },
    /// Discover and initialize maintained examples.
    Example {
        #[command(subcommand)]
        command: ExampleCommand,
    },
    /// Resolve and print a run plan as JSON.
    Plan(Selector),
    /// Diagnose a selected configuration without running it.
    Doctor(Selector),
    /// Execute one request through a selected configuration.
    Run {
        #[command(flatten)]
        selector: Selector,
        /// Inline request text.
        #[arg(long, default_value = "")]
        input: String,
    },
    /// Print the NeMo Fabric core version.
    Version,
}

#[derive(Debug, Subcommand)]
enum PresetCommand {
    /// List available presets.
    List,
    /// Describe one preset without serializing its configuration.
    Show {
        /// Preset name.
        name: String,
    },
}

#[derive(Debug, Subcommand)]
enum ExampleCommand {
    /// List maintained runnable examples.
    List,
    /// Describe one example and its complete variants.
    Show {
        /// Example name.
        name: String,
    },
    /// Generate an ordinary editable SDK application.
    Init {
        /// Example name.
        name: String,
        /// Destination directory; defaults to the example name.
        destination: Option<std::path::PathBuf>,
        /// Language API used by the generated application.
        #[arg(long, value_enum, default_value_t = Language::Python)]
        language: Language,
        /// Complete example variant rendered into source code.
        #[arg(long)]
        variant: Option<String>,
    },
}

#[derive(Debug, Args)]
#[command(group(
    ArgGroup::new("source")
        .required(true)
        .multiple(false)
        .args(["preset", "example"])
))]
struct Selector {
    /// Use a maintained, complete built-in preset.
    #[arg(long, value_name = "NAME")]
    preset: Option<String>,
    /// Use a maintained runnable example.
    #[arg(long, value_name = "NAME")]
    example: Option<String>,
    /// Select a complete example variant.
    #[arg(
        long,
        value_name = "NAME",
        requires = "example",
        conflicts_with = "preset"
    )]
    variant: Option<String>,
    /// Override the preset's default model identifier.
    #[arg(
        long,
        value_name = "MODEL",
        conflicts_with = "example",
        value_parser = clap::builder::NonEmptyStringValueParser::new()
    )]
    model: Option<String>,
    /// Override the preset's default model temperature.
    #[arg(
        long,
        value_name = "FLOAT",
        conflicts_with = "example",
        value_parser = parse_temperature
    )]
    temperature: Option<f64>,
}

fn parse_temperature(value: &str) -> Result<f64, String> {
    let temperature = value
        .parse::<f64>()
        .map_err(|error| format!("invalid temperature: {error}"))?;
    if !temperature.is_finite() {
        return Err("temperature must be finite".to_string());
    }
    Ok(temperature)
}

enum SelectedSource {
    Preset(presets::SelectedPreset),
    Example(examples::SelectedExample),
}

impl SelectedSource {
    fn config(&self) -> &nemo_fabric_core::FabricConfig {
        match self {
            Self::Preset(selected) => &selected.config,
            Self::Example(selected) => &selected.config,
        }
    }

    fn base_dir(&self) -> &std::path::Path {
        match self {
            Self::Preset(selected) => selected.base_dir(),
            Self::Example(selected) => selected.base_dir(),
        }
    }
}

/// Parse command-line arguments and execute the experimentation CLI.
pub fn execute_from<I, T>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(error)
            if matches!(
                error.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ) =>
        {
            error.print().map_err(|error| error.to_string())?;
            return Ok(());
        }
        Err(error) => return Err(error.to_string()),
    };
    run(cli).map_err(|error| error.to_string())
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Preset {
            command: PresetCommand::List,
        } => {
            for preset in presets::all() {
                println!("{:<12} {}", preset.name, preset.description);
            }
        }
        Command::Preset {
            command: PresetCommand::Show { name },
        } => {
            let preset = find_preset(&name)?;
            println!("{}\n  {}", preset.name, preset.description);
            if preset.required_env.is_empty() {
                println!("  credentials: none");
            } else {
                println!("  required environment: {}", preset.required_env.join(", "));
            }
        }
        Command::Example {
            command: ExampleCommand::List,
        } => {
            for example in examples::all() {
                println!("{:<12} {}", example.name, example.description);
            }
        }
        Command::Example {
            command: ExampleCommand::Show { name },
        } => {
            let example = find_example(&name)?;
            println!("{}\n  {}", example.name, example.description);
            println!("  default variant: {}", example.default_variant);
            println!("  variants: {}", example.variants.join(", "));
        }
        Command::Example {
            command:
                ExampleCommand::Init {
                    name,
                    destination,
                    language,
                    variant,
                },
        } => {
            let example = find_example(&name)?;
            let destination = destination.unwrap_or_else(|| std::path::PathBuf::from(&name));
            let destination = scaffold::init(example, variant.as_deref(), language, &destination)?;
            println!("created {}", destination.display());
        }
        Command::Plan(selector) => {
            let selected = select_source(&selector)?;
            let plan = resolve_run_plan_from_config(
                selected.config().clone(),
                ResolveContext::new(selected.base_dir()),
            )?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
        }
        Command::Doctor(selector) => {
            let selected = select_source(&selector)?;
            let plan = resolve_run_plan_from_config(
                selected.config().clone(),
                ResolveContext::new(selected.base_dir()),
            )?;
            println!("{}", serde_json::to_string_pretty(&doctor_plan(&plan))?);
        }
        Command::Run { selector, input } => {
            let selected = select_source(&selector)?;
            let plan = resolve_run_plan_from_config(
                selected.config().clone(),
                ResolveContext::new(selected.base_dir()),
            )?;
            let result = match run_plan(&plan, RunRequest::text(input)) {
                Ok(result) => result,
                Err(error) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "status": "failed",
                            "error": {
                                "code": "runtime_error",
                                "message": "run failed before a normalized result was available",
                            },
                        }))?
                    );
                    return Err(error.into());
                }
            };
            println!("{}", serde_json::to_string_pretty(&result)?);
            require_successful_run(result.status)?;
        }
        Command::Version => println!("{}", nemo_fabric_core::version()),
    }
    Ok(())
}

fn require_successful_run(status: RunStatus) -> Result<(), String> {
    match status {
        RunStatus::Succeeded => Ok(()),
        RunStatus::Failed => Err("run completed with status failed".to_string()),
        RunStatus::Cancelled => Err("run completed with status cancelled".to_string()),
    }
}

fn select_source(selector: &Selector) -> Result<SelectedSource, Box<dyn std::error::Error>> {
    if let Some(name) = &selector.preset {
        let mut selected = find_preset(name)?.stage()?;
        apply_model_overrides(&mut selected.config, selector)?;
        return Ok(SelectedSource::Preset(selected));
    }
    if let Some(name) = &selector.example {
        return Ok(SelectedSource::Example(
            find_example(name)?.select(selector.variant.as_deref())?,
        ));
    }
    unreachable!("Clap requires exactly one source selector")
}

fn apply_model_overrides(
    config: &mut nemo_fabric_core::FabricConfig,
    selector: &Selector,
) -> Result<(), String> {
    if selector.model.is_none() && selector.temperature.is_none() {
        return Ok(());
    }
    let default_model = config
        .models
        .get_mut("default")
        .ok_or_else(|| "the selected preset does not define a default model".to_string())?;
    if let Some(model) = &selector.model {
        default_model.model.clone_from(model);
    }
    if let Some(temperature) = selector.temperature {
        default_model.temperature = Some(temperature);
    }
    Ok(())
}

fn find_preset(name: &str) -> Result<presets::Preset, String> {
    presets::find(name).ok_or_else(|| {
        format!(
            "unknown preset {name:?}; available: {}",
            presets::all()
                .iter()
                .map(|preset| preset.name)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

fn find_example(name: &str) -> Result<examples::Example, String> {
    examples::find(name).ok_or_else(|| {
        format!(
            "unknown example {name:?}; available: {}",
            examples::all()
                .iter()
                .map(|example| example.name)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn missing_subcommand_is_an_error() {
        assert!(execute_from(["nemo-fabric"]).is_err());
    }

    #[test]
    fn explicit_help_is_successful() {
        assert!(execute_from(["nemo-fabric", "--help"]).is_ok());
    }

    #[test]
    fn failed_and_cancelled_runs_are_errors() {
        assert!(require_successful_run(RunStatus::Succeeded).is_ok());
        assert!(require_successful_run(RunStatus::Failed).is_err());
        assert!(require_successful_run(RunStatus::Cancelled).is_err());
    }

    #[test]
    fn parses_preset_plan() {
        let cli = Cli::try_parse_from(["nemo-fabric", "plan", "--preset", "scripted"])
            .expect("parse plan");
        assert!(matches!(cli.command, Command::Plan(_)));
    }

    #[test]
    fn preset_model_overrides_preserve_provider_settings() {
        let selector = Selector {
            preset: Some("hermes".to_string()),
            example: None,
            variant: None,
            model: Some("meta/llama-3.3-70b-instruct".to_string()),
            temperature: Some(0.2),
        };
        let selected = select_source(&selector).expect("select preset");
        let model = selected.config().models.get("default").expect("model");

        assert_eq!(model.provider, "nvidia");
        assert_eq!(model.model, "meta/llama-3.3-70b-instruct");
        assert_eq!(model.temperature, Some(0.2));
        assert_eq!(model.api_key_env.as_deref(), Some("NVIDIA_API_KEY"));
    }

    #[test]
    fn model_overrides_require_a_preset_with_a_default_model() {
        assert!(
            Cli::try_parse_from([
                "nemo-fabric",
                "plan",
                "--example",
                "code-review",
                "--model",
                "other-model",
            ])
            .is_err()
        );

        let selector = Selector {
            preset: Some("scripted".to_string()),
            example: None,
            variant: None,
            model: Some("other-model".to_string()),
            temperature: None,
        };
        assert_eq!(
            select_source(&selector)
                .err()
                .expect("scripted has no default model")
                .to_string(),
            "the selected preset does not define a default model"
        );

        assert!(
            Cli::try_parse_from([
                "nemo-fabric",
                "plan",
                "--preset",
                "hermes",
                "--temperature",
                "NaN",
            ])
            .is_err()
        );
    }

    #[test]
    fn parses_example_variant() {
        let cli = Cli::try_parse_from([
            "nemo-fabric",
            "run",
            "--example",
            "code-review",
            "--variant",
            "hermes",
        ])
        .expect("parse example run");
        assert!(matches!(cli.command, Command::Run { .. }));
    }

    #[test]
    fn selectors_are_mutually_exclusive() {
        assert!(
            Cli::try_parse_from([
                "nemo-fabric",
                "plan",
                "--preset",
                "scripted",
                "--example",
                "code-review",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "nemo-fabric",
                "plan",
                "--preset",
                "scripted",
                "--variant",
                "hermes",
            ])
            .is_err()
        );
    }

    #[test]
    fn plan_requires_a_preset() {
        assert!(Cli::try_parse_from(["nemo-fabric", "plan"]).is_err());
    }
}

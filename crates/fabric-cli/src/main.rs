// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NeMo Fabric command-line interface.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use fabric_core::{
    RunRequest, SchemaName, doctor_plan, generate_all_schemas, generate_schema_json,
    resolve_effective_config_with_profiles, resolve_run_plan_with_profiles, run_plan,
    validate_agent_directory, write_schema_snapshots,
};

#[derive(Debug, Parser)]
#[command(name = "fabric")]
#[command(about = "Harness-management CLI for NeMo Fabric")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate an agent directory or YAML config.
    Validate {
        /// Path to an agent directory or YAML config.
        path: PathBuf,
    },
    /// Resolve and print the effective Fabric config as JSON.
    Inspect {
        /// Path to an agent directory or YAML config.
        path: PathBuf,
        /// Profile name from configured profile directories, or a YAML profile path.
        #[arg(long = "profile")]
        profile: Vec<String>,
    },
    /// Resolve an agent/profile into a runnable plan.
    Plan {
        /// Path to an agent directory or YAML config.
        path: PathBuf,
        /// Profile name from configured profile directories, or a YAML profile path.
        #[arg(long = "profile")]
        profile: Vec<String>,
    },
    /// Diagnose a resolved agent/profile without installing or running it.
    Doctor {
        /// Path to an agent directory or YAML config.
        path: PathBuf,
        /// Profile name from configured profile directories, or a YAML profile path.
        #[arg(long = "profile")]
        profile: Vec<String>,
    },
    /// Run an agent/profile through its Fabric adapter.
    Run {
        /// Path to an agent directory or YAML config.
        path: PathBuf,
        /// Profile name from configured profile directories, or a YAML profile path.
        #[arg(long = "profile")]
        profile: Vec<String>,
        /// Inline request payload.
        #[arg(long, default_value = "")]
        input: String,
        /// Request payload file.
        #[arg(long)]
        input_file: Option<PathBuf>,
        /// Full Fabric RunRequest JSON payload.
        #[arg(long)]
        request_json: Option<String>,
        /// Full Fabric RunRequest JSON file.
        #[arg(long)]
        request_file: Option<PathBuf>,
    },
    /// Generate JSON Schema for Fabric config and runtime types.
    Schema {
        /// Schema name to print or write. Omit to include all schemas.
        #[arg(long)]
        name: Option<String>,
        /// Directory to write schema snapshot files to. Omit to print JSON.
        #[arg(long)]
        output_dir: Option<PathBuf>,
    },
    /// Print the Fabric core version.
    Version,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Validate { path }) => {
            validate_agent_directory(&path)?;
            println!("validated {}", path.display());
        }
        Some(Command::Inspect { path, profile }) => {
            let effective_config = resolve_effective_config_with_profiles(path, &profile)?;
            println!("{}", serde_json::to_string_pretty(&effective_config)?);
        }
        Some(Command::Plan { path, profile }) => {
            let plan = resolve_run_plan_with_profiles(path, &profile)?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
        }
        Some(Command::Doctor { path, profile }) => {
            let plan = resolve_run_plan_with_profiles(path, &profile)?;
            let report = doctor_plan(&plan);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Some(Command::Run {
            path,
            profile,
            input,
            input_file,
            request_json,
            request_file,
        }) => {
            let plan = resolve_run_plan_with_profiles(path, &profile)?;
            let request = match (request_file, request_json, input_file) {
                (Some(path), None, None) => {
                    serde_json::from_str::<RunRequest>(&std::fs::read_to_string(path)?)?
                }
                (None, Some(json), None) => serde_json::from_str::<RunRequest>(&json)?,
                (None, None, Some(path)) => RunRequest::text(std::fs::read_to_string(path)?),
                (None, None, None) => RunRequest::text(input),
                _ => {
                    return Err(
                        "--request-file, --request-json, and --input-file are mutually exclusive"
                            .into(),
                    );
                }
            };
            let result = run_plan(&plan, request)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            let _exit_code = result
                .metadata
                .get("exit_code")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0);
            if _exit_code != 0 {
                let message = result
                    .error
                    .as_ref()
                    .map(|error| error.message.clone())
                    .unwrap_or_else(|| format!("harness exited with an exit code of {_exit_code}"));
                return Err(message.into());
            }
        }
        Some(Command::Schema { name, output_dir }) => {
            if let Some(output_dir) = output_dir {
                if let Some(name) = name {
                    let schema = SchemaName::parse(&name)?;
                    let path = output_dir.join(schema.filename());
                    std::fs::create_dir_all(&output_dir)?;
                    std::fs::write(&path, generate_schema_json(schema)?)?;
                    println!("wrote {}", path.display());
                } else {
                    for path in write_schema_snapshots(&output_dir)? {
                        println!("wrote {}", path.display());
                    }
                }
            } else if let Some(name) = name {
                println!("{}", generate_schema_json(SchemaName::parse(&name)?)?);
            } else {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&generate_all_schemas()?)?
                );
            }
        }
        Some(Command::Version) | None => {
            println!("fabric {}", fabric_core::version());
        }
    }
    Ok(())
}

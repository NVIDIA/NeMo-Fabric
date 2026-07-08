// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NeMo Fabric command-line interface.

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver},
};
use std::thread;
use std::time::Duration;

use clap::{Parser, Subcommand};
use fabric_core::{
    AdapterKind, EffectiveConfig, RunPlan, RunRequest, RunResult, RunStatus, RuntimeHandle,
    SchemaName, doctor_plan, generate_all_schemas, generate_schema_json, invoke_runtime,
    resolve_effective_config_with_profiles, resolve_run_plan_from_effective_config,
    resolve_run_plan_with_profiles, run_plan, start_runtime, stop_runtime,
    validate_agent_directory, write_schema_snapshots,
};
use serde_json::{Map, Value};

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
        /// Ephemeral dotted config override, for example telemetry.enabled=true.
        #[arg(long = "set")]
        set: Vec<String>,
    },
    /// Resolve an agent/profile into a runnable plan.
    Plan {
        /// Path to an agent directory or YAML config.
        path: PathBuf,
        /// Profile name from configured profile directories, or a YAML profile path.
        #[arg(long = "profile")]
        profile: Vec<String>,
        /// Ephemeral dotted config override, for example telemetry.enabled=true.
        #[arg(long = "set")]
        set: Vec<String>,
    },
    /// Diagnose a resolved agent/profile without installing or running it.
    Doctor {
        /// Path to an agent directory or YAML config.
        path: PathBuf,
        /// Profile name from configured profile directories, or a YAML profile path.
        #[arg(long = "profile")]
        profile: Vec<String>,
        /// Ephemeral dotted config override, for example telemetry.enabled=true.
        #[arg(long = "set")]
        set: Vec<String>,
    },
    /// Start an interactive multi-turn runtime.
    Chat {
        /// Path to an agent directory or YAML config.
        path: PathBuf,
        /// Profile name from configured profile directories, or a YAML profile path.
        #[arg(long = "profile")]
        profile: Vec<String>,
        /// Ephemeral dotted config override, for example telemetry.enabled=true.
        #[arg(long = "set")]
        set: Vec<String>,
        /// Show per-turn runtime, invocation, artifact, and telemetry details.
        #[arg(long)]
        verbose: bool,
    },
    /// Run an agent/profile through its Fabric adapter.
    Run {
        /// Path to an agent directory or YAML config.
        path: PathBuf,
        /// Profile name from configured profile directories, or a YAML profile path.
        #[arg(long = "profile")]
        profile: Vec<String>,
        /// Ephemeral dotted config override, for example telemetry.enabled=true.
        #[arg(long = "set")]
        set: Vec<String>,
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
        /// Show adapter output.
        #[arg(long = "show-output")]
        show_output: bool,
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
        Some(Command::Inspect { path, profile, set }) => {
            let effective_config = resolve_effective_config_cli(path, &profile, &set)?;
            println!("{}", serde_json::to_string_pretty(&effective_config)?);
        }
        Some(Command::Plan { path, profile, set }) => {
            let plan = resolve_run_plan_cli(path, &profile, &set)?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
        }
        Some(Command::Doctor { path, profile, set }) => {
            let plan = resolve_run_plan_cli(path, &profile, &set)?;
            let report = doctor_plan(&plan);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Some(Command::Chat {
            path,
            profile,
            set,
            verbose,
        }) => {
            run_chat(path, &profile, &set, verbose)?;
        }
        Some(Command::Run {
            path,
            profile,
            set,
            input,
            input_file,
            request_json,
            request_file,
            show_output,
        }) => {
            let plan = resolve_run_plan_cli(path, &profile, &set)?;
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
            if show_output {
                if let Some(response) = result.output.get("response") {
                    println!("\nResponse:");
                    if let Some(response) = response.as_str() {
                        println!("{response}");
                    } else {
                        println!("{}", serde_json::to_string_pretty(response)?);
                    }
                }
            }

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

fn resolve_effective_config_cli(
    path: PathBuf,
    profiles: &[String],
    overrides: &[String],
) -> Result<EffectiveConfig, Box<dyn std::error::Error>> {
    let mut effective = resolve_effective_config_with_profiles(path, profiles)?;
    if overrides.is_empty() {
        return Ok(effective);
    }
    let mut config = serde_json::to_value(&effective.config)?;
    for override_spec in overrides {
        apply_cli_override(&mut config, override_spec)?;
    }
    effective.config = serde_json::from_value(config)?;
    effective.agent_name = effective.config.metadata.name.clone();
    Ok(effective)
}

fn resolve_run_plan_cli(
    path: PathBuf,
    profiles: &[String],
    overrides: &[String],
) -> Result<RunPlan, Box<dyn std::error::Error>> {
    if overrides.is_empty() {
        return Ok(resolve_run_plan_with_profiles(path, profiles)?);
    }
    let effective = resolve_effective_config_cli(path, profiles, overrides)?;
    Ok(resolve_run_plan_from_effective_config(effective)?)
}

fn apply_cli_override(
    config: &mut Value,
    override_spec: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some((path, raw_value)) = override_spec.split_once('=') else {
        return Err(format!("--set expects key=value, got `{override_spec}`").into());
    };
    let path = path.trim();
    if path.is_empty() {
        return Err("--set key path must not be empty".into());
    }
    let value = serde_json::from_str::<Value>(raw_value)
        .unwrap_or_else(|_| Value::String(raw_value.to_string()));
    let parts: Vec<&str> = path.split('.').filter(|part| !part.is_empty()).collect();
    if parts.is_empty() {
        return Err("--set key path must contain at least one field".into());
    }
    set_object_path(config, &parts, value)
}

fn set_object_path(
    current: &mut Value,
    parts: &[&str],
    value: Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some((head, tail)) = parts.split_first() else {
        return Err("empty --set path".into());
    };
    if tail.is_empty() {
        let object = current
            .as_object_mut()
            .ok_or_else(|| format!("cannot set `{head}` on a non-object value"))?;
        object.insert((*head).to_string(), value);
        return Ok(());
    }
    let object = current
        .as_object_mut()
        .ok_or_else(|| format!("cannot descend into `{head}` on a non-object value"))?;
    let child = object
        .entry((*head).to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if !child.is_object() {
        return Err(format!("cannot descend into non-object field `{head}`").into());
    }
    set_object_path(child, tail, value)
}

fn run_chat(
    path: PathBuf,
    profile: &[String],
    overrides: &[String],
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let plan = resolve_run_plan_cli(path, profile, overrides)?;
    let runtime = start_runtime(&plan)?;
    let chat_result = chat_loop(&plan, &runtime, verbose);
    let stop_result = stop_runtime(&plan, &runtime);
    if let Err(error) = chat_result {
        return Err(error);
    }
    stop_result?;
    Ok(())
}

fn chat_loop(
    plan: &RunPlan,
    runtime: &RuntimeHandle,
    mut verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = chat_prompt(plan, &runtime.runtime_id);
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted = Arc::clone(&interrupted);
        ctrlc::set_handler(move || {
            interrupted.store(true, Ordering::SeqCst);
        })?;
    }
    let input_is_terminal = io::stdin().is_terminal();
    let lines = stdin_lines();
    print_chat_info(plan, runtime);
    eprintln!();
    let mut turn_count = 0_u64;

    loop {
        eprint!("{prompt}> ");
        io::stderr().flush()?;

        let Some(line) = next_chat_line(&lines, &interrupted)? else {
            break;
        };
        let input = line;
        let command = input.trim();
        match command {
            "/exit" | "/quit" => break,
            "/help" => {
                print_chat_help();
                continue;
            }
            "/info" => {
                print_chat_info(plan, runtime);
                continue;
            }
            "/clear" => {
                eprint!("\x1b[2J\x1b[H");
                continue;
            }
            "/verbose" => {
                verbose = !verbose;
                eprintln!("verbose: {}", if verbose { "on" } else { "off" });
                continue;
            }
            "" => continue,
            _ => {}
        }
        if let Some(value) = command.strip_prefix("/verbose ") {
            match value.trim() {
                "on" => {
                    verbose = true;
                    eprintln!("verbose: on");
                    continue;
                }
                "off" => {
                    verbose = false;
                    eprintln!("verbose: off");
                    continue;
                }
                _ => {
                    eprintln!("usage: /verbose on|off");
                    continue;
                }
            }
        }
        if command.starts_with('/') {
            eprintln!("unknown command: {command}");
            eprintln!("type /help for available commands");
            continue;
        }

        let request = RunRequest::text(input);
        let result = invoke_runtime(plan, runtime, request)?;
        turn_count += 1;
        if !input_is_terminal {
            eprintln!();
        }
        print_chat_response(&result.output)?;
        if verbose {
            eprintln!();
            print_turn_verbose(turn_count, &result);
        }

        let exit_code = result
            .metadata
            .get("exit_code")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        if exit_code != 0 {
            let message = result
                .error
                .as_ref()
                .map(|error| error.message.clone())
                .unwrap_or_else(|| format!("harness exited with an exit code of {exit_code}"));
            return Err(message.into());
        }
    }
    Ok(())
}

fn print_chat_info(plan: &RunPlan, runtime: &RuntimeHandle) {
    eprintln!("+================================================================+");
    eprintln!("| NEMO FABRIC                                                    |");
    eprintln!("| interactive runtime                                            |");
    eprintln!("+----------------------------------------------------------------+");
    eprintln!("| agent: {}", plan.agent_name);
    eprintln!("| profile: {}", profile_label(plan));
    eprintln!("| harness: {}", runtime.harness);
    eprintln!("| adapter: {}", adapter_kind_label(runtime.adapter_kind));
    eprintln!("| runtime_id: {}", runtime.runtime_id);
    eprintln!("| commands: /help, /info, /verbose on|off, /clear, /exit, /quit");
    eprintln!("+----------------------------------------------------------------");
}

fn print_chat_help() {
    eprintln!("Commands:");
    eprintln!("  /help             show this help");
    eprintln!("  /info             show runtime info");
    eprintln!("  /verbose on|off   toggle per-turn metadata");
    eprintln!("  /clear            clear the terminal");
    eprintln!("  /exit, /quit      stop the runtime and exit");
    eprintln!("Type a non-empty message to invoke the same runtime.");
}

fn print_turn_verbose(turn: u64, result: &RunResult) {
    eprintln!("+-- turn {turn} metadata ---------------------------------------------");
    eprintln!("| status: {}", status_label(result.status));
    eprintln!("| request_id: {}", result.request_id);
    eprintln!("| invocation_id: {}", result.invocation_id);
    eprintln!("| runtime_id: {}", result.runtime_id);
    eprintln!("| artifact_count: {}", result.artifacts.artifacts.len());
    if let Some(telemetry) = result.telemetry.as_ref() {
        eprintln!("| telemetry: relay_enabled={}", telemetry.relay_enabled);
        if let Some(path) = telemetry
            .metadata
            .get("relay_config_path")
            .and_then(Value::as_str)
        {
            eprintln!("| telemetry_config: {path}");
        }
    }
    if let Some(error) = result.error.as_ref() {
        eprintln!("| error: {} {}", error.code, error.message);
    }
    eprintln!("+----------------------------------------------------------------");
}

fn chat_prompt(plan: &RunPlan, runtime_id: &str) -> String {
    format!(
        "you[{}:{}]",
        profile_label(plan),
        short_prompt_label(runtime_id)
    )
}

fn short_prompt_label(value: &str) -> String {
    let mut chars = value.chars();
    let short: String = chars.by_ref().take(24).collect();
    if chars.next().is_none() {
        short
    } else {
        format!("{short}...")
    }
}

fn profile_label(plan: &RunPlan) -> String {
    if !plan.profiles.is_empty() {
        return plan.profiles.join(", ");
    }
    "default".to_string()
}

fn adapter_kind_label(adapter_kind: AdapterKind) -> &'static str {
    match adapter_kind {
        AdapterKind::Process => "process",
        AdapterKind::Http => "http",
        AdapterKind::Python => "python",
        AdapterKind::NativePlugin => "native_plugin",
    }
}

fn status_label(status: RunStatus) -> &'static str {
    match status {
        RunStatus::Succeeded => "succeeded",
        RunStatus::Failed => "failed",
        RunStatus::Cancelled => "cancelled",
    }
}

fn stdin_lines() -> Receiver<io::Result<String>> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    receiver
}

fn next_chat_line(
    lines: &Receiver<io::Result<String>>,
    interrupted: &AtomicBool,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    loop {
        if interrupted.load(Ordering::SeqCst) {
            eprintln!();
            return Ok(None);
        }
        match lines.recv_timeout(Duration::from_millis(100)) {
            Ok(line) => return Ok(Some(line?)),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(None),
        }
    }
}

fn print_chat_response(output: &Value) -> Result<(), Box<dyn std::error::Error>> {
    let response = output.get("response").unwrap_or(output);
    if let Some(response) = response.as_str() {
        print_chat_text("agent", response);
    } else {
        let response = serde_json::to_string_pretty(response)?;
        print_chat_text("agent", &response);
    }
    Ok(())
}

fn print_chat_text(role: &str, text: &str) {
    let mut lines = text.lines();
    if let Some(first) = lines.next() {
        eprintln!("{role}> {first}");
        for line in lines {
            eprintln!("{:width$}{line}", "", width = role.len() + 2);
        }
    } else {
        eprintln!("{role}>");
    }
}

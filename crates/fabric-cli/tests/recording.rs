// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Failed recorded requests expose their exact retained journal without changing JSON stdout.

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn recorded_start_failure_prints_exact_journal_path_and_keeps_json_stdout() {
    let root = std::env::temp_dir().join(format!(
        "fabric-cli-record-path-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("artifact root");
    let output = Command::new(env!("CARGO_BIN_EXE_nemo-fabric"))
        .args(["run", "--preset", "scripted", "--record", "--artifacts"])
        .arg(&root)
        .env("PATH", root.join("no-executables"))
        .output()
        .expect("run CLI with unavailable adapter command");
    assert!(!output.status.success());
    let stdout: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("stdout stays JSON");
    assert_eq!(stdout["status"], "failed");
    assert_eq!(stdout["error"]["code"], "runtime_error");
    let stderr = String::from_utf8(output.stderr).expect("stderr text");
    let path = stderr
        .lines()
        .find_map(|line| line.strip_prefix("nemo-fabric: run file "))
        .expect("exact journal path");
    let path = std::path::Path::new(path);
    assert!(path.starts_with(&root));
    assert_eq!(
        path.file_name().and_then(|value| value.to_str()),
        Some("run.jsonl")
    );
    let text = std::fs::read_to_string(path).expect("journal exists");
    let rows: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).expect("JSONL record"))
        .collect();
    assert_eq!(rows.first().unwrap()["kind"], "request_started");
    assert_eq!(rows.last().unwrap()["kind"], "summary");
    assert!(rows.last().unwrap()["data"]["result"].is_null());
}

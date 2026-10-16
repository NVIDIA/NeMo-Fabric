// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI regression coverage for the run record switches.

use std::process::Command;

#[test]
fn run_with_record_writes_task_and_name_and_keeps_stdout_json() {
    let root = std::env::temp_dir().join(format!("fabric-cli-record-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);

    let output = Command::new(env!("CARGO_BIN_EXE_nemo-fabric"))
        .args([
            "run", "--preset", "scripted", "--input", "hello", "--record",
        ])
        .arg("--artifacts")
        .arg(&root)
        .args(["--name", "swe-scripted", "--task", "repo-io-fix"])
        .output()
        .expect("run CLI");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "scripted run failed: {stderr}");

    // The folder line comes first on standard error, and standard output is
    // only the JSON result.
    let folder = root.join(".fabric");
    assert_eq!(
        stderr.lines().next(),
        Some(format!("nemo-fabric: run record under {}", folder.display()).as_str()),
        "{stderr}"
    );
    let result: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("standard output is one JSON document");
    assert_eq!(result["status"], "succeeded");

    let records: Vec<_> = std::fs::read_dir(&folder)
        .expect("runtime folders")
        .flatten()
        .flat_map(|runtime| {
            std::fs::read_dir(runtime.path())
                .expect("invocations")
                .flatten()
        })
        .map(|invocation| invocation.path().join("run-record.jsonl"))
        .filter(|path| path.exists())
        .collect();
    assert_eq!(records.len(), 1, "{records:?}");
    let text = std::fs::read_to_string(&records[0]).expect("record");
    let start: serde_json::Value =
        serde_json::from_str(text.lines().next().expect("start line")).expect("json");
    assert_eq!(start["context"]["task_id"], "repo-io-fix");
    assert_eq!(start["agent_name"], "swe-scripted");
    assert!(
        text.lines()
            .last()
            .expect("end line")
            .contains("\"kind\":\"end\"")
    );
    let _ = std::fs::remove_dir_all(root);
}

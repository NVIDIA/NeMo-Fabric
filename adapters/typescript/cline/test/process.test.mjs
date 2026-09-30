// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

function context(workspace) {
  return {
    artifacts: {},
    environment: {
      control_location: "external_control",
      env: { TEST_API_KEY: "not-a-real-key" },
      environment_id: "environment-1",
      ownership: "caller_owned",
      provider: "local",
      workspace,
    },
    invocation_id: "start",
    request_id: "request-start",
    runtime_id: "runtime-1",
  };
}

async function exchange(workspace, requests) {
  const childEnv = { ...process.env };
  delete childEnv.NODE_TEST_CONTEXT;
  const child = spawn(process.execPath, [fileURLToPath(new URL("../dist/cli.js", import.meta.url))], {
    cwd: workspace,
    env: childEnv,
    stdio: ["pipe", "pipe", "pipe"],
  });
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => { stdout += chunk; });
  child.stderr.on("data", (chunk) => { stderr += chunk; });
  child.stdin.end(`${requests.map((request) => JSON.stringify(request)).join("\n")}\n`);
  const exitCode = await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("close", resolve);
  });
  return {
    exitCode,
    stderr,
    stdout,
    responses: stdout.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line)),
  };
}

test("returns a stable lifecycle error for an invalid Cline start", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "fabric-cline-invalid-"));
  try {
    const start = {
      operation: "start",
      payload: {
        agent_name: "cline-process-test",
        base_dir: workspace,
        config: {},
        runtime_context: context(workspace),
      },
    };
    const { exitCode, responses, stderr } = await exchange(workspace, [start]);
    assert.equal(exitCode, 0, stderr);
    assert.equal(responses[0].outcome.status, "failed");
    assert.equal(responses[0].outcome.error.code, "cline_model_required");
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { chmod, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { DeepAgentsCodeAcpSessionFactory } from "../dist/deepagents-code-acp.js";
import { DeepAgentsCodeAdapterRuntime } from "../dist/runtime.js";

const fixture = resolve(fileURLToPath(new URL("../../common/test/fixtures/acp-agent.mjs", import.meta.url)));

// A dcode stand-in on PATH: it records its arguments and environment and
// serves the shared ACP fixture.
async function fakeDcode(directory) {
  const executable = join(directory, "dcode");
  await writeFile(
    executable,
    [
      "#!/bin/sh",
      'printf "%s\\n" "$@" > "$TEST_ARGUMENTS"',
      'printf "home=%s\\nkey=%s\\nupdate=%s\\ndotenv=%s\\n" "$DEEPAGENTS_HOME" "$FABRIC_MODEL_KEY" "$DEEPAGENTS_CODE_NO_UPDATE_CHECK" "$DEEPAGENTS_CODE_READ_PROJECT_DOTENV" > "$TEST_ENVIRONMENT"',
      'exec "$TEST_NODE_EXECUTABLE" "$TEST_ACP_FIXTURE"',
      "",
    ].join("\n"),
  );
  await chmod(executable, 0o755);
}

async function startInput(settings) {
  const directory = await mkdtemp(join(tmpdir(), "fabric-deepagents-code-"));
  await fakeDcode(directory);
  process.env.FABRIC_MODEL_KEY = "model-key-value";
  return {
    directory,
    input: {
      agentName: "deepagents-code-test",
      baseDir: directory,
      config: {
        models: {
          default: {
            provider: "openai",
            model: "fabric-echo",
            base_url: "http://127.0.0.1:9/v1",
            api_key_env: "FABRIC_MODEL_KEY",
          },
        },
        harness: { settings },
      },
      runtimeContext: {
        runtime_id: "runtime-1",
        invocation_id: "start",
        request_id: "request",
        artifacts: {},
        environment: {
          environment_id: "environment",
          provider: "local",
          ownership: "caller_owned",
          control_location: "external_control",
          workspace: directory,
          env: {
            PATH: `${directory}${delimiter}${process.env.PATH}`,
            TEST_NODE_EXECUTABLE: process.execPath,
            TEST_ACP_FIXTURE: fixture,
            TEST_ARGUMENTS: join(directory, "arguments"),
            TEST_ENVIRONMENT: join(directory, "environment"),
          },
        },
      },
    },
  };
}

function runtimeDirectory(directory) {
  return join(directory, "artifacts/.fabric/deepagents-code/runtimes/runtime-1");
}

for (const [approvalMode, flags, permission] of [
  ["auto", ["--acp", "--no-tracing", "--auto-approve"], "reject"],
  ["yolo", ["--acp", "--no-tracing"], "allow"],
]) {
  test(`runs dcode over ACP with ${approvalMode} approvals and the runtime's configuration`, async () => {
    const { directory, input } = await startInput({ approval_mode: approvalMode });
    const runtime = new DeepAgentsCodeAdapterRuntime(new DeepAgentsCodeAcpSessionFactory());
    await runtime.start(input);
    let first;
    let second;
    let checks;
    try {
      first = await runtime.invoke({ input: "permission" }, input.runtimeContext);
      second = await runtime.invoke({ input: "again" }, input.runtimeContext);
      checks = await Promise.all(["live", "active", "ready"].map((level) => runtime.check(level)));
    } finally {
      await runtime.stop();
    }
    const home = join(runtimeDirectory(directory), "deepagents");

    assert.equal(first.status, "succeeded");
    assert.match(first.output.response, new RegExp(`^permission:1:.*:permission=${permission}:`, "u"));
    assert.match(second.output.response, /^again:2:/u);
    assert.equal(first.extensions.session_id, second.extensions.session_id);
    // Only process liveness is observable, so only live is claimed.
    assert.deepEqual(checks.map((check) => check.outcome), ["passed", "unsupported", "unsupported"]);
    assert.deepEqual((await readFile(join(directory, "arguments"), "utf8")).trim().split("\n"), flags);
    assert.equal(
      await readFile(join(directory, "environment"), "utf8"),
      `home=${home}\nkey=model-key-value\nupdate=1\ndotenv=0\n`,
    );
    assert.match(await readFile(join(home, "config.toml"), "utf8"), /api_key_env = "FABRIC_MODEL_KEY"/u);
    assert.doesNotMatch(await readFile(join(home, "config.toml"), "utf8"), /model-key-value/u);
  });
}

test("reports dcode missing from PATH", async () => {
  const { input } = await startInput({ approval_mode: "auto" });
  input.runtimeContext.environment.env.PATH = "/nonexistent";

  await assert.rejects(
    new DeepAgentsCodeAdapterRuntime(new DeepAgentsCodeAcpSessionFactory()).start(input),
    { code: "deepagents_code_harness_unavailable" },
  );
});

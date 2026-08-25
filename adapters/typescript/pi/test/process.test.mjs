// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { assertSupportedNodeVersion } from "../dist/node-version.js";

const [major, minor] = process.versions.node.split(".").map(Number);
const supportsPi = major > 22 || (major === 22 && minor >= 19);

function context(workspace, invocationId) {
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
    invocation_id: invocationId,
    request_id: `request-${invocationId}`,
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
  child.stdout.on("data", (chunk) => {
    stdout += chunk;
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk;
  });
  child.stdin.end(`${requests.map((request) => JSON.stringify(request)).join("\n")}\n`);

  const exitCode = await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("close", resolve);
  });
  const responses = stdout.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
  return { exitCode, responses, stderr, stdout };
}

test("enforces the package Node.js engine floor", () => {
  assert.doesNotThrow(() => assertSupportedNodeVersion("22.19.0", ">=22.19.0"));
  assert.doesNotThrow(() => assertSupportedNodeVersion("24.0.0", ">=22.19.0"));
  assert.throws(
    () => assertSupportedNodeVersion("22.18.9", ">=22.19.0"),
    /requires Node\.js >=22\.19\.0/,
  );
  assert.throws(
    () => assertSupportedNodeVersion("22.19.0-rc.1", ">=22.19.0"),
    /requires Node\.js >=22\.19\.0/,
  );
  assert.throws(
    () => assertSupportedNodeVersion("22.19.0", "^22.19.0"),
    /cannot enforce Node\.js engine requirement/,
  );
});

test(
  "launches the Pi process host and loads an explicit extension tool",
  { skip: supportsPi ? false : "Pi 0.84.2 requires Node 22.19 or newer" },
  async () => {
    const baseDir = await mkdtemp(join(tmpdir(), "fabric-pi-process-"));
    const workspace = join(baseDir, "workspace");
    try {
      await mkdir(join(baseDir, "skills", "code-review"), { recursive: true });
      await mkdir(workspace);
      await writeFile(
        join(baseDir, "skills", "code-review", "SKILL.md"),
        `---
name: code-review
description: Review code for correctness risks.
---

# Code Review

Read the implementation before reporting findings.
`,
        "utf8",
      );
      await writeFile(
        join(workspace, "extension.js"),
        `export default function (pi) {
  console.log("extension-loaded");
  pi.registerTool({
    name: "test_echo",
    label: "Test Echo",
    description: "Echo text for a process-host smoke test",
    parameters: {
      type: "object",
      properties: { text: { type: "string" } },
      required: ["text"],
      additionalProperties: false
    },
    async execute(_toolCallId, params) {
      return { content: [{ type: "text", text: params.text }], details: {} };
    }
  });
}
`,
        "utf8",
      );
      await writeFile(
        join(workspace, "fabric-tool.ts"),
        `export default function ({ name }) {
  return {
    name,
    label: "Fabric Echo",
    description: "Echo text from a Fabric tool definition",
    parameters: {
      type: "object",
      properties: { text: { type: "string" } },
      required: ["text"],
      additionalProperties: false
    },
    async execute(_toolCallId, params) {
      return { content: [{ type: "text", text: params.text }], details: {} };
    }
  };
}
`,
        "utf8",
      );

      const start = {
        operation: "start",
        payload: {
          agent_name: "pi-process-test",
          base_dir: baseDir,
          config: {
            harness: { settings: { extensions: ["extension.js"] } },
            models: {
              default: {
                api_key_env: "TEST_API_KEY",
                model: "gpt-4.1-mini",
                provider: "openai",
              },
            },
            skills: { paths: ["skills/code-review"] },
            tools: {
              definitions: {
                fabric_echo: { kind: "module", ref: "fabric-tool.ts" },
              },
              enabled: ["test_echo", "fabric_echo"],
            },
          },
          runtime_context: context(workspace, "start"),
        },
      };
      const stop = { operation: "stop", payload: { runtime_id: "runtime-1" } };
      const { exitCode, responses, stderr, stdout } = await exchange(workspace, [start, stop]);

      assert.equal(exitCode, 0, stderr);
      assert.equal(responses.length, 2, `stdout:\n${stdout}\nstderr:\n${stderr}`);
      assert.equal(responses[0].operation, "start");
      assert.equal(responses[0].outcome.status, "succeeded");
      assert.equal(responses[1].operation, "stop");
      assert.equal(responses[1].outcome.status, "succeeded");
      assert.match(stderr, /extension-loaded/);
      assert.doesNotMatch(stdout, /extension-loaded/);
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  },
);

test(
  "rejects an explicitly configured missing skill before session creation",
  { skip: supportsPi ? false : "Pi 0.84.2 requires Node 22.19 or newer" },
  async () => {
    const baseDir = await mkdtemp(join(tmpdir(), "fabric-pi-missing-skill-"));
    const workspace = join(baseDir, "workspace");
    try {
      await mkdir(workspace);
      const start = {
        operation: "start",
        payload: {
          agent_name: "pi-missing-skill-test",
          base_dir: baseDir,
          config: {
            models: {
              default: {
                api_key_env: "TEST_API_KEY",
                model: "gpt-4.1-mini",
                provider: "openai",
              },
            },
            skills: { paths: ["skills/missing"] },
            tools: { enabled: [] },
          },
          runtime_context: context(workspace, "start"),
        },
      };
      const { exitCode, responses, stderr } = await exchange(workspace, [start]);

      assert.equal(exitCode, 0, stderr);
      assert.equal(responses.length, 1);
      assert.equal(responses[0].operation, "start");
      assert.equal(responses[0].outcome.status, "failed");
      assert.equal(responses[0].outcome.error.code, "pi_skill_not_found");
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  },
);

test(
  "dispatches extension commands and rejects work after extension shutdown",
  { skip: supportsPi ? false : "Pi 0.84.2 requires Node 22.19 or newer" },
  async () => {
    const baseDir = await mkdtemp(join(tmpdir(), "fabric-pi-shutdown-"));
    const workspace = join(baseDir, "workspace");
    try {
      await mkdir(workspace);
      await writeFile(
        join(workspace, "shutdown.js"),
        `export default function (pi) {
  pi.registerCommand("shutdown-test", {
    description: "Request adapter shutdown",
    handler: async (_args, ctx) => ctx.shutdown()
  });
}
`,
        "utf8",
      );
      const start = {
        operation: "start",
        payload: {
          agent_name: "pi-shutdown-test",
          base_dir: baseDir,
          config: {
            harness: { settings: { extensions: ["shutdown.js"] } },
            models: {
              default: {
                api_key_env: "TEST_API_KEY",
                model: "gpt-4.1-mini",
                provider: "openai",
              },
            },
            tools: { enabled: [] },
          },
          runtime_context: context(workspace, "start"),
        },
      };
      const invoke = (invocationId, input) => ({
        operation: "invoke",
        payload: {
          request: { input },
          runtime_context: context(workspace, invocationId),
        },
      });
      const stop = { operation: "stop", payload: { runtime_id: "runtime-1" } };

      const { exitCode, responses, stderr } = await exchange(workspace, [
        start,
        invoke("shutdown", "/shutdown-test"),
        invoke("after-shutdown", "do not run"),
        stop,
      ]);

      assert.equal(exitCode, 0, stderr);
      assert.equal(responses.length, 4);
      assert.equal(responses[1].outcome.output.status, "cancelled");
      assert.equal(responses[1].outcome.output.error.code, "pi_extension_shutdown");
      assert.equal(responses[2].outcome.error.code, "pi_runtime_unusable");
      assert.equal(responses[3].outcome.status, "succeeded");
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  },
);

test(
  "returns a stable error for a missing configured extension",
  { skip: supportsPi ? false : "Pi 0.84.2 requires Node 22.19 or newer" },
  async () => {
    const baseDir = await mkdtemp(join(tmpdir(), "fabric-pi-missing-extension-"));
    const workspace = join(baseDir, "workspace");
    try {
      await mkdir(workspace);
      const start = {
        operation: "start",
        payload: {
          agent_name: "pi-missing-extension-test",
          base_dir: baseDir,
          config: {
            harness: { settings: { extensions: ["missing.ts"] } },
            models: {
              default: {
                api_key_env: "TEST_API_KEY",
                model: "gpt-4.1-mini",
                provider: "openai",
              },
            },
            tools: { enabled: [] },
          },
          runtime_context: context(workspace, "start"),
        },
      };

      const { exitCode, responses, stderr } = await exchange(workspace, [start]);

      assert.equal(exitCode, 0, stderr);
      assert.equal(responses.length, 1);
      assert.equal(responses[0].outcome.error.code, "pi_extension_not_found");
    } finally {
      await rm(baseDir, { recursive: true, force: true });
    }
  },
);

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { QwenAdapterRuntime } from "../dist/runtime.js";

function startInput(id = "runtime-1") {
  return {
    agentName: "qwen-test", baseDir: "/tmp", config: {},
    runtimeContext: {
      artifacts: {},
      environment: { control_location: "external_control", environment_id: "env-1", ownership: "caller_owned", provider: "local", workspace: "/tmp" },
      invocation_id: "start", request_id: "start", runtime_id: id,
    },
  };
}

test("retains a session for dependent invocations and closes once", async () => {
  const prompts = [];
  let closes = 0;
  const runtime = new QwenAdapterRuntime({ async create() {
    return { async prompt(text) { prompts.push(text); return { text: prompts.join(" "), usage: { input_tokens: 1, output_tokens: 2 } }; }, async close() { closes++; } };
  } });
  await runtime.start(startInput());
  assert.deepEqual(await runtime.invoke({ input: "one" }, startInput().runtimeContext), {
    status: "succeeded", output: { response: "one" }, usage: { input_tokens: 1, output_tokens: 2 },
  });
  assert.deepEqual((await runtime.invoke({ input: "two" }, startInput().runtimeContext)).output, { response: "one two" });
  await runtime.stop();
  await runtime.stop();
  assert.equal(closes, 1);
});

test("normalizes terminal failures without leaking upstream error text", async () => {
  const runtime = new QwenAdapterRuntime({ async create() {
    return { async prompt() { return { error: "error_during_execution", usage: { input_tokens: 3 } }; }, async close() {} };
  } });
  await runtime.start(startInput());
  const result = await runtime.invoke({ input: "fail" }, startInput().runtimeContext);
  assert.equal(result.status, "failed");
  assert.equal(result.error.code, "qwen_execution_failed");
  assert.deepEqual(result.usage, { input_tokens: 3 });
});

test("returns a stable failure when a configured MCP server is unavailable", async () => {
  const runtime = new QwenAdapterRuntime({ async create() {
    return { async prompt() { return { error: "error_mcp_unavailable" }; }, async close() {} };
  } });
  await runtime.start(startInput());
  const result = await runtime.invoke({ input: "use MCP" }, startInput().runtimeContext);
  assert.deepEqual(result, {
    status: "failed",
    output: null,
    error: {
      code: "qwen_mcp_unavailable",
      message: "A configured Qwen MCP server was unavailable",
      retryable: false,
    },
  });
});

test("invalidates a broken session and keeps stop safe", async () => {
  let closes = 0;
  const runtime = new QwenAdapterRuntime({ async create() {
    return { async prompt() { throw new Error("secret upstream detail"); }, async close() { closes++; } };
  } });
  await runtime.start(startInput());
  await assert.rejects(runtime.invoke({ input: "fail" }, startInput().runtimeContext),
    (error) => error.code === "qwen_session_failed" && !error.message.includes("secret"));
  await runtime.stop();
  assert.equal(closes, 1);
});

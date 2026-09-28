// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { LifecycleError } from "nemo-fabric-adapters-common";

import { ClineAdapterRuntime } from "../dist/runtime.js";

function startInput(runtimeId = "runtime-1") {
  return {
    agentName: "cline-test",
    baseDir: "/tmp",
    config: {},
    runtimeContext: {
      artifacts: {},
      environment: {
        control_location: "external_control",
        environment_id: "environment-1",
        ownership: "caller_owned",
        provider: "local",
        workspace: "/tmp",
      },
      invocation_id: "start",
      request_id: "request-start",
      runtime_id: runtimeId,
    },
  };
}

const context = startInput().runtimeContext;

test("enforces lifecycle ordering", async () => {
  const runtime = new ClineAdapterRuntime({
    async create() {
      return { async prompt() { return { status: "completed", text: "ok" }; }, async stop() {} };
    },
  });
  await assert.rejects(runtime.invoke({ input: "early" }, context), (error) => error.code === "cline_not_started");
  await runtime.start(startInput());
  await assert.rejects(runtime.start(startInput()), (error) => error.code === "cline_already_started");
  await runtime.stop();
});

test("reuses one session for ordered invocations", async () => {
  const prompts = [];
  const runtime = new ClineAdapterRuntime({
    async create() {
      return {
        async prompt(text) {
          prompts.push(text);
          return {
            status: "completed",
            text: prompts.length === 1 ? "remembered alpha" : `context:${prompts[0]}`,
            usage: { input_tokens: 3, output_tokens: 2, total_tokens: 5, cost_usd: 0.01 },
            extensions: { session_id: "session-1", finish_reason: "completed" },
          };
        },
        async stop() {},
      };
    },
  });
  await runtime.start(startInput());
  const first = await runtime.invoke({ input: "alpha" }, context);
  const second = await runtime.invoke({ input: "what did I say?" }, context);
  assert.equal(first.status, "succeeded");
  assert.deepEqual(second.output, { response: "context:alpha" });
  assert.deepEqual(second.extensions, { session_id: "session-1", finish_reason: "completed" });
  assert.deepEqual(prompts, ["alpha", "what did I say?"]);
});

test("isolates independent runtimes", async () => {
  const sessions = [];
  const factory = {
    async create(input) {
      const prompts = [];
      sessions.push({ runtimeId: input.runtimeContext.runtime_id, prompts });
      return {
        async prompt(text) {
          prompts.push(text);
          return { status: "completed", text: `${input.runtimeContext.runtime_id}:${prompts.join(",")}` };
        },
        async stop() {},
      };
    },
  };
  const first = new ClineAdapterRuntime(factory);
  const second = new ClineAdapterRuntime(factory);
  await first.start(startInput("runtime-one"));
  await second.start(startInput("runtime-two"));
  await first.invoke({ input: "one" }, { ...context, runtime_id: "runtime-one" });
  const result = await second.invoke({ input: "two" }, { ...context, runtime_id: "runtime-two" });
  assert.deepEqual(result.output, { response: "runtime-two:two" });
  assert.deepEqual(sessions, [
    { runtimeId: "runtime-one", prompts: ["one"] },
    { runtimeId: "runtime-two", prompts: ["two"] },
  ]);
});

test("normalizes Cline failures, cancellation, malformed output, and non-text input", async () => {
  const outcomes = [
    { status: "failed", errorMessage: "provider failed" },
    { status: "aborted" },
    { status: "completed", extensions: { session_id: "session-1", finish_reason: "completed" } },
  ];
  let calls = 0;
  const runtime = new ClineAdapterRuntime({
    async create() {
      return {
        async prompt() { calls += 1; return outcomes.shift(); },
        async stop() {},
      };
    },
  });
  await runtime.start(startInput());
  const unsupported = await runtime.invoke({ input: { task: "no" } }, context);
  const failed = await runtime.invoke({ input: "one" }, context);
  const cancelled = await runtime.invoke({ input: "two" }, context);
  const malformed = await runtime.invoke({ input: "three" }, context);
  assert.equal(unsupported.error.code, "cline_unsupported_input");
  assert.equal(failed.error.code, "cline_model_error");
  assert.equal(cancelled.status, "cancelled");
  assert.equal(malformed.error.code, "cline_no_assistant_response");
  assert.deepEqual(malformed.extensions, { session_id: "session-1", finish_reason: "completed" });
  assert.equal(calls, 3);
});

test("invalidates a session after a process or SDK failure", async () => {
  let stops = 0;
  const runtime = new ClineAdapterRuntime({
    async create() {
      return {
        async prompt() { throw new Error("connection lost"); },
        async stop() { stops += 1; },
      };
    },
  });
  await runtime.start(startInput());
  await assert.rejects(
    runtime.invoke({ input: "fail" }, context),
    (error) => error instanceof LifecycleError && error.code === "cline_session_failed" && error.retryable,
  );
  await assert.rejects(runtime.invoke({ input: "again" }, context), (error) => error.code === "cline_not_started");
  assert.equal(stops, 1);
});

test("stop is safe before start and idempotent after start", async () => {
  let stops = 0;
  const runtime = new ClineAdapterRuntime({
    async create() {
      return { async prompt() { return { status: "completed", text: "ok" }; }, async stop() { stops += 1; } };
    },
  });
  await runtime.stop();
  await runtime.start(startInput());
  await runtime.stop();
  await runtime.stop();
  assert.equal(stops, 1);
});

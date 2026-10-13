// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { LifecycleError } from "nemo-fabric-adapters-common";

import { DroidAdapterRuntime } from "../dist/runtime.js";

function startInput(runtimeId = "runtime-1") {
  return {
    agentName: "droid-test",
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

test("enforces lifecycle ordering and idempotent stop", async () => {
  let stops = 0;
  const runtime = new DroidAdapterRuntime({
    async create() {
      return { async prompt() { return { status: "completed", text: "ok" }; }, async stop() { stops += 1; } };
    },
  });
  await assert.rejects(runtime.invoke({ input: "early" }, context), (error) => error.code === "droid_not_started");
  await runtime.stop();
  await runtime.start(startInput());
  await assert.rejects(runtime.start(startInput()), (error) => error.code === "droid_already_started");
  await runtime.stop();
  await runtime.stop();
  assert.equal(stops, 1);
});

test("reuses one session for ordered invocations", async () => {
  const prompts = [];
  const runtime = new DroidAdapterRuntime({
    async create() {
      return {
        async prompt(text) {
          prompts.push(text);
          return {
            status: "completed",
            text: prompts.length === 1 ? "remembered alpha" : `context:${prompts[0]}`,
            extensions: { session_id: "session-1", finish_reason: "success", duration_ms: 1, turn_count: prompts.length },
          };
        },
        async stop() {},
      };
    },
  });
  await runtime.start(startInput());
  await runtime.invoke({ input: "alpha" }, context);
  const second = await runtime.invoke({ input: "what did I say?" }, context);
  assert.deepEqual(second.output, { response: "context:alpha" });
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
  const first = new DroidAdapterRuntime(factory);
  const second = new DroidAdapterRuntime(factory);
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

test("normalizes failures, interruption, malformed output, and non-text input", async () => {
  const outcomes = [
    { status: "failed", errorMessage: "provider failed" },
    { status: "interrupted" },
    { status: "completed" },
  ];
  let calls = 0;
  const runtime = new DroidAdapterRuntime({
    async create() {
      return {
        async prompt() { calls += 1; return outcomes.shift(); },
        async stop() {},
      };
    },
  });
  await runtime.start(startInput());
  assert.equal((await runtime.invoke({ input: { task: "no" } }, context)).error.code, "droid_unsupported_input");
  assert.equal((await runtime.invoke({ input: "one" }, context)).error.code, "droid_invocation_failed");
  assert.equal((await runtime.invoke({ input: "two" }, context)).status, "cancelled");
  assert.equal((await runtime.invoke({ input: "three" }, context)).error.code, "droid_no_assistant_response");
  assert.equal(calls, 3);
});

test("invalidates a runtime after an SDK or process failure", async () => {
  let stops = 0;
  const runtime = new DroidAdapterRuntime({
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
    (error) => error instanceof LifecycleError && error.code === "droid_session_failed" && error.retryable,
  );
  await assert.rejects(runtime.invoke({ input: "again" }, context), (error) => error.code === "droid_not_started");
  assert.equal(stops, 1);
});

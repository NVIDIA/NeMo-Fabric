// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { KiroAdapterRuntime } from "../dist/runtime.js";

const context = {
  runtime_id: "runtime",
  invocation_id: "invocation",
  request_id: "request",
  artifacts: {},
  environment: {
    environment_id: "environment",
    provider: "local",
    ownership: "caller_owned",
    control_location: "external_control",
    env: {},
  },
};

test("keeps one Kiro session across ordered invocations", async () => {
  const prompts = [];
  let stopped = 0;
  const runtime = new KiroAdapterRuntime({
    create: async () => ({
      id: "session",
      prompt: async (text) => {
        prompts.push(text);
        return {
          text: `answer:${text}`,
          stopReason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 2, total_tokens: 3 },
        };
      },
      stop: async () => {
        stopped += 1;
      },
    }),
  });
  await runtime.start({ agentName: "agent", baseDir: ".", config: {}, runtimeContext: context });
  assert.equal((await runtime.invoke({ input: "one" }, context)).output.response, "answer:one");
  assert.equal((await runtime.invoke({ input: "two" }, context)).output.response, "answer:two");
  await runtime.stop();
  assert.deepEqual(prompts, ["one", "two"]);
  assert.equal(stopped, 1);
});

test("normalizes terminal stop reasons and malformed output", async () => {
  const outcomes = [
    { text: "", stopReason: "refusal" },
    { text: "", stopReason: "end_turn" },
    { text: "", stopReason: "cancelled" },
  ];
  const runtime = new KiroAdapterRuntime({
    create: async () => ({ id: "session", prompt: async () => outcomes.shift(), stop: async () => {} }),
  });
  await runtime.start({ agentName: "agent", baseDir: ".", config: {}, runtimeContext: context });
  assert.equal((await runtime.invoke({ input: "one" }, context)).error.code, "kiro_refusal");
  assert.equal((await runtime.invoke({ input: "two" }, context)).error.code, "kiro_no_assistant_response");
  assert.equal((await runtime.invoke({ input: "three" }, context)).status, "cancelled");
});

test("invalidates a Kiro session after a transport failure", async () => {
  let stopped = 0;
  const runtime = new KiroAdapterRuntime({
    create: async () => ({
      id: "session",
      prompt: async () => {
        throw new Error("private transport detail");
      },
      stop: async () => {
        stopped += 1;
      },
    }),
  });
  await runtime.start({ agentName: "agent", baseDir: ".", config: {}, runtimeContext: context });
  await assert.rejects(
    runtime.invoke({ input: "one" }, context),
    (error) => error.code === "kiro_session_failed" && error.retryable === false,
  );
  await assert.rejects(runtime.invoke({ input: "two" }, context), (error) => error.code === "kiro_not_started");
  assert.equal(stopped, 1);
});

test("rejects an expired deadline before dispatch and keeps the session usable", async () => {
  const prompts = [];
  const runtime = new KiroAdapterRuntime({
    create: async () => ({
      id: "session",
      prompt: async (text) => {
        prompts.push(text);
        return { text: `answer:${text}`, stopReason: "end_turn" };
      },
      stop: async () => {},
    }),
  });
  await runtime.start({ agentName: "agent", baseDir: ".", config: {}, runtimeContext: context });

  const expired = await runtime.invoke(
    { input: "expired" },
    { ...context, deadline_millis: Date.now() - 1 },
  );
  assert.deepEqual(expired, {
    status: "failed",
    output: null,
    error: {
      code: "kiro_deadline_exceeded",
      message: "The Kiro invocation deadline has already elapsed",
      retryable: false,
    },
  });
  assert.deepEqual(prompts, []);
  assert.equal((await runtime.invoke({ input: "next" }, context)).output.response, "answer:next");
  assert.deepEqual(prompts, ["next"]);
});

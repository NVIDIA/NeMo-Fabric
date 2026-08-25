// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { PiAdapterRuntime } from "../dist/runtime.js";

function startInput() {
  return {
    agentName: "pi-test",
    baseDir: "/tmp",
    config: {},
    runtimeContext: {
      artifacts: {},
      environment: {
        control_location: "external_control",
        environment_id: "environment-1",
        ownership: "caller_owned",
        provider: "local",
      },
      invocation_id: "start",
      request_id: "request-start",
      runtime_id: "runtime-1",
    },
  };
}

const context = startInput().runtimeContext;

test("enforces the runtime start and invoke lifecycle guards", async () => {
  const runtime = new PiAdapterRuntime({
    async create() {
      return {
        async prompt() {
          return { accepted: true, text: "ok" };
        },
        async stop() {},
      };
    },
  });

  await assert.rejects(
    runtime.invoke({ input: "before start" }, context),
    (error) => error.code === "pi_not_started",
  );
  await runtime.start(startInput());
  await assert.rejects(runtime.start(startInput()), (error) => error.code === "pi_already_started");
  await runtime.stop();
});

test("normalizes successful plain-text prompts and reuses one session", async () => {
  const prompts = [];
  let stopped = false;
  const runtime = new PiAdapterRuntime({
    async create() {
      return {
        async prompt(text) {
          prompts.push(text);
          return { accepted: true, text: `reply:${text}`, stopReason: "stop" };
        },
        async stop() {
          stopped = true;
        },
      };
    },
  });

  await runtime.start(startInput());
  const first = await runtime.invoke({ input: "one" }, context);
  const second = await runtime.invoke({ input: "two" }, context);
  await runtime.stop();

  assert.deepEqual(prompts, ["one", "two"]);
  assert.deepEqual(first, { status: "succeeded", output: { response: "reply:one" } });
  assert.deepEqual(second, { status: "succeeded", output: { response: "reply:two" } });
  assert.equal(stopped, true);
});

test("does not invoke Pi again after an extension requests shutdown", async () => {
  let promptCount = 0;
  const runtime = new PiAdapterRuntime({
    async create() {
      return {
        async prompt() {
          promptCount += 1;
          return { accepted: true, stopReason: "aborted", shutdownRequested: true };
        },
        async stop() {},
      };
    },
  });
  await runtime.start(startInput());

  const first = await runtime.invoke({ input: "stop" }, context);
  assert.equal(first.status, "cancelled");
  assert.equal(first.error.code, "pi_extension_shutdown");
  await assert.rejects(
    runtime.invoke({ input: "again" }, context),
    (error) => error.code === "pi_runtime_unusable",
  );
  assert.equal(promptCount, 1);
});

test("rejects non-text input without invoking Pi", async () => {
  let prompted = false;
  const runtime = new PiAdapterRuntime({
    async create() {
      return {
        async prompt() {
          prompted = true;
          return { accepted: true, text: "unexpected" };
        },
        async stop() {},
      };
    },
  });
  await runtime.start(startInput());
  const result = await runtime.invoke({ input: { task: "not yet supported" } }, context);
  assert.equal(result.status, "failed");
  assert.equal(result.error.code, "pi_unsupported_input");
  assert.equal(prompted, false);
});

test("normalizes rejected, failed, empty, and aborted Pi outcomes", async () => {
  const outcomes = [
    { accepted: false },
    { accepted: true, stopReason: "error", errorMessage: "provider rejected the request" },
    { accepted: true, stopReason: "stop" },
    { accepted: true, stopReason: "aborted" },
  ];
  const runtime = new PiAdapterRuntime({
    async create() {
      return {
        async prompt() {
          return outcomes.shift();
        },
        async stop() {},
      };
    },
  });
  await runtime.start(startInput());

  const rejected = await runtime.invoke({ input: "one" }, context);
  const failed = await runtime.invoke({ input: "two" }, context);
  const empty = await runtime.invoke({ input: "three" }, context);
  const aborted = await runtime.invoke({ input: "four" }, context);

  assert.equal(rejected.status, "failed");
  assert.equal(rejected.error.code, "pi_prompt_rejected");
  assert.equal(failed.error.code, "pi_model_error");
  assert.equal(failed.error.message, "provider rejected the request");
  assert.equal(empty.error.code, "pi_no_assistant_response");
  assert.equal(aborted.status, "cancelled");
  assert.equal(aborted.error.code, "pi_aborted");
});

test("stop is safe before start and idempotent after start", async () => {
  let stopCount = 0;
  const runtime = new PiAdapterRuntime({
    async create() {
      return {
        async prompt() {
          return { accepted: true, text: "ok" };
        },
        async stop() {
          stopCount += 1;
        },
      };
    },
  });
  await runtime.stop();
  await runtime.start(startInput());
  await runtime.stop();
  await runtime.stop();
  assert.equal(stopCount, 1);
});

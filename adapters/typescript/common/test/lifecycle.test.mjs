// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { connect } from "node:net";
import { PassThrough } from "node:stream";
import test from "node:test";

import { serve } from "../dist/index.js";

function context(runtimeId, invocationId = "invocation-1") {
  return {
    artifacts: {},
    environment: {
      control_location: "external_control",
      environment_id: "environment-1",
      ownership: "caller_owned",
      provider: "local",
    },
    invocation_id: invocationId,
    request_id: `request-${invocationId}`,
    runtime_id: runtimeId,
  };
}

function start(runtimeId) {
  return {
    operation: "start",
    payload: {
      agent_name: "test-agent",
      base_dir: "/tmp",
      config: {},
      runtime_context: context(runtimeId, "start"),
    },
  };
}

function invoke(runtimeId, input, invocationId) {
  return {
    operation: "invoke",
    payload: {
      request: { input },
      runtime_context: context(runtimeId, invocationId),
    },
  };
}

function stop(runtimeId) {
  return { operation: "stop", payload: { runtime_id: runtimeId } };
}

async function exchange(factory, messages) {
  const input = new PassThrough();
  const output = new PassThrough();
  const diagnostics = new PassThrough();
  let encoded = "";
  output.setEncoding("utf8");
  output.on("data", (chunk) => {
    encoded += chunk;
  });
  const serving = serve(factory, { input, output, diagnostics });
  for (const message of messages) {
    input.write(`${JSON.stringify(message)}\n`);
  }
  input.end();
  await serving;
  return encoded.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
}

function responseReader(output) {
  let encoded = "";
  let wake;
  output.setEncoding("utf8");
  output.on("data", (chunk) => {
    encoded += chunk;
    wake?.();
    wake = undefined;
  });
  return async () => {
    while (!encoded.includes("\n")) {
      await new Promise((resolve) => {
        wake = resolve;
      });
    }
    const newline = encoded.indexOf("\n");
    const line = encoded.slice(0, newline);
    encoded = encoded.slice(newline + 1);
    return JSON.parse(line);
  };
}

async function checkHealth(control, runtimeId, timeoutMillis = 1000) {
  const socket = connect(control.port, control.host);
  socket.setEncoding("utf8");
  let encoded = "";
  const response = new Promise((resolve, reject) => {
    socket.on("data", (chunk) => {
      encoded += chunk;
      const newline = encoded.indexOf("\n");
      if (newline >= 0) {
        resolve(JSON.parse(encoded.slice(0, newline)));
      }
    });
    socket.once("error", reject);
    socket.once("end", () => {
      if (!encoded.includes("\n")) {
        reject(new Error("health connection ended without a response"));
      }
    });
  });
  await new Promise((resolve) => socket.once("connect", resolve));
  socket.write(`${JSON.stringify({
    protocol_version: "fabric.health/v1alpha1",
    token: control.token,
    runtime_id: runtimeId,
    timeout_millis: timeoutMillis,
  })}\n`);
  return response;
}

test("serves health independently while an invocation is busy", async () => {
  const input = new PassThrough();
  const output = new PassThrough();
  const diagnostics = new PassThrough();
  const nextResponse = responseReader(output);
  let releaseInvocation;
  let invocationStarted;
  const started = new Promise((resolve) => {
    invocationStarted = resolve;
  });
  const blocked = new Promise((resolve) => {
    releaseInvocation = resolve;
  });
  const runtime = {
    async start() {},
    async invoke() {
      invocationStarted();
      await blocked;
      return { status: "succeeded", output: null };
    },
    async stop() {},
  };
  const serving = serve(() => runtime, { input, output, diagnostics });
  input.write(`${JSON.stringify(start("runtime-1"))}\n`);
  const startResponse = await nextResponse();
  const control = startResponse.outcome.output.health_control;

  input.write(`${JSON.stringify(invoke("runtime-1", "one", "one"))}\n`);
  await started;
  const healthResponse = await checkHealth(control, "runtime-1");

  assert.equal(healthResponse.result.readiness.state, "not_ready");
  assert.equal(healthResponse.result.readiness.reason_code, "invocation_in_progress");
  assert.deepEqual(
    healthResponse.result.checks.map((check) => [check.name, check.status]),
    [
      ["dependency.inference", "unsupported"],
      ["adapter.health", "unsupported"],
    ],
  );

  releaseInvocation();
  await nextResponse();
  input.write(`${JSON.stringify(stop("runtime-1"))}\n`);
  await nextResponse();
  input.end();
  await serving;
});

test("serves two ordered invocations and stops one runtime", async () => {
  const calls = [];
  const runtime = {
    async start(input) {
      calls.push(["start", input.agentName]);
    },
    async invoke(request) {
      calls.push(["invoke", request.input]);
      return { status: "succeeded", output: { response: String(request.input) } };
    },
    async stop() {
      calls.push(["stop"]);
    },
  };

  const responses = await exchange(
    () => runtime,
    [start("runtime-1"), invoke("runtime-1", "one", "one"), invoke("runtime-1", "two", "two"), stop("runtime-1")],
  );

  assert.deepEqual(calls, [
    ["start", "test-agent"],
    ["invoke", "one"],
    ["invoke", "two"],
    ["stop"],
  ]);
  assert.equal(responses.length, 4);
  assert.equal(responses[2].outcome.output.output.response, "two");
});

test("rejects invalid typed configuration before adapter startup", async () => {
  let started = false;
  let stopped = false;
  const request = start("runtime-1");
  request.payload.config = { models: [] };

  const [response] = await exchange(
    () => ({
      async start() {
        started = true;
      },
      async invoke() {
        throw new Error("unreachable");
      },
      async stop() {
        stopped = true;
      },
    }),
    [request],
  );

  assert.equal(started, false);
  assert.equal(stopped, true);
  assert.equal(response.outcome.error.code, "lifecycle_invalid_config");
});

test("classifies factory failures as adapter startup failures", async () => {
  const [response] = await exchange(
    () => {
      throw new Error("private factory detail");
    },
    [start("runtime-1")],
  );

  assert.equal(response.outcome.error.code, "lifecycle_adapter_start_failed");
  assert.equal(response.outcome.error.message, "Adapter failed during lifecycle start");
});

test("retains a runtime for final cleanup when stop fails", async () => {
  let stopCount = 0;
  const responses = await exchange(
    () => ({
      async start() {},
      async invoke() {
        return { status: "succeeded", output: null };
      },
      async stop() {
        stopCount += 1;
        throw new Error("private stop detail");
      },
    }),
    [start("runtime-1"), stop("runtime-1")],
  );

  assert.equal(stopCount, 2);
  assert.equal(responses[1].outcome.error.code, "lifecycle_adapter_stop_failed");
});

test("marks a runtime unusable after an adapter invocation failure", async () => {
  let invokeCount = 0;
  const responses = await exchange(
    () => ({
      async start() {},
      async invoke() {
        invokeCount += 1;
        throw new Error("private target detail");
      },
      async stop() {},
    }),
    [start("runtime-1"), invoke("runtime-1", "one", "one"), invoke("runtime-1", "two", "two"), stop("runtime-1")],
  );

  assert.equal(invokeCount, 1);
  assert.equal(responses[1].outcome.error.code, "lifecycle_adapter_invoke_failed");
  assert.equal(responses[2].outcome.error.code, "lifecycle_runtime_failed");
});

test("marks a runtime unusable after an invalid adapter result", async () => {
  let invokeCount = 0;
  const responses = await exchange(
    () => ({
      async start() {},
      async invoke() {
        invokeCount += 1;
        return { status: "unknown", output: null };
      },
      async stop() {},
    }),
    [start("runtime-1"), invoke("runtime-1", "one", "one"), invoke("runtime-1", "two", "two"), stop("runtime-1")],
  );

  assert.equal(invokeCount, 1);
  assert.equal(responses[1].outcome.error.code, "lifecycle_invalid_response");
  assert.equal(responses[2].outcome.error.code, "lifecycle_runtime_failed");
});

test("normalizes an unencodable adapter result and marks the runtime unusable", async () => {
  let invokeCount = 0;
  const circular = {};
  circular.self = circular;
  const responses = await exchange(
    () => ({
      async start() {},
      async invoke() {
        invokeCount += 1;
        return { status: "succeeded", output: circular };
      },
      async stop() {},
    }),
    [start("runtime-1"), invoke("runtime-1", "one", "one"), invoke("runtime-1", "two", "two"), stop("runtime-1")],
  );

  assert.equal(invokeCount, 1);
  assert.equal(responses[1].outcome.error.code, "lifecycle_invalid_response");
  assert.equal(responses[2].outcome.error.code, "lifecycle_runtime_failed");
});

test("attempts cleanup when input ends without stop", async () => {
  let stopped = false;
  await exchange(
    () => ({
      async start() {},
      async invoke() {
        return { status: "succeeded", output: null };
      },
      async stop() {
        stopped = true;
      },
    }),
    [start("runtime-1")],
  );
  assert.equal(stopped, true);
});

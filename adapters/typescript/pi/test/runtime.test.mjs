// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdir, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { LifecycleError } from "nemo-fabric-adapters-common";

import { prepareRelayAtifMatchers } from "../dist/relay-config.js";
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

test("adds Relay details to results and stops the Pi session before the gateway", async () => {
  const order = [];
  const relay = {
    pluginConfig: { version: 1, components: [] },
    async output() {
      return {
        relay_runtime: { enabled: true, gateway_url: "http://127.0.0.1:41000" },
        relay_artifacts: [],
      };
    },
    async stop() {
      order.push("relay");
    },
  };
  const runtime = new PiAdapterRuntime({
    async create() {
      return {
        relay,
        async prompt() {
          return { accepted: true, text: "ok", stopReason: "stop" };
        },
        async stop() {
          order.push("session");
        },
      };
    },
  });

  await runtime.start(startInput());
  const result = await runtime.invoke({ input: "trace me" }, context);
  await runtime.stop();

  assert.deepEqual(result, {
    status: "succeeded",
    output: {
      response: "ok",
      relay_runtime: { enabled: true, gateway_url: "http://127.0.0.1:41000" },
      relay_artifacts: [],
    },
  });
  assert.deepEqual(order, ["session", "relay"]);
});

test("collects session-scoped ATIF after session shutdown and before Relay termination", async (t) => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-finalized-")));
  const atifDir = join(root, "atif");
  const atofDir = join(root, "atof");
  await mkdir(atifDir);
  await mkdir(atofDir);
  const atofPath = join(atofDir, "events.atof.jsonl");
  const atifPath = join(atifDir, "trajectory-session-1.atif.json");
  await writeFile(atofPath, "{}\n", "utf8");
  await writeFile(join(atifDir, "trajectory-stale.atif.json"), '{"session_id":"stale"}\n', "utf8");
  const order = [];
  const relay = {
    pluginConfig: {
      version: 1,
      components: [
        {
          kind: "observability",
          config: {
            atof: {
              enabled: true,
              sinks: [{ type: "file", output_directory: atofDir, filename: "events.atof.jsonl" }],
            },
            atif: {
              enabled: true,
              output_directory: atifDir,
              filename_template: "trajectory-{session_id}.atif.json",
            },
          },
        },
      ],
    },
    async output(artifacts) {
      if (artifacts.some((artifact) => artifact.kind === "atif")) {
        order.push("consume");
      }
      return { relay_artifacts: artifacts };
    },
    async stop() {
      order.push("terminate");
    },
  };
  relay.atifMatchers = await prepareRelayAtifMatchers(relay.pluginConfig);
  const factory = {
    async create() {
      return {
        relay,
        async prompt() {
          return { accepted: true, text: "ok", stopReason: "stop" };
        },
        async stop() {
          order.push("session_shutdown");
          await writeFile(atifPath, '{"session_id":"session-1"}\n', "utf8");
        },
      };
    },
  };
  assert.equal(new PiAdapterRuntime(factory).atifFinalizationTimeoutMs, 1_000);
  for (const invalidTimeout of [Number.NaN, Infinity, -1]) {
    assert.equal(
      new PiAdapterRuntime(factory, { atifFinalizationTimeoutMs: invalidTimeout }).atifFinalizationTimeoutMs,
      1_000,
    );
  }
  assert.equal(new PiAdapterRuntime(factory, { atifFinalizationTimeoutMs: 5_000 }).atifFinalizationTimeoutMs, 1_000);
  assert.equal(new PiAdapterRuntime(factory, { atifFinalizationTimeoutMs: 0 }).atifFinalizationTimeoutMs, 0);

  let stderr = "";
  t.mock.method(process.stderr, "write", (chunk) => {
    stderr += String(chunk);
    return true;
  });
  try {
    const runtime = new PiAdapterRuntime(factory);
    await runtime.start(startInput());
    const startedAt = performance.now();
    const invocation = await runtime.invoke({ input: "trace me" }, context);
    assert.ok(performance.now() - startedAt < 500, "invoke should not wait for session-scoped ATIF");
    assert.deepEqual(invocation.output.relay_artifacts, [{ kind: "atof", path: atofPath }]);

    const stopped = await runtime.stop();
    assert.deepEqual(stopped.relay_artifacts, [
      { kind: "atif", path: atifPath },
      { kind: "atof", path: atofPath },
    ]);
    assert.deepEqual(order, ["session_shutdown", "consume", "terminate"]);
    assert.equal(stderr, "");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("an ATIF finalization timeout at stop preserves per-invoke ATOF", async (t) => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-timeout-")));
  const atifDir = join(root, "atif");
  const atofDir = join(root, "atof");
  await mkdir(atifDir);
  await mkdir(atofDir);
  const atofPath = join(atofDir, "events.atof.jsonl");
  await writeFile(atofPath, "{}\n", "utf8");
  let promptCount = 0;
  const relay = {
    pluginConfig: {
      version: 1,
      components: [
        {
          kind: "observability",
          config: {
            atof: {
              enabled: true,
              sinks: [{ type: "file", output_directory: atofDir, filename: "events.atof.jsonl" }],
            },
            atif: {
              enabled: true,
              output_directory: atifDir,
              filename_template: "trajectory-{session_id}.atif.json",
            },
          },
        },
      ],
    },
    async output(artifacts) {
      return {
        relay_artifacts: artifacts ?? [{ kind: "atif", path: "unexpected" }],
      };
    },
    async stop() {},
  };
  relay.atifMatchers = await prepareRelayAtifMatchers(relay.pluginConfig);
  const factory = {
    async create() {
      return {
        relay,
        async prompt() {
          promptCount += 1;
          return { accepted: true, text: "ok", stopReason: "stop" };
        },
        async stop() {},
      };
    },
  };
  const runtime = new PiAdapterRuntime(factory, { atifFinalizationTimeoutMs: 25 });
  let stderr = "";
  t.mock.method(process.stderr, "write", (chunk) => {
    stderr += String(chunk);
    return true;
  });

  try {
    await runtime.start(startInput());
    const startedAt = performance.now();
    const timedOut = await runtime.invoke({ input: "trace me" }, context);
    assert.ok(performance.now() - startedAt < 500, "invoke should not wait for session-scoped ATIF");
    assert.equal(stderr, "");
    assert.deepEqual(timedOut.output.relay_artifacts, [{ kind: "atof", path: atofPath }]);

    const next = await runtime.invoke({ input: "still usable" }, context);
    assert.equal(next.status, "succeeded");
    assert.equal(promptCount, 2);
    const stopStartedAt = performance.now();
    const stopped = await runtime.stop();
    assert.ok(performance.now() - stopStartedAt < 500, "configured timeout should bound shutdown waiting");
    assert.deepEqual(stopped.relay_artifacts, [{ kind: "atof", path: atofPath }]);
    assert.equal(stderr, "NeMo Relay did not finalize an ATIF artifact within 25 ms after session shutdown\n");
  } finally {
    await runtime.stop();
    await rm(root, { recursive: true, force: true });
  }
});

test("an ATIF snapshot failure preserves the prompt and excludes ATIF", async (t) => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-snapshot-failure-")));
  const atifDir = join(root, "atif");
  const atofDir = join(root, "atof");
  try {
    await mkdir(atifDir);
    await mkdir(atofDir);
    const atofPath = join(atofDir, "events.atof.jsonl");
    await writeFile(atofPath, "{}\n", "utf8");
    const pluginConfig = {
      version: 1,
      components: [
        {
          kind: "observability",
          config: {
            atof: {
              enabled: true,
              sinks: [{ type: "file", output_directory: atofDir, filename: "events.atof.jsonl" }],
            },
            atif: {
              enabled: true,
              output_directory: atifDir,
              filename_template: "trajectory-{session_id}.atif.json",
            },
          },
        },
      ],
    };
    const atifMatchers = await prepareRelayAtifMatchers(pluginConfig);
    let promptCount = 0;
    const relay = {
      pluginConfig,
      atifMatchers,
      async output(artifacts) {
        return { relay_artifacts: artifacts ?? [{ kind: "atif", path: "unexpected" }] };
      },
      async stop() {},
    };
    const runtime = new PiAdapterRuntime({
      async create() {
        return {
          relay,
          async prompt() {
            promptCount += 1;
            return { accepted: true, text: "ok", stopReason: "stop" };
          },
          async stop() {},
        };
      },
    });
    let stderr = "";
    t.mock.method(process.stderr, "write", (chunk) => {
      stderr += String(chunk);
      return true;
    });

    await runtime.start(startInput());
    await rm(atifDir, { recursive: true, force: true });
    const first = await runtime.invoke({ input: "first" }, context);
    const second = await runtime.invoke({ input: "second" }, context);

    assert.equal(first.status, "succeeded");
    assert.equal(second.status, "succeeded");
    assert.equal(promptCount, 2);
    assert.deepEqual(first.output.relay_artifacts, [{ kind: "atof", path: atofPath }]);
    assert.deepEqual(second.output.relay_artifacts, [{ kind: "atof", path: atofPath }]);
    const stopped = await runtime.stop();
    assert.deepEqual(stopped.relay_artifacts, [{ kind: "atof", path: atofPath }]);
    assert.match(stderr, /^NeMo Relay ATIF artifact snapshot failed:/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("still stops Relay when Pi session shutdown fails", async () => {
  let sessionStopAttempts = 0;
  let relayStopAttempts = 0;
  const sessionFailure = new Error("session shutdown failed");
  const runtime = new PiAdapterRuntime({
    async create() {
      return {
        relay: {
          pluginConfig: { version: 1, components: [] },
          async output() {
            return {};
          },
          async stop() {
            relayStopAttempts += 1;
          },
        },
        async prompt() {
          return { accepted: true, text: "ok" };
        },
        async stop() {
          sessionStopAttempts += 1;
          if (sessionStopAttempts === 1) {
            throw sessionFailure;
          }
        },
      };
    },
  });
  await runtime.start(startInput());

  await assert.rejects(runtime.stop(), sessionFailure);
  assert.equal(relayStopAttempts, 1);
  await runtime.stop();
  assert.equal(sessionStopAttempts, 2);
  assert.equal(relayStopAttempts, 2);
});

test("preserves both Pi session and Relay shutdown failures", async () => {
  const sessionFailure = new Error("session shutdown failed");
  const relayFailure = new LifecycleError("pi_relay_stop_failed", "gateway shutdown failed", {
    metadata: {
      gateway_log_path: "/tmp/gateway.log",
      relay_error: "gateway still running",
    },
  });
  const runtime = new PiAdapterRuntime({
    async create() {
      return {
        relay: {
          pluginConfig: { version: 1, components: [] },
          async output() {
            return {};
          },
          async stop() {
            throw relayFailure;
          },
        },
        async prompt() {
          return { accepted: true, text: "ok" };
        },
        async stop() {
          throw sessionFailure;
        },
      };
    },
  });
  await runtime.start(startInput());

  await assert.rejects(
    runtime.stop(),
    (error) =>
      error instanceof LifecycleError &&
      error.code === "pi_relay_stop_failed" &&
      error.message === "Pi session and NeMo Relay cleanup failed" &&
      error.metadata.gateway_log_path === "/tmp/gateway.log" &&
      error.metadata.relay_error === "gateway still running" &&
      error.metadata.session_error === "session shutdown failed",
  );
});

test("preserves non-lifecycle cleanup failures in an AggregateError", async () => {
  const sessionFailure = new Error("session shutdown failed");
  const relayFailure = new Error("gateway shutdown failed");
  const runtime = new PiAdapterRuntime({
    async create() {
      return {
        relay: {
          pluginConfig: { version: 1, components: [] },
          async output() {
            return {};
          },
          async stop() {
            throw relayFailure;
          },
        },
        async prompt() {
          return { accepted: true, text: "ok" };
        },
        async stop() {
          throw sessionFailure;
        },
      };
    },
  });
  await runtime.start(startInput());

  await assert.rejects(
    runtime.stop(),
    (error) =>
      error instanceof AggregateError &&
      error.message === "Pi session and NeMo Relay cleanup failed" &&
      error.errors[0] === sessionFailure &&
      error.errors[1] === relayFailure,
  );
});

test("retries Relay cleanup when gateway shutdown fails", async () => {
  let promptCount = 0;
  let sessionStopAttempts = 0;
  let relayStopAttempts = 0;
  const relayFailure = new Error("gateway shutdown failed");
  const runtime = new PiAdapterRuntime({
    async create() {
      return {
        relay: {
          pluginConfig: { version: 1, components: [] },
          async output() {
            return {};
          },
          async stop() {
            relayStopAttempts += 1;
            if (relayStopAttempts === 1) {
              throw relayFailure;
            }
          },
        },
        async prompt() {
          promptCount += 1;
          return { accepted: true, text: "ok" };
        },
        async stop() {
          sessionStopAttempts += 1;
        },
      };
    },
  });
  await runtime.start(startInput());

  await assert.rejects(runtime.stop(), relayFailure);
  await assert.rejects(
    runtime.invoke({ input: "must not run" }, context),
    (error) => error.code === "pi_runtime_unusable",
  );
  assert.equal(promptCount, 0);
  await runtime.stop();
  await runtime.stop();
  assert.equal(sessionStopAttempts, 2);
  assert.equal(relayStopAttempts, 2);
});

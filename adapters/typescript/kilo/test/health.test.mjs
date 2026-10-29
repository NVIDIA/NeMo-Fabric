// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { KiloAdapterRuntime } from "../dist/runtime.js";

const input = {
  agentName: "health-test",
  baseDir: "/tmp",
  config: {},
  runtimeContext: {
    runtime_id: "runtime",
    invocation_id: "start",
    request_id: "start",
    artifacts: {},
    environment: {
      environment_id: "environment",
      provider: "local",
      ownership: "caller_owned",
      control_location: "external_control",
    },
  },
};
const passed = () => ({
  outcome: "passed",
  reason_code: null,
  activity: "unknown",
  source: "fabric.adapter.liveness",
  observed_at: new Date().toISOString(),
});

test("health reaches the native session during a blocked invocation", async () => {
  const entered = Promise.withResolvers();
  const release = Promise.withResolvers();
  let prompts = 0;
  const levels = [];
  const runtime = new KiloAdapterRuntime({
    async create() {
      return {
        id: "native-session",
        async prompt() {
          prompts += 1;
          entered.resolve();
          await release.promise;
          return { status: "completed", stopReason: "end_turn", text: "done" };
        },
        async check(level) {
          levels.push(level);
          return passed();
        },
        async stop() {},
      };
    },
  });
  await runtime.start(input);
  assert.equal((await runtime.check("live")).outcome, "passed");
  const invocation = runtime.invoke(
    { input: "PRIVATE-PROMPT" },
    input.runtimeContext,
  );
  try {
    await entered.promise;
    const observation = await runtime.check("live");
    assert.equal(observation.outcome, "passed");
    assert.equal(observation.activity, "busy");
    assert.equal(JSON.stringify(observation).includes("PRIVATE"), false);
    assert.equal(prompts, 1);
  } finally {
    release.resolve();
    await invocation;
  }
  assert.equal((await runtime.check("live")).outcome, "passed");
  assert.deepEqual(levels, ["live", "live", "live"]);
  await runtime.stop();
  assert.equal((await runtime.check("live")).reason_code, "session_closed");
});

test("health rejects absent sessions without constructing a native session", async () => {
  let created = 0;
  const runtime = new KiloAdapterRuntime({
    async create() {
      created += 1;
    },
  });
  assert.equal((await runtime.check("live")).reason_code, "session_closed");
  assert.equal(created, 0);
});

test("stop invalidates a pending observation without leaking native errors", async () => {
  const entered = Promise.withResolvers();
  const release = Promise.withResolvers();
  let probes = 0;
  const runtime = new KiloAdapterRuntime({
    async create() {
      return {
        id: "native-session",
        async prompt() {},
        async stop() {},
        async check() {
          probes += 1;
          entered.resolve();
          await release.promise;
          return passed();
        },
      };
    },
  });
  await runtime.start(input);
  const observation = runtime.check("live");
  await entered.promise;
  await runtime.stop();
  release.resolve();
  assert.equal((await observation).reason_code, "observation_changed");
  assert.equal((await runtime.check("live")).reason_code, "session_closed");
  assert.equal(probes, 1);
});

test("health errors have fixed safe output and preserve invocation state", async () => {
  let stops = 0;
  const runtime = new KiloAdapterRuntime({
    async create() {
      return {
        id: "native-session",
        async prompt() {
          return { status: "completed", stopReason: "end_turn", text: "done" };
        },
        async check() {
          throw new Error("PRIVATE-CREDENTIAL");
        },
        async stop() {
          stops += 1;
        },
      };
    },
  });
  await runtime.start(input);
  const result = await runtime.check("live");
  assert.equal(result.reason_code, "probe_error");
  assert.equal(JSON.stringify(result).includes("PRIVATE"), false);
  assert.equal(stops, 0);
  assert.equal(
    (await runtime.invoke({ input: "one" }, input.runtimeContext)).status,
    "succeeded",
  );
  await runtime.stop();
});

test("descriptor advertises exactly the qualified health levels", async () => {
  const descriptor = JSON.parse(
    await readFile(
      new URL("../kilo.fabric-adapter.json", import.meta.url),
      "utf8",
    ),
  );
  assert.deepEqual(descriptor.capabilities.metadata.health_checks, [
    "live",
    "active",
    "ready",
  ]);
});

test("a restart invalidates a check even when a factory reuses a session object", async () => {
  const entered = Promise.withResolvers();
  const release = Promise.withResolvers();
  const session = {
    id: "reused",
    async prompt() {},
    async stop() {},
    async check() {
      entered.resolve();
      await release.promise;
      return passed();
    },
  };
  const runtime = new KiloAdapterRuntime({
    async create() {
      return session;
    },
  });
  await runtime.start(input);
  const checking = runtime.check("live");
  await entered.promise;
  await runtime.stop();
  await runtime.start(input);
  release.resolve();
  assert.equal((await checking).reason_code, "observation_changed");
  await runtime.stop();
});

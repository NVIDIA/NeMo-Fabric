// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { ClineSdkSessionFactory } from "../dist/cline-sdk.js";
import { ClineAdapterRuntime } from "../dist/runtime.js";

const input = {
  agentName: "health",
  baseDir: "/tmp",
  config: {
    models: { default: { provider: "openai", model: "test", api_key_env: "TEST_KEY" } },
    tools: { enabled: [] },
  },
  runtimeContext: {
    artifacts: {},
    environment: {
      environment_id: "env",
      provider: "local",
      ownership: "caller_owned",
      control_location: "external_control",
      workspace: "/tmp",
      env: { TEST_KEY: "PRIVATE" },
    },
    runtime_id: "runtime",
    invocation_id: "start",
    request_id: "start",
  },
};

function nativeFixture() {
  const entered = Promise.withResolvers();
  const release = Promise.withResolvers();
  const agent = {
    shutdownCalled: false,
    running: false,
    getAgentId: () => "agent",
    canStartRun() {
      return !this.shutdownCalled && !this.running;
    },
  };
  const resident = { sessionId: "session", agent, status: "idle" };
  const sessions = new Map([["session", resident]]);
  const calls = { start: [], send: 0, get: 0 };
  const core = {
    host: { sessions },
    async start(value) {
      calls.start.push(value);
      return { sessionId: "session" };
    },
    async send() {
      calls.send++;
      agent.running = true;
      entered.resolve();
      await release.promise;
      agent.running = false;
      return { finishReason: "completed", text: "done" };
    },
    async get() {
      calls.get++;
      return { sessionId: "session", status: resident.status };
    },
    async stop() {
      agent.shutdownCalled = true;
      sessions.clear();
    },
    async dispose() {
      agent.shutdownCalled = true;
      sessions.clear();
    },
  };
  const loader = async () => ({
    ClineCore: { create: async () => core },
    ALL_DEFAULT_TOOL_NAMES: [],
    getClineDefaultSystemPrompt: () => "test",
    loadAgentPluginPackages: async () => ({ skills: [], mcpServers: [], diagnostics: [] }),
  });
  return { core, agent, resident, sessions, calls, loader, entered, release };
}

test("initializes a native session without a prompt and checks during the first blocked invocation", async () => {
  const fixture = nativeFixture();
  const handle = await new ClineSdkSessionFactory(fixture.loader).create(input);
  assert.equal(fixture.calls.start.length, 1);
  assert.equal("prompt" in fixture.calls.start[0], false);
  for (const level of ["live", "active", "ready"]) assert.equal((await handle.check(level)).outcome, "passed");
  const invocation = handle.prompt("PRIVATE prompt");
  await fixture.entered.promise;
  try {
    for (const level of ["live", "active", "ready"]) {
      const observation = await handle.check(level);
      assert.equal(observation.outcome, "passed");
      assert.equal(observation.activity, "busy");
      assert.equal(JSON.stringify(observation).includes("PRIVATE"), false);
    }
    assert.equal(fixture.calls.send, 1);
  } finally {
    fixture.release.resolve();
    await invocation;
    await handle.stop();
  }
});

test("historical get rows cannot prove a removed or replaced native session alive", async () => {
  for (const replacement of [undefined, {}]) {
    const fixture = nativeFixture();
    const handle = await new ClineSdkSessionFactory(fixture.loader).create(input);
    if (replacement === undefined) fixture.sessions.delete("session");
    else fixture.sessions.set("session", replacement);
    assert.notEqual((await handle.check("live")).outcome, "passed");
    assert.equal(fixture.calls.get, 0);
    await handle.stop();
  }
});

test("native shutdown rejects health even while the resident entry remains", async () => {
  const fixture = nativeFixture();
  const handle = await new ClineSdkSessionFactory(fixture.loader).create(input);
  fixture.agent.shutdownCalled = true;
  assert.equal((await handle.check("live")).reason_code, "session_closed");
  await handle.stop();
});

test("ready rechecks ownership on success and rejection and never reuses an old response", async () => {
  for (const rejected of [false, true]) {
    const fixture = nativeFixture();
    const reply = Promise.withResolvers();
    fixture.core.get = async () => {
      fixture.calls.get++;
      return reply.promise;
    };
    const handle = await new ClineSdkSessionFactory(fixture.loader).create(input);
    const check = handle.check("ready");
    assert.equal((await handle.check("ready")).reason_code, "probe_timeout");
    assert.equal(fixture.calls.get, 1);
    fixture.sessions.set("session", {});
    if (rejected) reply.reject(new Error("PRIVATE failure"));
    else reply.resolve({ sessionId: "session", status: "idle" });
    assert.equal((await check).reason_code, "observation_changed");
    fixture.sessions.set("session", fixture.resident);
    fixture.core.get = async () => {
      fixture.calls.get++;
      return { sessionId: "session", status: "idle" };
    };
    assert.equal((await handle.check("ready")).outcome, "passed");
    assert.equal(fixture.calls.get, 2);
    await handle.stop();
  }
});

test("runtime rechecks ownership when a check rejects during stop", async () => {
  const reply = Promise.withResolvers();
  const session = { prompt: async () => {}, stop: async () => {}, check: () => reply.promise };
  const runtime = new ClineAdapterRuntime({ create: async () => session });
  await runtime.start(input);
  const check = runtime.check("live");
  await runtime.stop();
  reply.reject(new Error("PRIVATE failure"));
  assert.equal((await check).reason_code, "observation_changed");
});

test("runtime replacement invalidates an observation even when a factory reuses a handle", async () => {
  const reply = Promise.withResolvers();
  const session = { prompt: async () => {}, stop: async () => {}, check: () => reply.promise };
  const runtime = new ClineAdapterRuntime({ create: async () => session });
  await runtime.start(input);
  const check = runtime.check("live");
  await runtime.stop();
  await runtime.start(input);
  reply.resolve({
    outcome: "passed",
    reason_code: null,
    activity: "idle",
    source: "fabric.adapter.liveness",
    observed_at: new Date().toISOString(),
  });
  assert.equal((await check).reason_code, "observation_changed");
  await runtime.stop();
});

test("native agent replacement invalidates checks even if the session and agent identifier stay the same", async () => {
  const fixture = nativeFixture();
  const handle = await new ClineSdkSessionFactory(fixture.loader).create(input);
  fixture.resident.agent = { ...fixture.agent };
  assert.equal((await handle.check("live")).reason_code, "observation_changed");
  await handle.stop();
});

test("native replacement takes precedence over the previous agent's closure", async () => {
  const fixture = nativeFixture();
  const handle = await new ClineSdkSessionFactory(fixture.loader).create(input);
  fixture.agent.shutdownCalled = true;
  fixture.sessions.set("session", { ...fixture.resident, agent: { ...fixture.agent, shutdownCalled: false } });
  const observation = await handle.check("live");
  assert.equal(observation.outcome, "unknown");
  assert.equal(observation.reason_code, "observation_changed");
  await handle.stop();
});

test("a replaced native host cannot turn a missing resident into confirmed closure", async () => {
  const fixture = nativeFixture();
  const handle = await new ClineSdkSessionFactory(fixture.loader).create(input);
  fixture.core.host = { sessions: new Map() };
  assert.equal((await handle.check("live")).reason_code, "observation_changed");
  await handle.stop();
});

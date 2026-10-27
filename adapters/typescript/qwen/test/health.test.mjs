// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { QwenSdkSession } from "../dist/qwen-sdk.js";
import { QwenAdapterRuntime } from "../dist/runtime.js";

const context = { runtime_id: "runtime", invocation_id: "invoke", request_id: "request", artifacts: {}, environment: { environment_id: "env", provider: "local", ownership: "caller_owned", control_location: "external_control", env: {} } };
const input = { agentName: "agent", baseDir: process.cwd(), config: {}, runtimeContext: context };
const passed = () => ({ outcome: "passed", reason_code: null, source: "fabric.adapter.liveness", observed_at: new Date().toISOString(), activity: "idle" });

test("health observes native state while an invocation remains blocked", async () => {
  let finish;
  const blocked = new Promise(resolve => { finish = resolve; });
  const levels = [];
  const native = { id: "session", check: async level => { levels.push(level); return passed(); }, prompt: async () => { await blocked; return { text: "answer", stopReason: "end_turn" }; }, stop: async () => { }, close: async () => { } };
  const runtime = new QwenAdapterRuntime({ create: async () => native });
  assert.equal((await runtime.check("live")).reason_code, "runtime_unavailable");
  await runtime.start(input);
  assert.equal((await runtime.check("live")).activity, "idle");
  const invocation = runtime.invoke({ input: "private prompt" }, context);
  assert.equal((await runtime.check("live")).activity, "busy");
  finish();
  assert.equal((await invocation).status, "succeeded");
  assert.equal((await runtime.check("live")).activity, "idle");
  assert.deepEqual(levels, ["live", "live", "live"]);
  await runtime.stop();
  assert.equal((await runtime.check("live")).reason_code, "runtime_unavailable");
});

test("health discards an observation after runtime ownership changes", async () => {
  let finish;
  const blocked = new Promise(resolve => { finish = resolve; });
  const native = { id: "session", check: async () => { await blocked; return passed(); }, prompt: async () => ({ text: "answer", stopReason: "end_turn" }), stop: async () => { }, close: async () => { } };
  const runtime = new QwenAdapterRuntime({ create: async () => native });
  await runtime.start(input);
  const observation = runtime.check("live");
  await runtime.stop();
  await runtime.start(input);
  finish();
  const result = await observation;
  assert.equal(result.outcome, "unknown");
  assert.equal(result.reason_code, "observation_changed");
  assert.equal(result.observed_at, null);
  await runtime.stop();
});

test("health sanitizes native errors without closing the session", async () => {
  let stopped = 0;
  const native = { id: "session", check: async () => { throw Error("PRIVATE_API_KEY=secret"); }, prompt: async () => ({ text: "answer", stopReason: "end_turn" }), stop: async () => { stopped++; }, close: async () => { stopped++; } };
  const runtime = new QwenAdapterRuntime({ create: async () => native });
  await runtime.start(input);
  const result = await runtime.check("live");
  assert.equal(result.reason_code, "probe_error");
  assert.doesNotMatch(JSON.stringify(result), /PRIVATE|secret/);
  assert.equal(stopped, 0);
  assert.equal((await runtime.invoke({ input: "still usable" }, context)).status, "succeeded");
  await runtime.stop();
});

test("native Qwen probes retain at most one outstanding request after caller timeouts", async () => {
  let calls = 0;
  let finish;
  const response = new Promise(resolve => { finish = resolve; });
  const query = { isClosed: () => false, getSessionId: () => "native", supportedCommands: () => { calls++; return response; } };
  const session = new QwenSdkSession(query, {}, "unused", {});
  const first = session.check("ready");
  const timed = await Promise.race([first, new Promise(resolve => setTimeout(() => resolve("timeout"), 5))]);
  assert.equal(timed, "timeout");
  const later = Array.from({ length: 20 }, () => session.check("live"));
  assert.equal(calls, 1);
  finish({ commands: ["supported_commands"] });
  assert.equal((await first).outcome, "passed");
  assert.equal((await Promise.all(later)).every(result => result.reason_code === "probe_timeout"), true);
});

test("Qwen rejects malformed native control responses and changed native identity", async () => {
  const query = { isClosed: () => false, getSessionId: () => "native", supportedCommands: async () => ({ commands: "PRIVATE native payload" }) };
  const session = new QwenSdkSession(query, {}, "unused", {});
  const result = await session.check("ready");
  assert.equal(result.reason_code, "probe_error");
  assert.doesNotMatch(JSON.stringify(result), /PRIVATE/);
  let finish;
  query.supportedCommands = () => new Promise(resolve => { finish = resolve; });
  const pending = session.check("live");
  query.getSessionId = () => "replacement";
  finish({ commands: [] });
  assert.equal((await pending).reason_code, "observation_changed");
});

test("later Qwen checks reject a delayed observation from an expired caller", async () => {
  const delayed = Promise.withResolvers();
  let calls = 0;
  const query = {
    isClosed: () => false,
    getSessionId: () => "native",
    supportedCommands: () => {
      calls++;
      return calls === 1 ? delayed.promise : Promise.resolve({ commands: "invalid current response" });
    },
  };
  const session = new QwenSdkSession(query, {}, "unused", {});
  const first = session.check("ready");
  assert.equal(await Promise.race([first, new Promise(resolve => setTimeout(() => resolve("timeout"), 5))]), "timeout");
  const later = session.check("ready");
  delayed.resolve({ commands: [] });
  await first;
  assert.equal((await later).reason_code, "probe_timeout");
  assert.equal(calls, 1);
  assert.equal((await session.check("ready")).outcome, "unknown");
  assert.equal(calls, 2);
});

test("Qwen validates native closure when the health request rejects", async () => {
  const delayed = Promise.withResolvers();
  let closed = false;
  const query = { isClosed: () => closed, getSessionId: () => "native", supportedCommands: () => delayed.promise };
  const session = new QwenSdkSession(query, {}, "unused", {});
  const pending = session.check("ready");
  closed = true;
  delayed.reject(new Error("PRIVATE native failure"));
  const result = await pending;
  assert.equal(result.reason_code, "observation_changed");
  assert.equal(result.observed_at, null);
  assert.doesNotMatch(JSON.stringify(result), /PRIVATE/);
  assert.equal((await session.check("ready")).reason_code, "session_closed");
});

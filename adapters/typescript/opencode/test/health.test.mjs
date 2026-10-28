// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { OpenCodeAdapterRuntime } from "../dist/runtime.js";
import { LifecycleError } from "nemo-fabric-adapters-common";

const context = { runtime_id: "runtime", invocation_id: "invoke", request_id: "request", artifacts: {}, environment: { environment_id: "env", provider: "local", ownership: "caller_owned", control_location: "external_control", env: {} } };
const input = { agentName: "agent", baseDir: process.cwd(), config: {}, runtimeContext: context };
const passed = () => ({ outcome: "passed", reason_code: null, source: "fabric.adapter.liveness", observed_at: new Date().toISOString(), activity: "idle" });

test("health observes native state while an invocation remains blocked", async () => {
  let finish;
  const blocked = new Promise(resolve => { finish = resolve; });
  const levels = [];
  const native = { id: "session", check: async level => { levels.push(level); return passed(); }, prompt: async () => { await blocked; return { text: "answer", stopReason: "end_turn" }; }, stop: async () => { }, close: async () => { } };
  const runtime = new OpenCodeAdapterRuntime({ create: async () => native });
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
  const runtime = new OpenCodeAdapterRuntime({ create: async () => native });
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
  const runtime = new OpenCodeAdapterRuntime({ create: async () => native });
  await runtime.start(input);
  const result = await runtime.check("live");
  assert.equal(result.reason_code, "probe_error");
  assert.doesNotMatch(JSON.stringify(result), /PRIVATE|secret/);
  assert.equal(stopped, 0);
  assert.equal((await runtime.invoke({ input: "still usable" }, context)).status, "succeeded");
  await runtime.stop();
});

test("health activity belongs to the current session generation", async () => {
  const finishes = new Map();
  const native = {
    id: "session",
    check: async () => passed(),
    prompt: async (text) => {
      await new Promise((resolve) => finishes.set(text, resolve));
      return { text: "answer" };
    },
    stop: async () => {},
  };
  const runtime = new OpenCodeAdapterRuntime({ create: async () => native });
  await runtime.start(input);
  const oldInvocation = runtime.invoke({ input: "old" }, context);
  let newInvocation;
  try {
    assert.equal((await runtime.check("live")).activity, "busy");
    await runtime.stop();
    await runtime.start(input);
    assert.equal((await runtime.check("live")).activity, "idle");
    newInvocation = runtime.invoke({ input: "new" }, context);
    assert.equal((await runtime.check("live")).activity, "busy");
    finishes.get("old")();
    assert.equal((await oldInvocation).status, "succeeded");
    assert.equal((await runtime.check("live")).activity, "busy");
    finishes.get("new")();
    assert.equal((await newInvocation).status, "succeeded");
    assert.equal((await runtime.check("live")).activity, "idle");
  } finally {
    for (const finish of finishes.values()) finish();
    await Promise.allSettled([oldInvocation, newInvocation]);
    await runtime.stop();
  }
});

for (const fails of [false, true]) {
  test(`health remains busy until artifact writing ${fails ? "fails" : "completes"}`, async (t) => {
    let finish;
    let entered;
    const blocked = new Promise((resolve) => { finish = resolve; });
    const writing = new Promise((resolve) => { entered = resolve; });
    const native = {
      id: "session",
      check: async () => passed(),
      prompt: async () => ({ text: "answer", patch: "private patch" }),
      stop: async () => { },
    };
    const runtime = new OpenCodeAdapterRuntime({ create: async () => native });
    const artifacts = [{ kind: "patch", media_type: "text/x-diff", name: "opencode-diff-1", path: "session.patch" }];
    t.mock.method(runtime, "writePatch", async () => {
      entered();
      await blocked;
      if (fails) throw new LifecycleError("opencode_artifact_write_failed", "OpenCode could not write its session diff artifact");
      return artifacts;
    });
    await runtime.start(input);
    const invocation = runtime.invoke({ input: "private prompt" }, { ...context, artifacts: { root: "/artifacts" } });
    try {
      await writing;
      assert.equal((await runtime.check("live")).activity, "busy");
      finish();
      if (fails) {
        await assert.rejects(invocation, (error) => error.code === "opencode_artifact_write_failed");
      } else {
        const result = await invocation;
        assert.equal(result.status, "succeeded");
        assert.deepEqual(result.artifacts, artifacts);
      }
      assert.equal((await runtime.check("live")).activity, "idle");
    } finally {
      finish();
      await Promise.allSettled([invocation]);
      await runtime.stop();
    }
  });
}

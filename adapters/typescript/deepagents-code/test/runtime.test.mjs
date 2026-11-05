// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { DeepAgentsCodeAdapterRuntime } from "../dist/runtime.js";

// A session whose first stop fails, as when signalling the ACP child is denied.
function flakySession(prompt = async () => ({ text: "ok", stopReason: "end_turn" })) {
  const session = {
    id: "session-1",
    stops: 0,
    check: async () => ({ outcome: "passed", reason_code: null, activity: "unknown", source: "fabric.adapter.liveness", observed_at: "" }),
    prompt,
    async stop() {
      session.stops += 1;
      if (session.stops === 1) throw new Error("EPERM");
    },
  };
  return session;
}

const context = { runtime_id: "runtime-1", invocation_id: "invoke", request_id: "request", artifacts: {} };

test("a failed stop keeps the session so a later stop retries it", async () => {
  const session = flakySession();
  const runtime = new DeepAgentsCodeAdapterRuntime({ create: async () => session });
  await runtime.start({});

  await assert.rejects(runtime.stop(), /EPERM/u);
  await assert.rejects(runtime.invoke({ input: "hello" }, context), { code: "deepagents_code_not_started" });
  await runtime.stop();

  assert.equal(session.stops, 2);
  await runtime.stop();
  assert.equal(session.stops, 2);
});

test("a session whose cleanup failed after a broken prompt is stopped again by stop", async () => {
  const session = flakySession(async () => {
    throw new Error("broken pipe");
  });
  const runtime = new DeepAgentsCodeAdapterRuntime({ create: async () => session });
  await runtime.start({});

  await assert.rejects(runtime.invoke({ input: "hello" }, context), { code: "deepagents_code_session_failed" });
  await runtime.stop();

  assert.equal(session.stops, 2);
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { KiloSdkSessionFactory } from "../dist/kilo-sdk.js";
const input = {
  agentName: "health",
  baseDir: "/tmp",
  config: {
    models: {
      default: {
        provider: "openai",
        model: "test",
        api_key_env: "TEST_MODEL_KEY",
      },
    },
  },
  runtimeContext: {
    runtime_id: "runtime",
    invocation_id: "start",
    request_id: "start",
    artifacts: {},
    environment: {
      environment_id: "env",
      provider: "local",
      ownership: "caller_owned",
      control_location: "external_control",
      workspace: "/tmp",
      env: { TEST_MODEL_KEY: "PRIVATE-KEY" },
    },
  },
};

async function session(t, overrides = {}) {
  const state = { alive: true, calls: [], aborts: 0, closed: 0, healthy: true };
  const client = {
    global: {
      async health(options) {
        state.calls.push(["health", options]);
        return {
          data: {
            healthy: state.healthy,
            version: "7.7.12",
            PRIVATE: "PRIVATE-STATE",
          },
        };
      },
    },
    session: {
      async create() {
        return { data: { id: "session-1" } };
      },
      async get(parameters, options) {
        state.calls.push(["get", parameters, options]);
        return { data: { id: "session-1", PRIVATE: "PRIVATE-STATE" } };
      },
      async prompt() {
        return { data: { info: {}, parts: [{ type: "text", text: "done" }] } };
      },
      async abort() {
        state.aborts += 1;
      },
      async delete() {},
    },
  };
  Object.assign(client.session, overrides.session);
  const factory = new KiloSdkSessionFactory(
    async () => () => client,
    async () => ({
      url: "http://127.0.0.1:1234",
      isAlive: () => state.alive,
      async close() {
        state.closed += 1;
      },
    }),
  );
  const handle = await factory.create(input);
  t.after(() => handle.stop());
  return { handle, state, client };
}

test("native process, selected session query, and health response use distinct evidence", async (t) => {
  const { handle, state, client } = await session(t);
  for (const level of ["live", "active", "ready"]) {
    const result = await handle.check(level);
    assert.equal(result.outcome, "passed");
    assert.ok(Number.isFinite(Date.parse(result.observed_at)));
    assert.equal(JSON.stringify(result).includes("PRIVATE"), false);
  }
  assert.equal(state.calls[0][0], "get");
  assert.deepEqual(state.calls[0][1], {
    sessionID: "session-1",
    directory: await import("node:fs/promises").then((fs) =>
      fs.realpath("/tmp"),
    ),
  });
  assert.ok(state.calls[0][2].signal instanceof AbortSignal);
  assert.equal(state.calls[1][0], "health");
  const nativeHealth = client.global.health;
  client.global.health = async () => ({ data: { healthy: "PRIVATE-INVALID" } });
  assert.equal((await handle.check("ready")).reason_code, "probe_error");
  client.global.health = nativeHealth;
  state.healthy = false;
  assert.equal((await handle.check("ready")).reason_code, "health_failed");
  state.alive = false;
  assert.equal((await handle.check("live")).reason_code, "agent_exited");
});

test("native checks are bounded while a request hangs and do not abort prompts", async (t) => {
  const release = Promise.withResolvers();
  const entered = Promise.withResolvers();
  const { handle, state } = await session(t, {
    session: {
      async get() {
        entered.resolve();
        return release.promise;
      },
    },
  });
  const pending = handle.check("active");
  await entered.promise;
  assert.equal((await handle.check("active")).reason_code, "probe_timeout");
  assert.equal((await handle.check("ready")).reason_code, "probe_timeout");
  assert.equal(state.aborts, 0);
  assert.equal(state.closed, 0);
  assert.equal((await handle.prompt("continue")).text, "done");
  await handle.stop();
  release.resolve({ data: { id: "session-1" } });
  assert.equal((await pending).reason_code, "observation_changed");
});

test("native query failure remains unknown and never leaks errors", async (t) => {
  const { handle, state } = await session(t, {
    session: {
      async get() {
        throw new Error("PRIVATE-ERROR");
      },
    },
  });
  const result = await handle.check("active");
  assert.equal(result.reason_code, "probe_error");
  assert.equal(JSON.stringify(result).includes("PRIVATE"), false);
  assert.equal(state.closed, 0);
});

test("native health request deadlines do not abort or close an invocation", async (t) => {
  const keepAlive = setInterval(() => {}, 1000);
  t.after(() => clearInterval(keepAlive));
  const { handle, state } = await session(t, {
    session: {
      async get(_parameters, options) {
        return await new Promise((_, reject) => {
          options.signal.addEventListener(
            "abort",
            () => reject(options.signal.reason),
            { once: true },
          );
        });
      },
    },
  });
  const result = await handle.check("active");
  assert.equal(result.reason_code, "probe_timeout");
  assert.equal(state.aborts, 0);
  assert.equal(state.closed, 0);
  assert.equal((await handle.prompt("continue")).text, "done");
});


for (const level of ["active", "ready"]) {
  test(`native server exit invalidates a rejected ${level} probe`, async (t) => {
    const reply = Promise.withResolvers();
    const { handle, state, client } = await session(t);
    if (level === "active") client.session.get = () => reply.promise;
    else client.global.health = () => reply.promise;
    const pending = handle.check(level);
    state.alive = false;
    reply.reject(new Error("PRIVATE-native-connection-error"));
    const result = await pending;
    assert.equal(result.outcome, "unknown");
    assert.equal(result.reason_code, "observation_changed");
    assert.equal(result.observed_at, null);
    assert.equal(JSON.stringify(result).includes("PRIVATE"), false);
    assert.equal(state.aborts, 0);
    assert.equal(state.closed, 0);
    assert.equal((await handle.check("live")).reason_code, "agent_exited");
  });
}

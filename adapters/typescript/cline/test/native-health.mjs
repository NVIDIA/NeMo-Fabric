// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Run with the supported @cline/sdk@0.0.83 installed. CLINE_NATIVE_SDK can
// select that installation's dist/index.js without changing package manifests.
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { ClineSdkSessionFactory } from "../dist/cline-sdk.js";

const sdk = await import(process.env.CLINE_NATIVE_SDK ?? "@cline/sdk");

test(
  "real Cline observes a resident native session before, during and after the first model request",
  { timeout: 30000 },
  async (t) => {
    const workspace = await mkdtemp(join(tmpdir(), "cline-native-health-"));
    t.after(() => rm(workspace, { recursive: true, force: true }));
    const entered = Promise.withResolvers();
    const release = Promise.withResolvers();
    let modelCalls = 0;
    let core;
    const loader = async () => ({
      ...sdk,
      ClineCore: {
        async create(options) {
          core = await sdk.ClineCore.create({
            ...options,
            fetch: async () => {
              modelCalls++;
              entered.resolve();
              await release.promise;
              return new Response(
                'data: {"id":"test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":null}]}\n\ndata: {"id":"test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
                { headers: { "content-type": "text/event-stream" } },
              );
            },
          });
          return core;
        },
      },
    });
    const handle = await new ClineSdkSessionFactory(loader).create({
      agentName: "health",
      baseDir: workspace,
      config: {
        models: {
          default: { provider: "nvidia", model: "test", api_key_env: "TEST_KEY", base_url: "http://127.0.0.1:1/v1" },
        },
        tools: { enabled: [] },
      },
      runtimeContext: {
        artifacts: {},
        environment: {
          environment_id: "env",
          provider: "local",
          ownership: "caller_owned",
          control_location: "external_control",
          workspace,
          env: { TEST_KEY: "PRIVATE" },
        },
        runtime_id: "runtime",
        invocation_id: "start",
        request_id: "start",
      },
    });
    t.after(() => handle.stop());
    const [sessionId, resident] = [...core.host.sessions][0];
    const observe = async (activity) => {
      const messages = JSON.stringify(resident.agent.getMessages());
      for (const level of ["live", "active", "ready"]) {
        const check = await handle.check(level);
        assert.equal(check.outcome, "passed");
        assert.equal(check.activity, activity);
        assert.doesNotMatch(JSON.stringify(check), /PRIVATE/);
      }
      assert.equal(JSON.stringify(resident.agent.getMessages()), messages);
      assert.equal(core.host.sessions.get(sessionId), resident);
    };
    await observe("idle");
    assert.equal(modelCalls, 0);
    let completed = false;
    const invocation = handle.prompt("PRIVATE prompt").finally(() => {
      completed = true;
    });
    await entered.promise;
    try {
      await observe("busy");
      assert.equal(completed, false);
      assert.equal(modelCalls, 1);
      const originalGet = core.get;
      const reply = Promise.withResolvers();
      let reads = 0;
      core.get = () => {
        reads++;
        return reply.promise;
      };
      const pending = handle.check("ready");
      assert.equal(
        await Promise.race([pending, new Promise((resolve) => setTimeout(() => resolve("deadline"), 20))]),
        "deadline",
      );
      for (let count = 0; count < 10; count++) assert.equal((await handle.check("ready")).reason_code, "probe_timeout");
      assert.equal(reads, 1);
      assert.equal(completed, false);
      reply.resolve(await originalGet(sessionId));
      await pending;
      core.get = originalGet;
      await observe("busy");
    } finally {
      release.resolve();
    }
    assert.equal((await invocation).status, "completed");
    await observe("idle");
    assert.equal(modelCalls, 1);
    await core.stop(sessionId);
    assert.ok(await core.get(sessionId), "persisted history remains readable after stop");
    assert.equal((await handle.check("live")).reason_code, "session_closed");
    await core.start({
      config: {
        providerId: "nvidia",
        modelId: "test",
        apiKey: "PRIVATE",
        baseUrl: "http://127.0.0.1:1/v1",
        cwd: workspace,
        workspaceRoot: workspace,
        mode: "act",
        enableTools: false,
        enableSpawnAgent: false,
        enableAgentTeams: false,
        sessionId,
      },
      interactive: true,
      source: "sdk",
      toolPolicies: {},
      localRuntime: { configExtensions: [] },
    });
    assert.notEqual(core.host.sessions.get(sessionId), resident);
    const replaced = await handle.check("live");
    assert.equal(replaced.outcome, "unknown");
    assert.equal(replaced.reason_code, "observation_changed");
    assert.equal(modelCalls, 1);
  },
);

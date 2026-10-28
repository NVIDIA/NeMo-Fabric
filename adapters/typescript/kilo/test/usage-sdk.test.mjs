// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createKiloClient } from "@kilocode/sdk/v2";
import { KiloSdkSessionFactory } from "../dist/kilo-sdk.js";
import { KiloAdapterRuntime } from "../dist/runtime.js";

const evidenceKey = "nvidia.fabric.kilo.usage_evidence.v1";

for (const mode of ["normal", "retry", "lost-events", "failed", "memory-enabled", "memory-unknown", "memory-event", "custom-config", "native-runtime"]) {
  test(`pinned SDK preserves invocation steps with ${mode} coverage`, async () => {
    const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-kilo-usage-")));
    const history = [];
    let events;
    let promptCount = 0;
    const send = value => { if (events && !events.writableEnded) events.write(`data: ${JSON.stringify(value)}\n\n`); };
    const server = createServer(async (request, response) => {
      const url = new URL(request.url, "http://localhost");
      for await (const _body of request) { /* Drain the SDK request. */ }
      if (url.pathname === "/event") {
        events = response;
        response.writeHead(200, { "content-type": "text/event-stream" });
        send({ type: "server.connected", properties: {} });
        return;
      }
      response.setHeader("content-type", "application/json");
      if (request.method === "POST" && url.pathname === "/session") {
        response.end(JSON.stringify({ id: "session-wire" }));
      } else if (url.pathname === "/memory/status") {
        response.end(JSON.stringify(mode === "memory-unknown" ? {} : { state: { enabled: mode === "memory-enabled" } }));
      } else if (request.method === "GET" && url.pathname === "/session/session-wire/message") {
        response.end(JSON.stringify(url.searchParams.has("limit") ? history.slice(-1) : history));
      } else if (request.method === "POST" && url.pathname === "/session/session-wire/message") {
        const user = `u${++promptCount}`;
        history.push({ info: { id: user, role: "user", sessionID: "session-wire" }, parts: [{ type: "text", text: "fixture" }] });
        for (const index of [1, 2]) {
          const info = { id: `${user}-a${index}`, role: "assistant", sessionID: "session-wire", parentID: user, time: { created: index } };
          send({ type: "message.updated", properties: { info: structuredClone(info) } });
          if (mode === "retry" && index === 1) send({ type: "session.status", properties: { sessionID: "session-wire", status: { type: "retry", attempt: 1 } } });
          if (mode === "memory-event" && index === 1) {
            send({ type: "memory.status", properties: { enabled: true } });
            send({ type: "memory.status", properties: { enabled: false } });
          }
          info.time.completed = index + 1;
          // Kilo stores input without cache and output without reasoning.
          info.tokens = { input: 10, output: 20, total: 36, reasoning: 3, cache: { read: 2, write: 1 } };
          info.cost = 9;
          const finished = { info, parts: [{ id: `${user}-step${index}`, type: "step-finish", tokens: info.tokens, cost: 9 }, { type: "text", text: "done" }] };
          history.push(finished);
          if (mode === "lost-events") events?.end();
          else send({ type: "message.updated", properties: { info: structuredClone(info) } });
        }
        if (mode === "failed") response.statusCode = 500;
        response.end(JSON.stringify(history.at(-1)));
      } else if (request.method === "DELETE" && url.pathname === "/session/session-wire") response.end("true");
      else { response.statusCode = 404; response.end("{}"); }
    });
    await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
    const url = `http://127.0.0.1:${server.address().port}`;
    let runtime;
    try {
      runtime = new KiloAdapterRuntime(new KiloSdkSessionFactory(async () => options => createKiloClient(options), async () => ({ url, close: async () => { } })));
      await runtime.start({
        agentName: "fixture", baseDir: workspace,
        config: { models: { default: { provider: "openai", model: "fixture", api_key_env: "MODEL_KEY" } } },
        runtimeContext: {
          runtime_id: "r", invocation_id: "i", artifacts: {}, environment: {
            environment_id: "e", provider: "local", ownership: "caller_owned", control_location: "external_control", workspace, env: {
              MODEL_KEY: "fixture", ...(mode === "custom-config" ? { KILO_CONFIG: "/fixture/custom-config.json" } : {}),
              ...(mode === "native-runtime" ? { KILO_EXPERIMENTAL_NATIVE_LLM: "true" } : {}),
            },
          },
        },
      });
      for (const _invocation of mode === "failed" ? [1] : [1, 2]) {
        const result = await runtime.invoke({ input: "fixture" }, {});
        assert.equal(result.status, mode === "failed" ? "failed" : "succeeded");
        if (mode !== "failed") assert.deepEqual(result.output, { response: "done" });
        assert.equal(result.usage.input_tokens, mode === "normal" ? 26 : undefined);
        assert.equal(result.usage.output_tokens, mode === "normal" ? 46 : undefined);
        assert.equal(result.usage.cost_usd, undefined);
        assert.deepEqual(result.usage.extensions[evidenceKey].observed, {
          input_tokens: 26, output_tokens: 46, total_tokens: 72, cached_input_tokens: 4, cache_write_input_tokens: 2, reasoning_tokens: 6,
        });
        assert.equal(result.usage.extensions[evidenceKey].gaps.length === 0, mode === "normal");
      }
    } finally {
      await runtime?.stop();
      events?.end();
      server.closeAllConnections();
      await new Promise(resolve => server.close(resolve));
      await rm(workspace, { recursive: true, force: true });
    }
  });
}

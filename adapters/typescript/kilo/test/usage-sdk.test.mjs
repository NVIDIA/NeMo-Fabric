// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import {createServer} from "node:http";
import {mkdir, mkdtemp, realpath, rename} from "node:fs/promises";
import {homedir, tmpdir} from "node:os";
import {basename, join} from "node:path";
import test from "node:test";
import {createKiloClient} from "@kilocode/sdk/v2";
import {KiloSdkSessionFactory} from "../dist/kilo-sdk.js";

for (const mode of ["normal", "retry", "lost-events"]) {
  test(`pinned Kilo SDK event wire preserves ${mode} accounting`, async () => {
    const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-kilo-sdk-")));
    const history = [];
    const requests = [];
    let events;
    const send = value => events?.write(`data: ${JSON.stringify(value)}\n\n`);
    const server = createServer(async (request, response) => {
      const url = new URL(request.url, "http://localhost");
      requests.push({method: request.method, path: url.pathname});
      for await (const _body of request) { /* Drain the fake control request. */ }
      if (url.pathname === "/event") {
        assert.equal(url.searchParams.get("directory"), workspace);
        events = response;
        response.writeHead(200, {"content-type": "text/event-stream"});
        send({type: "server.connected", properties: {}});
        return;
      }
      response.setHeader("content-type", "application/json");
      if (request.method === "POST" && url.pathname === "/session") {
        response.end(JSON.stringify({id: "session-wire"}));
      } else if (request.method === "GET" && url.pathname === "/session/session-wire/message") {
        response.end(JSON.stringify(url.searchParams.has("limit") ? history.slice(-1) : history));
      } else if (request.method === "POST" && url.pathname === "/session/session-wire/message") {
        const info = {id: "a", role: "assistant", sessionID: "session-wire", parentID: "u",
          providerID: "openai", modelID: "gpt-test", time: {created: 2}};
        history.push({info: {id: "u", role: "user", sessionID: "session-wire", time: {created: 1}},
          parts: [{type: "text", text: "fixture"}]});
        send({type: "message.updated", properties: {info: structuredClone(info)}});
        if (mode === "retry") send({type: "session.status", properties: {sessionID: "session-wire", status: {type: "retry", attempt: 1}}});
        info.time.completed = 3;
        const finished = {info, parts: [{id: "step", type: "step-finish", tokens: {input: 10, output: 20,
          reasoning: 0, cache: {read: 0, write: 0}}, cost: 9}, {type: "text", text: "done"}]};
        history.push(finished);
        if (mode === "lost-events") events.end();
        else send({type: "message.updated", properties: {info: structuredClone(info)}});
        response.end(JSON.stringify(finished));
      } else if (request.method === "DELETE" && url.pathname === "/session/session-wire") response.end("true");
      else { response.statusCode = 404; response.end("{}"); }
    });
    await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
    const url = `http://127.0.0.1:${server.address().port}`;
    let handle;
    try {
      const factory = new KiloSdkSessionFactory(async () => options => createKiloClient(options), async () => ({url, close: async () => {}}));
      handle = await factory.create({agentName: "fixture", baseDir: workspace,
        config: {models: {default: {provider: "openai", model: "gpt-test", api_key_env: "MODEL_KEY"}}},
        runtimeContext: {runtime_id: "r", invocation_id: "i", artifacts: {}, environment: {
          environment_id: "e", provider: "local", ownership: "caller_owned", control_location: "external_control",
          workspace, env: {MODEL_KEY: "local-not-a-key"},
        }},
      });
      const outcome = await handle.prompt("fixture");
      assert.equal(outcome.text, "done", "observability does not fail execution");
      assert.equal(outcome.usage.input_tokens, mode === "normal" ? 10 : undefined);
      assert.equal(outcome.usage.extensions["nemo_fabric.usage_accounting.v1"].observed.input_tokens, 10);
      assert.equal(outcome.usage.cost_usd, undefined);
      assert.ok(requests.some(request => request.path === "/event"));
    } finally {
      await handle?.stop();
      events?.end();
      server.closeAllConnections();
      await new Promise(resolve => server.close(resolve));
      const trash = join(homedir(), "trash", "nemo-fabric");
      await mkdir(trash, {recursive: true});
      await rename(workspace, join(trash, `${basename(workspace)}-${Date.now()}`));
    }
  });
}

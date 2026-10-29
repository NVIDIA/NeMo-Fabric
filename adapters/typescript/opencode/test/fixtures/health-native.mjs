// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { OpenCodeAdapterRuntime } from "../../dist/runtime.js";
import { OpenCodeSdkSessionFactory } from "../../dist/opencode-sdk.js";

const workspace = await mkdtemp(join(tmpdir(), "fabric-opencode-health-"));
let entered;
const enteredModel = new Promise(resolve => { entered = resolve; });
let finish;
const blocked = new Promise(resolve => { finish = resolve; });
let requests = 0;
const server = createServer(async (request, response) => {
  for await (const chunk of request) { void chunk; }
  requests++;
  entered();
  await blocked;
  response.writeHead(200, { "content-type": "text/event-stream" });
  const common = { id: "chatcmpl-health", object: "chat.completion.chunk", created: 0, model: "fabric-test-model" };
  response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: { role: "assistant", content: "answer" }, finish_reason: null }] })}\n\n`);
  response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } })}\n\n`);
  response.end("data: [DONE]\n\n");
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let native;
const factory = new OpenCodeSdkSessionFactory();
const runtime = new OpenCodeAdapterRuntime({ create: async input => { native = await factory.create(input); return native; } });
const context = { artifacts: {}, environment: { control_location: "external_control", env: { FABRIC_OPENCODE_TEST_KEY: "local-test-key" }, environment_id: "env-health", ownership: "caller_owned", provider: "local", workspace }, invocation_id: "invoke", request_id: "request", runtime_id: "runtime-health" };
try {
  await runtime.start({ agentName: "health", baseDir: workspace, config: { models: { default: { provider: "openai", model: "fabric-test-model", api_key_env: "FABRIC_OPENCODE_TEST_KEY", base_url: `http://127.0.0.1:${server.address().port}/v1` } } }, runtimeContext: context });
  for (const level of ["live", "active", "ready"]) assert.equal((await runtime.check(level)).outcome, "passed", level);
  assert.equal(requests, 0);
  const invocation = runtime.invoke({ input: "private prompt" }, context);
  await enteredModel;
  for (const level of ["live", "active", "ready"]) {
    const result = await runtime.check(level);
    assert.equal(result.outcome, "passed", level);
    assert.equal(result.activity, "busy");
    assert.doesNotMatch(JSON.stringify(result), /private prompt|local-test-key|sessionID/);
  }
  finish();
  assert.equal((await invocation).status, "succeeded");
  assert.equal((await runtime.check("ready")).activity, "idle");
  await native.stop();
  assert.equal((await runtime.check("live")).reason_code, "session_closed");
} finally {
  finish();
  await runtime.stop();
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
  await rm(workspace, { recursive: true, force: true });
}
assert.equal((await runtime.check("live")).reason_code, "runtime_unavailable");
console.log("native OpenCode health passed before, during, and after a blocked model response");

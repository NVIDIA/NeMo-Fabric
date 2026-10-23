// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import {mkdtemp} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import test from "node:test";
import {KiloSdkSessionFactory} from "../dist/kilo-sdk.js";
import {KiloAdapterRuntime} from "../dist/runtime.js";

const key = "nemo_fabric.usage_accounting.v1";
const assistant = (id, parentID, input) => ({info: {id, role: "assistant", sessionID: "root", parentID, modelID: "gpt-test", providerID: "openai", time: {created: 2, completed: 3}}, parts: [{id: id+"-step", type: "step-finish", tokens: {input, output: 2, reasoning: 0, cache: {read: 0, write: 0}}}]});
async function fixture(mode = "failure") {
  const workspace = await mkdtemp(join(tmpdir(), "kilo-usage-failure-"));
  const old = {info: {id: "previous", role: "user"}, parts: [{type: "text", text: "previous"}]};
  const own = assistant("own", "current", 10);
  const history = [old, assistant("previous-answer", "previous", 999)];
  let sends = 0;
  const client = {app: {skills: async () => ({data: []})}, mcp: {status: async () => ({data: {}})}, session: {
    create: async () => ({data: {id: "root"}}),
    messages: async ({limit}) => ({data: limit ? history.slice(-limit) : history}),
    prompt: async () => {
      sends++;
      history.push({info: {id: "current", role: "user"}, parts: [{type: "text", text: "current"}]}, own);
      if (mode === "timeout") return await new Promise(() => {});
      throw new Error("controlled transport loss after a billed step");
    },
    abort: async () => {}, delete: async () => {},
  }};
  const input = {agentName: "fixture", baseDir: workspace,
    config: {models: {default: {provider: "openai", model: "gpt-test", api_key_env: "MODEL_KEY"}}},
    runtimeContext: {runtime_id: "r", invocation_id: "start", request_id: "start", artifacts: {}, environment: {environment_id: "e", provider: "local", ownership: "caller_owned", control_location: "external_control", workspace, env: {MODEL_KEY: "fixture-only"}}},
  };
  const factory = new KiloSdkSessionFactory(async () => () => client, async () => ({url: "http://127.0.0.1:1", close: async () => {}}), mode === "timeout" ? 0 : 60_000);
  const handle = await factory.create(input);
  return {handle, input, own, sends: () => sends};
}
for (const mode of ["failure", "timeout"]) {
  test(`${mode} retains frozen owned usage without previous-step transfer`, async () => {
    const f = await fixture(mode);
    const error = await f.handle.prompt("current").then(() => assert.fail("prompt must fail"), value => value);
    assert.equal(error.code, mode === "timeout" ? "kilo_prompt_timeout" : "kilo_session_failed");
    assert.equal(error.retryable, false);
    assert.ok(error.usage, "failed SDK exchange must carry its observable billed steps");
    assert.equal(error.usage.input_tokens, undefined);
    assert.equal(error.usage.extensions[key].observed.input_tokens, 10);
    assert.ok(error.usage.extensions[key].gaps.some(gap => gap.reason === "unfinished_work"));
    const snapshot = structuredClone(error.usage);
    f.own.parts[0].tokens.input = 1000;
    assert.deepEqual(error.usage, snapshot);
    assert.equal(f.sends(), 1);
    // No process/server was started. Keep fixture files for audit; never delete.
  });
}
test("runtime preserves failure code and evidence while invalidating without replay", async () => {
  const f = await fixture();
  let stopped = 0;
  const runtime = new KiloAdapterRuntime({create: async () => ({id: f.handle.id, prompt: (...args) => f.handle.prompt(...args), stop: async () => {stopped++;}})});
  await runtime.start(f.input);
  const result = await runtime.invoke({input: "current"}, {});
  assert.equal(result.status, "failed");
  assert.equal(result.error.code, "kilo_session_failed");
  assert.equal(result.error.retryable, false);
  assert.equal(result.usage.extensions[key].observed.input_tokens, 10);
  assert.equal(stopped, 1);
  await assert.rejects(runtime.invoke({input: "again"}, {}), error => error.code === "kilo_not_started");
  assert.equal(f.sends(), 1);
});

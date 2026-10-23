// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import test from "node:test";
import {OpenCodeUsageCollector} from "../dist/usage.js";

const key = "nemo_fabric.usage_accounting.v1";
async function observed(requestURL, responseURL) {
  const callbacks = new Map();
  const collector = new OpenCodeUsageCollector();
  const cleanup = await collector.plugin().setup({session: {
    get: async () => ({}),
    hook: async (name, callback) => {callbacks.set(name, callback); return {dispose: async () => {}};},
  }});
  const owner = collector.begin("root", new Map([["root", {running: false}]]));
  const request = new Request(requestURL, {method: "POST", body: JSON.stringify({model: "gpt-4.1-mini"})});
  await callbacks.get("http.request")({sessionID: "root", request});
  const response = new Response(JSON.stringify({model: "gpt-4.1-mini", usage: {prompt_tokens: 10, completion_tokens: 2, total_tokens: 12}}));
  Object.defineProperty(response, "url", {value: responseURL});
  const event = {sessionID: "root", request, response};
  await callbacks.get("http.response")(event);
  await event.response.text();
  const result = owner.finish();
  await cleanup();
  return result;
}
for (const [name, requestURL, responseURL, provider] of [
  ["native", "https://api.openai.com/v1/chat/completions", "https://api.openai.com/v1/chat/completions", "openai"],
  ["native redirect", "https://api.openai.com/v1/chat/completions", "https://custom.fixture.invalid/final", undefined],
  ["custom redirect to native", "https://custom.fixture.invalid/v1/chat/completions", "https://api.openai.com/v1/chat/completions", undefined],
  ["nonstandard port", "https://api.openai.com:8443/v1/chat/completions", "https://api.openai.com:8443/v1/chat/completions", undefined],
]) {
  test(`${name} preserves model/counts without inventing provider pricing identity`, async () => {
    const result = await observed(requestURL, responseURL);
    assert.equal(result.input_tokens, 10);
    assert.equal(result.output_tokens, 2);
    const model = result.extensions[key].models[0];
    assert.equal(model.model, "gpt-4.1-mini");
    assert.equal(model.provider, provider);
  });
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { stream as streamOpenAI } from "@earendil-works/pi-ai/api/openai-completions";
import { openaiProvider } from "@earendil-works/pi-ai/providers/openai";
import { PiUsageCollector } from "../dist/usage.js";

const KEY = "nemo_fabric.usage_accounting.v1";
const native = openaiProvider().getModels().find(model => model.id === "gpt-4.1-mini");
assert.ok(native, "the locked Pi catalog contains the fixture model");
const events = { subscribe: () => () => {} };
const raw = { choices: [{finish_reason: "stop"}], model: native.id, usage: {prompt_tokens: 10, completion_tokens: 2, total_tokens: 12} };
const final = {role: "assistant", model: native.id, provider: "openai", stopReason: "stop", usage: {}};
const models = result => result.usage.extensions[KEY].models;

for (const [name, baseUrl, expected] of [
  ["native", native.baseUrl, "openai"],
  ["custom", "https://custom.fixture.invalid/v1", undefined],
]) {
  test(`actual Pi1.0.3 OpenAI parser keeps ${name} endpoint accounting identity accurate`, async () => {
    const requests = [];
    const selected = {...native, api: "openai-completions", baseUrl};
    const original = structuredClone(selected);
    const agent = {streamFunction: streamOpenAI};
    const collector = new PiUsageCollector(agent);
    const result = await collector.prompt(async () => {
      await (await agent.streamFunction(selected, {messages: [{role: "user", content: "fixture", timestamp: 0}]}, {
        apiKey: "controlled-fixture",
        fetch: async input => {
          const request = input instanceof Request ? input : new Request(input);
          requests.push(request.url);
          const chunk = value => `data: ${JSON.stringify({id: "fixture", object: "chat.completion.chunk", created: 0, model: native.id, ...value})}\n\n`;
          return new Response(chunk({choices: [{index: 0, delta: {role: "assistant", content: "done"}, finish_reason: null}]}) + chunk({choices: [{index: 0, delta: {}, finish_reason: "stop"}]}) + chunk({choices: [], usage: raw.usage}) + "data: [DONE]\n\n", {headers: {"content-type": "text/event-stream"}});
        },
      })).result();
    }, events);
    assert.equal(result.ok, true);
    assert.equal(result.usage.input_tokens, 10);
    assert.equal(result.usage.output_tokens, 2);
    assert.equal(models(result)[0].model, native.id);
    assert.equal(models(result)[0].provider, expected);
    assert.equal(requests.length, 1);
    assert.ok(requests[0].startsWith(baseUrl));
    assert.deepEqual(selected, original);
  });
}

test("callback model overrides and final nominal provider cannot restore custom endpoint pricing", async () => {
  const override = {...native, api: "openai-completions", baseUrl: "https://override.fixture.invalid/v1"};
  const agent = {streamFunction: async (_selected, _context, options) => {
    await options.onPayload({}, override);
    await options.onProviderStreamEvent(raw, override);
    return {result: async () => final};
  }};
  const collector = new PiUsageCollector(agent);
  const result = await collector.prompt(async () => {await (await agent.streamFunction(native, {}, {})).result();}, events);
  assert.equal(result.usage.input_tokens, 10);
  assert.equal(models(result)[0].provider, undefined);
});

test("sequential warmups independently qualify their overridden endpoint", async () => {
  const selected = {...native, api: "openai-completions"};
  let callbacks;
  const agent = {streamFunction: async (_selected, _context, options) => {
    callbacks = options;
    await options.onPayload({}, selected);
    await options.onProviderStreamEvent(raw, selected);
    return {result: async () => final};
  }};
  const collector = new PiUsageCollector(agent);
  const result = await collector.prompt(async () => {
    await (await agent.streamFunction(selected, {}, {})).result();
    await callbacks.onPayload({}, {...selected, baseUrl: "https://warmup.fixture.invalid/v1"});
    await callbacks.onProviderStreamEvent(raw, {...selected, baseUrl: "https://warmup.fixture.invalid/v1"});
  }, events);
  assert.equal(result.usage.input_tokens, 20);
  assert.equal(models(result).find(model => model.provider === "openai").requests.length, 1);
  assert.equal(models(result).find(model => model.provider === undefined).requests.length, 1);
});

test("the actual fetch route outranks a native-looking model and nominal event identity", async () => {
  const selected = {...native, api: "openai-completions"};
  const agent = {streamFunction: async (_selected, _context, options) => {
    await options.onPayload({}, selected);
    await options.fetch("https://routed.fixture.invalid/v1/chat/completions");
    await options.onProviderStreamEvent(raw, selected);
    return {result: async () => final};
  }};
  const collector = new PiUsageCollector(agent);
  const result = await collector.prompt(async () => {
    await (await agent.streamFunction(selected, {}, {fetch: async () => new Response("")})).result();
  }, events);
  assert.equal(result.usage.input_tokens, 10);
  assert.equal(models(result)[0].provider, undefined);
});

test("only HTTPS official endpoint and matching protocol qualify a model route", async () => {
  const routes = [
    ["openai-completions", "https://api.openai.com:443/v1", "openai"],
    ["anthropic-messages", "https://api.anthropic.com", "anthropic"],
    ["google-generative-ai", "https://generativelanguage.googleapis.com/v1beta", "google"],
    ["mistral-conversations", "https://api.mistral.ai", "mistral"],
    ["openai-completions", "http://api.openai.com/v1", undefined],
    ["openai-completions", "https://api.openai.com:8443/v1", undefined],
    ["openai-completions", "https://api.openai.com.other.invalid/v1", undefined],
    ["anthropic-messages", "https://api.openai.com/v1", undefined],
    ["openai-codex-responses", "https://chatgpt.com/backend-api", undefined],
    ["pi-messages", "https://api.openai.com/v1", undefined],
    ["openai-completions", undefined, undefined],
  ];
  for (const [api, baseUrl, expected] of routes) {
    const selected = {...native, api, baseUrl};
    const frame = api === "anthropic-messages"
      ? {type: "message_start", message: {usage: {input_tokens: 10}}}
      : api === "google-generative-ai" ? {usageMetadata: {promptTokenCount: 10}}
      : api === "openai-codex-responses" ? {type: "response.completed", response: {usage: {input_tokens: 10}}}
      : api === "pi-messages" ? {type: "done", usage: {input: 10}}
      : raw;
    const agent = {streamFunction: async (_selected, _context, options) => {
      await options.onPayload({}, selected);
      await options.onProviderStreamEvent(frame, selected);
      return {result: async () => final};
    }};
    const collector = new PiUsageCollector(agent);
    const result = await collector.prompt(async () => {await (await agent.streamFunction(selected, {}, {})).result();}, events);
    assert.equal(models(result)[0].provider, expected, `${api} ${baseUrl}`);
    assert.equal(result.usage.extensions[KEY].observed.input_tokens, 10);
  }
});

test("mixed retry routes and a custom redirect cannot recover native pricing identity", async () => {
  for (const redirect of [false, true]) {
    const selected = {...native, api: "openai-completions"};
    const agent = {streamFunction: async (_selected, _context, options) => {
      await options.onPayload({}, selected);
      if (!redirect) await options.fetch("https://custom.fixture.invalid/v1/chat/completions");
      await options.fetch("https://api.openai.com/v1/chat/completions");
      await options.onProviderStreamEvent(raw, selected);
      return {result: async () => final};
    }};
    const collector = new PiUsageCollector(agent);
    const result = await collector.prompt(async () => {
      await (await agent.streamFunction(selected, {}, {fetch: async () => {
        const response = new Response("");
        if (redirect) Object.defineProperty(response, "url", {value: "https://redirect.fixture.invalid/final"});
        return response;
      }})).result();
    }, events);
    assert.equal(models(result)[0].provider, undefined);
    assert.equal(result.usage.extensions[KEY].observed.input_tokens, 10);
  }
});

test("a native protocol override is qualified before its physical request", async () => {
  const override = {...native, id: "claude-fixture", api: "anthropic-messages", provider: "anthropic", baseUrl: "https://api.anthropic.com"};
  const agent = {streamFunction: async (_selected, _context, options) => {
    await options.onPayload({}, override);
    await options.fetch("https://api.anthropic.com/v1/messages");
    await options.onProviderStreamEvent({type: "message_start", message: {model: override.id, usage: {input_tokens: 10}}}, override);
    await options.onProviderStreamEvent({type: "message_delta", usage: {output_tokens: 2}, delta: {stop_reason: "end_turn"}}, override);
    return {result: async () => ({...final, model: override.id, provider: "anthropic"})};
  }};
  const collector = new PiUsageCollector(agent);
  const result = await collector.prompt(async () => {await (await agent.streamFunction(native, {}, {fetch: async () => new Response("")})).result();}, events);
  assert.equal(result.usage.input_tokens, 10);
  assert.equal(result.usage.output_tokens, 2);
  assert.equal(models(result)[0].provider, "anthropic");
  assert.equal(models(result)[0].model, override.id);
});

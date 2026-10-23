// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { OpenCodeUsageCollector } from "../dist/usage.js";

const KEY = "nemo_fabric.usage_accounting.v1";
const raw = (usage = {prompt_tokens: 10, completion_tokens: 2, total_tokens: 12, prompt_tokens_details: {cached_tokens: 0}}) => ({model: "model-a", usage});
const sse = value => `data: ${JSON.stringify(value)}\n\ndata: [DONE]\n\n`;

async function fixture(parents = new Map([["root", undefined], ["child", "root"]])) {
  const callbacks = new Map();
  const context = {session: {
    get: async ({sessionID}) => ({parentID: parents.get(sessionID)}),
    hook: async (name, callback) => {callbacks.set(name, callback); return {dispose: async () => callbacks.delete(name)};},
  }};
  const collector = new OpenCodeUsageCollector();
  const cleanup = await collector.plugin().setup(context);
  const before = new Map([["root", {running: false}]]);
  const owner = collector.begin("root", before);
  return {collector, owner, callbacks, cleanup, context};
}

async function call(fixture, {sessionID = "root", url = "https://example.invalid/v1/chat/completions", response = new Response(JSON.stringify(raw())), payload = {model: "model-a"}} = {}) {
  const request = new Request(url, {method: "POST", body: JSON.stringify(payload)});
  await fixture.callbacks.get("http.request")({sessionID, request});
  const event = {sessionID, request, response};
  await fixture.callbacks.get("http.response")(event);
  await event.response.text();
  return event;
}

test("literal field presence keeps reported zeros and independent omitted dimensions", async () => {
  const f = await fixture();
  await call(f, {response: new Response(JSON.stringify(raw({prompt_tokens: 0, completion_tokens: 2, total_tokens: 2})))});
  const result = f.owner.finish();
  assert.equal(result.input_tokens, 0);
  assert.equal(result.output_tokens, 2);
  assert.equal(result.cached_input_tokens, undefined);
  await f.cleanup();
});

test("main, compaction and a foreground child are physical calls counted once", async () => {
  const f = await fixture();
  await call(f);
  await call(f);
  await f.callbacks.get("prompt")({sessionID: "child"});
  await call(f, {sessionID: "child"});
  const result = f.owner.finish();
  assert.equal(result.input_tokens, 30);
  assert.equal(result.output_tokens, 6);
  assert.equal(result.cached_input_tokens, 0);
  assert.equal(result.extensions[KEY].models[0].requests.length, 3);
  assert.equal(result.cost_usd, undefined);
  await f.cleanup();
});

test("HTTP failure before success retains observed usage without inferred free failure", async () => {
  const f = await fixture();
  await call(f, {response: new Response("rejected", {status: 429})});
  await call(f);
  const result = f.owner.finish();
  assert.equal(result.input_tokens, undefined);
  assert.equal(result.extensions[KEY].observed.input_tokens, 10);
  await f.cleanup();
});

test("streaming is lazy, preserves bytes and requires the terminal marker", async () => {
  for (const complete of [true, false]) {
    const f = await fixture();
    const payload = complete ? sse(raw()) : `data: ${JSON.stringify(raw())}\n\n`;
    let reads = 0;
    const bytes = new TextEncoder().encode(payload);
    const stream = new ReadableStream({pull(controller) {reads++; controller.enqueue(bytes); controller.close();}}, {highWaterMark: 0});
    const request = new Request("https://example.invalid/v1/chat/completions", {method: "POST", body: '{"model":"model-a","stream":true}'});
    await f.callbacks.get("http.request")({sessionID: "root", request});
    const event = {sessionID: "root", request, response: new Response(stream, {headers: {"content-type": "text/event-stream"}})};
    await f.callbacks.get("http.response")(event);
    assert.equal(reads, 0);
    assert.equal(await event.response.text(), payload);
    const result = f.owner.finish();
    assert.equal(result.input_tokens, complete ? 10 : undefined);
    assert.equal(result.extensions[KEY].observed.input_tokens, 10);
    await f.cleanup();
  }
});

test("malformed later stream evidence cannot certify an earlier usage snapshot", async () => {
  const f = await fixture();
  await call(f, {response: new Response(`data: ${JSON.stringify(raw())}\n\ndata: {bad json}\n\ndata: [DONE]\n\n`, {headers: {"content-type": "text/event-stream"}})});
  const result = f.owner.finish();
  assert.equal(result.input_tokens, undefined);
  assert.equal(result.extensions[KEY].observed.input_tokens, 10);
  assert.ok(result.extensions[KEY].gaps.some(gap => gap.reason === "scope_unverified"));
  await f.cleanup();
});

test("a native alias on a custom route does not authorize native provider pricing", async () => {
  for (const [url, expected] of [["https://api.openai.com/v1/chat/completions", "openai"], ["https://other.invalid/v1/chat/completions", undefined], ["http://api.openai.com/v1/chat/completions", undefined]]) {
    const f = await fixture();
    await call(f, {url, payload: {model: "gpt-4.1-mini"}, response: new Response(JSON.stringify({model: "gpt-4.1-mini", usage: raw().usage}))});
    assert.equal(f.owner.calls[0].provider, expected);
    assert.equal(f.owner.calls[0].model, "gpt-4.1-mini");
    await f.cleanup();
  }
});

test("a late child response stays with its initiating owner and cannot change a frozen result", async () => {
  const f = await fixture();
  await f.callbacks.get("prompt")({sessionID: "child"});
  const request = new Request("https://example.invalid/v1/chat/completions", {method: "POST", body: '{"model":"model-a"}'});
  await f.callbacks.get("http.request")({sessionID: "child", request});
  const frozen = structuredClone(f.owner.finish());
  const second = f.collector.begin("root", new Map([["root", {running: false}], ["child", {running: true}]]));
  const event = {sessionID: "child", request, response: new Response(JSON.stringify(raw()))};
  await f.callbacks.get("http.response")(event);
  await event.response.text();
  assert.deepEqual(f.owner.finish(), frozen);
  assert.equal(second.calls.length, 0);
  // A later request by the same unfinished child still belongs to the old owner.
  await call(f, {sessionID: "child"});
  assert.equal(second.calls.length, 0);
  await f.cleanup();
});

test("explicitly reusing a settled child captures the new initiating invocation", async () => {
  const f = await fixture();
  await f.callbacks.get("prompt")({sessionID: "child"});
  await call(f, {sessionID: "child"});
  f.owner.finish();
  const second = f.collector.begin("root", new Map([["root", {running: false}], ["child", {running: false}]]));
  await f.callbacks.get("prompt")({sessionID: "child"});
  await call(f, {sessionID: "child"});
  assert.equal(second.finish().input_tokens, 10);
  await f.cleanup();
});

test("native Anthropic stream input and final output stay independently present", async () => {
  const f = await fixture();
  const events = [
    {type: "message_start", message: {model: "claude-fixture", usage: {input_tokens: 10, cache_read_input_tokens: 0, cache_creation_input_tokens: 0, output_tokens: 0}}},
    {type: "message_delta", usage: {output_tokens: 2}},
    {type: "message_stop"},
  ];
  await call(f, {url: "https://api.anthropic.com/v1/messages", response: new Response(events.map(event => `data: ${JSON.stringify(event)}\n\n`).join(""), {headers: {"content-type": "text/event-stream"}})});
  const result = f.owner.finish();
  assert.equal(result.input_tokens, 10);
  assert.equal(result.input_tokens_include_cache, false);
  assert.equal(result.output_tokens, 2);
  assert.equal(result.cached_input_tokens, 0);
  await f.cleanup();
});

test("Responses usage is read from the terminal response without losing a reported zero", async () => {
  const f = await fixture();
  await call(f, {url: "https://api.openai.com/v1/responses", response: new Response(`data: ${JSON.stringify({type: "response.completed", response: {model: "fixture", usage: {input_tokens: 0, output_tokens: 2, total_tokens: 2}}})}\n\n`, {headers: {"content-type": "text/event-stream"}})});
  assert.equal(f.owner.finish().input_tokens, 0);
  assert.equal(f.owner.finish().output_tokens, 2);
  await f.cleanup();
});

test("an actual WebSocket dispatch leaves scope unknown without inventing an HTTP call", async () => {
  const f = await fixture();
  await f.callbacks.get("experimental.ws.handshake")({sessionID: "root"});
  const result = f.owner.finish();
  assert.equal(result.input_tokens, undefined);
  assert.ok(result.extensions[KEY].gaps.some(gap => gap.reason === "scope_unverified"));
  assert.equal(f.owner.calls.length, 0);
  await f.cleanup();
});

test("a declared usage extension validates the accounting transport rather than allowing arbitrary keys", async () => {
  const {readFile} = await import("node:fs/promises");
  const descriptor = JSON.parse(await readFile(new URL("../opencode.fabric-adapter.json", import.meta.url), "utf8"));
  const schema = descriptor.extension_schemas.usage;
  assert.equal(schema.additionalProperties, false);
  assert.ok(schema.properties[KEY]);
  assert.ok(schema.$defs.UsageGap);
});

test("earlier background completion suppresses ambiguous parent continuations while retaining earlier owned calls", async () => {
  const f = await fixture();
  await f.callbacks.get("prompt")({sessionID: "child"});
  f.owner.finish();
  let history = [{id: "boundary", type: "assistant"}];
  f.context.session.context = async () => history;
  const owner = f.collector.begin("root", new Map([["root", {running: false, newestMessageId: "boundary"}], ["child", {running: true}]]), new Set(["boundary"]));
  await call(f);
  history = [...history, {id: "late-notice", type: "synthetic", metadata: {source: "subagent", childID: "child"}}];
  await call(f);
  const result = owner.finish();
  assert.equal(result.input_tokens, undefined);
  assert.equal(result.extensions[KEY].observed.input_tokens, 10);
  assert.equal(result.extensions[KEY].models[0].requests.length, 1);
  // A later invocation's pre-existing notice does not poison normal owned work.
  const later = f.collector.begin("root", new Map([["root", {running: false}], ["child", {running: false}]]), new Set(["boundary", "late-notice"]));
  await call(f);
  assert.equal(later.finish().input_tokens, 10);
  await f.cleanup();
});

test("unreadable continuation context cannot transfer possibly earlier-owned observations", async () => {
  const f = await fixture();
  await f.callbacks.get("prompt")({sessionID: "child"});
  f.owner.finish();
  f.context.session.context = async () => {throw new Error("unavailable");};
  const owner = f.collector.begin("root", new Map([["root", {running: false}], ["child", {running: true}]]));
  await call(f);
  const result = owner.finish();
  assert.equal(result.extensions[KEY].observed.input_tokens, undefined);
  assert.ok(result.extensions[KEY].gaps.some(gap => gap.reason === "scope_unverified"));
  await f.cleanup();
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createServer } from "node:http";
import test from "node:test";
import { crc32 } from "node:zlib";

import { encodeKiroEvent, startKiroByomBridge, translateKiroRequest } from "../dist/byom.js";

// Narrow version of the Kiro 2.34.2 tool-cycle request: retain protocol fields,
// without copying native system instructions or private workspace paths.
function requestBody() {
  return { conversationState: {
    history: [
      { userInputMessage: { content: "System instructions supplied by Kiro." } },
      { assistantResponseMessage: { content: "Understood." } },
      { userInputMessage: { content: "Inspect the workspace." } },
      { assistantResponseMessage: { content: "", toolUses: [
        { toolUseId: "tool-1", name: "execute_bash", input: { command: "pwd" } },
      ] } },
    ],
    currentMessage: { userInputMessage: { content: "", userInputMessageContext: {
      toolResults: [{ toolUseId: "tool-1", content: [{ text: "Output:\n/workspace" }, { text: "Exit Code: 0" }], status: "success" }],
      tools: [{ toolSpecification: {
        name: "execute_bash", description: "Run a command",
        inputSchema: { json: { type: "object", properties: { command: { type: "string" } }, required: ["command"] } },
      } }],
    } } },
  } };
}

function completion(usage = { prompt_tokens: 10, completion_tokens: 4 }, overrides = {}) {
  return { choices: [{ message: { role: "assistant", content: "Finished." }, finish_reason: "stop" }], usage, ...overrides };
}

function readEvents(bytes) {
  const events = [];
  for (let start = 0; start < bytes.length;) {
    const size = bytes.readUInt32BE(start);
    const headersSize = bytes.readUInt32BE(start + 4);
    assert.ok(size >= 16 && start + size <= bytes.length, "complete EventStream frame");
    const frame = bytes.subarray(start, start + size);
    assert.equal(frame.readUInt32BE(8), crc32(frame.subarray(0, 8)), "prelude CRC");
    assert.equal(frame.readUInt32BE(size - 4), crc32(frame.subarray(0, size - 4)), "message CRC");
    const headers = {};
    for (let offset = 12; offset < 12 + headersSize;) {
      const nameSize = frame.readUInt8(offset++);
      const name = frame.subarray(offset, offset + nameSize).toString();
      offset += nameSize;
      assert.equal(frame.readUInt8(offset++), 7, "string EventStream header");
      const valueSize = frame.readUInt16BE(offset);
      offset += 2;
      headers[name] = frame.subarray(offset, offset + valueSize).toString();
      offset += valueSize;
      assert.ok(offset <= 12 + headersSize);
    }
    assert.equal(headers[":message-type"], "event");
    assert.equal(headers[":content-type"], "application/json");
    events.push({ type: headers[":event-type"], payload: JSON.parse(frame.subarray(12 + headersSize, size - 4).toString()) });
    start += size;
  }
  return events;
}

async function bridgeFixture(t, handler) {
  const requests = [];
  const upstream = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    requests.push({ path: request.url, authorization: request.headers.authorization, body: JSON.parse(Buffer.concat(chunks).toString()) });
    await handler(requests.length, response);
  });
  await new Promise(resolve => upstream.listen(0, "127.0.0.1", resolve));
  const bridge = await startKiroByomBridge({
    model: "configured-model", baseUrl: `http://127.0.0.1:${upstream.address().port}/v1`, apiKey: "provider-test-key",
  });
  let stopped;
  const stopBridge = () => stopped ??= bridge.stop();
  t.after(async () => {
    await stopBridge();
    upstream.closeAllConnections();
    await new Promise(resolve => upstream.close(resolve));
  });
  const send = (body = requestBody(), options = {}) => fetch(`${bridge.url}${options.path ?? "/generateAssistantResponse"}`, {
    method: options.method ?? "POST",
    headers: { authorization: `Bearer ${bridge.token}`, "content-type": "application/json", ...options.headers },
    body: options.method === "GET" ? undefined : JSON.stringify(body),
    ...(options.signal === undefined ? {} : { signal: options.signal }),
  });
  return { bridge, requests, send, stopBridge };
}

function jsonReply(response, payload) {
  response.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify(payload));
}

test("Kiro tool history and results preserve native order and JSON schemas", () => {
  const body = requestBody();
  const original = structuredClone(body);
  assert.deepEqual(translateKiroRequest(body, "selected-model"), {
    model: "selected-model", stream: false,
    messages: [
      { role: "user", content: "System instructions supplied by Kiro." },
      { role: "assistant", content: "Understood." },
      { role: "user", content: "Inspect the workspace." },
      { role: "assistant", content: "", tool_calls: [{ id: "tool-1", type: "function", function: { name: "execute_bash", arguments: '{"command":"pwd"}' } }] },
      { role: "tool", tool_call_id: "tool-1", content: "Output:\n/workspace\nExit Code: 0" },
    ],
    tools: [{ type: "function", function: {
      name: "execute_bash", description: "Run a command",
      parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"] },
    } }],
  });
  assert.deepEqual(body, original, "translation must not mutate retained Kiro history");
});

for (const field of ["images", "documents"]) {
  test(`Kiro unsupported ${field} fail closed`, () => {
    const body = requestBody();
    body.conversationState.currentMessage.userInputMessage[field] = [{ data: "unsupported" }];
    assert.throws(() => translateKiroRequest(body, "selected-model"));
  });
}

test("Kiro EventStream encoding has independently verified lengths, CRCs and UTF-8 payload", () => {
  const payload = { content: "Résumé λ", modelId: "configured-model" };
  const frame = encodeKiroEvent("assistantResponseEvent", payload);
  assert.equal(frame.readUInt32BE(0), frame.length);
  assert.deepEqual(readEvents(frame), [{ type: "assistantResponseEvent", payload }]);
});

test("Kiro bridge authenticates routes and fences generation outside invocations", async t => {
  const { bridge, requests, send } = await bridgeFixture(t, (_number, response) => jsonReply(response, completion()));
  assert.equal((await send(undefined, { headers: { authorization: "Bearer wrong-token" } })).status, 401);
  assert.equal((await send(undefined, { path: "/unknown" })).status, 404);
  assert.equal((await send(undefined, { method: "GET" })).status, 405);
  assert.equal((await send()).status, 409);
  assert.equal(requests.length, 0);
  assert.equal(bridge.endInvocation(), undefined);
  bridge.beginInvocation();
  assert.throws(() => bridge.beginInvocation());
  const models = await send(undefined, { method: "GET", path: "/List-Available-Models" });
  assert.equal(models.status, 200);
  assert.equal((await models.json()).models[0].modelId, "configured-model");
  assert.equal(bridge.endInvocation(), undefined, "metadata requests must not fabricate model usage");
});

test("Kiro bridge sums every provider call once and keeps cache/reasoning inside parent totals", async t => {
  const { bridge, requests, send } = await bridgeFixture(t, (number, response) => jsonReply(response, completion({
    prompt_tokens: number * 10, completion_tokens: number * 4,
    prompt_tokens_details: { cached_tokens: number * 3 }, completion_tokens_details: { reasoning_tokens: number * 2 },
  })));
  bridge.beginInvocation();
  const first = await send(undefined, { path: "/", headers: { "x-amz-target": "KiroService.GenerateAssistantResponse" } });
  assert.equal(first.status, 200);
  assert.match(first.headers.get("content-type"), /amazon\.eventstream/);
  const events = readEvents(Buffer.from(await first.arrayBuffer()));
  assert.deepEqual(events.map(event => event.type), ["assistantResponseEvent", "metadataEvent"]);
  assert.equal(events.at(-1).payload.stopReason, "END_TURN");
  const second = await send();
  assert.equal(second.status, 200);
  await second.arrayBuffer();
  assert.equal(requests.length, 2);
  for (const request of requests) {
    assert.equal(request.path, "/v1/chat/completions");
    assert.equal(request.authorization, "Bearer provider-test-key");
    assert.deepEqual(request.body, translateKiroRequest(requestBody(), "configured-model"));
  }
  assert.deepEqual(bridge.endInvocation(), {
    input_tokens: 30, output_tokens: 12, total_tokens: 42, cached_input_tokens: 9, input_tokens_include_cache: true,
  });
  bridge.beginInvocation();
  const third = await send();
  await third.arrayBuffer();
  assert.equal(bridge.endInvocation().input_tokens, 30, "new invocation excludes previous calls");
});

test("Kiro bridge emits native fragmented tool-use events and tool stop reason", async t => {
  const { bridge, send } = await bridgeFixture(t, (_number, response) => jsonReply(response, completion(undefined, {
    choices: [{ message: { role: "assistant", content: null, tool_calls: [
      { id: "tool-next", type: "function", function: { name: "execute_bash", arguments: '{"command":"ls"}' } },
    ] }, finish_reason: "tool_calls" }],
  })));
  bridge.beginInvocation();
  const response = await send();
  assert.equal(response.status, 200);
  assert.deepEqual(readEvents(Buffer.from(await response.arrayBuffer())), [
    { type: "toolUseEvent", payload: { name: "execute_bash", toolUseId: "tool-next" } },
    { type: "toolUseEvent", payload: { name: "execute_bash", toolUseId: "tool-next", input: '{"command":"ls"}' } },
    { type: "toolUseEvent", payload: { name: "execute_bash", toolUseId: "tool-next", stop: true } },
    { type: "metadataEvent", payload: { stopReason: "TOOL_USE" } },
  ]);
});

for (const [name, usage, expected] of [
  ["explicit zero", { prompt_tokens: 0, completion_tokens: 0, prompt_tokens_details: { cached_tokens: 0 } }, { input_tokens: 0, output_tokens: 0, total_tokens: 0, cached_input_tokens: 0, input_tokens_include_cache: true }],
  ["missing cache", { prompt_tokens: 8, completion_tokens: 3 }, { input_tokens: 8, output_tokens: 3, total_tokens: 11, input_tokens_include_cache: true }],
  ["invalid cache", { prompt_tokens: 8, completion_tokens: 3, prompt_tokens_details: { cached_tokens: 9 } }, { input_tokens: 8, output_tokens: 3, total_tokens: 11, input_tokens_include_cache: true }],
  ["missing input", { completion_tokens: 3 }, undefined],
  ["missing output", { prompt_tokens: 8 }, undefined],
  ["negative input", { prompt_tokens: -1, completion_tokens: 3 }, undefined],
  ["fractional output", { prompt_tokens: 8, completion_tokens: 0.5 }, undefined],
]) {
  test(`Kiro provider usage distinguishes ${name}`, async t => {
    const { bridge, send } = await bridgeFixture(t, (_number, response) => jsonReply(response, completion(usage)));
    bridge.beginInvocation();
    const response = await send();
    assert.equal(response.status, 200);
    await response.arrayBuffer();
    assert.deepEqual(bridge.endInvocation(), expected);
  });
}

test("Kiro missing usage on one request invalidates the whole invocation sum", async t => {
  const { bridge, send } = await bridgeFixture(t, (number, response) => jsonReply(response, {
    ...completion(), ...(number === 1 ? {} : { usage: undefined }),
  }));
  bridge.beginInvocation();
  for (let number = 0; number < 2; number++) await (await send()).arrayBuffer();
  assert.equal(bridge.endInvocation(), undefined);
});

for (const failure of ["http", "invalid-json", "unsupported-finish", "oversized-body"]) {
  test(`Kiro provider ${failure} marks invocation usage unknown`, async t => {
    const { bridge, send } = await bridgeFixture(t, (number, response) => {
      if (number === 1) return jsonReply(response, completion());
      if (failure === "http") return response.writeHead(500).end("private provider explanation");
      if (failure === "invalid-json") return response.writeHead(200).end("not JSON");
      if (failure === "oversized-body") return response.writeHead(200).end("x".repeat(8 * 1024 * 1024 + 1));
      return jsonReply(response, completion(undefined, { choices: [{ message: { content: "blocked" }, finish_reason: "content_filter" }] }));
    });
    bridge.beginInvocation();
    await (await send()).arrayBuffer();
    const failed = await send();
    assert.equal(failed.status, 502);
    assert.doesNotMatch(await failed.text(), /private provider explanation/);
    assert.equal(bridge.endInvocation(), undefined);
  });
}

test("Kiro oversized request fails before any provider call", async t => {
  const { bridge, requests, send } = await bridgeFixture(t, (_number, response) => jsonReply(response, completion()));
  bridge.beginInvocation();
  const body = requestBody();
  body.conversationState.currentMessage.userInputMessage.content = "x".repeat(8 * 1024 * 1024 + 1);
  const response = await send(body);
  assert.ok(response.status >= 400);
  await response.arrayBuffer();
  assert.equal(requests.length, 0);
  assert.equal(bridge.endInvocation(), undefined);
});

test("Kiro endInvocation aborts a pending provider request and leaves usage unknown", { timeout: 5000 }, async t => {
  const started = Promise.withResolvers();
  const closed = Promise.withResolvers();
  const { bridge, send } = await bridgeFixture(t, (_number, response) => {
    response.on("close", closed.resolve);
    started.resolve();
  });
  bridge.beginInvocation();
  const pending = send();
  await started.promise;
  assert.equal(bridge.endInvocation(), undefined);
  assert.throws(() => bridge.beginInvocation(), "an aborted provider request still belongs to the ended invocation");
  await closed.promise;
  assert.equal((await pending).status, 502);
});

test("Kiro client abort cancels upstream work and cannot report a complete invocation", { timeout: 5000 }, async t => {
  const started = Promise.withResolvers();
  const closed = Promise.withResolvers();
  const { bridge, send } = await bridgeFixture(t, (_number, response) => {
    response.on("close", closed.resolve);
    started.resolve();
  });
  bridge.beginInvocation();
  const controller = new AbortController();
  const pending = send(undefined, { signal: controller.signal });
  const rejected = assert.rejects(pending, { name: "AbortError" });
  await started.promise;
  controller.abort();
  await rejected;
  await closed.promise;
  assert.equal(bridge.endInvocation(), undefined);
});

test("Kiro stopped bridge aborts provider work and rejects a new invocation", { timeout: 5000 }, async t => {
  const started = Promise.withResolvers();
  const closed = Promise.withResolvers();
  const { bridge, send, stopBridge } = await bridgeFixture(t, (_number, response) => {
    response.on("close", closed.resolve);
    started.resolve();
  });
  bridge.beginInvocation();
  const pending = send().then(response => response.status, error => error.name);
  await started.promise;
  await stopBridge();
  await closed.promise;
  assert.ok([502, "TypeError"].includes(await pending), "stopping must not return a successful model completion");
  assert.throws(() => bridge.beginInvocation());
});

test("Kiro provider error does not poison the next completed invocation", async t => {
  const { bridge, send } = await bridgeFixture(t, (number, response) => {
    if (number === 1) return response.writeHead(500).end();
    jsonReply(response, completion({ prompt_tokens: 6, completion_tokens: 2 }));
  });
  bridge.beginInvocation();
  assert.equal((await send()).status, 502);
  assert.equal(bridge.endInvocation(), undefined);
  bridge.beginInvocation();
  const response = await send();
  await response.arrayBuffer();
  assert.deepEqual(bridge.endInvocation(), {
    input_tokens: 6, output_tokens: 2, total_tokens: 8, input_tokens_include_cache: true,
  });
});

for (const [name, input] of [
  ["invoke_sub_agent", { prompt: "Inspect a file" }],
  ["execute_bash", { command: "echo ready", run_in_background: true }],
]) {
  test(`Kiro ${name} asynchronous work latches session accounting unknown`, async t => {
    const { bridge, send } = await bridgeFixture(t, (number, response) => {
      const choices = number === 1 ? [{ message: { role: "assistant", content: null, tool_calls: [
        { id: "async-tool", type: "function", function: { name, arguments: JSON.stringify(input) } },
      ] }, finish_reason: "tool_calls" }] : completion().choices;
      jsonReply(response, completion({ prompt_tokens: 10, completion_tokens: 4 }, { choices }));
    });
    bridge.beginInvocation();
    const first = await send();
    assert.equal(first.status, 200);
    await first.arrayBuffer();
    assert.equal(bridge.endInvocation(), undefined, "root ACP terminal does not fence asynchronous work");
    bridge.beginInvocation();
    const second = await send();
    assert.equal(second.status, 200);
    await second.arrayBuffer();
    assert.equal(bridge.endInvocation(), undefined, "later invocation must not absorb asynchronous model usage");
  });
}

test("Kiro tool failures retain their error status in OpenAI tool content", () => {
  const body = requestBody();
  body.conversationState.currentMessage.userInputMessage.userInputMessageContext.toolResults[0].status = "error";
  const translated = translateKiroRequest(body, "selected-model");
  assert.deepEqual(translated.messages.at(-1), {
    role: "tool", tool_call_id: "tool-1", content: "Tool error:\nOutput:\n/workspace\nExit Code: 0",
  });
});

for (const status of [undefined, "cancelled", "other"]) {
  test(`Kiro unknown tool-result status fails closed: ${status}`, () => {
    const body = requestBody();
    body.conversationState.currentMessage.userInputMessage.userInputMessageContext.toolResults[0].status = status;
    assert.throws(() => translateKiroRequest(body, "selected-model"));
  });
}

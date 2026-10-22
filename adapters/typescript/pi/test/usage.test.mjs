// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtemp, realpath, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { PiUsageCollector } from "../dist/usage.js";
import { PiSdkSessionFactory } from "../dist/pi-sdk.js";
import { PiAdapterRuntime } from "../dist/runtime.js";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";

const KEY = "nemo_fabric.usage_accounting.v1";
const model = { id: "m", provider: "test", api: "openai-completions" };
const raw = (input = 10, output = 2) => ({ choices: [], usage: { prompt_tokens: input, completion_tokens: output, prompt_tokens_details: { cached_tokens: 0 } } });
const message = (usage = { input: 10, output: 2, cacheRead: 0, cacheWrite: 0 }) => ({ role: "assistant", content: [{ type: "text", text: "done" }], stopReason: "stop", model: "m", provider: "test", usage });
function source(stream, options = {}) {
  const listeners = new Set();
  const agent = { streamFunction: stream };
  const events = { subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); } };
  const emit = event => { for (const listener of listeners) listener(event); };
  const collector = new PiUsageCollector(agent, options);
  const call = async (selected = model, callOptions = {}) => {
    const result = await (await agent.streamFunction(selected, { messages: [] }, callOptions)).result();
    emit({ type: "message_end", message: result });
    return result;
  };
  return { agent, events, emit, collector, call };
}
function streamFor(frames, final = message()) {
  return async (selected, _context, options) => {
    await options.onPayload?.({}, selected);
    for (const frame of frames) await options.onProviderStreamEvent?.(frame, selected);
    return { result: async () => final };
  };
}

test("reported zero stays zero while a Pi-zero-filled omitted field stays unknown", async () => {
  const omitted = source(streamFor([{ usage: { prompt_tokens: 10 } }]));
  const first = await omitted.collector.prompt(omitted.call, omitted.events);
  assert.equal(first.usage.input_tokens, 10);
  assert.equal(first.usage.output_tokens, undefined);
  assert.equal(first.usage.cached_input_tokens, undefined);
  const zeros = source(streamFor([raw(0, 0)], message({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 })));
  const second = await zeros.collector.prompt(zeros.call, zeros.events);
  assert.equal(second.usage.input_tokens, 0);
  assert.equal(second.usage.output_tokens, 0);
  assert.equal(second.usage.cached_input_tokens, 0);
});

test("a failed terminal call keeps literal usage; a rejection without usage never fabricates zero", async () => {
  const failed = source(streamFor([{ ...raw(), choices: [{ finish_reason: "stop" }] }], { ...message(), stopReason: "error" }));
  const counted = await failed.collector.prompt(failed.call, failed.events);
  assert.equal(counted.usage.input_tokens, 10);
  assert.equal(counted.usage.output_tokens, 2);
  const rejected = source(streamFor([], { ...message(), stopReason: "error" }));
  const absent = await rejected.collector.prompt(rejected.call, rejected.events);
  assert.equal(absent.usage.input_tokens, undefined);
  assert.equal(absent.usage.output_tokens, undefined);
  assert.deepEqual(absent.usage.extensions[KEY].observed, {});
});

test("Bedrock usage stays observed because SDK attempt counts are unavailable", async () => {
  const selected = { ...model, api: "bedrock-converse-stream" };
  const harness = source(streamFor([
    { messageStop: { stopReason: "end_turn" } },
    { metadata: { usage: { inputTokens: 10, outputTokens: 2, totalTokens: 12, cacheReadInputTokens: 0, cacheWriteInputTokens: 0 } } },
  ]));
  const result = await harness.collector.prompt(() => harness.call(selected), harness.events);
  assert.equal(result.usage.input_tokens, undefined);
  assert.equal(result.usage.output_tokens, undefined);
  assert.equal(result.usage.extensions[KEY].observed.input_tokens, 10);
  assert.equal(result.usage.extensions[KEY].observed.output_tokens, 2);
  assert(result.usage.extensions[KEY].gaps.some(gap => gap.reason === "scope_unverified"));
});

for (const api of ["google-generative-ai", "google-vertex"]) {
  test(`${api} only certifies usage when effective provider retries are disabled`, async () => {
    const selected = { ...model, api };
    const frames = [{ candidates: [{ finishReason: "STOP" }], usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 2, totalTokenCount: 12, cachedContentTokenCount: 0 } }];
    const harness = source(streamFor(frames), { providerMaxRetries: () => 1 });
    const hidden = await harness.collector.prompt(() => harness.call(selected), harness.events);
    assert.equal(hidden.usage.input_tokens, undefined);
    assert.equal(hidden.usage.extensions[KEY].observed.input_tokens, 10);
    assert(hidden.usage.extensions[KEY].gaps.some(gap => gap.reason === "scope_unverified"));
    const disabled = await harness.collector.prompt(() => harness.call(selected, { maxRetries: 0 }), harness.events);
    assert.equal(disabled.usage.input_tokens, 10);
    assert.equal(disabled.usage.output_tokens, 2);
    const defaultHarness = source(streamFor(frames));
    const normal = await defaultHarness.collector.prompt(() => defaultHarness.call(selected), defaultHarness.events);
    assert.equal(normal.usage.input_tokens, 10);
    assert.equal(normal.usage.output_tokens, 2);
  });
}

for (const code of ["previous_response_not_found", "websocket_connection_limit_reached"]) {
  test(`Codex WebSocket ${code} retains recovered usage without certifying hidden attempts`, async () => {
    const selected = { ...model, api: "openai-codex-responses" };
    for (const error of [{ type: "error", code }, { type: "response.failed", response: { error: { code } } }]) {
      const harness = source(streamFor([
        error,
        { type: "response.completed", response: { usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12 } } },
      ]));
      const result = await harness.collector.prompt(() => harness.call(selected), harness.events);
      assert.equal(result.usage.input_tokens, undefined);
      assert.equal(result.usage.output_tokens, undefined);
      assert.equal(result.usage.extensions[KEY].observed.input_tokens, 10);
      assert.equal(result.usage.extensions[KEY].observed.output_tokens, 2);
      assert(result.usage.extensions[KEY].gaps.some(gap => gap.reason === "scope_unverified"));
    }
    const normal = source(streamFor([
      { type: "response.completed", response: { usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12 } } },
    ]));
    const known = await normal.collector.prompt(() => normal.call(selected), normal.events);
    assert.equal(known.usage.input_tokens, 10);
  });
}

for (const api of ["openai-responses", "openai-codex-responses"]) {
  test(`${api} keeps reported usage from a failed terminal response`, async () => {
    const selected = { ...model, api };
    const harness = source(streamFor([
      { type: "response.failed", response: { error: { code: "server_error" }, usage: { input_tokens: 10, output_tokens: 2, total_tokens: 12 } } },
    ], { ...message(), stopReason: "error" }));
    const result = await harness.collector.prompt(() => harness.call(selected), harness.events);
    assert.equal(result.usage.input_tokens, 10);
    assert.equal(result.usage.output_tokens, 2);
    assert.equal(result.usage.total_tokens, 12);
  });
}

test("raw provider total and cache reads survive independently of input", async () => {
  const selected = { ...model, api: "openai-responses" };
  const harness = source(streamFor([{ type: "response.completed", response: { usage: { output_tokens: 2, total_tokens: 17, input_tokens_details: { cached_tokens: 5 } } } }]));
  const result = await harness.collector.prompt(() => harness.call(selected), harness.events);
  assert.equal(result.usage.input_tokens, undefined);
  assert.equal(result.usage.cached_input_tokens, 5);
  assert.equal(result.usage.output_tokens, 2);
  assert.equal(result.usage.total_tokens, 17);
});

test("unfinished work freezes the owner snapshot and never transfers into the next prompt", async () => {
  let release;
  const late = new Promise(resolve => { release = resolve; });
  let calls = 0;
  const harness = source(async (selected, _context, options) => {
    await options.onPayload?.({}, selected);
    if (++calls === 2) return { result: () => late };
    await options.onProviderStreamEvent?.(raw(), selected);
    const final = message();
    return { result: async () => final };
  });
  const first = await harness.collector.prompt(async () => {
    await harness.call();
    await harness.agent.streamFunction(model, { messages: [] }, {});
  }, harness.events);
  assert.equal(first.usage.input_tokens, undefined);
  assert.equal(first.usage.extensions[KEY].observed.input_tokens, 10);
  assert(first.usage.extensions[KEY].gaps.some(gap => gap.reason === "unfinished_work"));
  const frozen = structuredClone(first.usage);
  release(message({ input: 900, output: 90, cacheRead: 0, cacheWrite: 0 }));
  const second = await harness.collector.prompt(harness.call, harness.events);
  assert.equal(second.usage.input_tokens, 10);
  assert.equal(second.usage.output_tokens, 2);
  assert.deepEqual(first.usage, frozen);
});

test("compaction summaries are observed once per call and ignore the combined event", async () => {
  const harness = source(streamFor([raw()]));
  const result = await harness.collector.prompt(async () => {
    harness.emit({ type: "compaction_start" });
    await harness.call();
    await harness.call();
    harness.emit({ type: "compaction_end", aborted: false, result: { usage: { input: 20, output: 4, cacheRead: 0, cacheWrite: 0 } } });
  }, harness.events);
  assert.equal(result.usage.input_tokens, 20);
  assert.equal(result.usage.output_tokens, 4);
  assert.equal(result.usage.extensions[KEY].models[0].requests.length, 2);
});

test("tool reports retain independent fields and are deduplicated by source identity", async () => {
  const harness = source(streamFor([raw()]));
  const usage = { cacheRead: 7, output: 3 };
  const result = await harness.collector.prompt(async () => {
    await harness.call();
    const event = { type: "message_end", message: { role: "toolResult", usage } };
    harness.emit(event);
    harness.emit(event);
  }, harness.events);
  assert.equal(result.usage.input_tokens, undefined);
  assert.equal(result.usage.cached_input_tokens, 7);
  assert.equal(result.usage.output_tokens, 5);
});

test("warm-up callbacks remain attached to their initiating invocation", async () => {
  let callbacks;
  const harness = source(async (selected, _context, options) => {
    callbacks = options;
    await options.onPayload?.({}, selected);
    await options.onProviderStreamEvent?.(raw(), selected);
    const final = message();
    return { result: async () => final };
  });
  const first = await harness.collector.prompt(async () => {
    await harness.call();
    await callbacks.onPayload({}, model);
    await callbacks.onProviderStreamEvent(raw(100, 1), model);
  }, harness.events);
  assert.equal(first.usage.input_tokens, undefined);
  assert.equal(first.usage.extensions[KEY].observed.input_tokens, 110);
  const originalCallbacks = callbacks;
  const frozen = structuredClone(first.usage);
  const second = await harness.collector.prompt(async () => {
    await originalCallbacks.onProviderStreamEvent(raw(900, 1), model);
    await harness.call();
  }, harness.events);
  assert.equal(second.usage.input_tokens, 10);
  assert.deepEqual(first.usage, frozen);
});

test("a source usage object reused by a later tool call is counted in that invocation", async () => {
  const harness = source(streamFor([raw()]));
  const usage = { input: 3, cacheRead: 0, cacheWrite: 0, output: 1 };
  const run = async () => harness.emit({ type: "message_end", message: { role: "toolResult", usage } });
  const first = await harness.collector.prompt(run, harness.events);
  const second = await harness.collector.prompt(run, harness.events);
  assert.equal(first.usage.input_tokens, 3);
  assert.equal(second.usage.input_tokens, 3);
});

test("an extension's replacement stream is preserved and its unobserved messages leave a scope gap", async () => {
  const harness = source(streamFor([raw()]));
  const replacement = streamFor([raw(500, 50)]);
  harness.agent.streamFunction = replacement;
  const result = await harness.collector.prompt(harness.call, harness.events);
  assert.equal(harness.agent.streamFunction, replacement);
  assert.equal(result.usage.input_tokens, undefined);
  assert(result.usage.extensions[KEY].gaps.some(gap => gap.reason === "scope_unverified"));
});

test("runtime keeps completed-call usage when the prompt itself raises", async () => {
  const harness = source(streamFor([raw()]));
  const accounted = await harness.collector.prompt(async () => { await harness.call(); throw new Error("tool processing failed"); }, harness.events);
  assert.equal(accounted.ok, false);
  const runtime = new PiAdapterRuntime({ async create() { return { async prompt() { return { accepted: true, turnStarted: true, turnCount: 1, promptFailed: true, errorMessage: accounted.error.message, usage: accounted.usage }; }, async stop() {} }; } });
  await runtime.start({});
  const result = await runtime.invoke({ input: "one" }, {});
  assert.equal(result.status, "failed");
  assert.equal(result.error.code, "pi_prompt_failed");
  assert.equal(result.error.message, "tool processing failed");
  assert.equal(result.usage.input_tokens, 10);
  assert.equal(result.usage.output_tokens, 2);
});

test("overlapping warm-up frames cannot contaminate the original call or inflate observations", async () => {
  const harness = source(async (selected, _context, options) => {
    await options.onPayload({}, selected);
    await options.onProviderStreamEvent(raw(), selected);
    await options.onPayload({}, selected);
    await options.onProviderStreamEvent(raw(100, 1), selected);
    // The original stream's tail has the same callback and model identity.
    await options.onProviderStreamEvent(raw(), selected);
    const final = message();
    return { result: async () => final };
  });
  const result = await harness.collector.prompt(harness.call, harness.events);
  assert.equal(result.usage.input_tokens, undefined);
  assert.equal(result.usage.extensions[KEY].observed.input_tokens, 10);
  assert(result.usage.extensions[KEY].gaps.some(gap => gap.reason === "scope_unverified"));
});

test("a failed stream keeps a nonterminal cumulative snapshot observed, not complete", async () => {
  const harness = source(streamFor([raw(), { type: "error", error: { message: "stream dropped" } }], { ...message(), stopReason: "error" }));
  const result = await harness.collector.prompt(harness.call, harness.events);
  assert.equal(result.usage.input_tokens, undefined);
  assert.equal(result.usage.output_tokens, undefined);
  assert.equal(result.usage.extensions[KEY].observed.input_tokens, 10);
  assert.equal(result.usage.extensions[KEY].observed.output_tokens, 2);
});

test("internal retries retain an aligned final-attempt snapshot for observed pricing", async () => {
  let attempts = 0;
  const ownFetch = async () => new Response("", { status: ++attempts === 1 ? 503 : 200 });
  const harness = source(async (selected, _context, options) => {
    await options.onPayload({}, selected);
    await options.fetch("https://unused.test");
    await options.fetch("https://unused.test");
    await options.onProviderStreamEvent(raw(), selected);
    const final = message();
    return { result: async () => final };
  });
  const result = await harness.collector.prompt(async () => {
    const final = await (await harness.agent.streamFunction(model, { messages: [] }, { fetch: ownFetch })).result();
    harness.emit({ type: "message_end", message: final });
  }, harness.events);
  assert.equal(result.usage.input_tokens, undefined);
  const group = result.usage.extensions[KEY].models[0];
  assert.equal(group.observed_tokens_aligned, true);
  assert.equal(group.requests[0].usage.input_tokens, 10);
  assert.equal(group.requests[0].gap, "scope_unverified");
});

test("Pi's actual resolved result promise is recorded before message_end without a timing wait", async () => {
  const harness = source(async (selected, _context, options) => {
    await options.onPayload({}, selected);
    await options.onProviderStreamEvent(raw(), selected);
    const stream = createAssistantMessageEventStream();
    stream.push({ type: "done", reason: "stop", message: message() });
    return stream;
  });
  const result = await harness.collector.prompt(harness.call, harness.events);
  assert.equal(result.usage.input_tokens, 10);
  assert.equal(result.usage.output_tokens, 2);
  assert.equal(result.usage.extensions[KEY].models[0].requests.length, 1);
});

test("pi-messages normalized zeros stay observed when upstream presence was lost", async () => {
  const selected = { ...model, api: "pi-messages" };
  const harness = source(streamFor([{ type: "done", usage: { input: 0, cacheRead: 0, cacheWrite: 0, output: 0, totalTokens: 0 } }]));
  const result = await harness.collector.prompt(() => harness.call(selected), harness.events);
  assert.equal(result.usage.input_tokens, undefined);
  assert.equal(result.usage.output_tokens, undefined);
  assert.equal(result.usage.total_tokens, undefined);
  assert.equal(result.usage.extensions[KEY].observed.input_tokens, 0);
  assert.equal(result.usage.extensions[KEY].observed.total_tokens, undefined);
  assert(result.usage.extensions[KEY].gaps.some(gap => gap.reason === "scope_unverified"));
});

test("distinct tool messages sharing one usage object each count once", async () => {
  const harness = source(streamFor([raw()]));
  const usage = { input: 3, output: 1, cacheRead: 0, cacheWrite: 0 };
  const one = { type: "message_end", message: { role: "toolResult", toolCallId: "one", usage } };
  const two = { type: "message_end", message: { role: "toolResult", toolCallId: "two", usage } };
  const result = await harness.collector.prompt(async () => { harness.emit(one); harness.emit(two); harness.emit(one); }, harness.events);
  assert.equal(result.usage.input_tokens, 6);
  assert.equal(result.usage.output_tokens, 2);
});

test("a sequential completed Anthropic warm-up keeps its owned counts known", async () => {
  const selected = { ...model, api: "anthropic-messages" };
  let callbacks;
  const frames = async (options, input, output) => {
    await options.onProviderStreamEvent({ type: "message_start", message: { model: "served", usage: { input_tokens: input, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 } } }, selected);
    await options.onProviderStreamEvent({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: output } }, selected);
    await options.onProviderStreamEvent({ type: "message_stop" }, selected);
  };
  const harness = source(async (_model, _context, options) => {
    callbacks = options;
    await options.onPayload({}, selected);
    await frames(options, 5, 2);
    const final = { ...message(), responseModel: "served" };
    return { result: async () => final };
  });
  const result = await harness.collector.prompt(async () => {
    await harness.call(selected);
    await callbacks.onPayload({}, selected);
    await frames(callbacks, 7, 1);
    // CacheWarmer awaits the refresh before scheduling the next one.
    await callbacks.onPayload({}, selected);
    await frames(callbacks, 9, 1);
  }, harness.events);
  assert.equal(result.usage.input_tokens, 21);
  assert.equal(result.usage.output_tokens, 4);
  assert.equal(result.usage.extensions[KEY].models[0].model, "served");
  assert.equal(result.usage.extensions[KEY].models[0].requests.length, 3);
});

test("active idle warming stays with the original session-request owner", async () => {
  let status = { state: "scheduled" };
  const harness = source(streamFor([raw()]), { sessionId: "session", warmingStatus: () => status });
  const first = await harness.collector.prompt(() => harness.call(model, { sessionId: "session" }), harness.events);
  assert.equal(first.usage.input_tokens, undefined);
  assert(first.usage.extensions[KEY].gaps.some(gap => gap.reason === "unfinished_work"));
  const second = await harness.collector.prompt(async () => {
    harness.emit({ type: "message_end", message: { role: "toolResult", usage: { input: 3, cacheRead: 0, cacheWrite: 0, output: 1 } } });
  }, harness.events);
  assert.equal(second.usage.input_tokens, 3);
  assert.equal(second.usage.output_tokens, 1);
  status = { state: "inactive" };
  const third = await harness.collector.prompt(() => harness.call(model, { sessionId: "session" }), harness.events);
  assert.equal(third.usage.input_tokens, 10);
});

test("custom tool dispatch without a usage report leaves a scope gap", async () => {
  const harness = source(streamFor([raw()]), { customToolNames: new Set(["custom_llm"]) });
  const unused = await harness.collector.prompt(harness.call, harness.events);
  assert.equal(unused.usage.input_tokens, 10);
  const used = await harness.collector.prompt(async () => {
    await harness.call();
    harness.emit({ type: "tool_execution_start", toolCallId: "one", toolName: "custom_llm" });
    harness.emit({ type: "message_end", message: { role: "toolResult", toolCallId: "one", toolName: "custom_llm" } });
  }, harness.events);
  assert.equal(used.usage.input_tokens, undefined);
  assert.equal(used.usage.extensions[KEY].observed.input_tokens, 10);
  assert(used.usage.extensions[KEY].gaps.some(gap => gap.reason === "scope_unverified"));
});

test("an explicit custom-tool report closes only that dispatch's coverage gap", async () => {
  const harness = source(streamFor([raw()]), { customToolNames: new Set(["custom_llm"]) });
  const result = await harness.collector.prompt(async () => {
    harness.emit({ type: "tool_execution_start", toolCallId: "one", toolName: "custom_llm" });
    harness.emit({ type: "message_end", message: { role: "toolResult", toolCallId: "one", toolName: "custom_llm", usage: { input: 3, cacheRead: 0, cacheWrite: 0, output: 1 } } });
  }, harness.events);
  assert.equal(result.usage.input_tokens, 3);
  assert.equal(result.usage.output_tokens, 1);
  assert(!result.usage.extensions[KEY].gaps.some(gap => gap.reason === "scope_unverified"));
});

/** Real Pi1.0.3 session, its provider parser, retry loop, tools and compaction. */
async function realPrompts(replies, prompts, extensions) {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-accounting-")));
  await writeFile(join(workspace, "note.txt"), "nonce\n");
  let served = 0;
  const provider = createServer(async (request, response) => {
    for await (const _chunk of request);
    const reply = replies[served++];
    if (reply === undefined) { response.writeHead(500); response.end("unexpected request"); return; }
    if (reply.status !== undefined) {
      response.writeHead(reply.status, { "content-type": "application/json" });
      response.end(JSON.stringify({ error: { message: reply.status === 400 ? "request rejected" : "rate limited", type: reply.status === 400 ? "invalid_request_error" : "rate_limit_error" } }));
      return;
    }
    const common = { id: `chatcmpl-${served}`, object: "chat.completion.chunk", created: 0, model: "openai/gpt-oss-20b" };
    const chunk = fields => `data: ${JSON.stringify({ ...common, ...fields })}\n\n`;
    const delta = reply.readNote ? { role: "assistant", tool_calls: [{ index: 0, id: `call-${served}`, type: "function", function: { name: "read", arguments: JSON.stringify({ path: "note.txt" }) } }] } : { role: "assistant", content: reply.content };
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.end(chunk({ choices: [{ index: 0, delta, finish_reason: null }] }) + chunk({ choices: [{ index: 0, delta: {}, finish_reason: reply.readNote ? "tool_calls" : "stop" }] }) + (reply.usage === undefined ? "" : chunk({ choices: [], usage: reply.usage })) + "data: [DONE]\n\n");
  });
  await new Promise((resolve, reject) => { provider.once("error", reject); provider.listen(0, "127.0.0.1", resolve); });
  const url = `http://127.0.0.1:${provider.address().port}/v1`;
  let handle;
  try {
    handle = await new PiSdkSessionFactory().create({
      agentName: "pi-accounting-test", baseDir: workspace,
      config: { models: { default: { provider: "nvidia", model: "openai/gpt-oss-20b", api_key_env: "TEST_API_KEY", base_url: url, ...(extensions === undefined ? {} : { extensions }) } }, tools: { enabled: null } },
      runtimeContext: { artifacts: {}, environment: { control_location: "external_control", environment_id: "env", ownership: "caller_owned", provider: "local", workspace, env: { TEST_API_KEY: "not-real" } }, invocation_id: "start", request_id: "req", runtime_id: "rt" },
    });
    const results = [];
    for (const prompt of prompts) results.push(await handle.prompt(prompt));
    assert.equal(served, replies.length);
    return results;
  } finally {
    await handle?.stop();
    provider.closeAllConnections();
    await new Promise(resolve => provider.close(resolve));
  }
}

test("real Pi parser keeps normal tool turns known and independent across prompts", async () => {
  const [first, second, zeros] = await realPrompts([
    { readNote: true, usage: { prompt_tokens: 100, completion_tokens: 10, prompt_tokens_details: { cached_tokens: 40 } } },
    { content: "first", usage: { prompt_tokens: 150, completion_tokens: 0 } },
    { content: "second", usage: { completion_tokens: 5 } },
    { content: "zero", usage: { prompt_tokens: 0, completion_tokens: 0, prompt_tokens_details: { cached_tokens: 0 } } },
  ], ["read the note", "again", "zero"]);
  assert.equal(first.usage.input_tokens, 250);
  assert.equal(first.usage.output_tokens, 10);
  assert.equal(first.usage.cached_input_tokens, undefined);
  assert.equal(second.usage.input_tokens, undefined);
  assert.equal(second.usage.output_tokens, 5);
  assert.equal(zeros.usage.input_tokens, 0);
  assert.equal(zeros.usage.output_tokens, 0);
  assert.equal(zeros.usage.cached_input_tokens, 0);
});

for (const status of [400, 429, 503]) {
  test(`real Pi ${status} response preserves gaps without fabricated zero`, async () => {
    const replies = status === 400 ? [{ status }] : [{ status }, { content: "recovered", usage: { prompt_tokens: 50, completion_tokens: 5 } }];
    const [result] = await realPrompts(replies, ["one"]);
    assert.equal(result.usage.input_tokens, undefined);
    assert.equal(result.usage.output_tokens, undefined);
    if (status !== 400) {
      assert.equal(result.text, "recovered");
      assert.equal(result.usage.extensions[KEY].observed.input_tokens, 50);
    }
  });
}

test("real split-turn compaction counts both summaries once", async () => {
  const [, result] = await realPrompts([
    { content: "first", usage: { prompt_tokens: 100, completion_tokens: 5 } },
    { readNote: true, usage: { prompt_tokens: 200, completion_tokens: 5 } },
    { content: "x".repeat(100_000), usage: { prompt_tokens: 90_000, completion_tokens: 5 } },
    { content: "history summary", usage: { prompt_tokens: 20, completion_tokens: 1 } },
    { content: "turn prefix summary", usage: { prompt_tokens: 20, completion_tokens: 1 } },
  ], ["one", "read the note"], { context_window: 100_000, max_tokens: 1_000 });
  assert.equal(result.usage.input_tokens, 90_240);
  assert.equal(result.usage.output_tokens, 12);
  assert.equal(result.usage.extensions[KEY].models[0].requests.length, 4);
});

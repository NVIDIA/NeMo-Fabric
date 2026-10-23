// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { QwenUsageTracker, QwenStreamUsage } from "../dist/qwen-sdk.js";

const key = "nemo_fabric.usage_accounting.v1";
const next = (tracker, usage, session_id = "session") => tracker.next({ usage, session_id });
const source = usage => usage.extensions[key];

test("a positive terminal aggregate stays observed when hidden calls may omit usage", () => {
  const result = next(new QwenUsageTracker(), { input_tokens: 5, cache_read_input_tokens: 3, output_tokens: 2, total_tokens: 7 });
  assert.equal(result.input_tokens, undefined);
  assert.equal(result.cached_input_tokens, undefined);
  assert.equal(result.output_tokens, undefined);
  assert.equal(result.total_tokens, undefined);
  assert.equal(result.cost_usd, undefined);
  assert.equal(source(result).observed.input_tokens, 5);
  assert.equal(source(result).observed.cached_input_tokens, 3);
  assert.ok(source(result).gaps.some(gap => gap.reason === "scope_unverified"));
});

test("cache-only growth survives independently and zero growth never becomes a measured zero", () => {
  const tracker = new QwenUsageTracker();
  next(tracker, { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0 });
  const cacheOnly = next(tracker, { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 6 });
  assert.equal(source(cacheOnly).observed.cached_input_tokens, 6);
  assert.equal(source(cacheOnly).observed.input_tokens, undefined);
  assert.equal(source(cacheOnly).observed.output_tokens, undefined);
  assert.equal(source(cacheOnly).observed.total_tokens, undefined);
});

test("native zero cumulative total establishes a baseline without deriving a provider total", () => {
  const tracker = new QwenUsageTracker();
  const first = next(tracker, { input_tokens: 5, output_tokens: 2, cache_read_input_tokens: 0 });
  assert.equal(source(first).observed.total_tokens, undefined);
  const second = next(tracker, { input_tokens: 10, output_tokens: 4, cache_read_input_tokens: 0, total_tokens: 7 });
  assert.equal(source(second).observed.total_tokens, 7);
});

test("reset and replay rebaseline only their own invocation; warm snapshots remain frozen", () => {
  const tracker = new QwenUsageTracker();
  const first = next(tracker, { input_tokens: 40, output_tokens: 8, cache_read_input_tokens: 6, total_tokens: 48 });
  const frozen = structuredClone(first);
  assert.deepEqual(source(next(tracker, { input_tokens: 90, output_tokens: 20, cache_read_input_tokens: 10, total_tokens: 110 }, "resumed")).observed, {});
  const warm = next(tracker, { input_tokens: 94, output_tokens: 22, cache_read_input_tokens: 11, total_tokens: 116 }, "resumed");
  assert.equal(source(warm).observed.input_tokens, 4);
  assert.equal(source(warm).observed.output_tokens, 2);
  assert.equal(source(warm).observed.cached_input_tokens, 1);
  assert.deepEqual(first, frozen);
  assert.deepEqual(source(next(tracker, { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0 }, "resumed")).observed, {});
  assert.equal(source(next(tracker, { input_tokens: 5, output_tokens: 2, cache_read_input_tokens: 0, total_tokens: 7 }, "resumed")).observed.input_tokens, 5);
});

test("interrupted stream keeps deduplicated source receipts without model or zero inventions", () => {
  const evidence = new QwenStreamUsage();
  const message = { type: "assistant", message: { id: "a", model: "root-model", usage: { input_tokens: 40, output_tokens: 8, cache_read_input_tokens: 0, total_tokens: 48 } } };
  evidence.consume(message); evidence.consume(message);
  evidence.consume({ type: "assistant", parent_tool_use_id: "child", message: { id: "b", model: "root-model", usage: { input_tokens: 0, output_tokens: 0 } } });
  const result = evidence.finish();
  assert.equal(source(result).observed.input_tokens, 40);
  assert.equal(source(result).observed.output_tokens, 8);
  assert.equal(source(result).observed.cached_input_tokens, undefined);
  assert.equal(source(result).models, undefined);
  assert.equal(result.input_tokens, undefined);
  assert.ok(source(result).gaps.some(gap => gap.reason === "unfinished_work"));
});

// Source qualification for the pinned SDK 0.1.16 / CLI 0.24.6 emitter. A model
// response can have fresh IDs for thinking, text and tools, so ID dedup alone
// is insufficient unless only one of those messages carries positive usage.
for (const finished of [false, true]) {
  test(`pinned Qwen emitter reports split-response usage once when ${finished ? "finished" : "interrupted"}`, async () => {
    const emitterUrl = new URL("./cli/chunks/chunk-IROCR6WJ.js", import.meta.resolve("@qwen-code/sdk"));
    const { StreamJsonOutputAdapter } = await import(emitterUrl.href);
    const frames = [];
    const emitter = new StreamJsonOutputAdapter(
      { getSessionId: () => "session", getModel: () => "root-model" },
      true,
      { write: frame => frames.push(JSON.parse(frame)) },
    );
    emitter.startAssistantMessage();
    emitter.processEvent({ type: "thought", value: { subject: "Plan", description: "inspect" } });
    emitter.processEvent({ type: "content", value: "I will inspect." });
    emitter.processEvent({ type: "tool_call_request", value: { callId: "read", name: "read_file", args: { path: "a" } } });
    if (finished) emitter.processEvent({ type: "finished", value: { usageMetadata: {
      promptTokenCount: 40, candidatesTokenCount: 8, cachedContentTokenCount: 6, totalTokenCount: 48,
    } } });
    // An interrupted attempt may be finalized before a retry or stream close.
    // Repeated finalization must not re-emit a receipt.
    emitter.finalizeAssistantMessage();
    emitter.finalizeAssistantMessage();
    const receipts = frames.filter(frame => frame.type === "assistant");
    assert.equal(receipts.length, 3);
    assert.equal(new Set(receipts.map(frame => frame.message.id)).size, 3);
    assert.deepEqual(receipts.map(frame => frame.message.content[0].type), ["thinking", "text", "tool_use"]);
    assert.deepEqual(receipts.map(frame => frame.message.usage), [
      { input_tokens: 0, output_tokens: 0 },
      { input_tokens: 0, output_tokens: 0 },
      finished ? { input_tokens: 40, output_tokens: 8, cache_read_input_tokens: 6, total_tokens: 48 }
        : { input_tokens: 0, output_tokens: 0 },
    ]);
    // Losing the query transport after these frames gives no terminal result.
    // The failure collector must keep just the emitted evidence, once.
    const evidence = new QwenStreamUsage();
    for (const frame of frames) evidence.consume(frame);
    const observed = source(evidence.finish()).observed;
    assert.equal(observed.input_tokens, finished ? 40 : undefined);
    assert.equal(observed.output_tokens, finished ? 8 : undefined);
    assert.equal(observed.cached_input_tokens, finished ? 6 : undefined);
    assert.equal(observed.total_tokens, finished ? 48 : undefined);
  });
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { extractInvocationUsage, findPrompt, reverseStoredUsage } from "../dist/usage.js";
import { KiloUsageEvents } from "../dist/usage-events.js";

const evidenceKey = "nvidia.fabric.kilo.usage_evidence.v1";
const detailKey = "nemo_fabric.detailed_token_usage.v1";
const tokens = { input: 10, output: 20, total: 36, reasoning: 3, cache: { read: 2, write: 1 } };
const user = id => ({ info: { id, role: "user", sessionID: "s" }, parts: [{ type: "text", text: "prompt" }] });
const assistant = (id, parentID, overrides = {}) => ({
  info: { id, parentID, role: "assistant", sessionID: "s", time: { completed: 1 } },
  parts: [{ id: `${id}-step`, type: "step-finish", tokens, cost: 9 }], ...overrides,
});
const collect = (messages, options = {}) => extractInvocationUsage(messages, {
  sessionId: "s", prompt: "u", prompts: new Set(["u"]), retried: false, eventCoverage: true, unfinished: false, ...options,
});

test("reconstructs reversible counters and preserves a supplied total zero", () => {
  assert.deepEqual(reverseStoredUsage(tokens), { input_tokens: 13, output_tokens: 23, total_tokens: 36, cached_input_tokens: 2, cache_write_input_tokens: 1, reasoning_tokens: 3 });
  assert.deepEqual(reverseStoredUsage({ input: 0, output: 0, total: 0, reasoning: 0, cache: { read: 0, write: 0 } }), { total_tokens: 0 });
  assert.deepEqual(reverseStoredUsage({ input: 0, output: 0, reasoning: 3, cache: { read: 2, write: 1 } }), { cached_input_tokens: 2, cache_write_input_tokens: 1, reasoning_tokens: 3 });
  assert.deepEqual(reverseStoredUsage({ input: Number.MAX_SAFE_INTEGER, output: NaN, total: -1, cache: { read: 1, write: 0 } }), { cached_input_tokens: 1 });
});

test("keeps a complete input when output is unknown and never publishes inserted cache zero", () => {
  const result = collect([user("u"), assistant("a", "u", { parts: [{ id: "step", type: "step-finish", tokens: { input: 10, output: 0, reasoning: 0, cache: { read: 0, write: 0 } } }] })]);
  assert.equal(result.input_tokens, 10);
  assert.equal(result.output_tokens, undefined);
  assert.deepEqual(result.extensions[detailKey], { input_tokens: 10, input_tokens_include_cache: true });
  assert.deepEqual(result.extensions[evidenceKey].observed, { input_tokens: 10 });
  assert.ok(result.extensions[evidenceKey].gaps.includes("missing_usage"));
});

test("deduplicates identical steps and drops conflicting step evidence", () => {
  const message = assistant("a", "u");
  assert.equal(collect([user("u"), message, structuredClone(message)]).input_tokens, 13);
  const changed = structuredClone(message);
  changed.parts[0].tokens.input = 11;
  const result = collect([user("u"), message, changed, message]);
  assert.equal(result.input_tokens, undefined);
  assert.deepEqual(result.extensions[evidenceKey].observed, {});
  assert.ok(result.extensions[evidenceKey].gaps.includes("scope_unverified"));
});

test("does not transfer late work from a previous invocation", () => {
  const result = collect([user("old"), assistant("a-old", "old"), user("u"), assistant("late-old", "old"), assistant("a", "u")], { prompts: new Set(["old", "u"]) });
  assert.equal(result.input_tokens, 13);
  assert.equal(result.extensions[evidenceKey].observed.input_tokens, 13);
});

test("does not guess prompt ownership when another user root arrived during the invocation", () => {
  assert.equal(findPrompt([user("previous"), user("scheduled"), user("u")], "previous"), undefined);
  assert.equal(findPrompt([user("previous"), user("u")], "previous"), "u");
});

test("preserves direct steps with a gap for child or internal turns", () => {
  for (const extra of [
    { ...user("internal"), parts: [{ type: "text", text: "continue", synthetic: true }] },
    ...["task", "schedule_wakeup", "cron_create", "goal"].map(tool => ({ ...assistant("a", "u"), parts: [...assistant("a", "u").parts, { type: "tool", tool }] })),
  ]) {
    const result = collect([user("u"), assistant("a", "u"), extra]);
    assert.equal(result.input_tokens, undefined);
    assert.equal(result.extensions[evidenceKey].observed.input_tokens, 13);
    assert.ok(result.extensions[evidenceKey].gaps.includes("scope_unverified"));
  }
});

test("keeps usage after an assistant error but an unfinished request remains partial", () => {
  const failed = assistant("a", "u", { info: { id: "a", role: "assistant", parentID: "u", sessionID: "s", error: { name: "APIError" } } });
  assert.equal(collect([user("u"), failed]).input_tokens, 13);
  const result = collect([user("u"), failed], { unfinished: true });
  assert.equal(result.input_tokens, undefined);
  assert.equal(result.extensions[evidenceKey].observed.input_tokens, 13);
  assert.ok(result.extensions[evidenceKey].gaps.includes("unfinished_work"));
});

test("does not expose a rounded aggregate outside the safe integer range", () => {
  const message = id => assistant(id, "u", { parts: [{ id, type: "step-finish", tokens: { input: Number.MAX_SAFE_INTEGER, output: 1, reasoning: 0, cache: { read: 0, write: 0 } } }] });
  const result = collect([user("u"), message("a"), message("b")]);
  assert.equal(result.input_tokens, undefined);
  assert.equal(result.extensions[evidenceKey].observed.input_tokens, undefined);
  assert.equal(result.output_tokens, 2);
});

test("an error update alone does not settle retry coverage", async () => {
  const events = new KiloUsageEvents();
  events.observe({ type: "message.updated", properties: { info: { id: "a", role: "assistant", sessionID: "s", error: { name: "APIError" } } } });
  assert.equal(await events.through("a", 0), false);
  assert.equal(events.retryCount("s"), 0);
  events.observe({ type: "session.status", properties: { sessionID: "s", status: { type: "retry" } } });
  assert.equal(events.retryCount("s"), 1);
  events.observe({ type: "message.updated", properties: { info: { id: "a", role: "assistant", sessionID: "s", time: { completed: 1 } } } });
  assert.equal(await events.through("a", 0), true);
});

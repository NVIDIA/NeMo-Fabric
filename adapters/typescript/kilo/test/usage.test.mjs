// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";
import {extractInvocationUsage, reverseStoredUsage} from "../dist/usage.js";
import {KiloUsageEvents} from "../dist/usage-events.js";

const key = "nemo_fabric.usage_accounting.v1";
const tokens = (input, output, read = 0, write = 0, reasoning = 0) => ({input, output, reasoning, cache: {read, write}});
const user = {info: {id: "u", role: "user", sessionID: "root", time: {created: 1}}, parts: []};
const assistant = (id, parts, more = {}) => ({info: {id, role: "assistant", sessionID: "root", parentID: "u", modelID: "gpt-test", providerID: "openai", time: {created: 2, completed: 3}, ...more}, parts});
const step = (id, counts) => ({type: "step-finish", id, tokens: counts, cost: 100});
const usage = (messages, source = {}) => extractInvocationUsage(messages, "u", new Set(["u"]), new Map(), source);

// Pinned Session.getUsage stores input=max(inclusive-read-write,0),
// output=max(inclusive-reasoning,0). Positive adjusted main proves no clamp.
test("reverses the source transform without claiming zero-filled cache presence", () => {
  assert.deepEqual(reverseStoredUsage(tokens(10, 20)), {
    input_tokens: 10, input_tokens_include_cache: true,
    output_tokens: 20, output_tokens_include_reasoning: true,
  });
  assert.deepEqual(reverseStoredUsage(tokens(10, 20, 4, 3, 2)), {
    input_tokens: 17, input_tokens_include_cache: true,
    cached_input_tokens: 4, cache_write_input_tokens: 3,
    output_tokens: 22, output_tokens_include_reasoning: true, reasoning_tokens: 2,
  });
});

test("clamped or omitted main counters cannot become known from positive components", () => {
  for (const main of [0, undefined]) {
    const raw = tokens(main, main, 40, 20, 5);
    const evidence = reverseStoredUsage(raw);
    assert.equal(evidence.input_tokens, undefined);
    assert.equal(evidence.output_tokens, undefined);
    assert.equal(evidence.cached_input_tokens, 40);
    assert.equal(evidence.cache_write_input_tokens, 20);
    assert.equal(evidence.reasoning_tokens, 5);
  }
});

test("invalid operands and source-inverse overflow never fabricate inclusive totals", () => {
  for (const invalid of [-1, true, 1.5, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
    const evidence = reverseStoredUsage(tokens(10, 20, invalid));
    assert.equal(evidence.input_tokens_include_cache, false);
    assert.equal(evidence.input_tokens, 10, "exclusive source observation remains useful");
    assert.equal(evidence.cached_input_tokens, undefined);
    assert.equal(evidence.output_tokens, 20);
  }
  const overflow = reverseStoredUsage(tokens(Number.MAX_SAFE_INTEGER, 1, 1));
  assert.equal(overflow.input_tokens_include_cache, false);
  assert.equal(overflow.input_tokens, Number.MAX_SAFE_INTEGER);
});

test("unfinished owned work keeps settled observations and gaps complete totals", () => {
  const data = usage([user, assistant("done", [step("s", tokens(10, 20, 3))]),
    assistant("running", [], {time: {created: 4}})]);
  assert.equal(data.input_tokens, undefined);
  assert.equal(data.extensions[key].observed.input_tokens, 13);
  assert.equal(data.extensions[key].observed.output_tokens, 20);
  assert.ok(data.extensions[key].gaps.some(gap => gap.reason === "unfinished_work"));
});

test("source input/output remain useful independently of omitted cache counts", () => {
  const data = usage([user, assistant("done", [step("s", tokens(10, 20))])]);
  assert.equal(data.input_tokens, 10);
  assert.equal(data.output_tokens, 20);
  assert.equal(data.cached_input_tokens, undefined);
  assert.equal(data.total_tokens, undefined, "core derives normalized totals");
  assert.equal(data.cost_usd, undefined, "mixed catalog/provider source cost is diagnostic");
  assert.equal(data.extensions[key].models[0].model, "gpt-test");
  assert.equal(data.extensions[key].models[0].provider, "openai");
});

test("custom provider overrides retain model identity without native pricing identity", () => {
  const data = usage([user, assistant("done", [step("s", tokens(10, 20))])], {customProvider: "openai"});
  assert.equal(data.extensions[key].models[0].model, "gpt-test");
  assert.equal(data.extensions[key].models[0].provider, undefined);
});

test("cache reads stay independently known when input was clamped", () => {
  const data = usage([user, assistant("done", [step("s", tokens(0, 20, 30))])]);
  assert.equal(data.input_tokens, undefined);
  assert.equal(data.cached_input_tokens, 30);
  assert.equal(data.output_tokens, 20);
});

test("only explicit preflight false proves a no-request assistant before compaction", () => {
  const ended = (overflow) => [user, assistant("done", [step("s", tokens(10, 20))]), assistant("empty", []),
    {info: {id: "compact", role: "user"}, parts: [{type: "compaction", overflow}]},
    {info: {id: "summary", role: "assistant", parentID: "compact", time: {created: 5, completed: 6}}, parts: [step("summary", tokens(2, 3))]}];
  assert.equal(usage(ended(false)).input_tokens, 12);
  for (const overflow of [true, undefined]) {
    const data = usage(ended(overflow));
    assert.equal(data.input_tokens, undefined);
    assert.equal(data.extensions[key].observed.input_tokens, 12);
  }
});

test("each retained source step is counted once", () => {
  const repeated = step("same-step", tokens(10, 20));
  const data = usage([user, assistant("done", [repeated, structuredClone(repeated)])]);
  assert.equal(data.input_tokens, 10);
  assert.equal(data.output_tokens, 20);
});

test("actual transient retry belongs to its message, never another invocation", async () => {
  const events = new KiloUsageEvents();
  events.observe({type: "server.connected"});
  events.observe({type: "message.updated", properties: {info: {id: "done", role: "assistant", sessionID: "root"}}});
  events.observe({type: "session.status", properties: {sessionID: "root", status: {type: "retry", attempt: 1}}});
  events.observe({type: "message.updated", properties: {info: {id: "done", role: "assistant", sessionID: "root", time: {completed: 3}}}});
  assert.equal(await events.through("done", 0), true);
  const data = usage([user, assistant("done", [step("s", tokens(10, 20))])], {events, eventCoverage: true});
  assert.equal(data.input_tokens, undefined);
  assert.equal(data.extensions[key].observed.input_tokens, 10);
  const next = extractInvocationUsage([user, assistant("done", [step("s", tokens(10, 20))]),
    {info: {id: "next-user", role: "user"}, parts: []},
    assistant("next", [step("next-step", tokens(2, 3))], {parentID: "next-user"})],
    "next-user", new Set(["u", "next-user"]), new Map(), {events, eventCoverage: true});
  assert.equal(next.input_tokens, 2);
  assert.equal(data.extensions[key].observed.input_tokens, 10, "returned snapshot stays detached");
});

test("missing event watermark remains an observational gap", async () => {
  const events = new KiloUsageEvents();
  events.observe({type: "server.connected"});
  assert.equal(await events.through("missing", 0), false);
  events.end();
  const data = usage([user, assistant("done", [step("s", tokens(10, 20))])], {events, eventCoverage: false});
  assert.equal(data.input_tokens, undefined);
  assert.equal(data.extensions[key].observed.input_tokens, 10);
  assert.ok(data.extensions[key].gaps.some(gap => gap.reason === "scope_unverified"));
});

test("leaf descriptor declares the shared accounting schema", async () => {
  const leaf = await readFile(new URL("../kilo.fabric-adapter.json", import.meta.url));
  const descriptor = JSON.parse(leaf);
  assert.ok(descriptor.extension_schemas.usage.properties[key]);
});


test("source inversion uncertainty gaps only its affected dimension", () => {
  const data = usage([user, assistant("done", [step("s", {input: 10, output: 20, reasoning: 0})])]);
  assert.equal(data.input_tokens, undefined);
  assert.equal(data.output_tokens, 20);
  assert.equal(data.extensions[key].observed.input_tokens, 10);
  assert.ok(data.extensions[key].gaps.some(gap => gap.reason === "scope_unverified" && gap.dimensions.includes("input_tokens") && !gap.dimensions.includes("output_tokens")));
});

test("conflicting duplicate source step cannot publish a complete count", () => {
  const data = usage([user, assistant("done", [step("same", tokens(10, 20)), step("same", tokens(30, 40))])]);
  assert.equal(data.input_tokens, undefined);
  assert.equal(data.output_tokens, undefined);
  assert.equal(data.extensions[key].observed.input_tokens, undefined);
});

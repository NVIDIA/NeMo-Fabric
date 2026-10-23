// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { ClineInvocationUsage, clineUsageHooks } from "../dist/usage.js";

const key = "nemo_fabric.usage_accounting.v1";
const message = (id, metrics, modelInfo = { id: "test-model", provider: "openai" }) => ({ id, metrics, modelInfo });

async function collect(messages) {
  const usage = new ClineInvocationUsage();
  await usage.run(async () => {
    for (const value of messages) {
      clineUsageHooks.beforeModel();
      clineUsageHooks.afterModel({ assistantMessage: value });
    }
  });
  return usage.finish();
}

test("per-call omissions cannot be hidden by another call's positive aggregate", async () => {
  const result = await collect([
    message("first", { inputTokens: 40, outputTokens: 8, cacheReadTokens: 6 }),
    message("second", { inputTokens: 20, outputTokens: 0, cacheReadTokens: 2 }),
  ]);
  assert.equal(result.input_tokens, undefined);
  assert.equal(result.extensions["nemo_fabric.usage_accounting.v1"].observed.input_tokens, 60);
  assert.equal(result.cached_input_tokens, undefined);
  assert.equal(result.extensions["nemo_fabric.usage_accounting.v1"].observed.cached_input_tokens, 8);
  assert.equal(result.output_tokens, undefined);
  assert.equal(result.extensions[key].observed.output_tokens, 8);
  assert.equal(result.extensions[key].models[0].requests_exhaustive, false);
});

test("zero-filled counts and client costs never become measured zeros or provider dollars", async () => {
  const result = await collect([message("first", { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, totalCost: 2 })]);
  assert.equal(result.input_tokens, undefined);
  assert.equal(result.output_tokens, undefined);
  assert.equal(result.cost_usd, undefined);
  assert.deepEqual(result.extensions[key].observed, {});
});

test("duplicate message identities deduplicate; conflicting duplicates add a gap", async () => {
  const value = message("first", { inputTokens: 40, outputTokens: 8 });
  const usage = new ClineInvocationUsage();
  await usage.run(async () => {
    clineUsageHooks.beforeModel();
    clineUsageHooks.afterModel({ assistantMessage: value });
    clineUsageHooks.afterModel({ assistantMessage: value });
  });
  assert.equal(usage.finish().input_tokens, undefined);
  assert.equal(usage.finish().extensions["nemo_fabric.usage_accounting.v1"].observed.input_tokens, 40);
  const conflict = new ClineInvocationUsage();
  await conflict.run(async () => {
    clineUsageHooks.beforeModel();
    clineUsageHooks.afterModel({ assistantMessage: value });
    clineUsageHooks.afterModel({ assistantMessage: message("first", { inputTokens: 60, outputTokens: 8 }) });
  });
  const result = conflict.finish();
  assert.equal(result.input_tokens, undefined);
  assert.equal(result.extensions[key].observed.input_tokens, 40);
});

test("a completion without an id closes its pending call", async () => {
  const usage = new ClineInvocationUsage();
  await usage.run(async () => {
    clineUsageHooks.beforeModel();
    clineUsageHooks.afterModel({ assistantMessage: message(undefined, { inputTokens: 40, outputTokens: 8 }) });
  });
  const gaps = usage.finish().extensions[key].gaps.map(gap => gap.reason);
  assert.ok(gaps.includes("scope_unverified"));
  assert.ok(!gaps.includes("unfinished_work"));
});

test("an unfinished cumulative snapshot needs a baseline after completed calls", async () => {
  const usage = new ClineInvocationUsage();
  await usage.run(async () => {
    clineUsageHooks.beforeModel();
    clineUsageHooks.afterModel({ assistantMessage: message("first", { inputTokens: 40, outputTokens: 8 }) });
    clineUsageHooks.beforeModel();
    clineUsageHooks.onEvent({ type: "usage-updated", usage: { inputTokens: 100, outputTokens: 18 } });
  });
  const result = usage.finish();
  assert.equal(result.extensions[key].observed.input_tokens, 40);
  assert.equal(result.extensions[key].observed.output_tokens, 8);
  assert.ok(result.extensions[key].gaps.some(gap => gap.reason === "unfinished_work"));
});

test("compaction preserves observed source metrics without certifying hidden calls", async () => {
  for (const phase of ["skipped", "started"]) {
    const usage = new ClineInvocationUsage();
    await usage.run(async () => {
      clineUsageHooks.onEvent({ type: "status-notice", metadata: { kind: "auto_compaction", phase } });
      clineUsageHooks.beforeModel();
      clineUsageHooks.afterModel({ assistantMessage: message("first", { inputTokens: 40, outputTokens: 8 }) });
    });
    const result = usage.finish();
    assert.equal(result.input_tokens, undefined);
    assert.equal(result.extensions[key].observed.input_tokens, 40);
  }
});

test("unfinished work freezes at return and late callbacks cannot enter a later owner", async () => {
  let release;
  const blocked = new Promise(resolve => { release = resolve; });
  const first = new ClineInvocationUsage();
  let late;
  await first.run(async () => {
    clineUsageHooks.beforeModel();
    late = blocked.then(() => clineUsageHooks.afterModel({ assistantMessage: message("late", { inputTokens: 100, outputTokens: 20 }) }));
  });
  const frozen = first.finish();
  assert.ok(frozen.extensions[key].gaps.some(gap => gap.reason === "unfinished_work"));
  const next = new ClineInvocationUsage();
  await next.run(async () => {
    release();
    await late;
    clineUsageHooks.beforeModel();
    clineUsageHooks.afterModel({ assistantMessage: message("next", { inputTokens: 4, outputTokens: 2 }) });
  });
  assert.equal(next.finish().input_tokens, undefined);
  assert.equal(next.finish().extensions["nemo_fabric.usage_accounting.v1"].observed.input_tokens, 4);
  assert.deepEqual(first.finish(), frozen);
});

test("terminal-only aggregates remain observations when hook coverage is unavailable", () => {
  const result = new ClineInvocationUsage().finish({ inputTokens: 40, outputTokens: 8 });
  assert.equal(result.input_tokens, undefined);
  assert.equal(result.extensions[key].observed.input_tokens, 40);
  assert.ok(result.extensions[key].gaps.some(gap => gap.reason === "scope_unverified"));
});


test("native OpenAI client IDs normalize only when the source endpoint proves the provider", async () => {
  const value = message("first", { inputTokens: 40, outputTokens: 8 }, { id: "gpt-test", provider: "openai-native" });
  const known = await collect([value]);
  assert.equal(known.extensions[key].models[0].provider, "openai");
  const custom = new ClineInvocationUsage(false);
  await custom.run(async () => {
    clineUsageHooks.beforeModel(); clineUsageHooks.afterModel({ assistantMessage: value });
  });
  assert.equal(custom.finish().extensions[key].models[0].provider, undefined);
});

test("unfinished usage snapshots own their scalar values before the SDK can mutate them", async () => {
  const baseline = { inputTokens: 10, outputTokens: 2 };
  const latest = { inputTokens: 50, outputTokens: 10 };
  const usage = new ClineInvocationUsage();
  await usage.run(async () => {
    clineUsageHooks.beforeModel({ snapshot: { usage: baseline } });
    baseline.inputTokens = 30;
    clineUsageHooks.onEvent({ type: "usage-updated", usage: latest });
    latest.inputTokens = 100;
  });
  const result = usage.finish();
  assert.equal(result.extensions[key].observed.input_tokens, 40);
  assert.equal(result.extensions[key].observed.output_tokens, 8);
  assert.equal(result.input_tokens, undefined);
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { invocationUsage } from "../dist/usage.js";
import { PiAdapterRuntime } from "../dist/runtime.js";

const usage = (input = 10, output = 3, cacheRead = 2, cacheWrite = 1) => ({
  input,
  output,
  cacheRead,
  cacheWrite,
  totalTokens: input + output + cacheRead + cacheWrite,
  reasoning: 1,
  cost: { total: 123 },
});
const message = (id, value = usage()) => ({
  id,
  type: "message",
  message: { role: "assistant", usage: value },
});

test("counts the invocation's raw ledger including retries, tools, summaries and all branches", () => {
  const entries = [
    message("old"),
    message("failed-attempt"),
    message("retry"),
    {
      id: "tool",
      type: "message",
      message: { role: "toolResult", usage: usage() },
    },
    { id: "ordinary-tool", type: "message", message: { role: "toolResult" } },
    { id: "summary", type: "compaction", usage: usage() },
    { id: "branch", type: "branch_summary", usage: usage() },
    { id: "auxiliary", type: "usage", usage: usage() },
  ];
  const result = invocationUsage(["old"], entries);
  assert.equal(result.input_tokens, 78);
  assert.equal(result.output_tokens, 18);
  assert.equal(result.total_tokens, 96);
  assert.equal(result.cached_input_tokens, 12);
  assert.equal(result.input_tokens_include_cache, true);
  assert.equal(result.cost_usd, undefined);
  assert.deepEqual(result.extensions["nemo_fabric.detailed_token_usage.v1"], {
    input_tokens: 78,
    output_tokens: 18,
    total_tokens: 96,
    cached_input_tokens: 12,
    cache_write_input_tokens: 6,
    input_tokens_include_cache: true,
    output_tokens_include_reasoning: true,
    reasoning_tokens: 6,
  });
  // The next prompt cannot reuse any preceding work.
  assert.equal(
    invocationUsage(
      entries.map((entry) => entry.id),
      [...entries, message("next")],
    ).input_tokens,
    13,
  );
});

test("keeps incomplete, default-zero, reset, invalid and unsafe usage unknown", () => {
  const invalid = [
    undefined,
    usage(0, 0, 0, 0),
    { ...usage(), input: undefined },
    { ...usage(), totalTokens: 5 },
    { ...usage(), output: -1 },
    { ...usage(), reasoning: 4 },
    { ...usage(), input: Number.MAX_SAFE_INTEGER + 1 },
  ];
  for (const value of invalid) {
    const entry = {
      id: "missing",
      type: "message",
      message: { role: "assistant", usage: value },
    };
    assert.equal(invocationUsage([], [message("observed"), entry]), undefined);
  }
  assert.equal(
    invocationUsage([], [{ id: "c", type: "compaction" }]),
    undefined,
  );
  assert.equal(invocationUsage(["old"], [message("reset")]), undefined);
  assert.equal(invocationUsage(["old"], []), undefined);
  assert.equal(invocationUsage([], []), undefined);
});

test("preserves real zero dimensions and does not invent unavailable reasoning", () => {
  const value = { ...usage(10, 0, 0, 0), reasoning: undefined };
  const result = invocationUsage([], [message("zero-output", value)]);
  assert.equal(result.output_tokens, 0);
  assert.equal(result.cached_input_tokens, 0);
  assert.equal(
    result.extensions["nemo_fabric.detailed_token_usage.v1"].reasoning_tokens,
    undefined,
  );
});

test("retains observed invocation usage on failed and cancelled results", async () => {
  for (const stopReason of ["error", "aborted", "stop"]) {
    const observed = invocationUsage([], [message("billed")]);
    const runtime = new PiAdapterRuntime({
      async create() {
        return {
          async prompt() {
            return {
              accepted: true,
              turnStarted: true,
              turnCount: 1,
              stopReason,
              usage: observed,
            };
          },
          async stop() {},
        };
      },
    });
    await runtime.start({});
    const result = await runtime.invoke({ input: "test" }, {});
    assert.notEqual(result.status, "succeeded");
    assert.deepEqual(result.usage, observed);
    await runtime.stop();
  }
});

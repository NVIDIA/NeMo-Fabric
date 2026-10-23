// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { usageFromObservations, USAGE_ACCOUNTING_KEY } from "../dist/index.js";

const cases = JSON.parse(await readFile(new URL("../../../tests/fixtures/usage_accounting/reducer-cases.json", import.meta.url), "utf8"));
for (const vector of cases) {
  test(`shared raw reducer: ${vector.name}`, () => {
    const usage = usageFromObservations(vector.records);
    const accounting = usage.extensions[USAGE_ACCOUNTING_KEY];
    for (const [field, value] of Object.entries(vector.expected)) {
      const actual = field === "exhaustive" ? accounting.models_exhaustive
        : field.startsWith("observed_") ? accounting.observed[field.slice(9) + "_tokens"]
        : field === "peak_request_input_tokens" ? usage.extensions["nemo_fabric.detailed_token_usage.v1"][field]
        : usage[field];
      assert.equal(actual ?? null, value, field);
    }
  });
}
test("unsafe count sums do not erase independent output", () => {
  const usage = usageFromObservations([1, 2].map(() => ({ reported: {
    input_tokens: Number.MAX_SAFE_INTEGER, input_tokens_include_cache: true,
    output_tokens: 1, output_tokens_include_reasoning: true,
  } })));
  assert.equal(usage.input_tokens, undefined);
  assert.equal(usage.output_tokens, 2);
});
test("negative and unsafe source counts are rejected", () => {
  for (const input_tokens of [-1, 0.5, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => usageFromObservations([{ reported: { input_tokens } }]), RangeError);
  }
});

test("scoped request gaps retain independent fields without certifying cache", () => {
  const vector = cases.find(vector => vector.name === "actual_call_with_scoped_cache_gap");
  const accounting = usageFromObservations(vector.records).extensions[USAGE_ACCOUNTING_KEY];
  const model = accounting.models[0];
  assert.equal(model.observed.cached_input_tokens, 20);
  assert.equal(model.reported.cached_input_tokens, undefined);
  assert.equal(model.requests[0].usage.cached_input_tokens, undefined);
  assert.equal(model.requests[0].usage.input_tokens, 100);
  assert.equal(model.requests[0].usage.output_tokens, 5);
  assert.equal(model.requests[0].gap, undefined);
  assert.equal(model.requests[0].tokens_aligned, true);
});

test("request usage omits provider-specific keys outside the wire schema", () => {
  const usage = usageFromObservations([{ model: "m", is_model_call: true,
    reported: { input_tokens: 7, input_tokens_include_cache: true, prompt_tokens: 7 },
  }]);
  const request = usage.extensions[USAGE_ACCOUNTING_KEY].models[0].requests[0].usage;
  assert.deepEqual(request, { input_tokens: 7, input_tokens_include_cache: true });
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { configToml, selectModel, selectSettings } from "../dist/configuration.js";

function config(settings, model = { provider: "openai", model: "fabric-echo" }) {
  return { models: { default: model }, harness: { settings } };
}

test("turns the terminal on when it is present and not disabled", () => {
  for (const [interfaces, terminal] of [[undefined, false], [{}, false], [{ terminal: {} }, true], [{ terminal: { enabled: false } }, false]]) {
    assert.equal(selectSettings(config({ approval_mode: "auto", interfaces })).terminal, terminal);
  }
});

test("maps an OpenAI-compatible endpoint to Chat Completions", () => {
  const model = selectModel(
    config({ approval_mode: "auto" }, {
      provider: "openai",
      model: "fabric-echo",
      base_url: "http://127.0.0.1:9/v1",
      api_key_env: "FABRIC_MODEL_KEY",
    }),
  );

  assert.equal(
    configToml(model),
    [
      "[models]",
      'default = "openai:fabric-echo"',
      "",
      "[models.providers.openai]",
      'models = ["fabric-echo"]',
      'api_key_env = "FABRIC_MODEL_KEY"',
      'base_url = "http://127.0.0.1:9/v1"',
      "",
      "[models.providers.openai.params]",
      "use_responses_api = false",
      "",
    ].join("\n"),
  );
});

test("uses OpenAI's own endpoint and key variable without a base_url", () => {
  assert.deepEqual(selectModel(config({ approval_mode: "auto" })), { model: "fabric-echo", apiKeyEnv: "OPENAI_API_KEY" });
});

for (const [model, code] of [
  [{ provider: "anthropic", model: "claude" }, "deepagents_code_model_invalid"],
  [{ provider: "openai", model: "m", base_url: "http://models.example.test/v1" }, "deepagents_code_model_endpoint_invalid"],
]) {
  test(`rejects unsupported model ${JSON.stringify(model)}`, () => {
    assert.throws(() => selectModel(config({ approval_mode: "auto" }, model)), { code });
  });
}

test("accepts an HTTP endpoint on a trusted host", () => {
  const settings = { approval_mode: "auto", insecure_http_model_hosts: ["inference.local"] };
  const model = { provider: "openai", model: "m", base_url: "http://inference.local/v1" };

  assert.equal(selectModel(config(settings, model)).baseUrl, "http://inference.local/v1");
});

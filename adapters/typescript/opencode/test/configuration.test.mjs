// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { selectModel } from "../dist/configuration.js";

test("selects the default OpenCode model role", () => {
  assert.deepEqual(
    selectModel({
      models: {
        default: { provider: "openai", model: "gpt-4.1-mini", api_key_env: "OPENAI_API_KEY" },
        reviewer: { provider: "anthropic", model: "claude-sonnet-4", api_key_env: "ANTHROPIC_API_KEY" },
      },
    }),
    { provider: "openai", model: "gpt-4.1-mini", apiKeyEnv: "OPENAI_API_KEY" },
  );
});

test("selects a sole OpenCode model role", () => {
  assert.deepEqual(
    selectModel({
      models: {
        coding: { provider: "openai", model: "gpt-4.1-mini", api_key_env: "OPENAI_API_KEY" },
      },
    }),
    { provider: "openai", model: "gpt-4.1-mini", apiKeyEnv: "OPENAI_API_KEY" },
  );
});

test("selects an OpenAI-compatible endpoint when configured", () => {
  assert.deepEqual(
    selectModel({
      models: {
        default: {
          provider: "local-test",
          model: "test-model",
          api_key_env: "LOCAL_TEST_KEY",
          base_url: "http://127.0.0.1:8080/v1",
        },
      },
    }),
    {
      provider: "local-test",
      model: "test-model",
      apiKeyEnv: "LOCAL_TEST_KEY",
      baseUrl: "http://127.0.0.1:8080/v1",
    },
  );
});

test("selects an HTTPS OpenAI-compatible endpoint when configured", () => {
  assert.deepEqual(
    selectModel({
      models: {
        default: {
          provider: "hosted-test",
          model: "test-model",
          api_key_env: "HOSTED_TEST_KEY",
          base_url: "https://provider.example/v1",
        },
      },
    }),
    {
      provider: "hosted-test",
      model: "test-model",
      apiKeyEnv: "HOSTED_TEST_KEY",
      baseUrl: "https://provider.example/v1",
    },
  );
});

test("selects sampling settings for an OpenAI-compatible endpoint", () => {
  assert.deepEqual(
    selectModel({
      models: {
        default: {
          provider: "local-test",
          model: "test-model",
          api_key_env: "LOCAL_TEST_KEY",
          base_url: "http://127.0.0.1:8080/v1",
          temperature: 0.25,
          top_p: 0.8,
        },
      },
    }),
    {
      provider: "local-test",
      model: "test-model",
      apiKeyEnv: "LOCAL_TEST_KEY",
      baseUrl: "http://127.0.0.1:8080/v1",
      sampling: { temperature: 0.25, topP: 0.8 },
    },
  );
});

test("rejects sampling settings for native OpenCode providers", () => {
  assert.throws(
    () =>
      selectModel({
        models: {
          default: {
            provider: "openai",
            model: "gpt-4.1-mini",
            api_key_env: "OPENAI_API_KEY",
            temperature: 0.25,
          },
        },
      }),
    (error) => error.code === "opencode_sampling_requires_base_url",
  );
});

test("rejects unsupported OpenCode maximum-token settings", () => {
  assert.throws(
    () =>
      selectModel({
        models: {
          default: {
            provider: "local-test",
            model: "test-model",
            api_key_env: "LOCAL_TEST_KEY",
            base_url: "http://127.0.0.1:8080/v1",
            max_tokens: 128,
          },
        },
      }),
    (error) => error.code === "opencode_max_tokens_unsupported",
  );
});

test("rejects missing and ambiguous OpenCode model selection", () => {
  assert.throws(() => selectModel({}), (error) => error.code === "opencode_model_required");
  assert.throws(
    () =>
      selectModel({
        models: {
          coding: { provider: "openai", model: "gpt-4.1-mini", api_key_env: "OPENAI_API_KEY" },
          review: { provider: "anthropic", model: "claude-sonnet-4", api_key_env: "ANTHROPIC_API_KEY" },
        },
      }),
    (error) => error.code === "opencode_model_ambiguous",
  );
});

test("rejects a non-portable OpenCode credential environment-variable name", () => {
  assert.throws(
    () =>
      selectModel({
        models: {
          default: { provider: "openai", model: "gpt-4.1-mini", api_key_env: "MODEL-API-KEY" },
        },
      }),
    (error) => error.code === "opencode_invalid_model",
  );
});

test("rejects non-HTTP OpenCode model endpoints", () => {
  assert.throws(
    () =>
      selectModel({
        models: {
          default: {
            provider: "local-test",
            model: "test-model",
            api_key_env: "LOCAL_TEST_KEY",
            base_url: "ftp://provider.example/v1",
          },
        },
      }),
    (error) => error.code === "opencode_invalid_model",
  );
});

test("rejects non-loopback HTTP OpenCode model endpoints", () => {
  assert.throws(
    () =>
      selectModel({
        models: {
          default: {
            provider: "hosted-test",
            model: "test-model",
            api_key_env: "HOSTED_TEST_KEY",
            base_url: "http://provider.example/v1",
          },
        },
      }),
    (error) => error.code === "opencode_invalid_model",
  );
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { selectModel } from "../dist/configuration.js";

test("selects the default OpenCode model role", () => {
  assert.deepEqual(
    selectModel({
      models: {
        default: { provider: "nvidia", model: "nvidia/nemotron-3.5-lightning-30b-a3b", api_key_env: "NVIDIA_API_KEY" },
        reviewer: { provider: "nvidia", model: "nvidia/nemotron-3-ultra-550b-a55b", api_key_env: "NVIDIA_API_KEY" },
      },
    }),
    { provider: "nvidia", model: "nvidia/nemotron-3.5-lightning-30b-a3b", apiKeyEnv: "NVIDIA_API_KEY" },
  );
});

test("selects a sole OpenCode model role", () => {
  assert.deepEqual(
    selectModel({
      models: {
        coding: { provider: "nvidia", model: "nvidia/nemotron-3.5-lightning-30b-a3b", api_key_env: "NVIDIA_API_KEY" },
      },
    }),
    { provider: "nvidia", model: "nvidia/nemotron-3.5-lightning-30b-a3b", apiKeyEnv: "NVIDIA_API_KEY" },
  );
});

test("selects an OpenAI-compatible endpoint when configured", () => {
  assert.deepEqual(
    selectModel({
      models: {
        default: {
          provider: "nvidia",
          model: "nvidia/nemotron-3.5-lightning-30b-a3b",
          api_key_env: "NVIDIA_API_KEY",
          base_url: "http://127.0.0.1:8080/v1",
        },
      },
    }),
    {
      provider: "nvidia",
      model: "nvidia/nemotron-3.5-lightning-30b-a3b",
      apiKeyEnv: "NVIDIA_API_KEY",
      baseUrl: "http://127.0.0.1:8080/v1",
    },
  );
});

test("selects an HTTPS OpenAI-compatible endpoint when configured", () => {
  assert.deepEqual(
    selectModel({
      models: {
        default: {
          provider: "nvidia",
          model: "nvidia/nemotron-3.5-lightning-30b-a3b",
          api_key_env: "NVIDIA_API_KEY",
          base_url: "https://provider.example/v1",
        },
      },
    }),
    {
      provider: "nvidia",
      model: "nvidia/nemotron-3.5-lightning-30b-a3b",
      apiKeyEnv: "NVIDIA_API_KEY",
      baseUrl: "https://provider.example/v1",
    },
  );
});

test("selects sampling settings for an OpenAI-compatible endpoint", () => {
  assert.deepEqual(
    selectModel({
      models: {
        default: {
          provider: "nvidia",
          model: "nvidia/nemotron-3.5-lightning-30b-a3b",
          api_key_env: "NVIDIA_API_KEY",
          base_url: "http://127.0.0.1:8080/v1",
          temperature: 0.25,
          top_p: 0.8,
        },
      },
    }),
    {
      provider: "nvidia",
      model: "nvidia/nemotron-3.5-lightning-30b-a3b",
      apiKeyEnv: "NVIDIA_API_KEY",
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
            provider: "nvidia",
            model: "nvidia/nemotron-3.5-lightning-30b-a3b",
            api_key_env: "NVIDIA_API_KEY",
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
            provider: "nvidia",
            model: "nvidia/nemotron-3.5-lightning-30b-a3b",
            api_key_env: "NVIDIA_API_KEY",
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
          coding: { provider: "nvidia", model: "nvidia/nemotron-3.5-lightning-30b-a3b", api_key_env: "NVIDIA_API_KEY" },
          review: { provider: "nvidia", model: "nvidia/nemotron-3-ultra-550b-a55b", api_key_env: "NVIDIA_API_KEY" },
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
          default: { provider: "nvidia", model: "nvidia/nemotron-3.5-lightning-30b-a3b", api_key_env: "MODEL-API-KEY" },
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
            provider: "nvidia",
            model: "nvidia/nemotron-3.5-lightning-30b-a3b",
            api_key_env: "NVIDIA_API_KEY",
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
            provider: "nvidia",
            model: "nvidia/nemotron-3.5-lightning-30b-a3b",
            api_key_env: "NVIDIA_API_KEY",
            base_url: "http://provider.example/v1",
          },
        },
      }),
    (error) => error.code === "opencode_invalid_model",
  );
});

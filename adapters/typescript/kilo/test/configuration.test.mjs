// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import {selectMaxTurns, selectModel, selectSystemInstruction} from "../dist/configuration.js";

test("selects the default Kilo Code model and normalized sampling", () => {
  assert.deepEqual(selectModel({models: {default: {
    provider: "nvidia", model: "nvidia/nemotron", api_key_env: "NVIDIA_API_KEY",
    base_url: "https://integrate.api.nvidia.com/v1", temperature: 0.2, top_p: 0.9,
  }}}), {
    provider: "nvidia", model: "nvidia/nemotron", apiKeyEnv: "NVIDIA_API_KEY",
    baseUrl: "https://integrate.api.nvidia.com/v1", temperature: 0.2, topP: 0.9,
  });
});

test("selects a sole model role", () => {
  assert.equal(selectModel({models: {coding: {provider: "openai", model: "gpt-5", api_key_env: "OPENAI_API_KEY"}}}).model, "gpt-5");
});

test("rejects ambiguous, unsafe, and unsupported model configuration", () => {
  assert.throws(() => selectModel({models: {a: {provider: "x", model: "a", api_key_env: "KEY"}, b: {provider: "x", model: "b", api_key_env: "KEY"}}}), (error) => error.code === "kilo_model_ambiguous");
  assert.throws(() => selectModel({models: {default: {provider: "x", model: "a", api_key_env: "BAD-KEY"}}}), (error) => error.code === "kilo_invalid_model");
  assert.throws(() => selectModel({models: {default: {provider: "x", model: "a", api_key_env: "KEY", base_url: "http://example.com/v1"}}}), (error) => error.code === "kilo_invalid_model");
  assert.throws(() => selectModel({models: {default: {provider: "x", model: "a", api_key_env: "KEY", max_tokens: 42}}}), (error) => error.code === "kilo_max_tokens_unsupported");
});

test("maps replacement instructions and positive maximum turns", () => {
  assert.equal(selectSystemInstruction({instructions: {system: {content: "Review carefully.", mode: "replace"}}}), "Review carefully.");
  assert.equal(selectMaxTurns({runtime: {max_turns: 12}}), 12);
  assert.throws(() => selectSystemInstruction({instructions: {system: {content: "Extra", mode: "append"}}}), (error) => error.code === "unsupported_system_instruction_mode");
  assert.throws(() => selectMaxTurns({runtime: {max_turns: 0}}), (error) => error.code === "kilo_invalid_max_turns");
});

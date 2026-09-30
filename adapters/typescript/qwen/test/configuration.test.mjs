// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { selectModel, selectSystemPrompt, selectPermissionMode, selectBlockedTools, selectMcpServers, rejectUnclaimedConfig } from "../dist/configuration.js";

const descriptor = JSON.parse(await readFile(new URL("../qwen.fabric-adapter.json", import.meta.url)));
const model = { provider: "openai", model: "test-model", api_key_env: "TEST_KEY", base_url: "http://127.0.0.1:1234/v1" };

test("descriptor claims only implemented normalized fields", () => {
  assert.equal(descriptor.adapter_id, "nvidia.fabric.qwen");
  assert.deepEqual(descriptor.config.accepts, [
    "models", "models.base_url", "models.temperature", "models.top_p", "instructions.system", "tools.blocked", "skills", "mcp", "mcp.tool_filters",
  ]);
  assert.deepEqual(descriptor.config.system_instruction_modes, ["replace", "append"]);
  assert.deepEqual(descriptor.capabilities, { streaming: false, cancellation: false, updates: false, service: false });
});

test("selects and validates an OpenAI-compatible model", () => {
  assert.deepEqual(selectModel({ models: { default: { ...model, temperature: 0.2, top_p: 0.8 } } }), {
    id: "test-model", apiKeyEnv: "TEST_KEY", baseUrl: "http://127.0.0.1:1234/v1", temperature: 0.2, topP: 0.8,
  });
  assert.throws(() => selectModel({}), (error) => error.code === "qwen_model_required");
  assert.throws(() => selectModel({ models: { one: model, two: model } }), (error) => error.code === "qwen_model_ambiguous");
  assert.throws(() => selectModel({ models: { default: { ...model, provider: "anthropic" } } }), (error) => error.code === "qwen_invalid_model");
  assert.throws(() => selectModel({ models: { default: { ...model, base_url: "http://example.com/v1" } } }), (error) => error.code === "qwen_invalid_model");
  assert.throws(() => selectModel({ models: { default: { ...model, max_tokens: 100 } } }), (error) => error.code === "qwen_unsupported_model_field");
});

test("maps both system instruction modes and rejects unclaimed config", () => {
  assert.equal(selectSystemPrompt({ instructions: { system: { content: "review carefully" } } }), "review carefully");
  assert.deepEqual(selectSystemPrompt({ instructions: { system: { content: "review carefully", mode: "append" } } }), {
    type: "preset", preset: "qwen_code", append: "review carefully",
  });
  assert.throws(() => selectSystemPrompt({ instructions: { system: { content: "x", mode: "prepend" } } }),
    (error) => error.code === "unsupported_system_instruction_mode");
  assert.deepEqual(selectBlockedTools({ tools: { blocked: ["Bash"] } }), ["Bash"]);
  assert.throws(() => selectBlockedTools({ tools: { enabled: [] } }), (error) => error.code === "qwen_unsupported_tools");
  assert.equal(selectPermissionMode({ harness: { settings: { permission_mode: "yolo" } } }), "yolo");
  assert.throws(() => selectPermissionMode({ harness: { settings: { permission_mode: "other" } } }), (error) => error.code === "qwen_invalid_settings");
  assert.throws(() => rejectUnclaimedConfig({ runtime: { max_turns: 1 } }), (error) => error.code === "qwen_unsupported_config");
});

test("maps normalized stdio and streamable HTTP MCP servers", () => {
  assert.deepEqual({ ...selectMcpServers({ mcp: { servers: {
    local: {
      transport: "stdio", url: "node", args: ["server.mjs"], env: { MODE: "test" },
      allowed_tools: ["echo"], blocked_tools: ["hidden"],
    },
    remote: {
      transport: "streamable-http", url: "https://mcp.example.test/api",
      custom_headers: { Authorization: "Bearer ${MCP_TOKEN}" },
    },
  } } }, { MCP_TOKEN: "configured-token" }, {}) }, {
    local: {
      command: "node", args: ["server.mjs"], env: { MODE: "test" },
      includeTools: ["echo"], excludeTools: ["hidden"],
    },
    remote: {
      httpUrl: "https://mcp.example.test/api",
      headers: { Authorization: "Bearer configured-token" },
    },
  });
  assert.throws(() => selectMcpServers({ mcp: { servers: {
    remote: { transport: "streamable-http", url: "http://mcp.example.test/api" },
  } } }, {}, {}), (error) => error.code === "qwen_mcp_invalid_server");
  assert.throws(() => selectMcpServers({ mcp: { servers: {
    legacy: { transport: "sse", url: "https://mcp.example.test/sse" },
  } } }, {}, {}), (error) => error.code === "qwen_mcp_transport_unsupported");
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  selectMcpServers,
  selectModel,
  selectSettings,
  systemInstructions,
} from "../dist/configuration.js";

test("maps ACP-native MCP transports and expands configured headers", () => {
  const servers = selectMcpServers(
    {
      mcp: {
        servers: {
          local: {
            transport: "stdio",
            url: "/usr/bin/node",
            args: ["server.mjs"],
            env: { TOKEN: "scoped" },
          },
          remote: {
            transport: "streamable-http",
            url: "https://mcp.example.test",
            custom_headers: { Authorization: "Bearer ${MCP_TOKEN}" },
          },
        },
      },
    },
    { MCP_TOKEN: "secret" },
    {},
  );
  assert.deepEqual(servers, [
    {
      name: "local",
      command: "/usr/bin/node",
      args: ["server.mjs"],
      env: [{ name: "TOKEN", value: "scoped" }],
    },
    {
      type: "http",
      name: "remote",
      url: "https://mcp.example.test",
      headers: [{ name: "Authorization", value: "Bearer secret" }],
    },
  ]);
});

test("rejects MCP fields Strands ACP cannot enforce", () => {
  assert.throws(
    () =>
      selectMcpServers(
        {
          mcp: {
            servers: {
              filtered: {
                transport: "stdio",
                url: "/bin/echo",
                allowed_tools: ["read"],
              },
            },
          },
        },
        {},
        {},
      ),
    (error) => error.code === "strands_mcp_tool_filters_unsupported",
  );
  assert.throws(
    () =>
      selectMcpServers(
        {
          mcp: {
            servers: {
              insecure: {
                transport: "streamable-http",
                url: "http://example.test",
              },
            },
          },
        },
        {},
        {},
      ),
    (error) => error.code === "strands_mcp_invalid_server",
  );
});

test("maps Bedrock and explicit Chat Completions endpoints without changing model IDs", () => {
  const settings = selectSettings({});
  assert.equal(
    selectModel(
      {
        models: {
          default: { provider: "bedrock", model: "us.amazon.nova-pro-v1:0" },
        },
      },
      settings,
    ).name,
    "bedrock/us.amazon.nova-pro-v1:0",
  );
  assert.deepEqual(
    selectModel(
      {
        models: {
          default: {
            provider: "nvidia",
            model: "nvidia/nemotron",
            base_url: "https://integrate.api.nvidia.com/v1",
            api_key_env: "NVIDIA_API_KEY",
          },
        },
      },
      settings,
    ),
    {
      name: "litellm/nvidia/nemotron",
      endpoint: "https://integrate.api.nvidia.com/v1",
      keyEnvironment: "NVIDIA_API_KEY",
    },
  );
  assert.equal(
    selectModel(
      {
        models: {
          default: {
            provider: "openai",
            model: "local",
            base_url: "http://localhost:8000/v1",
          },
        },
      },
      settings,
    ).name,
    "litellm/local",
  );
  assert.throws(
    () =>
      selectModel(
        {
          models: {
            default: {
              provider: "openai",
              model: "local",
              base_url: "http://10.1.2.3:8000/v1",
            },
          },
        },
        settings,
      ),
    { code: "strands_model_invalid" },
  );
  const trusted = selectSettings({
    harness: { settings: { insecure_http_model_hosts: ["10.1.2.3"] } },
  });
  assert.ok(
    selectModel(
      {
        models: {
          default: {
            provider: "openai",
            model: "local",
            base_url: "http://10.1.2.3:8000/v1",
          },
        },
      },
      trusted,
    ),
  );
});

test("rejects unsupported model roles, credential URLs, settings, and instruction replacement", () => {
  for (const models of [
    { other: { provider: "bedrock", model: "x" } },
    {
      default: {
        provider: "openai",
        model: "x",
        base_url: "https://user:secret@example.test/v1",
      },
    },
    { default: { provider: "bedrock", model: "x", api_key_env: "KEY" } },
  ]) {
    assert.throws(() => selectModel({ models }, selectSettings({})));
  }
  for (const settings of [
    { unknown: true },
    { builtin_tools: ["subagent"] },
    { max_invocations: 0 },
  ]) {
    assert.throws(() => selectSettings({ harness: { settings } }), {
      code: "strands_settings_invalid",
    });
  }
  assert.throws(
    () =>
      systemInstructions({
        instructions: { system: { mode: "replace", content: "x" } },
      }),
    { code: "strands_instruction_mode_unsupported" },
  );
  assert.equal(
    systemInstructions({
      instructions: { system: { mode: "append", content: "review" } },
    }),
    "review",
  );
});

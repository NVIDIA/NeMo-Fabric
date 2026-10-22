// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { selectMcpServers, selectModel, selectSettings } from "../dist/configuration.js";

test("selects the narrow Kiro model and harness settings", () => {
  assert.equal(selectModel({ models: { default: { provider: "kiro", model: "model-a" } } }), "model-a");
  assert.deepEqual(
    selectSettings({ harness: { settings: { mode_id: "plan", effort: "high", permission_mode: "allow_once" } } }),
    { modeId: "plan", effort: "high", permissionMode: "allow_once" },
  );
});

test("maps ACP-native MCP transports and expands configured headers", () => {
  const servers = selectMcpServers(
    {
      mcp: {
        servers: {
          local: { transport: "stdio", url: "/usr/bin/node", args: ["server.mjs"], env: { TOKEN: "scoped" } },
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

test("rejects MCP fields Kiro ACP cannot enforce", () => {
  assert.throws(
    () =>
      selectMcpServers(
        { mcp: { servers: { filtered: { transport: "stdio", url: "/bin/echo", allowed_tools: ["read"] } } } },
        {},
        {},
      ),
    (error) => error.code === "kiro_mcp_tool_filters_unsupported",
  );
  assert.throws(
    () =>
      selectMcpServers(
        { mcp: { servers: { insecure: { transport: "streamable-http", url: "http://example.test" } } } },
        {},
        {},
      ),
    (error) => error.code === "kiro_mcp_invalid_server",
  );
});

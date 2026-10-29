// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { selectByomModel, selectMcpServers, selectModel, selectSettings } from "../dist/configuration.js";

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

test("experimental BYOM preserves the full model ID and resolves explicit credentials", () => {
  const config = {
    harness: { settings: { experimental_byom: true } },
    models: {
      default: {
        provider: "openai",
        model: "nvidia/nemotron-model",
        base_url: "https://model.example.test/v1",
        api_key_env: "MODEL_KEY",
      },
    },
  };
  assert.equal(selectModel(config), "nvidia/nemotron-model");
  assert.deepEqual(selectByomModel(config, { MODEL_KEY: "configured" }, { MODEL_KEY: "inherited" }), {
    model: "nvidia/nemotron-model",
    baseUrl: "https://model.example.test/v1",
    apiKey: "configured",
  });
  assert.equal(selectByomModel(config, {}, { MODEL_KEY: "inherited" }).apiKey, "inherited");
  assert.throws(
    () => selectByomModel(config, { MODEL_KEY: "" }, { MODEL_KEY: "inherited" }),
    (error) => error.code === "kiro_byom_credential_missing",
  );
  delete config.models.default.api_key_env;
  assert.deepEqual(selectByomModel(config, {}, {}), {
    model: "nvidia/nemotron-model",
    baseUrl: "https://model.example.test/v1",
  });
  assert.equal(selectByomModel({ models: { default: { provider: "kiro", model: "native" } } }, {}, {}), undefined);
});

test("BYOM requires an explicit opt-in, compatible provider, and endpoint", () => {
  const config = {
    models: {
      default: {
        provider: "openai",
        model: "served/model",
        base_url: "https://model.example.test/v1",
      },
    },
  };
  assert.throws(
    () => selectModel(config),
    (error) => error.code === "kiro_byom_opt_in_required",
  );
  for (const provider of ["kiro", "nvidia", "anthropic"]) {
    assert.throws(
      () =>
        selectModel({
          harness: { settings: { experimental_byom: true } },
          models: { default: { provider, model: "model" } },
        }),
      (error) => error.code === "kiro_byom_model_required",
    );
  }
  assert.throws(
    () =>
      selectModel({
        harness: { settings: { experimental_byom: true } },
        models: { default: { provider: "openai", model: "model" } },
      }),
    (error) => error.code === "kiro_byom_endpoint_invalid",
  );
  assert.throws(
    () => selectSettings({ harness: { settings: { experimental_byom: "true" } } }),
    (error) => error.code === "kiro_settings_invalid",
  );
});

test("BYOM follows the shared model HTTP trust boundary and never echoes rejected URLs", () => {
  function config(baseUrl, hosts) {
    return {
      harness: {
        settings: {
          experimental_byom: true,
          ...(hosts === undefined ? {} : { insecure_http_model_hosts: hosts }),
        },
      },
      models: {
        default: {
          provider: "openai",
          model: "served/model",
          base_url: baseUrl,
        },
      },
    };
  }
  for (const url of ["http://127.0.0.1:8000/v1", "http://localhost:8000/v1", "https://model.example.test/v1"]) {
    assert.equal(selectModel(config(url)), "served/model");
  }
  assert.equal(selectModel(config("http://10.86.19.10:8000/v1", ["10.86.19.10"])), "served/model");
  for (const url of [
    "http://10.86.19.10:8000/v1",
    "https://secret@model.example.test/v1",
    "https://model.example.test/v1?key=secret",
    "https://model.example.test/v1#secret",
    "file:///secret",
  ]) {
    assert.throws(
      () => selectModel(config(url)),
      (error) =>
        error.code === "kiro_byom_endpoint_invalid" &&
        !error.message.includes(url) &&
        !error.message.includes("secret"),
    );
  }
  assert.throws(
    () => selectSettings(config("https://model.example.test", ["http://bad.example.test"])),
    (error) => error.code === "kiro_settings_invalid",
  );
});

test("BYOM credential errors omit environment names and values", () => {
  const config = {
    harness: { settings: { experimental_byom: true } },
    models: {
      default: {
        provider: "openai",
        model: "served/model",
        base_url: "https://model.example.test/v1",
        api_key_env: "PRIVATE_MODEL_KEY",
      },
    },
  };
  for (const value of [undefined, "", "secret\nvalue"]) {
    assert.throws(
      () => selectByomModel(config, {}, { PRIVATE_MODEL_KEY: value }),
      (error) =>
        error.code === "kiro_byom_credential_missing" &&
        !error.message.includes("PRIVATE_MODEL_KEY") &&
        !error.message.includes("secret"),
    );
  }
  config.models.default.api_key_env = "invalid-secret-name";
  assert.throws(
    () => selectByomModel(config, {}, {}),
    (error) => error.code === "kiro_byom_credential_invalid" && !error.message.includes("invalid-secret-name"),
  );
});

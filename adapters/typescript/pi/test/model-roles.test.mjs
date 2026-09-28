// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { InMemoryCredentialStore, InMemoryModelsStore } from "@earendil-works/pi-ai";
import { AgentSession, ModelRuntime, SettingsManager } from "@earendil-works/pi-coding-agent";

import { loadConfiguredModels } from "../dist/pi-model.js";
import { PiSdkSessionFactory } from "../dist/pi-sdk.js";
import { PiAdapterRuntime } from "../dist/runtime.js";

const pi = { ModelRuntime, InMemoryModelsStore };
const endpoint = "https://inference.local/v1";
const credential = () => "fixture-key";

/** Resolve model roles, pass them to check, and remove Pi's generated models.json. */
async function withRoles(models, check) {
  const loaded = await loadConfiguredModels(pi, new InMemoryCredentialStore(), models, "default", {
    relayEnabled: false,
    credential,
  });
  try {
    await check(loaded.roles);
  } finally {
    await loaded.cleanup();
  }
}

const load = (models) => withRoles(models, () => {});

function described(model, metadata, extra = {}) {
  return {
    provider: "openai",
    model,
    base_url: endpoint,
    api_key_env: "MODEL_KEY",
    api: "openai-completions",
    settings: { model_metadata: metadata },
    ...extra,
  };
}

test("a catalog model keeps its provider and uses the configured endpoint", async () => {
  const catalog = { provider: "openai", model: "gpt-4o-mini", base_url: endpoint, api_key_env: "MODEL_KEY" };
  await withRoles({ default: catalog }, (roles) => {
    const model = roles.get("default");
    assert.deepEqual([model.provider, model.id, model.baseUrl], ["openai", "gpt-4o-mini", endpoint]);
  });
});

test("model_metadata describes a model that the Pi catalog does not know", async () => {
  const metadata = { contextWindow: 8192, maxTokens: 2048, compat: { supportsDeveloperRole: false } };
  await withRoles({ default: described("qwen3:4b", metadata) }, (roles) => {
    const model = roles.get("default");
    assert.deepEqual(
      [model.id, model.baseUrl, model.api, model.contextWindow, model.maxTokens],
      ["qwen3:4b", endpoint, "openai-completions", 8192, 2048],
    );
    assert.equal(model.compat.supportsDeveloperRole, false);
    assert.equal(model.reasoning, false, "Pi applies its own defaults");
  });
});

test("Pi validates model_metadata, and an unknown model needs it", async () => {
  await assert.rejects(load({ default: described("qwen3:4b", { contextWindow: "large" }) }), /models\.json/);
  await assert.rejects(
    load({ default: { provider: "openai", model: "custom-model", api_key_env: "MODEL_KEY" } }),
    (error) => error.code === "pi_model_unknown" && /model_metadata/.test(error.message),
  );
});

test("identical roles share a provider and distinct roles get their own", async () => {
  const fast = described("fast", { contextWindow: 8192 });
  const smart = described("smart", { contextWindow: 8192 }, { base_url: "https://smart.local/v1" });
  await withRoles({ default: fast, primary: fast, smart }, (roles) => {
    assert.deepEqual(
      [...roles].map(([role, model]) => [role, model.provider, model.baseUrl]),
      [
        ["default", "openai", endpoint],
        ["primary", "openai", endpoint],
        ["smart", "openai-smart", "https://smart.local/v1"],
      ],
    );
  });
});

/** Start Pi with identical default and fast roles plus a distinct smart role. */
async function withRoleRuntime({ telemetry, factory = new PiSdkSessionFactory(), smartContextWindow = 8192 }, body) {
  const requests = [];
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    requests.push({
      path: request.url,
      authorization: request.headers.authorization,
      body: JSON.parse(Buffer.concat(chunks).toString()),
    });
    const chunk = {
      id: "fixture",
      object: "chat.completion.chunk",
      created: 0,
      model: "fixture",
      choices: [{ index: 0, delta: { role: "assistant", content: "FOUR" }, finish_reason: "stop" }],
    };
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.end(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const workspace = await mkdtemp(join(tmpdir(), "pi-model-roles-"));
  const runtime = new PiAdapterRuntime(factory);
  try {
    const { port } = server.address();
    const role = (path, key, contextWindow = 8192) => ({
      provider: "openai",
      model: "fixture",
      base_url: `http://127.0.0.1:${port}/${path}/v1`,
      api_key_env: key,
      api: "openai-completions",
      settings: { model_metadata: { contextWindow, maxTokens: 128 } },
    });
    const fast = role("fast", "FAST_KEY");
    const runtimeContext = {
      artifacts: {},
      environment: {
        control_location: "external_control",
        environment_id: "fixture",
        ownership: "caller_owned",
        provider: "local",
        workspace,
        env: { FAST_KEY: "fixture-fast", SMART_KEY: "fixture-smart" },
      },
      invocation_id: "start",
      request_id: "start",
      runtime_id: "fixture",
      ...(telemetry === undefined ? {} : { telemetry }),
    };
    await runtime.start({
      agentName: "main",
      baseDir: workspace,
      config: {
        models: { default: fast, fast, smart: role("smart", "SMART_KEY", smartContextWindow) },
        tools: { enabled: [] },
      },
      runtimeContext,
    });
    await body({ invoke: (input) => runtime.invoke({ input }, runtimeContext), requests, runtime });
  } finally {
    await runtime.stop();
    await rm(workspace, { recursive: true, force: true });
    await new Promise((resolve) => server.close(() => resolve()));
  }
}

test("invocations select declared roles and keep the conversation", { timeout: 30000 }, async () => {
  await withRoleRuntime({}, async ({ invoke, requests }) => {
    for (const input of [{ prompt: "Reply FOUR", model: "fast" }, { prompt: "Reply FOUR", model: "smart" }, "Again"]) {
      const result = await invoke(input);
      assert.equal(result.status, "succeeded", JSON.stringify(result));
    }
    assert.deepEqual(
      requests.map((entry) => [entry.path, entry.authorization]),
      [
        ["/fast/v1/chat/completions", "Bearer fixture-fast"],
        ["/smart/v1/chat/completions", "Bearer fixture-smart"],
        ["/smart/v1/chat/completions", "Bearer fixture-smart"],
      ],
      "a selected role stays active for later plain-text invocations",
    );
    assert(requests[1].body.messages.length > requests[0].body.messages.length, "switching keeps history");
    const unknown = await invoke({ prompt: "Do not send", model: "missing" });
    assert.equal(unknown.status, "failed");
    assert.equal(unknown.error.code, "pi_model_selection_failed");
    assert.equal(unknown.error.extensions?.reason, "pi_model_unknown");
    assert.equal(requests.length, 3, "an unknown role fails before inference");
  });
});

test("with Relay, roles that resolve to the active model stay selectable", { timeout: 30000 }, async () => {
  const factory = new PiSdkSessionFactory({ start: async () => undefined });
  await withRoleRuntime({ telemetry: { relay_enabled: true }, factory }, async ({ invoke, requests }) => {
    for (const model of ["default", "fast"]) {
      const result = await invoke({ prompt: "Reply FOUR", model });
      assert.equal(result.status, "succeeded", `${model}: ${JSON.stringify(result)}`);
    }
    const switched = await invoke({ prompt: "Do not send", model: "smart" });
    assert.equal(switched.status, "failed");
    assert.equal(switched.error.code, "pi_model_selection_failed");
    assert.equal(switched.error.extensions?.reason, "pi_model_switch_unsupported");
    assert.deepEqual(
      requests.map((entry) => entry.path),
      ["/fast/v1/chat/completions", "/fast/v1/chat/completions"],
    );
  });
});

test("a failed role switch keeps the active role's session settings", { timeout: 30000 }, async () => {
  const setModel = AgentSession.prototype.setModel;
  const applyOverrides = SettingsManager.prototype.applyOverrides;
  const reserves = [];
  AgentSession.prototype.setModel = async function () {
    throw new Error("fixture setModel failure");
  };
  SettingsManager.prototype.applyOverrides = function (overrides) {
    if (overrides?.compaction?.reserveTokens !== undefined) {
      reserves.push(overrides.compaction.reserveTokens);
    }
    return applyOverrides.call(this, overrides);
  };
  try {
    await withRoleRuntime({ smartContextWindow: 65536 }, async ({ invoke, requests }) => {
      const switched = await invoke({ prompt: "Do not send", model: "smart" });
      assert.equal(switched.status, "failed");
      assert.equal(switched.error.code, "pi_model_selection_failed");
      assert.deepEqual(reserves, [4096, 16384, 4096], "the smart role's settings are rolled back");
      const result = await invoke("Reply FOUR");
      assert.equal(result.status, "succeeded", JSON.stringify(result));
      assert.deepEqual(requests.map((entry) => entry.path), ["/fast/v1/chat/completions"]);
    });
  } finally {
    AgentSession.prototype.setModel = setModel;
    SettingsManager.prototype.applyOverrides = applyOverrides;
  }
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { InMemoryCredentialStore, InMemoryModelsStore } from "@earendil-works/pi-ai";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";

import { loadConfiguredModels } from "../dist/pi-model.js";
import { PiSdkSessionFactory } from "../dist/pi-sdk.js";
import { PiAdapterRuntime } from "../dist/runtime.js";

const pi = { ModelRuntime, InMemoryModelsStore };
const endpoint = "https://inference.local/v1";
const credential = () => "fixture-key";

async function load(models, options = {}) {
  return loadConfiguredModels(pi, new InMemoryCredentialStore(), models, "default", {
    relayEnabled: false,
    credential,
    ...options,
  });
}

function withMetadata(model, metadata, extra = {}) {
  return {
    provider: "openai",
    model,
    base_url: endpoint,
    api_key_env: "MODEL_KEY",
    settings: { model_metadata: metadata },
    ...extra,
  };
}

test("catalog models keep their provider and use the configured endpoint", async () => {
  const loaded = await load({
    default: { provider: "openai", model: "gpt-4o-mini", base_url: endpoint, api_key_env: "MODEL_KEY" },
  });
  try {
    const model = loaded.roles.get("default");
    assert.equal(model.provider, "openai");
    assert.equal(model.id, "gpt-4o-mini");
    assert.equal(model.baseUrl, endpoint);
  } finally {
    await loaded.cleanup();
  }
});

test("model_metadata defines a model that the Pi catalog does not know", async () => {
  const loaded = await load({
    default: withMetadata("qwen3:4b", { contextWindow: 8192, maxTokens: 2048, compat: { supportsDeveloperRole: false } }, {
      api: "openai-completions",
    }),
  });
  try {
    const model = loaded.roles.get("default");
    assert.equal(model.provider, "openai");
    assert.equal(model.id, "qwen3:4b");
    assert.equal(model.baseUrl, endpoint);
    assert.equal(model.api, "openai-completions");
    assert.equal(model.contextWindow, 8192);
    assert.equal(model.maxTokens, 2048);
    assert.equal(model.compat.supportsDeveloperRole, false);
    assert.equal(model.reasoning, false, "Pi applies its own defaults");
  } finally {
    await loaded.cleanup();
  }
});

test("Pi validates model_metadata and a conflicting protocol fails", async () => {
  await assert.rejects(
    load({ default: withMetadata("qwen3:4b", { api: "openai-completions", contextWindow: "large" }) }),
    /models\.json/,
  );
  await assert.rejects(
    load({ default: withMetadata("qwen3:4b", { api: "anthropic-messages" }, { api: "openai-completions" }) }),
    (error) => error.code === "pi_model_api_conflict",
  );
});

test("an unknown model without metadata fails instead of borrowing another model", async () => {
  await assert.rejects(
    load({ default: { provider: "openai", model: "custom-model", api_key_env: "MODEL_KEY" } }),
    (error) => error.code === "pi_model_unknown" && /model_metadata/.test(error.message),
  );
});

test("identical roles share a provider and distinct roles get their own", async () => {
  const fast = withMetadata("fast", { api: "openai-completions", contextWindow: 8192, maxTokens: 128 });
  const smart = withMetadata("smart", { api: "openai-completions", contextWindow: 8192, maxTokens: 128 }, {
    base_url: "https://smart.local/v1",
  });
  const loaded = await load({ default: fast, primary: fast, smart });
  try {
    assert.deepEqual(
      [...loaded.roles].map(([role, model]) => [role, model.provider, model.baseUrl]),
      [
        ["default", "openai", endpoint],
        ["primary", "openai", endpoint],
        ["smart", "openai-smart", "https://smart.local/v1"],
      ],
    );
  } finally {
    await loaded.cleanup();
  }
});

test("invocations select declared roles and keep the conversation", { timeout: 30000 }, async () => {
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
  const runtime = new PiAdapterRuntime(new PiSdkSessionFactory());
  try {
    const { port } = server.address();
    const role = (path, key) => ({
      provider: "openai",
      model: "fixture",
      base_url: `http://127.0.0.1:${port}/${path}/v1`,
      api_key_env: key,
      api: "openai-completions",
      settings: { model_metadata: { contextWindow: 8192, maxTokens: 128 } },
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
    };
    await runtime.start({
      agentName: "main",
      baseDir: workspace,
      config: { models: { default: fast, fast, smart: role("smart", "SMART_KEY") }, tools: { enabled: [] } },
      runtimeContext,
    });
    const invoke = (input) => runtime.invoke({ input }, runtimeContext);
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
    assert.equal(requests.length, 3, "an unknown role fails before inference");
  } finally {
    await runtime.stop();
    await rm(workspace, { recursive: true, force: true });
    await new Promise((resolve) => server.close(() => resolve()));
  }
});

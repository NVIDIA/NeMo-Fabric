// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import test from "node:test";

import { DroidSdkSessionFactory } from "../dist/droid-sdk.js";

function input(overrides = {}) {
  return {
    agentName: "droid-test",
    baseDir: "/tmp",
    config: {
      models: {
        default: { provider: "factory", model: "auto", api_key_env: "FACTORY_API_KEY" },
      },
      instructions: { system: { content: "Be concise.", mode: "append" } },
      tools: { enabled: ["Read"], blocked: ["Execute"] },
      ...overrides,
    },
    runtimeContext: {
      artifacts: {},
      environment: {
        control_location: "external_control",
        env: { FACTORY_API_KEY: "secret" },
        environment_id: "environment-1",
        ownership: "caller_owned",
        provider: "local",
        workspace: "/tmp",
      },
      invocation_id: "start",
      request_id: "request-start",
      runtime_id: "runtime-1",
    },
  };
}

function result(overrides = {}) {
  return {
    type: "result",
    sessionId: "droid-session-1",
    durationMs: 12,
    tokenUsage: {
      inputTokens: 4,
      outputTokens: 3,
      cacheCreationTokens: 1,
      cacheReadTokens: 2,
      thinkingTokens: 5,
      factoryCredits: 0.25,
    },
    text: "done",
    turnCount: 1,
    success: true,
    subtype: "success",
    interrupted: false,
    error: null,
    ...overrides,
  };
}

function fakeSdk(options = {}) {
  const calls = {
    create: [], createEnvironment: [], stream: [], close: 0, listTools: 0, updateSettings: [], listMcpTools: 0, listSkills: 0,
  };
  const results = [...(options.results ?? [result(), result({ text: "second", turnCount: 2 })])];
  const observers = new Set();
  let turn = 0;
  const session = {
    id: "droid-session-1",
    settings: { modelId: options.modelId },
    async *stream(prompt) {
      calls.stream.push(prompt);
      if (options.streamError) throw options.streamError;
      const index = turn++;
      yield { type: "assistant", text: "intermediate" };
      options.duringTurn?.(index, session);
      // Droid's SDK delivers the turn's completion notification before the result.
      const notification = {
        type: "agent_turn_completed",
        reason: "completed",
        turnId: `turn-${index}`,
        tokenUsage: results[0]?.tokenUsage,
        ...(options.childUsage?.[index] === undefined ? {} : { childTokenUsage: options.childUsage[index] }),
      };
      for (const observer of observers) {
        observer({ jsonrpc: "2.0", method: "droid.session_notification", params: { sessionId: session.id, notification } });
      }
      yield results.shift();
    },
    onNotification(callback) {
      observers.add(callback);
      return () => observers.delete(callback);
    },
    // Droid's SDK applies settings_updated to session.settings and passes it to observers.
    changeModel(modelId) {
      session.settings.modelId = modelId;
      const notification = { type: "settings_updated", settings: { modelId } };
      for (const observer of observers) {
        observer({ jsonrpc: "2.0", method: "droid.session_notification", params: { sessionId: session.id, notification } });
      }
    },
    async close() { calls.close += 1; },
    async listTools() {
      calls.listTools += 1;
      if (options.toolError) throw options.toolError;
      return options.tools ?? [{ id: "Read" }, { id: "Edit" }, { id: "Execute" }];
    },
    async updateSettings(value) { calls.updateSettings.push(value); },
    async listMcpTools() {
      calls.listMcpTools += 1;
      if (typeof options.mcpTools === "function") return options.mcpTools(calls.listMcpTools);
      return options.mcpTools ?? (calls.create.at(-1)?.mcpServers ?? []).map((server) => ({
        serverName: server.name,
        name: "test-tool",
      }));
    },
    async listSkills() {
      calls.listSkills += 1;
      return typeof options.skillReport === "function"
        ? options.skillReport(calls.create.at(-1))
        : options.skillReport ?? { skills: [] };
    },
  };
  return {
    calls,
    observers,
    session,
    loader: async () => ({
      async createSession(value) {
        calls.create.push(value);
        calls.createEnvironment.push({ ...process.env });
        if (options.createError) throw options.createError;
        return session;
      },
    }),
  };
}

test("maps model, credential, append instructions, and an exact built-in tool policy", async () => {
  const sdk = fakeSdk();
  const handle = await new DroidSdkSessionFactory(sdk.loader).create(input());
  const first = await handle.prompt("one");
  const second = await handle.prompt("two");

  assert.deepEqual(sdk.calls.create, [{
    cwd: await realpath("/tmp"),
    modelId: "auto",
    apiKey: "secret",
    autonomyLevel: "high",
    env: { FACTORY_API_KEY: "secret" },
    systemPrompt: { type: "preset", preset: "droid", append: "Be concise." },
  }]);
  assert.deepEqual(sdk.calls.updateSettings, [{ disabledToolIds: ["Edit", "Execute"] }]);
  assert.deepEqual(sdk.calls.stream, ["one", "two"]);
  assert.deepEqual(first.usage, {
    input_tokens: 4,
    cached_input_tokens: 2,
    output_tokens: 3,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        input_tokens: 4,
        cached_input_tokens: 2,
        cache_write_input_tokens: 1,
        output_tokens: 3,
        output_tokens_include_reasoning: true,
      },
      thinking_tokens: 5,
      factory_credits: 0.25,
    },
  });
  assert.deepEqual(first.extensions, {
    session_id: "droid-session-1",
    finish_reason: "success",
    duration_ms: 12,
    turn_count: 1,
  });
  assert.equal(second.text, "second");
  await handle.stop();
  await handle.stop();
  assert.equal(sdk.calls.close, 1);
});

test("isolates the Droid child environment and restores the adapter environment", async () => {
  const ambientName = "DROID_TEST_AMBIENT_SECRET";
  const explicitName = "DROID_TEST_EXPLICIT_VALUE";
  const previousAmbient = process.env[ambientName];
  process.env[ambientName] = "must-not-leak";
  try {
    const configured = input();
    configured.runtimeContext.environment.env[explicitName] = "configured";
    const sdk = fakeSdk();
    const handle = await new DroidSdkSessionFactory(sdk.loader).create(configured);

    assert.equal(sdk.calls.createEnvironment[0][ambientName], undefined);
    assert.equal(sdk.calls.createEnvironment[0][explicitName], "configured");
    assert.equal(sdk.calls.createEnvironment[0].FACTORY_API_KEY, "secret");
    assert.equal(process.env[ambientName], "must-not-leak");
    await handle.stop();

    const failing = fakeSdk({ createError: new Error("failed") });
    await assert.rejects(
      new DroidSdkSessionFactory(failing.loader).create(configured),
      (error) => error.code === "droid_start_failed",
    );
    assert.equal(failing.calls.createEnvironment[0][ambientName], undefined);
    assert.equal(process.env[ambientName], "must-not-leak");
  } finally {
    if (previousAmbient === undefined) delete process.env[ambientName];
    else process.env[ambientName] = previousAmbient;
  }
});

test("maps replacement instructions and native stdio, HTTP, and SSE MCP configuration", async () => {
  const sdk = fakeSdk();
  const configured = input({
    instructions: { system: { content: "Replace the prompt.", mode: "replace" } },
    tools: undefined,
    mcp: {
      servers: {
        local: { transport: "stdio", url: "node", args: ["server.js"], env: { MODE: "test" } },
        remote: {
          transport: "streamable-http",
          url: "https://mcp.example.test/rpc",
          custom_headers: { Authorization: "Bearer token" },
        },
        events: { transport: "sse", url: "http://127.0.0.1:3000/sse" },
        ipv6: { transport: "sse", url: "http://[::1]:3001/sse" },
      },
    },
  });
  const handle = await new DroidSdkSessionFactory(sdk.loader).create(configured);
  const { env, ...created } = sdk.calls.create[0];
  assert.deepEqual(created, {
    cwd: await realpath("/tmp"),
    modelId: "auto",
    apiKey: "secret",
    autonomyLevel: "high",
    mcpServers: [
      { name: "local", command: "node", args: ["server.js"], env: { MODE: "test" } },
      {
        name: "remote",
        type: "http",
        url: "https://mcp.example.test/rpc",
        headers: [{ name: "Authorization", value: "Bearer token" }],
      },
      { name: "events", type: "sse", url: "http://127.0.0.1:3000/sse", headers: [] },
      { name: "ipv6", type: "sse", url: "http://[::1]:3001/sse", headers: [] },
    ],
    systemPrompt: "Replace the prompt.",
  });
  assert.equal(env.HOME, env.USERPROFILE);
  assert.equal(env.HOME, env.FACTORY_HOME_OVERRIDE);
  assert.equal(env.FACTORY_API_KEY, "secret");
  await assert.rejects(readFile(join(env.HOME, ".factory", "mcp.json"), "utf8"), { code: "ENOENT" });
  assert.equal(sdk.calls.listMcpTools, 1);
  await handle.stop();
});

test("preserves Factory settings and its local overlay without copying ambient MCP configuration", async (t) => {
  const sourceHome = await mkdtemp(join(tmpdir(), "fabric-droid-factory-home-"));
  t.after(() => rm(sourceHome, { recursive: true, force: true }));
  const sourceFactory = join(sourceHome, ".factory");
  await mkdir(sourceFactory, { recursive: true });
  const settings = { cloudSessionSync: false };
  const localSettings = {
    customModels: [{
      model: "nvidia/nemotron-3-super-120b-a12b",
      displayName: "Local NVIDIA Nemotron 3 Super",
      baseUrl: "https://integrate.api.nvidia.com/v1",
      apiKey: "${NVIDIA_API_KEY}",
      provider: "generic-chat-completion-api",
    }],
  };
  await writeFile(join(sourceFactory, "settings.json"), `${JSON.stringify(settings)}\n`);
  await writeFile(join(sourceFactory, "settings.local.json"), `${JSON.stringify(localSettings)}\n`);
  await writeFile(join(sourceFactory, "mcp.json"), '{"mcpServers":{"ambient":{"type":"stdio"}}}\n');
  const sdk = fakeSdk();

  const configured = input({
    tools: undefined,
    mcp: { servers: { configured: { transport: "stdio", url: "node" } } },
  });
  configured.config.models.default.model = "custom:nvidia/nemotron-3-super-120b-a12b";
  configured.runtimeContext.environment.env.FACTORY_HOME_OVERRIDE = sourceHome;
  const handle = await new DroidSdkSessionFactory(
    sdk.loader,
    10_000,
    { FACTORY_HOME_OVERRIDE: join(sourceHome, "missing"), NVIDIA_API_KEY: "provider-secret" },
  ).create(configured);
  const runtimeFactory = join(sdk.calls.create[0].env.HOME, ".factory");
  assert.equal(sdk.calls.create[0].env.FACTORY_API_KEY, "secret");
  assert.equal(sdk.calls.create[0].env.NVIDIA_API_KEY, "provider-secret");
  assert.deepEqual(JSON.parse(await readFile(join(runtimeFactory, "settings.json"), "utf8")), settings);
  assert.deepEqual(
    JSON.parse(await readFile(join(runtimeFactory, "settings.local.json"), "utf8")),
    localSettings,
  );
  assert.deepEqual(sdk.calls.create[0].mcpServers, [
    { name: "configured", command: "node", args: [], env: {} },
  ]);
  await assert.rejects(readFile(join(runtimeFactory, "mcp.json"), "utf8"), { code: "ENOENT" });

  await handle.stop();
});

test("waits for configured MCP servers to expose tools", async () => {
  const sdk = fakeSdk({
    mcpTools(call) {
      return call === 1 ? [] : [{ serverName: "test", name: "echo" }];
    },
  });
  const handle = await new DroidSdkSessionFactory(sdk.loader).create(
    input({ tools: undefined, mcp: { servers: { test: { transport: "stdio", url: "node" } } } }),
  );

  assert.equal(sdk.calls.listMcpTools, 2);
  await handle.stop();
});

test("keeps configured MCP tools outside the built-in tool policy", async () => {
  const sdk = fakeSdk({
    tools: [{ id: "Read" }, { id: "Edit" }, { id: "test___echo" }],
  });
  const handle = await new DroidSdkSessionFactory(sdk.loader).create(
    input({
      tools: { enabled: ["Read"] },
      mcp: { servers: { test: { transport: "stdio", url: "node" } } },
    }),
  );

  assert.deepEqual(sdk.calls.updateSettings, [{ disabledToolIds: ["Edit"] }]);
  await handle.stop();
});

test("rejects an MCP server that does not expose a tool", async () => {
  const sdk = fakeSdk({
    mcpTools: [],
  });

  await assert.rejects(
    new DroidSdkSessionFactory(sdk.loader, 1).create(
      input({ tools: undefined, mcp: { servers: { test: { transport: "stdio", url: "node" } } } }),
    ),
    (error) =>
      error.code === "droid_mcp_load_failed" &&
      error.metadata?.missing?.includes("test"),
  );
  assert.equal(sdk.calls.close, 1);
});

test("times out when Droid does not return MCP tool status", async () => {
  const sdk = fakeSdk({
    mcpTools: () => new Promise(() => undefined),
  });

  await assert.rejects(
    new DroidSdkSessionFactory(sdk.loader, 1).create(
      input({ tools: undefined, mcp: { servers: { test: { transport: "stdio", url: "node" } } } }),
    ),
    (error) => error.code === "droid_mcp_load_failed",
  );
  assert.equal(sdk.calls.close, 1);
});

test("stages configured skills in an isolated Droid home and verifies discovery", async (t) => {
  const source = await mkdtemp(join(tmpdir(), "fabric-droid-skill-source-"));
  t.after(() => rm(source, { recursive: true, force: true }));
  await writeFile(join(source, "SKILL.md"), "---\nname: review-code\ndescription: Review code.\n---\n");
  const sdk = fakeSdk({
    skillReport: (created) => ({
      skills: [{
        name: "review-code",
        filePath: join(created.env.HOME, ".agents", "skills", basename(source), "SKILL.md"),
        enabled: true,
      }],
    }),
  });

  const handle = await new DroidSdkSessionFactory(sdk.loader).create(input({
    tools: undefined,
    skills: { paths: [source] },
  }));
  const created = sdk.calls.create[0];
  assert.equal(created.env.HOME, created.env.USERPROFILE);
  assert.equal(created.env.HOME, created.env.FACTORY_HOME_OVERRIDE);
  const stagedEntrypoint = join(created.env.HOME, ".agents", "skills", basename(source), "SKILL.md");
  assert.equal((await stat(stagedEntrypoint)).isFile(), true);
  assert.equal(sdk.calls.listSkills, 1);

  await handle.stop();
  await assert.rejects(stat(dirname(created.env.HOME)), { code: "ENOENT" });
});

test("rejects invalid, colliding, disabled, and extended skill configuration", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "fabric-droid-skills-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const first = join(root, "first", "same");
  const second = join(root, "second", "same");
  await mkdir(first, { recursive: true });
  await mkdir(second, { recursive: true });
  await writeFile(join(first, "SKILL.md"), "---\nname: first\ndescription: First.\n---\n");
  await writeFile(join(second, "SKILL.md"), "---\nname: second\ndescription: Second.\n---\n");

  await assert.rejects(
    new DroidSdkSessionFactory(fakeSdk().loader).create(input({ skills: { paths: [join(root, "missing")] } })),
    (error) => error.code === "droid_skill_invalid",
  );
  await assert.rejects(
    new DroidSdkSessionFactory(fakeSdk().loader).create(input({ skills: { paths: [first, second] } })),
    (error) => error.code === "droid_skill_collision",
  );
  await assert.rejects(
    new DroidSdkSessionFactory(fakeSdk().loader).create(input({ skills: { extensions: {} } })),
    (error) => error.code === "droid_skill_extensions_unsupported",
  );

  const disabled = fakeSdk({
    skillReport: (created) => ({
      skills: [{
        name: "first",
        filePath: join(created.env.HOME, ".agents", "skills", "same", "SKILL.md"),
        enabled: false,
      }],
    }),
  });
  await assert.rejects(
    new DroidSdkSessionFactory(disabled.loader).create(input({ tools: undefined, skills: { paths: [first] } })),
    (error) => error.code === "droid_skill_load_failed",
  );
  assert.equal(disabled.calls.close, 1);
});

test("normalizes interrupted and failed terminal results", async () => {
  const sdk = fakeSdk({ results: [
    result({ success: false, subtype: "interrupted", interrupted: true, text: "stopped" }),
    result({
      success: false,
      subtype: "error_during_execution",
      interrupted: false,
      text: "",
      error: { message: "provider failed" },
    }),
  ] });
  const handle = await new DroidSdkSessionFactory(sdk.loader).create(input({ tools: undefined }));
  assert.equal((await handle.prompt("one")).status, "interrupted");
  assert.deepEqual(await handle.prompt("two"), {
    status: "failed",
    text: "",
    errorMessage: "provider failed",
    usage: {
      input_tokens: 4,
      cached_input_tokens: 2,
      output_tokens: 3,
      extensions: {
        "nemo_fabric.detailed_token_usage.v1": {
          input_tokens: 4,
          cached_input_tokens: 2,
          cache_write_input_tokens: 1,
          output_tokens: 3,
          output_tokens_include_reasoning: true,
        },
        thinking_tokens: 5,
        factory_credits: 0.25,
      },
    },
    extensions: {
      session_id: "droid-session-1",
      finish_reason: "error_during_execution",
      duration_ms: 12,
      turn_count: 1,
    },
  });
  await handle.stop();
});

async function usageForModel(t, model, tokenUsage, customModel) {
  const configured = input({ tools: undefined });
  configured.config.models.default.model = model;
  if (customModel !== undefined) {
    const home = await mkdtemp(join(tmpdir(), "droid-usage-home-"));
    t.after(() => rm(home, { recursive: true, force: true }));
    await mkdir(join(home, ".factory"), { recursive: true });
    await writeFile(join(home, ".factory", "settings.json"), JSON.stringify({ customModels: [customModel] }));
    configured.runtimeContext.environment.env.FACTORY_HOME_OVERRIDE = home;
  }
  const sdk = fakeSdk({ results: [result({ tokenUsage: { thinkingTokens: 0, ...tokenUsage } })] });
  const handle = await new DroidSdkSessionFactory(sdk.loader, 10_000, {}).create(configured);
  try {
    return (await handle.prompt("one")).usage;
  } finally {
    await handle.stop();
  }
}

function inclusiveUsage({ input, cached, written, output }) {
  const cache = {
    ...(cached === undefined ? {} : { cached_input_tokens: cached }),
  };
  return {
    input_tokens: input,
    ...cache,
    input_tokens_include_cache: true,
    output_tokens: output,
    total_tokens: input + output,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        input_tokens: input,
        ...cache,
        ...(written === undefined ? {} : { cache_write_input_tokens: written }),
        input_tokens_include_cache: true,
        output_tokens: output,
        output_tokens_include_reasoning: true,
      },
    },
  };
}

const chatModel = {
  model: "nvidia/nemotron-3-super-120b-a12b",
  baseUrl: "https://integrate.api.nvidia.com/v1",
  provider: "generic-chat-completion-api",
};

test("normalizes input to include cache on each established Droid provider path", async (t) => {
  // Gemini: Droid persists promptTokenCount, which already includes the cache reads.
  assert.deepEqual(
    await usageForModel(t, "gemini-3-pro-preview", { inputTokens: 100, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 80 }),
    inclusiveUsage({ input: 100, cached: 80, output: 5 }),
  );
  assert.deepEqual(
    await usageForModel(t, "gemini-3-pro-preview", { inputTokens: 100, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 0 }),
    inclusiveUsage({ input: 100, cached: 0, output: 5 }),
  );
  // Anthropic Messages: input excludes cache reads and writes.
  assert.deepEqual(
    await usageForModel(t, "claude-sonnet-4-5-20250929", { inputTokens: 10, outputTokens: 5, cacheCreationTokens: 20, cacheReadTokens: 80 }),
    inclusiveUsage({ input: 110, cached: 80, written: 20, output: 5 }),
  );
  // OpenAI Responses: Droid subtracts the cache reads from input_tokens.
  assert.deepEqual(
    await usageForModel(t, "gpt-5.1", { inputTokens: 20, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 80 }),
    inclusiveUsage({ input: 100, cached: 80, output: 5 }),
  );
  // Factory-hosted model on the Chat Completions handler: Droid subtracts the cache reads.
  assert.deepEqual(
    await usageForModel(t, "glm-5", { inputTokens: 20, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 80 }),
    inclusiveUsage({ input: 100, cached: 80, output: 5 }),
  );
  // Custom Chat Completions model: Droid subtracts the cache reads.
  assert.deepEqual(
    await usageForModel(t, "custom:nvidia/nemotron-3-super-120b-a12b", { inputTokens: 20, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 80 }, chatModel),
    inclusiveUsage({ input: 100, cached: 80, output: 5 }),
  );
  // Custom Bedrock Converse model: input excludes cache reads and writes.
  assert.deepEqual(
    await usageForModel(
      t,
      "custom:anthropic.claude-sonnet-4-5",
      { inputTokens: 10, outputTokens: 5, cacheCreationTokens: 20, cacheReadTokens: 80 },
      { model: "anthropic.claude-sonnet-4-5", provider: "bedrock-converse" },
    ),
    inclusiveUsage({ input: 110, cached: 80, written: 20, output: 5 }),
  );
});

async function usageForRun(model, turns) {
  const configured = input({ tools: undefined });
  configured.config.models.default.model = model;
  const sdk = fakeSdk({ results: turns.map((tokenUsage) => result({ tokenUsage: { thinkingTokens: 0, ...tokenUsage } })) });
  const handle = await new DroidSdkSessionFactory(sdk.loader, 10_000, {}).create(configured);
  try {
    const usage = [];
    for (const [index] of turns.entries()) usage.push((await handle.prompt(`turn ${index}`)).usage);
    return usage;
  } finally {
    await handle.stop();
  }
}

test("keeps every count known across a normal cached Claude and Gemini run", async () => {
  // First turn writes the prompt cache, later turns read it.
  assert.deepEqual(
    await usageForRun("claude-sonnet-4-5-20250929", [
      { inputTokens: 5_000, outputTokens: 200, cacheCreationTokens: 4_000, cacheReadTokens: 0 },
      { inputTokens: 300, outputTokens: 150, cacheCreationTokens: 500, cacheReadTokens: 9_000 },
    ]),
    [
      inclusiveUsage({ input: 9_000, cached: 0, written: 4_000, output: 200 }),
      inclusiveUsage({ input: 9_800, cached: 9_000, written: 500, output: 150 }),
    ],
  );
  assert.deepEqual(
    await usageForRun("gemini-3-pro-preview", [
      { inputTokens: 6_000, outputTokens: 120, cacheCreationTokens: 0, cacheReadTokens: 0 },
      { inputTokens: 6_400, outputTokens: 90, cacheCreationTokens: 0, cacheReadTokens: 5_800 },
    ]),
    [
      inclusiveUsage({ input: 6_000, cached: 0, output: 120 }),
      inclusiveUsage({ input: 6_400, cached: 5_800, output: 90 }),
    ],
  );
});

async function usageForSession(t, { model, droidDefinition, turns, beforeTurn, ...sdkOptions }) {
  const root = await mkdtemp(join(tmpdir(), "droid-usage-session-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const home = join(root, "home");
  const workspace = join(root, "workspace");
  await mkdir(join(home, ".factory"), { recursive: true });
  await mkdir(workspace, { recursive: true });
  if (droidDefinition !== undefined) {
    await mkdir(join(workspace, ".factory", "droids"), { recursive: true });
    await writeFile(join(workspace, ".factory", "droids", "reviewer.md"), droidDefinition);
  }
  const configured = input({ tools: undefined });
  configured.config.models.default.model = model;
  configured.runtimeContext.environment.workspace = workspace;
  configured.runtimeContext.environment.env.FACTORY_HOME_OVERRIDE = home;
  const sdk = fakeSdk({
    ...sdkOptions,
    results: turns.map((tokenUsage) => result({ tokenUsage: { thinkingTokens: 0, ...tokenUsage } })),
  });
  const handle = await new DroidSdkSessionFactory(sdk.loader, 10_000, {}).create(configured);
  try {
    const usage = [];
    for (const [index] of turns.entries()) {
      beforeTurn?.(index, sdk.session);
      usage.push((await handle.prompt(`turn ${index}`)).usage);
    }
    return usage;
  } finally {
    await handle.stop();
  }
}

function reportedUsage({ input, cached, written, output }) {
  return {
    input_tokens: input,
    cached_input_tokens: cached,
    output_tokens: output,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        input_tokens: input,
        cached_input_tokens: cached,
        ...(written === undefined ? {} : { cache_write_input_tokens: written }),
        output_tokens: output,
        output_tokens_include_reasoning: true,
      },
    },
  };
}

// Droid sums a turn's parent and child usage: parent Claude 10 uncached plus 20
// cached, child Gemini 100 inclusive with 80 cached, outputs 3 and 5.
const MIXED_TURN = { inputTokens: 110, outputTokens: 8, cacheCreationTokens: 0, cacheReadTokens: 100 };
const GEMINI_CHILD = { inputTokens: 100, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 80, thinkingTokens: 0 };

test("keeps the reported counters of a turn with subagent usage", async (t) => {
  // Droid definitions can name a child model in YAML forms a simple scan misses;
  // the adapter does not read them and keeps every turn with child usage raw.
  for (const droidDefinition of [
    '---\nname: worker\ndescription: Worker\n"model": gemini-3-pro-preview\n---\nWork.\n',
    "---\n  name: worker\n  description: Worker\n  model: gemini-3-pro-preview\n---\nWork.\n",
  ]) {
    assert.deepEqual(
      await usageForSession(t, {
        model: "claude-sonnet-4-5-20250929",
        droidDefinition,
        turns: [MIXED_TURN],
        childUsage: [GEMINI_CHILD],
      }),
      [reportedUsage({ input: 110, cached: 100, output: 8 })],
    );
  }
  // A subagent on the session's own model is kept raw too: the result does not
  // say which model the child used.
  assert.deepEqual(
    await usageForSession(t, {
      model: "claude-sonnet-4-5-20250929",
      turns: [{ inputTokens: 40, outputTokens: 8, cacheCreationTokens: 0, cacheReadTokens: 60 }],
      childUsage: [{ inputTokens: 30, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 40, thinkingTokens: 0 }],
    }),
    [reportedUsage({ input: 40, cached: 60, output: 8 })],
  );
  // A turn without child usage normalizes as usual.
  assert.deepEqual(
    await usageForSession(t, {
      model: "claude-sonnet-4-5-20250929",
      turns: [{ inputTokens: 10, outputTokens: 3, cacheCreationTokens: 0, cacheReadTokens: 20 }],
      childUsage: [{ inputTokens: 0, outputTokens: 0, cacheCreationTokens: 0, cacheReadTokens: 0, thinkingTokens: 0 }],
    }),
    [inclusiveUsage({ input: 30, cached: 20, output: 3 })],
  );
});

test("follows the session's effective model and model changes", async (t) => {
  const googleTurn = { inputTokens: 100, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 80 };
  const claudeTurn = { inputTokens: 10, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 20 };
  // Droid reports Gemini as the session model although Claude was configured.
  assert.deepEqual(
    await usageForSession(t, { model: "claude-sonnet-4-5-20250929", modelId: "gemini-3-pro-preview", turns: [googleTurn] }),
    [inclusiveUsage({ input: 100, cached: 80, output: 5 })],
  );
  // The model changes between turns, then from Claude to Gemini and back within one turn.
  assert.deepEqual(
    await usageForSession(t, {
      model: "claude-sonnet-4-5-20250929",
      turns: [claudeTurn, googleTurn, MIXED_TURN],
      beforeTurn(index, session) {
        if (index === 1) session.changeModel("gemini-3-pro-preview");
        if (index === 2) session.changeModel("claude-sonnet-4-5-20250929");
      },
      duringTurn(index, session) {
        if (index !== 2) return;
        session.changeModel("gemini-3-pro-preview");
        session.changeModel("claude-sonnet-4-5-20250929");
      },
    }),
    [
      inclusiveUsage({ input: 30, cached: 20, output: 5 }),
      inclusiveUsage({ input: 100, cached: 80, output: 5 }),
      reportedUsage({ input: 110, cached: 100, output: 8 }),
    ],
  );
});

test("keeps the reported counters for Droid's router models", async (t) => {
  for (const model of ["auto", "auto-fast"]) {
    assert.deepEqual(
      await usageForSession(t, { model, turns: [{ inputTokens: 100, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 80 }] }),
      [reportedUsage({ input: 100, cached: 80, output: 5 })],
    );
  }
});

test("leaves inclusive input unknown when a 0 cache read may be an omission", async (t) => {
  const uncachedTurn = { inputTokens: 20, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 0 };
  // Factory-hosted endpoints are not shown to report cache reads on every
  // response, so Droid's 0 may hide cache hits that belong in the input.
  assert.deepEqual(await usageForModel(t, "glm-5", uncachedTurn), {
    input_tokens: 20,
    output_tokens: 5,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": { input_tokens: 20, output_tokens: 5, output_tokens_include_reasoning: true },
    },
  });
  // Anthropic reports cache reads on every response, so its 0 is a real 0.
  assert.deepEqual(
    await usageForModel(t, "claude-sonnet-4-5-20250929", uncachedTurn),
    inclusiveUsage({ input: 20, cached: 0, output: 5 }),
  );
  // A fallback turn derives no total either: its 0 cache reads are not established.
  assert.deepEqual(
    await usageForSession(t, {
      model: "glm-5",
      turns: [uncachedTurn],
      childUsage: [{ inputTokens: 8, outputTokens: 2, cacheCreationTokens: 0, cacheReadTokens: 0, thinkingTokens: 0 }],
    }),
    [{
      input_tokens: 20,
      output_tokens: 5,
      extensions: {
        "nemo_fabric.detailed_token_usage.v1": { input_tokens: 20, output_tokens: 5, output_tokens_include_reasoning: true },
      },
    }],
  );
});

test("keeps Droid input as reported when the provider convention is not established", async (t) => {
  const reported = (input, cached, written) => ({
    input_tokens: input,
    cached_input_tokens: cached,
    output_tokens: 5,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        input_tokens: input,
        cached_input_tokens: cached,
        ...(written === undefined ? {} : { cache_write_input_tokens: written }),
        output_tokens: 5,
        output_tokens_include_reasoning: true,
      },
    },
  });
  // Responses cache writes are a nonstandard field Droid does not subtract.
  assert.deepEqual(
    await usageForModel(t, "gpt-5.1", { inputTokens: 20, outputTokens: 5, cacheCreationTokens: 10, cacheReadTokens: 80 }),
    reported(20, 80, 10),
  );
  // Droid can route a Gemini model configured as Chat Completions to its Gemini handler.
  assert.deepEqual(
    await usageForModel(
      t,
      "custom:google/gemini-3-pro",
      { inputTokens: 100, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 80 },
      { ...chatModel, model: "google/gemini-3-pro" },
    ),
    reported(100, 80),
  );
  // Without a matching settings entry, the custom model's provider is unknown.
  assert.deepEqual(
    await usageForModel(t, "custom:nvidia/nemotron-3-super-120b-a12b", { inputTokens: 100, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 80 }),
    reported(100, 80),
  );
  // Droid selects the model for auto.
  assert.deepEqual(
    await usageForModel(t, "auto", { inputTokens: 100, outputTokens: 5, cacheCreationTokens: 0, cacheReadTokens: 80 }),
    reported(100, 80),
  );
});

test("leaves token counts that Droid filled with zero unknown", async () => {
  const zeroUsage = { inputTokens: 0, outputTokens: 0, cacheCreationTokens: 0, cacheReadTokens: 0, thinkingTokens: 0 };
  const sdk = fakeSdk({ results: [
    result({ tokenUsage: { ...zeroUsage, inputTokens: 9 } }),
    result({ tokenUsage: { ...zeroUsage, outputTokens: 2, cacheReadTokens: 6 } }),
    result({ tokenUsage: { ...zeroUsage, thinkingTokens: 3, factoryCredits: 0.5 } }),
    result({ tokenUsage: { ...zeroUsage, factoryCredits: 0 } }),
  ] });
  const handle = await new DroidSdkSessionFactory(sdk.loader).create(input({ tools: undefined }));
  assert.deepEqual((await handle.prompt("one")).usage, {
    input_tokens: 9,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": { input_tokens: 9 },
    },
  });
  assert.deepEqual((await handle.prompt("two")).usage, {
    cached_input_tokens: 6,
    output_tokens: 2,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        cached_input_tokens: 6,
        output_tokens: 2,
        output_tokens_include_reasoning: true,
      },
    },
  });
  assert.deepEqual((await handle.prompt("three")).usage, {
    extensions: { thinking_tokens: 3, factory_credits: 0.5 },
  });
  assert.equal((await handle.prompt("four")).usage, undefined);
  await handle.stop();
});

test("rejects unknown tools and cleans up after partial startup", async () => {
  const sdk = fakeSdk();
  await assert.rejects(
    new DroidSdkSessionFactory(sdk.loader).create(input({ tools: { enabled: ["Invented"] } })),
    (error) => error.code === "droid_tool_unknown",
  );
  assert.equal(sdk.calls.close, 1);

  const failed = fakeSdk({ toolError: new Error("transport closed") });
  await assert.rejects(
    new DroidSdkSessionFactory(failed.loader).create(input()),
    (error) => error.code === "droid_start_failed",
  );
  assert.equal(failed.calls.close, 1);
});

test("rejects unsupported or incompatible normalized MCP fields", async () => {
  const cases = [
    [{ transport: "stdio", url: "node", custom_headers: { X: "value" } }, "droid_mcp_headers_unsupported"],
    [{ transport: "streamable-http", url: "https://example.test", args: ["no"] }, "droid_mcp_process_fields_unsupported"],
    [{ transport: "streamable-http", url: "http://example.test" }, "droid_mcp_url_insecure"],
    [{ transport: "stdio", url: "node", allowed_tools: ["one"] }, "droid_mcp_allowlist_unsupported"],
    [{ transport: "stdio", url: "node", blocked_tools: ["one"] }, "droid_mcp_blocklist_unsupported"],
    [{ transport: "stdio", url: "node", authentication: { type: "none" } }, "droid_mcp_authentication_unsupported"],
    [{ transport: "stdio", url: "node", extensions: { custom: true } }, "droid_mcp_extensions_unsupported"],
  ];
  for (const [server, code] of cases) {
    await assert.rejects(
      new DroidSdkSessionFactory(fakeSdk().loader).create(input({ mcp: { servers: { test: server } } })),
      (error) => error.code === code,
    );
  }
});

test("rejects invalid startup inputs and missing or malformed terminal results", async () => {
  const missing = input();
  missing.runtimeContext.environment.env = {};
  await assert.rejects(
    new DroidSdkSessionFactory(fakeSdk().loader).create(missing),
    (error) => error.code === "droid_credential_missing",
  );

  const provider = input();
  provider.config.models.default.provider = "openai";
  await assert.rejects(
    new DroidSdkSessionFactory(fakeSdk().loader).create(provider),
    (error) => error.code === "droid_provider_unsupported",
  );

  const mcp = fakeSdk({ mcpTools: [] });
  await assert.rejects(
    new DroidSdkSessionFactory(mcp.loader, 1).create(input({ mcp: { servers: { test: { transport: "stdio", url: "missing" } } } })),
    (error) => error.code === "droid_mcp_load_failed",
  );
  assert.equal(mcp.calls.close, 1);

  const missingTerminal = fakeSdk({ results: [] });
  const missingTerminalHandle = await new DroidSdkSessionFactory(missingTerminal.loader).create(
    input({ tools: undefined }),
  );
  await assert.rejects(
    missingTerminalHandle.prompt("one"),
    (error) => error.code === "droid_session_failed" && error.retryable,
  );
  await missingTerminalHandle.stop();

  const malformed = fakeSdk({ results: [{ type: "result", text: "not terminal" }] });
  const handle = await new DroidSdkSessionFactory(malformed.loader).create(input({ tools: undefined }));
  await assert.rejects(handle.prompt("one"), (error) => error.code === "droid_malformed_result");
  await handle.stop();
});

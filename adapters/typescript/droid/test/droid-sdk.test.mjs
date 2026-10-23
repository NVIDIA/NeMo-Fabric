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
        turnId: options.turnIds?.[index] ?? `turn-${index}`,
        tokenUsage: options.omitCompletionUsage ? undefined : results[0]?.tokenUsage,
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
  assert.equal(first.usage.input_tokens, undefined);
  assert.equal(first.usage.output_tokens, undefined);
  const observed = first.usage.extensions["nemo_fabric.usage_accounting.v1"].observed;
  assert.equal(observed.input_tokens, 4);
  assert.equal(observed.output_tokens, 3);
  assert.equal(observed.cached_input_tokens, 2);
  assert.equal(first.usage.extensions.thinking_tokens, 5);
  assert.equal(first.usage.extensions.factory_credits, 0.25);
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
  const failed = await handle.prompt("two");
  assert.equal(failed.status, "failed");
  assert.equal(failed.text, "");
  assert.equal(failed.errorMessage, "provider failed");
  assert.equal(failed.usage.extensions["nemo_fabric.usage_accounting.v1"].observed.input_tokens, 4);
  assert.equal(failed.usage.input_tokens, undefined);
  assert.equal(failed.extensions.finish_reason, "error_during_execution");
  await handle.stop();
});

async function usageFor(model, turns, options = {}) {
  const configured = input({ tools: undefined });
  configured.config.models.default.model = model;
  const sdk = fakeSdk({ ...options, results: turns.map(tokenUsage => result({ tokenUsage: {
    inputTokens: 0, outputTokens: 0, cacheCreationTokens: 0, cacheReadTokens: 0, thinkingTokens: 0, ...tokenUsage,
  } })) });
  const handle = await new DroidSdkSessionFactory(sdk.loader).create(configured);
  try {
    const usages = [];
    for (let index = 0; index < turns.length; index++) usages.push((await handle.prompt(`turn ${index}`)).usage);
    return usages;
  } finally { await handle.stop(); }
}

const accounting = usage => usage.extensions["nemo_fabric.usage_accounting.v1"];

test("terminal aggregate omissions cannot become complete positive totals", async () => {
  // The same terminal40/8 can represent one fully measured call or an input-
  // only call plus an output-only call. Public terminal data cannot decide.
  const [usage] = await usageFor("gemini-3-pro-preview", [{ inputTokens: 40, outputTokens: 8, cacheReadTokens: 6 }]);
  assert.equal(usage.input_tokens, undefined);
  assert.equal(usage.output_tokens, undefined);
  assert.equal(usage.total_tokens, undefined);
  assert.equal(usage.cost_usd, undefined);
  assert.equal(accounting(usage).observed.input_tokens, 40);
  assert.equal(accounting(usage).observed.input_tokens_include_cache, true);
  assert.equal(accounting(usage).observed.output_tokens, 8);
  assert.equal(accounting(usage).observed.cached_input_tokens, 6);
  assert.ok(accounting(usage).gaps.some(gap => gap.reason === "scope_unverified"));
});

test("literal provider conventions remain raw for core normalization", async () => {
  const [claude] = await usageFor("claude-sonnet-4-5-20250929", [{ inputTokens: 10, outputTokens: 5, cacheReadTokens: 80, cacheCreationTokens: 20 }]);
  assert.equal(accounting(claude).observed.input_tokens, 10);
  assert.equal(accounting(claude).observed.input_tokens_include_cache, false);
  const [googleTotal] = await usageFor("gemini-3-pro-preview", [{ inputTokens: 100, outputTokens: 5, cacheReadTokens: 80, cacheCreationTokens: 10 }]);
  // Its source promptTokenCount is a total; inclusion is not inferred from a
  // zero-filled cache-write field. No write0 is synthesized.
  assert.equal(accounting(googleTotal).observed.input_tokens, 100);
  assert.equal(accounting(googleTotal).observed.input_tokens_include_cache, true);
  assert.equal(accounting(googleTotal).observed.cache_write_input_tokens, 10);
  const [responses] = await usageFor("gpt-5.1", [{ inputTokens: 20, outputTokens: 5, cacheReadTokens: 80 }]);
  assert.equal(accounting(responses).observed.input_tokens, 20);
  assert.equal(accounting(responses).observed.input_tokens_include_cache, undefined);
  const [router] = await usageFor("auto", [{ inputTokens: 100, outputTokens: 5, cacheReadTokens: 80 }]);
  assert.equal(accounting(router).observed.input_tokens_include_cache, undefined);
  assert.equal(accounting(router).models, undefined);
});

test("zero-filled and independently omitted dimensions remain absent", async () => {
  const usages = await usageFor("gemini-3-pro-preview", [
    { inputTokens: 9 }, { outputTokens: 2, cacheReadTokens: 6 }, {},
  ]);
  assert.equal(accounting(usages[0]).observed.input_tokens, 9);
  assert.equal(accounting(usages[0]).observed.output_tokens, undefined);
  assert.equal(accounting(usages[1]).observed.input_tokens, undefined);
  assert.equal(accounting(usages[1]).observed.cached_input_tokens, 6);
  assert.equal(accounting(usages[1]).observed.output_tokens, 2);
  assert.deepEqual(accounting(usages[2]).observed, {});
});

test("matching terminal turns never carry a preceding turn's counts forward", async () => {
  const usages = await usageFor("gemini-3-pro-preview", [
    { inputTokens: 40, outputTokens: 8 }, { inputTokens: 4, outputTokens: 2 },
  ]);
  assert.equal(accounting(usages[0]).observed.input_tokens, 40);
  assert.equal(accounting(usages[1]).observed.input_tokens, 4);
});

test("descendant-inclusive totals count once without allocating children to the parent model", async () => {
  const [usage] = await usageFor("claude-sonnet-4-5-20250929", [{ inputTokens: 110, outputTokens: 15, cacheReadTokens: 80 }], {
    childUsage: [{ inputTokens: 100, outputTokens: 10, cacheReadTokens: 0, cacheCreationTokens: 0, thinkingTokens: 0 }],
  });
  assert.equal(accounting(usage).observed.input_tokens, 110);
  assert.equal(accounting(usage).models, undefined);
  assert.equal(accounting(usage).observed.input_tokens_include_cache, undefined);
});

test("model changes prevent assigning mixed counters to one model", async () => {
  const [usage] = await usageFor("claude-sonnet-4-5-20250929", [{ inputTokens: 110, outputTokens: 15, cacheReadTokens: 80 }], {
    duringTurn(_index, session) { session.changeModel("gemini-3-pro-preview"); },
  });
  assert.equal(accounting(usage).observed.input_tokens, 110);
  assert.equal(accounting(usage).models, undefined);
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


test("missing raw completion usage cannot inherit the SDK's session-lifetime fallback", async () => {
  const [usage] = await usageFor("gemini-3-pro-preview", [{ inputTokens: 1000, outputTokens: 200 }], { omitCompletionUsage: true });
  assert.deepEqual(accounting(usage).observed, {});
  assert.equal(usage.input_tokens, undefined);
  assert.ok(accounting(usage).gaps.some(gap => gap.reason === "missing_usage"));
});

test("a duplicate completion cannot move a preceding turn's usage into the next", async () => {
  const usages = await usageFor("gemini-3-pro-preview", [
    { inputTokens: 40, outputTokens: 8 }, { inputTokens: 40, outputTokens: 8 },
  ], { turnIds: ["same-turn", "same-turn"] });
  assert.equal(accounting(usages[0]).observed.input_tokens, 40);
  assert.deepEqual(accounting(usages[1]).observed, {});
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { access, mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";

import { ClineSdkSessionFactory } from "../dist/cline-sdk.js";

function input(overrides = {}) {
  return {
    agentName: "cline-test",
    baseDir: "/tmp",
    config: {
      models: {
        default: {
          provider: "openai",
          model: "gpt-test",
          api_key_env: "TEST_API_KEY",
          base_url: "https://example.test/v1",
        },
      },
      instructions: { system: { content: "NeMo Fabric system prompt", mode: "replace" } },
      tools: { enabled: ["read_files"], blocked: ["run_commands"] },
      ...overrides,
    },
    runtimeContext: {
      artifacts: {},
      environment: {
        control_location: "external_control",
        env: { TEST_API_KEY: "secret" },
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

function fakeSdk(options = {}) {
  const calls = { create: [], start: [], send: [], stop: [], dispose: [], prompts: [], plugins: [] };
  const results = options.results ?? [
    { finishReason: "completed", text: "first", usage: { inputTokens: 2, outputTokens: 3, totalCost: 0.5 } },
    { finishReason: "completed", text: "second", usage: { inputTokens: 4, outputTokens: 1 } },
  ];
  const core = {
    async start(value) {
      calls.start.push(value);
      return { sessionId: "cline-session-1", result: results.shift() };
    },
    async send(value) {
      calls.send.push(value);
      if (options.sendError !== undefined) throw options.sendError;
      return results.shift();
    },
    async stop(value) { calls.stop.push(value); },
    async dispose(value) { calls.dispose.push(value); },
  };
  return {
    calls,
    loader: async () => ({
      ALL_DEFAULT_TOOL_NAMES: ["read_files", "search_codebase", "run_commands"],
      ClineCore: { async create(value) {
        calls.create.push(value);
        await options.onCreate?.(process.env.CLINE_DATA_DIR);
        return core;
      } },
      getClineDefaultSystemPrompt(value) { calls.prompts.push(value); return `system:${value.overridePrompt ?? "default"}`; },
      async loadAgentPluginPackages(value) {
        calls.plugins.push(value);
        return { skills: [], mcpServers: [], diagnostics: [] };
      },
    }),
  };
}

test("maps model, endpoint, credential, system instruction, and built-in tool policy", async () => {
  const sdk = fakeSdk();
  const workspace = await realpath("/tmp");
  const handle = await new ClineSdkSessionFactory(sdk.loader).create(input());
  const first = await handle.prompt("one");
  const second = await handle.prompt("two");

  assert.deepEqual(sdk.calls.create, [{
    clientName: "nemo-fabric",
    backendMode: "local",
    toolPolicies: {
      read_files: { enabled: true, autoApprove: true },
      search_codebase: { enabled: false, autoApprove: true },
      run_commands: { enabled: false, autoApprove: true },
    },
  }]);
  assert.deepEqual(sdk.calls.prompts, []);
  assert.deepEqual(sdk.calls.start[0], {
    config: {
      providerId: "openai",
      modelId: "gpt-test",
      apiKey: "secret",
      baseUrl: "https://example.test/v1",
      cwd: workspace,
      workspaceRoot: workspace,
      mode: "act",
      enableTools: true,
      enableSpawnAgent: false,
      enableAgentTeams: false,
      systemPrompt: "NeMo Fabric system prompt",
    },
    interactive: true,
    source: "sdk",
    toolPolicies: {
      read_files: { enabled: true, autoApprove: true },
      search_codebase: { enabled: false, autoApprove: true },
      run_commands: { enabled: false, autoApprove: true },
    },
    localRuntime: { configExtensions: [] },
    prompt: "one",
  });
  assert.deepEqual(sdk.calls.send, [{ sessionId: "cline-session-1", prompt: "two" }]);
  assert.deepEqual(first.usage, {
    input_tokens: 2,
    input_tokens_include_cache: true,
    output_tokens: 3,
    total_tokens: 5,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        input_tokens: 2,
        input_tokens_include_cache: true,
        output_tokens: 3,
        output_tokens_include_reasoning: true,
      },
    },
  });
  assert.deepEqual(second.usage, {
    input_tokens: 4,
    input_tokens_include_cache: true,
    output_tokens: 1,
    total_tokens: 5,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        input_tokens: 4,
        input_tokens_include_cache: true,
        output_tokens: 1,
        output_tokens_include_reasoning: true,
      },
    },
  });
  await handle.stop();
});

test("uses Cline's default prompt only when NeMo Fabric supplies no system instruction", async () => {
  const sdk = fakeSdk();
  const configured = input();
  delete configured.config.instructions;
  const workspace = await realpath("/tmp");
  const handle = await new ClineSdkSessionFactory(sdk.loader).create(configured);

  await handle.prompt("one");

  assert.deepEqual(sdk.calls.prompts, [{
    workspaceRoot: workspace,
    providerId: "openai",
  }]);
  assert.equal(sdk.calls.start[0].config.systemPrompt, "system:default");
  await handle.stop();
});

test("stops the retained Cline session and disposes its core", async () => {
  const sdk = fakeSdk();
  const handle = await new ClineSdkSessionFactory(sdk.loader).create(input());
  await handle.prompt("one");
  await handle.stop();
  await handle.stop();
  assert.deepEqual(sdk.calls.stop, ["cline-session-1"]);
  assert.deepEqual(sdk.calls.dispose, ["NeMo Fabric runtime stopped"]);
});

test("disposes Cline even when no invocation started a session", async () => {
  const sdk = fakeSdk();
  const handle = await new ClineSdkSessionFactory(sdk.loader).create(input());
  await handle.stop();
  assert.deepEqual(sdk.calls.stop, []);
  assert.deepEqual(sdk.calls.dispose, ["NeMo Fabric runtime stopped"]);
});

test("cleans up the core, runtime files, and environment after partial startup", async () => {
  const sdk = fakeSdk();
  const original = process.env.CLINE_DATA_DIR;
  const module = await sdk.loader();
  let stagedDataDir;
  const loader = async () => ({
    ...module,
    ClineCore: {
      async create(value) {
        stagedDataDir = process.env.CLINE_DATA_DIR;
        return module.ClineCore.create(value);
      },
    },
    getClineDefaultSystemPrompt() {
      throw new Error("prompt setup failed");
    },
  });
  const configured = input();
  delete configured.config.instructions;

  await assert.rejects(
    new ClineSdkSessionFactory(loader).create(configured),
    (error) => error.code === "cline_start_failed",
  );

  assert.deepEqual(sdk.calls.dispose, ["NeMo Fabric startup failed"]);
  assert.equal(process.env.CLINE_DATA_DIR, original);
  assert.ok(stagedDataDir);
  await assert.rejects(access(dirname(stagedDataDir)));
});

test("serializes core creation while scoping each Cline data directory", async () => {
  const original = process.env.CLINE_DATA_DIR;
  const observedDataDirs = [];
  let activeCreates = 0;
  let maximumActiveCreates = 0;
  const onCreate = async (dataDir) => {
    observedDataDirs.push(dataDir);
    activeCreates += 1;
    maximumActiveCreates = Math.max(maximumActiveCreates, activeCreates);
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 10));
    activeCreates -= 1;
  };
  const firstSdk = fakeSdk({ onCreate });
  const secondSdk = fakeSdk({ onCreate });

  const handles = await Promise.all([
    new ClineSdkSessionFactory(firstSdk.loader).create(input()),
    new ClineSdkSessionFactory(secondSdk.loader).create(input()),
  ]);
  try {
    assert.equal(maximumActiveCreates, 1);
    assert.equal(observedDataDirs.length, 2);
    assert.notEqual(observedDataDirs[0], observedDataDirs[1]);
    assert.ok(observedDataDirs.every((dataDir) => typeof dataDir === "string"));
    assert.equal(process.env.CLINE_DATA_DIR, original);
  } finally {
    await Promise.all(handles.map((handle) => handle.stop()));
  }
});

test("reports a Cline turn timeout and still permits cleanup", async () => {
  const sdk = fakeSdk({ sendError: new Error("provider request timed out") });
  const handle = await new ClineSdkSessionFactory(sdk.loader).create(input());
  await handle.prompt("one");

  await assert.rejects(
    handle.prompt("two"),
    (error) => error.code === "cline_session_failed" && error.retryable && error.message.includes("timed out"),
  );
  await handle.stop();

  assert.deepEqual(sdk.calls.stop, ["cline-session-1"]);
  assert.deepEqual(sdk.calls.dispose, ["NeMo Fabric runtime stopped"]);
});

test("rejects unsupported instruction mode, unknown tools, and missing credentials", async () => {
  const sdk = fakeSdk();
  await assert.rejects(
    new ClineSdkSessionFactory(sdk.loader).create(input({ instructions: { system: { content: "append", mode: "append" } } })),
    (error) => error.code === "unsupported_system_instruction_mode",
  );
  await assert.rejects(
    new ClineSdkSessionFactory(sdk.loader).create(input({ tools: { enabled: ["invented"] } })),
    (error) => error.code === "cline_tool_unknown",
  );
  const missing = input();
  missing.runtimeContext.environment.env = {};
  await assert.rejects(
    new ClineSdkSessionFactory(sdk.loader).create(missing),
    (error) => error.code === "cline_credential_missing",
  );
});

test("rejects malformed Cline results", async () => {
  const sdk = fakeSdk({ results: [{ finishReason: "completed", text: { invalid: true } }] });
  const handle = await new ClineSdkSessionFactory(sdk.loader).create(input());
  await assert.rejects(handle.prompt("one"), (error) => error.code === "cline_malformed_result");
  await handle.stop();

  const invalidUsage = fakeSdk({
    results: [{ finishReason: "completed", text: "done", usage: { inputTokens: -1, outputTokens: 1 } }],
  });
  const usageHandle = await new ClineSdkSessionFactory(invalidUsage.loader).create(input());
  await assert.rejects(usageHandle.prompt("one"), (error) => error.code === "cline_malformed_result");
  await usageHandle.stop();
});

test("maps Cline terminal finish reasons", async () => {
  const sdk = fakeSdk({ results: [
    { finishReason: "aborted", text: "stopped", usage: { inputTokens: 1, outputTokens: 0 } },
    { finishReason: "error", text: "provider failed", usage: { inputTokens: 2, outputTokens: 1 } },
    { finishReason: "max_iterations", text: "iteration limit reached" },
  ] });
  const handle = await new ClineSdkSessionFactory(sdk.loader).create(input());
  assert.deepEqual(await handle.prompt("one"), {
    status: "aborted",
    text: "stopped",
    usage: {
      input_tokens: 1,
      input_tokens_include_cache: true,
      extensions: {
        "nemo_fabric.detailed_token_usage.v1": { input_tokens: 1, input_tokens_include_cache: true },
      },
    },
    extensions: { session_id: "cline-session-1", finish_reason: "aborted" },
  });
  assert.deepEqual(await handle.prompt("two"), {
    status: "failed",
    text: "provider failed",
    errorMessage: "provider failed",
    usage: {
      input_tokens: 2,
      input_tokens_include_cache: true,
      output_tokens: 1,
      total_tokens: 3,
      extensions: {
        "nemo_fabric.detailed_token_usage.v1": {
          input_tokens: 2,
          input_tokens_include_cache: true,
          output_tokens: 1,
          output_tokens_include_reasoning: true,
        },
      },
    },
    extensions: { session_id: "cline-session-1", finish_reason: "error" },
  });
  assert.deepEqual(await handle.prompt("three"), {
    status: "failed",
    text: "iteration limit reached",
    errorMessage: "iteration limit reached",
    extensions: { session_id: "cline-session-1", finish_reason: "max_iterations" },
  });
  await handle.stop();
});

test("normalizes Cline usage and result metadata", async () => {
  const sdk = fakeSdk({ results: [{
    finishReason: "completed",
    text: "done",
    iterations: 2,
    model: { id: "model-a", provider: "provider-a", ignored: true },
    // Cline counts input as the AI SDK does: cache reads and writes included.
    usage: {
      inputTokens: 1200,
      outputTokens: 50,
      cacheReadTokens: 900,
      cacheWriteTokens: 200,
      totalCost: 0.25,
    },
  }] });
  const handle = await new ClineSdkSessionFactory(sdk.loader).create(input());

  assert.deepEqual(await handle.prompt("one"), {
    status: "completed",
    text: "done",
    usage: {
      input_tokens: 1200,
      cached_input_tokens: 900,
      input_tokens_include_cache: true,
      output_tokens: 50,
      total_tokens: 1250,
      extensions: {
        "nemo_fabric.detailed_token_usage.v1": {
          input_tokens: 1200,
          cached_input_tokens: 900,
          cache_write_input_tokens: 200,
          input_tokens_include_cache: true,
          output_tokens: 50,
          output_tokens_include_reasoning: true,
        },
      },
    },
    extensions: {
      session_id: "cline-session-1",
      finish_reason: "completed",
      iterations: 2,
      model: { id: "model-a", provider: "provider-a" },
    },
  });
  await handle.stop();

  // Cline leaves out a cache write count of 0.
  const readOnly = fakeSdk({ results: [{
    finishReason: "completed",
    text: "done",
    usage: { inputTokens: 1000, outputTokens: 40, cacheReadTokens: 800 },
  }] });
  const readOnlyHandle = await new ClineSdkSessionFactory(readOnly.loader).create(input());
  assert.deepEqual((await readOnlyHandle.prompt("one")).usage, {
    input_tokens: 1000,
    cached_input_tokens: 800,
    input_tokens_include_cache: true,
    output_tokens: 40,
    total_tokens: 1040,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        input_tokens: 1000,
        cached_input_tokens: 800,
        input_tokens_include_cache: true,
        output_tokens: 40,
        output_tokens_include_reasoning: true,
      },
    },
  });
  await readOnlyHandle.stop();

  // A provider without prompt caching: Cline leaves out both cache counts.
  const uncached = fakeSdk({ results: [{
    finishReason: "completed",
    text: "done",
    usage: { inputTokens: 700, outputTokens: 30 },
  }] });
  const uncachedHandle = await new ClineSdkSessionFactory(uncached.loader).create(input());
  assert.deepEqual((await uncachedHandle.prompt("one")).usage, {
    input_tokens: 700,
    input_tokens_include_cache: true,
    output_tokens: 30,
    total_tokens: 730,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        input_tokens: 700,
        input_tokens_include_cache: true,
        output_tokens: 30,
        output_tokens_include_reasoning: true,
      },
    },
  });
  await uncachedHandle.stop();
});

test("leaves Cline token counts of 0 unknown", async () => {
  const sdk = fakeSdk({ results: [
    // Cline records a count the provider omitted as 0.
    { finishReason: "completed", text: "done", iterations: 1, usage: { inputTokens: 0, outputTokens: 0, totalCost: 0 } },
    { finishReason: "completed", text: "done", iterations: 1, usage: { inputTokens: 12, outputTokens: 0 } },
    // Cline's result for an aborted interactive turn reports all zeros.
    {
      finishReason: "aborted",
      text: "",
      iterations: 0,
      usage: { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, totalCost: 0 },
    },
  ] });
  const handle = await new ClineSdkSessionFactory(sdk.loader).create(input());

  assert.equal((await handle.prompt("one")).usage, undefined);
  assert.deepEqual((await handle.prompt("two")).usage, {
    input_tokens: 12,
    input_tokens_include_cache: true,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": { input_tokens: 12, input_tokens_include_cache: true },
    },
  });
  assert.equal((await handle.prompt("three")).usage, undefined);
  await handle.stop();
});

test("keeps Cline cache counts when the input total is unknown", async () => {
  const sdk = fakeSdk({ results: [
    // An OpenAI-compatible response can omit prompt_tokens and still report cached tokens.
    { finishReason: "completed", text: "done", usage: { inputTokens: 0, outputTokens: 0, cacheReadTokens: 900, cacheWriteTokens: 200 } },
    { finishReason: "completed", text: "done", usage: { inputTokens: 0, outputTokens: 50, cacheReadTokens: 900, cacheWriteTokens: 200 } },
  ] });
  const handle = await new ClineSdkSessionFactory(sdk.loader).create(input());

  assert.deepEqual((await handle.prompt("one")).usage, {
    cached_input_tokens: 900,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": { cached_input_tokens: 900, cache_write_input_tokens: 200 },
    },
  });
  assert.deepEqual((await handle.prompt("two")).usage, {
    cached_input_tokens: 900,
    output_tokens: 50,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        cached_input_tokens: 900,
        cache_write_input_tokens: 200,
        output_tokens: 50,
        output_tokens_include_reasoning: true,
      },
    },
  });
  await handle.stop();
});

test("leaves a Cline input total below its cache counts unknown", async () => {
  // One model call omitted prompt_tokens but reported 900 cached tokens; another reported 150 input.
  const sdk = fakeSdk({ results: [
    { finishReason: "completed", text: "done", usage: { inputTokens: 150, outputTokens: 15, cacheReadTokens: 900 } },
    { finishReason: "completed", text: "done", usage: { inputTokens: 150, outputTokens: 15, cacheReadTokens: 100, cacheWriteTokens: 60 } },
  ] });
  const handle = await new ClineSdkSessionFactory(sdk.loader).create(input());

  assert.deepEqual((await handle.prompt("one")).usage, {
    cached_input_tokens: 900,
    output_tokens: 15,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        cached_input_tokens: 900,
        output_tokens: 15,
        output_tokens_include_reasoning: true,
      },
    },
  });
  assert.deepEqual((await handle.prompt("two")).usage, {
    cached_input_tokens: 100,
    output_tokens: 15,
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        cached_input_tokens: 100,
        cache_write_input_tokens: 60,
        output_tokens: 15,
        output_tokens_include_reasoning: true,
      },
    },
  });
  await handle.stop();
});

test("declares every usage extension field the adapter returns", async () => {
  const sdk = fakeSdk({ results: [{
    finishReason: "completed",
    text: "done",
    usage: { inputTokens: 30, outputTokens: 5, cacheReadTokens: 10, cacheWriteTokens: 20 },
  }] });
  const handle = await new ClineSdkSessionFactory(sdk.loader).create(input());
  const { extensions } = (await handle.prompt("one")).usage;
  await handle.stop();

  const schema = JSON.parse(await readFile(new URL("../cline.fabric-adapter.json", import.meta.url), "utf8"))
    .extension_schemas.usage;
  assert.equal(schema.additionalProperties, false);
  for (const [key, value] of Object.entries(extensions)) {
    const declared = schema.properties[key];
    assert.ok(declared, `undeclared usage extension ${key}`);
    assert.equal(declared.additionalProperties, false);
    for (const field of Object.keys(value)) {
      assert.ok(declared.properties[field], `undeclared field ${key}.${field}`);
    }
  }
});

test("stages NeMo Fabric skills and MCP servers as one runtime-scoped native Cline plugin", async () => {
  const root = await mkdtemp(join(tmpdir(), "fabric-cline-native-config-"));
  const skill = join(root, "skills", "review-code");
  try {
    await mkdir(skill, { recursive: true });
    await writeFile(
      join(skill, "SKILL.md"),
      "---\nname: review-code\ndescription: Review code carefully.\n---\n\n# Review code\n",
      "utf8",
    );
    const sdk = fakeSdk();
    const baseModule = await sdk.loader();
    sdk.loader = async () => ({
      ...baseModule,
      async loadAgentPluginPackages(value) {
        sdk.calls.plugins.push(value);
        return { skills: [{}], mcpServers: [{}, {}], diagnostics: [] };
      },
    });
    const configured = input({
      skills: { paths: ["skills/review-code"] },
      mcp: {
        servers: {
          local: {
            transport: "stdio",
            url: "node",
            args: ["server.js"],
            env: { MODE: "test" },
          },
          remote: {
            transport: "streamable-http",
            url: "https://mcp.example.test/rpc",
            custom_headers: { "X-Test": "value" },
          },
        },
      },
    });
    configured.baseDir = root;
    const handle = await new ClineSdkSessionFactory(sdk.loader).create(configured);
    await handle.prompt("one");

    const pluginRoot = sdk.calls.start[0].config.agentPluginPaths[0];
    const manifest = JSON.parse(await readFile(join(pluginRoot, "plugin.json"), "utf8"));
    const mcp = JSON.parse(await readFile(join(pluginRoot, "mcp.json"), "utf8"));
    assert.equal(manifest.name, "nvidia.fabric.cline");
    assert.equal(
      await readFile(join(pluginRoot, "skills", "review-code", "SKILL.md"), "utf8"),
      "---\nname: review-code\ndescription: Review code carefully.\n---\n\n# Review code\n",
    );
    assert.deepEqual(mcp.mcpServers, {
      local: { type: "stdio", command: "node", args: ["server.js"], env: { MODE: "test" } },
      remote: {
        type: "streamable-http",
        url: "https://mcp.example.test/rpc",
        headers: { "X-Test": "value" },
      },
    });
    await handle.stop();
    await assert.rejects(access(pluginRoot));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects unsupported MCP authentication and tool filters", async () => {
  const sdk = fakeSdk();
  await assert.rejects(
    new ClineSdkSessionFactory(sdk.loader).create(input({
      mcp: {
        servers: {
          remote: {
            transport: "streamable-http",
            url: "https://mcp.example.test/rpc",
            authentication: { type: "oauth2" },
          },
        },
      },
    })),
    (error) => error.code === "cline_mcp_authentication_unsupported",
  );
  await assert.rejects(
    new ClineSdkSessionFactory(sdk.loader).create(input({
      mcp: {
        servers: {
          local: {
            transport: "stdio",
            url: "node",
            allowed_tools: ["read"],
          },
        },
      },
    })),
    (error) => error.code === "cline_mcp_allowlist_unsupported",
  );
  await assert.rejects(
    new ClineSdkSessionFactory(sdk.loader).create(input({
      mcp: {
        servers: {
          local: {
            transport: "stdio",
            url: "node",
            blocked_tools: ["delete"],
          },
        },
      },
    })),
    (error) => error.code === "cline_mcp_blocklist_unsupported",
  );
});

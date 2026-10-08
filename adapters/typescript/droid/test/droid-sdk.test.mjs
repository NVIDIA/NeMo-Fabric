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
    create: [], stream: [], close: 0, listTools: 0, updateSettings: [], listMcpServers: 0, listSkills: 0,
  };
  const results = [...(options.results ?? [result(), result({ text: "second", turnCount: 2 })])];
  const session = {
    id: "droid-session-1",
    async *stream(prompt) {
      calls.stream.push(prompt);
      if (options.streamError) throw options.streamError;
      yield { type: "assistant", text: "intermediate" };
      yield results.shift();
    },
    async close() { calls.close += 1; },
    async listTools() {
      calls.listTools += 1;
      if (options.toolError) throw options.toolError;
      return options.tools ?? [{ id: "Read" }, { id: "Edit" }, { id: "Execute" }];
    },
    async updateSettings(value) { calls.updateSettings.push(value); },
    async listMcpServers() {
      calls.listMcpServers += 1;
      return typeof options.mcpReport === "function"
        ? options.mcpReport(calls.listMcpServers)
        : options.mcpReport ?? { servers: [] };
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
    loader: async () => ({
      async createSession(value) {
        calls.create.push(value);
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
    output_tokens: 3,
    total_tokens: 7,
    extensions: {
      cache_read_tokens: 2,
      cache_write_tokens: 1,
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

test("maps replacement instructions and native stdio, HTTP, and SSE MCP configuration", async () => {
  const sdk = fakeSdk({
    mcpReport: {
      servers: [
        { name: "local", status: "connected" },
        { name: "remote", status: "connected" },
        { name: "events", status: "connected" },
        { name: "ipv6", status: "connected" },
      ],
    },
  });
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
    systemPrompt: "Replace the prompt.",
  });
  assert.equal(env.HOME, env.USERPROFILE);
  assert.equal(env.HOME, env.FACTORY_HOME_OVERRIDE);
  assert.equal(env.FACTORY_API_KEY, "secret");
  assert.deepEqual(
    JSON.parse(await readFile(join(env.HOME, ".factory", "mcp.json"), "utf8")),
    {
      mcpServers: {
        local: { type: "stdio", command: "node", args: ["server.js"], env: { MODE: "test" }, disabled: false },
        remote: {
          type: "http",
          url: "https://mcp.example.test/rpc",
          headers: { Authorization: "Bearer token" },
          disabled: false,
        },
        events: { type: "sse", url: "http://127.0.0.1:3000/sse", headers: {}, disabled: false },
        ipv6: { type: "sse", url: "http://[::1]:3001/sse", headers: {}, disabled: false },
      },
    },
  );
  assert.equal(sdk.calls.listMcpServers, 1);
  await handle.stop();
});

test("preserves Factory settings without copying ambient MCP configuration", async (t) => {
  const sourceHome = await mkdtemp(join(tmpdir(), "fabric-droid-factory-home-"));
  t.after(() => rm(sourceHome, { recursive: true, force: true }));
  const sourceFactory = join(sourceHome, ".factory");
  await mkdir(sourceFactory, { recursive: true });
  const settings = {
    customModels: [{
      model: "nvidia/nemotron-3-super-120b-a12b",
      displayName: "NVIDIA Nemotron 3 Super",
      baseUrl: "https://integrate.api.nvidia.com/v1",
      apiKey: "${NVIDIA_API_KEY}",
      provider: "generic-chat-completion-api",
      maxOutputTokens: 4096,
    }],
  };
  await writeFile(join(sourceFactory, "settings.json"), `${JSON.stringify(settings)}\n`);
  await writeFile(join(sourceFactory, "mcp.json"), '{"mcpServers":{"ambient":{"type":"stdio"}}}\n');
  const sdk = fakeSdk({ mcpReport: { servers: [{ name: "configured", status: "connected" }] } });

  const configured = input({
    tools: undefined,
    mcp: { servers: { configured: { transport: "stdio", url: "node" } } },
  });
  configured.runtimeContext.environment.env.FACTORY_HOME_OVERRIDE = sourceHome;
  const handle = await new DroidSdkSessionFactory(
    sdk.loader,
    10_000,
    { FACTORY_HOME_OVERRIDE: join(sourceHome, "missing") },
  ).create(configured);
  const runtimeFactory = join(sdk.calls.create[0].env.HOME, ".factory");
  assert.equal(sdk.calls.create[0].env.FACTORY_API_KEY, "secret");
  assert.deepEqual(JSON.parse(await readFile(join(runtimeFactory, "settings.json"), "utf8")), settings);
  assert.deepEqual(JSON.parse(await readFile(join(runtimeFactory, "mcp.json"), "utf8")), {
    mcpServers: {
      configured: { type: "stdio", command: "node", args: [], env: {}, disabled: false },
    },
  });

  await handle.stop();
});

test("waits for configured MCP servers to connect", async () => {
  const sdk = fakeSdk({
    mcpReport(call) {
      return {
        servers: [{ name: "test", status: call === 1 ? "connecting" : "connected" }],
      };
    },
  });
  const handle = await new DroidSdkSessionFactory(sdk.loader).create(
    input({ tools: undefined, mcp: { servers: { test: { transport: "stdio", url: "node" } } } }),
  );

  assert.equal(sdk.calls.listMcpServers, 2);
  await handle.stop();
});

test("keeps configured MCP tools outside the built-in tool policy", async () => {
  const sdk = fakeSdk({
    tools: [{ id: "Read" }, { id: "Edit" }, { id: "test___echo" }],
    mcpReport: { servers: [{ name: "test", status: "connected" }] },
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

test("rejects an MCP server that does not finish connecting", async () => {
  const sdk = fakeSdk({
    mcpReport: { servers: [{ name: "test", status: "connecting" }] },
  });

  await assert.rejects(
    new DroidSdkSessionFactory(sdk.loader, 1).create(
      input({ tools: undefined, mcp: { servers: { test: { transport: "stdio", url: "node" } } } }),
    ),
    (error) =>
      error.code === "droid_mcp_load_failed" &&
      error.metadata?.connecting?.includes("test"),
  );
  assert.equal(sdk.calls.close, 1);
});

test("times out when Droid does not return MCP status", async () => {
  const sdk = fakeSdk({
    mcpReport: () => new Promise(() => undefined),
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
      output_tokens: 3,
      total_tokens: 7,
      extensions: {
        cache_read_tokens: 2,
        cache_write_tokens: 1,
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

  const mcp = fakeSdk({ mcpReport: { servers: [{ name: "test", status: "failed", error: "spawn failed" }] } });
  await assert.rejects(
    new DroidSdkSessionFactory(mcp.loader).create(input({ mcp: { servers: { test: { transport: "stdio", url: "missing" } } } })),
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

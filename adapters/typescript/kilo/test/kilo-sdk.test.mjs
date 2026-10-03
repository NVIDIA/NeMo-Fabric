// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import {EventEmitter} from "node:events";
import test from "node:test";
import {mkdtemp, mkdir, rm, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {PassThrough} from "node:stream";

import {classifySpawnError, closeChild, extractPromptOutcome, KiloSdkSessionFactory, startKiloServer} from "../dist/kilo-sdk.js";

test("extracts Kilo Code text, usage, and cost", () => {
  assert.deepEqual(extractPromptOutcome({
    info: {tokens: {input: 4, output: 6, total: 10}, cost: 0.05},
    parts: [{type: "text", text: "hello"}, {type: "tool", name: "read"}, {type: "text", text: " world"}],
  }), {text: "hello world", usage: {input_tokens: 4, output_tokens: 6, total_tokens: 10, cost_usd: 0.05}});
});

test("marks upstream assistant errors without exposing their content", () => {
  assert.deepEqual(extractPromptOutcome({info: {error: {name: "ProviderError", data: {message: "secret"}}}, parts: []}), {error: true});
});

test("distinguishes a missing Kilo executable from other launch failures", () => {
  assert.equal(classifySpawnError(Object.assign(new Error("missing"), {code: "ENOENT"})).code, "kilo_harness_unavailable");
  assert.equal(classifySpawnError(Object.assign(new Error("denied"), {code: "EACCES"})).code, "kilo_start_failed");
  assert.equal(classifySpawnError(new Error("unknown")).code, "kilo_start_failed");
});

test("waits for process exit after escalating shutdown", async () => {
  const child = new EventEmitter();
  child.pid = 42;
  child.exitCode = null;
  child.signalCode = null;
  child.kill = (signal) => {
    child.signals.push(signal);
    if (signal === "SIGKILL") {
      setImmediate(() => {
        child.signalCode = signal;
        child.emit("exit", null, signal);
      });
    }
    return true;
  };
  child.signals = [];

  await closeChild(child, 1);
  assert.deepEqual(child.signals, ["SIGTERM", "SIGKILL"]);
  assert.equal(child.signalCode, "SIGKILL");
});

test("waits for server cleanup before rejecting startup", async () => {
  const child = new EventEmitter();
  child.pid = 42;
  child.exitCode = null;
  child.signalCode = null;
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  child.signals = [];
  child.kill = (signal) => {
    child.signals.push(signal);
    return true;
  };
  let spawned;
  const didSpawn = new Promise((resolveSpawn) => { spawned = resolveSpawn; });
  let rejected = false;
  const result = startKiloServer(process.cwd(), process.env, "kilo", () => {
    spawned();
    return child;
  }, async () => 4123).catch((error) => {
    rejected = true;
    return error;
  });

  await didSpawn;
  child.emit("error", Object.assign(new Error("denied"), {code: "EACCES"}));
  await new Promise((resolveImmediate) => setImmediate(resolveImmediate));
  assert.equal(rejected, false);
  assert.deepEqual(child.signals, ["SIGTERM"]);

  child.signalCode = "SIGTERM";
  child.emit("exit", null, "SIGTERM");
  const error = await result;
  assert.equal(rejected, true);
  assert.equal(error.code, "kilo_start_failed");
});

test("projects normalized configuration into an isolated Kilo Code server", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "kilo-adapter-test-"));
  const skill = join(workspace, "review-skill");
  await mkdir(skill);
  await writeFile(join(skill, "SKILL.md"), "# Review\n", "utf8");
  let serverEnvironment;
  let createParameters;
  let deleteParameters;
  let deleteSignal;
  const client = {
    app: {skills: async () => ({data: [{name: "review", location: join(skill, "SKILL.md"), content: "# Review"}]})},
    mcp: {status: async () => ({data: {local: {status: "connected"}}})},
    session: {
      create: async (parameters) => { createParameters = parameters; return {data: {id: "session-1"}}; },
      prompt: async () => ({data: {info: {}, parts: [{type: "text", text: "done"}]}}),
      delete: async (parameters, options) => { deleteParameters = parameters; deleteSignal = options.signal; },
    },
  };
  const factory = new KiloSdkSessionFactory(
    async () => () => client,
    async (_workspace, environment) => { serverEnvironment = environment; return {url: "http://127.0.0.1:4123", close: async () => {}}; },
  );
  try {
    const handle = await factory.create({
      agentName: "reviewer",
      baseDir: workspace,
      config: {
        models: {default: {provider: "nvidia", model: "nemotron", api_key_env: "MODEL_KEY", base_url: "https://provider.example/v1", temperature: 0.1, top_p: 0.8}},
        instructions: {system: {content: "Review carefully.", mode: "replace"}},
        runtime: {max_turns: 9},
        tools: {enabled: ["read", "grep"], blocked: ["bash"]},
        skills: {paths: ["review-skill"]},
        mcp: {servers: {local: {transport: "stdio", url: "mcp-server", args: ["--stdio"], env: {MODE: "test"}}}},
      },
      runtimeContext: {runtime_id: "r", invocation_id: "i", request_id: "q", artifacts: {}, environment: {environment_id: "e", provider: "local", ownership: "caller_owned", control_location: "external_control", workspace, env: {MODEL_KEY: "secret"}}},
    });
    const projected = JSON.parse(serverEnvironment.KILO_CONFIG_CONTENT);
    assert.deepEqual(projected.provider.nvidia, {npm: "@ai-sdk/openai-compatible", options: {apiKey: "{env:MODEL_KEY}", baseURL: "https://provider.example/v1"}, models: {nemotron: {}}});
    assert.deepEqual(projected.permission, {question: "deny", external_directory: "deny", doom_loop: "deny", read: {"*": "allow", "*.env": "deny", "*.env.*": "deny", "*.env.example": "allow"}, "*": "deny", grep: "allow", bash: "deny"});
    assert.deepEqual(projected.agent, {build: {model: "nvidia/nemotron", prompt: "Review carefully.", steps: 9, temperature: 0.1, top_p: 0.8}});
    assert.deepEqual(projected.skills.paths, [skill]);
    assert.deepEqual(projected.mcp.local, {type: "local", command: ["mcp-server", "--stdio"], environment: {MODE: "test"}});
    assert.equal(serverEnvironment.KILO_DISABLE_PROJECT_CONFIG, "1");
    assert.equal(serverEnvironment.MODEL_KEY, "secret");
    assert.equal(serverEnvironment.NVIDIA_API_KEY, undefined);
    assert.deepEqual(createParameters, {directory: workspace, agent: "build", model: {providerID: "nvidia", id: "nemotron"}});
    await handle.stop();
    assert.deepEqual(deleteParameters, {sessionID: "session-1", directory: workspace});
    assert.ok(deleteSignal instanceof AbortSignal);
  } finally {
    await rm(workspace, {recursive: true, force: true});
  }
});

test("denies interactive permissions when normalized tools are omitted", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "kilo-adapter-test-"));
  let serverEnvironment;
  const client = {
    app: {skills: async () => ({data: []})},
    mcp: {status: async () => ({data: {}})},
    session: {
      create: async () => ({data: {id: "session-1"}}),
      prompt: async () => ({data: {info: {}, parts: []}}),
      delete: async () => {},
    },
  };
  const factory = new KiloSdkSessionFactory(
    async () => () => client,
    async (_workspace, environment) => {
      serverEnvironment = environment;
      return {url: "http://127.0.0.1:4123", close: async () => {}};
    },
  );
  try {
    const handle = await factory.create({
      agentName: "reviewer",
      baseDir: workspace,
      config: {
        models: {default: {provider: "nvidia", model: "nemotron", api_key_env: "MODEL_KEY"}},
      },
      runtimeContext: {
        runtime_id: "r",
        invocation_id: "i",
        request_id: "q",
        artifacts: {},
        environment: {
          environment_id: "e",
          provider: "local",
          ownership: "caller_owned",
          control_location: "external_control",
          workspace,
          env: {MODEL_KEY: "secret"},
        },
      },
    });
    const projected = JSON.parse(serverEnvironment.KILO_CONFIG_CONTENT);
    assert.deepEqual(projected.permission, {
      question: "deny",
      external_directory: "deny",
      doom_loop: "deny",
      read: {"*": "allow", "*.env": "deny", "*.env.*": "deny", "*.env.example": "allow"},
    });
    assert.deepEqual(projected.agent, {build: {model: "nvidia/nemotron"}});
    await handle.stop();
  } finally {
    await rm(workspace, {recursive: true, force: true});
  }
});

test("aborts the Kilo session when a prompt reaches its deadline", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "kilo-adapter-test-"));
  let promptSignal;
  let abortParameters;
  let abortSignal;
  const client = {
    app: {skills: async () => ({data: []})},
    mcp: {status: async () => ({data: {}})},
    session: {
      create: async () => ({data: {id: "session-timeout"}}),
      prompt: async (_parameters, options) => {
        promptSignal = options.signal;
        return await new Promise(() => {});
      },
      abort: async (parameters, options) => {
        abortParameters = parameters;
        abortSignal = options.signal;
      },
      delete: async () => {},
    },
  };
  const factory = new KiloSdkSessionFactory(
    async () => () => client,
    async () => ({url: "http://127.0.0.1:4123", close: async () => {}}),
    0,
  );
  try {
    const handle = await factory.create({
      agentName: "reviewer",
      baseDir: workspace,
      config: {models: {default: {provider: "nvidia", model: "nemotron", api_key_env: "MODEL_KEY"}}},
      runtimeContext: {
        runtime_id: "r",
        invocation_id: "i",
        request_id: "q",
        artifacts: {},
        environment: {
          environment_id: "e",
          provider: "local",
          ownership: "caller_owned",
          control_location: "external_control",
          workspace,
          env: {MODEL_KEY: "secret"},
        },
      },
    });
    await assert.rejects(handle.prompt("wait forever"), (error) => error.code === "kilo_prompt_timeout");
    assert.equal(promptSignal.aborted, true);
    assert.deepEqual(abortParameters, {sessionID: "session-timeout", directory: workspace});
    assert.ok(abortSignal instanceof AbortSignal);
    await handle.stop();
  } finally {
    await rm(workspace, {recursive: true, force: true});
  }
});

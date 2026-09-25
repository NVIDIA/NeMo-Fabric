// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import {mkdtemp, mkdir, rm, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";

import {extractPromptOutcome, KiloSdkSessionFactory} from "../dist/kilo-sdk.js";

test("extracts Kilo Code text, usage, and cost", () => {
  assert.deepEqual(extractPromptOutcome({
    info: {tokens: {input: 4, output: 6, total: 10}, cost: 0.05},
    parts: [{type: "text", text: "hello"}, {type: "tool", name: "read"}, {type: "text", text: " world"}],
  }), {text: "hello world", usage: {input_tokens: 4, output_tokens: 6, total_tokens: 10, cost_usd: 0.05}});
});

test("marks upstream assistant errors without exposing their content", () => {
  assert.deepEqual(extractPromptOutcome({info: {error: {name: "ProviderError", data: {message: "secret"}}}, parts: []}), {error: true});
});

test("projects normalized configuration into an isolated Kilo Code server", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "kilo-adapter-test-"));
  const skill = join(workspace, "review-skill");
  await mkdir(skill);
  await writeFile(join(skill, "SKILL.md"), "# Review\n", "utf8");
  let serverEnvironment;
  let createParameters;
  let deleteParameters;
  const client = {
    app: {skills: async () => ({data: [{name: "review", location: join(skill, "SKILL.md"), content: "# Review"}]})},
    mcp: {status: async () => ({data: {local: {status: "connected"}}})},
    session: {
      create: async (parameters) => { createParameters = parameters; return {data: {id: "session-1"}}; },
      prompt: async () => ({data: {info: {}, parts: [{type: "text", text: "done"}]}}),
      delete: async (parameters) => { deleteParameters = parameters; },
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
    assert.deepEqual(projected.agent.build, {model: "nvidia/nemotron", permission: {question: "deny", external_directory: "deny", read: {"*": "allow", "*.env": "deny", "*.env.*": "deny", "*.env.example": "allow"}, "*": "deny", grep: "allow", bash: "deny"}, prompt: "Review carefully.", steps: 9, temperature: 0.1, top_p: 0.8});
    assert.deepEqual(projected.skills.paths, [skill]);
    assert.deepEqual(projected.mcp.local, {type: "local", command: ["mcp-server", "--stdio"], environment: {MODE: "test"}});
    assert.equal(serverEnvironment.KILO_DISABLE_PROJECT_CONFIG, "1");
    assert.equal(serverEnvironment.MODEL_KEY, "secret");
    assert.equal(serverEnvironment.NVIDIA_API_KEY, undefined);
    assert.deepEqual(createParameters, {directory: workspace, agent: "build", model: {providerID: "nvidia", id: "nemotron"}});
    await handle.stop();
    assert.deepEqual(deleteParameters, {sessionID: "session-1", directory: workspace});
  } finally {
    await rm(workspace, {recursive: true, force: true});
  }
});

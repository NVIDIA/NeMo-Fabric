// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

function context(workspace, invocationId, environment = {}) {
  return {
    artifacts: {},
    environment: {
      control_location: "external_control",
      env: { TEST_API_KEY: "not-a-real-key", ...environment },
      environment_id: "environment-1",
      ownership: "caller_owned",
      provider: "local",
      workspace,
    },
    invocation_id: invocationId,
    request_id: `request-${invocationId}`,
    runtime_id: "runtime-1",
  };
}

async function exchange(workspace, requests, environment = {}, timeoutMs) {
  const childEnv = { ...process.env, ...environment };
  delete childEnv.NODE_TEST_CONTEXT;
  const child = spawn(process.env.BUN_EXECUTABLE ?? "bun", [fileURLToPath(new URL("../dist/cli.js", import.meta.url))], {
    cwd: workspace,
    env: childEnv,
    stdio: ["pipe", "pipe", "pipe"],
  });
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    stdout += chunk;
  });
  child.stderr.on("data", (chunk) => {
    stderr += chunk;
  });
  child.stdin.end(`${requests.map((request) => JSON.stringify(request)).join("\n")}\n`);

  let timeout;
  try {
    const exitCode = await new Promise((resolve, reject) => {
      child.once("error", reject);
      child.once("close", resolve);
      if (timeoutMs !== undefined) {
        timeout = setTimeout(() => {
          child.kill();
          reject(new Error(`adapter process did not exit within ${timeoutMs}ms`));
        }, timeoutMs);
      }
    });
    const lines = stdout.trim().split("\n").filter(Boolean);
    const responses = lines.map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        throw new Error(`adapter emitted non-JSON stdout: ${line}`);
      }
    });
    return {
      exitCode,
      responses,
      stderr,
    };
  } finally {
    clearTimeout(timeout);
  }
}

async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
  const address = server.address();
  assert.notEqual(address, null);
  assert.notEqual(typeof address, "string");
  return `http://127.0.0.1:${address.port}`;
}

async function close(server) {
  await new Promise((resolve, reject) => {
    server.close((error) => (error === undefined ? resolve() : reject(error)));
  });
}

function openAiStream(text) {
  const chunk = (delta, finishReason = null) =>
    `data: ${JSON.stringify({
      id: "chatcmpl-opencode-test",
      object: "chat.completion.chunk",
      created: 0,
      model: "nvidia/nemotron-3.5-lightning-30b-a3b",
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    })}\n\n`;
  return `${chunk({ role: "assistant" })}${chunk({ content: text })}${chunk({}, "stop")}data: [DONE]\n\n`;
}

function openAiToolCall(name, input) {
  const chunk = (delta, finishReason = null) =>
    `data: ${JSON.stringify({
      id: "chatcmpl-opencode-tool-test",
      object: "chat.completion.chunk",
      created: 0,
      model: "nvidia/nemotron-3.5-lightning-30b-a3b",
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    })}\n\n`;
  return `${chunk({
    role: "assistant",
    tool_calls: [{
      index: 0,
      id: "call-read-env",
      type: "function",
      function: { name, arguments: JSON.stringify(input) },
    }],
  })}${chunk({}, "tool_calls")}data: [DONE]\n\n`;
}

test("runs two real OpenCode SDK prompts through the process host", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "fabric-opencode-process-"));
  const fakeHome = await mkdtemp(join(tmpdir(), "fabric-opencode-home-"));
  let prompts = 0;
  const providerRequests = [];
  const mcpRequests = [];
  const endpoint = createServer(async (request, response) => {
    if (request.method !== "POST" || request.url !== "/v1/chat/completions") {
      response.writeHead(404).end();
      return;
    }
    const chunks = [];
    for await (const chunk of request) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    const payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    providerRequests.push(payload);
    const messages = Array.isArray(payload.messages) ? payload.messages : [];
    const userMessages = messages.filter((message) => message?.role === "user");
    const latest = userMessages.at(-1)?.content;
    prompts += 1;
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.end(
      openAiStream(
        typeof latest === "string"
          ? `reply:user_count=${userMessages.length} latest=${latest}`
          : `reply-${prompts}`,
      ),
    );
  });
  const endpointUrl = await listen(endpoint);
  const mcpEndpoint = createServer(async (request, response) => {
    if (request.method !== "POST" || new URL(request.url, "http://localhost").pathname !== "/mcp") {
      response.writeHead(404).end();
      return;
    }
    const chunks = [];
    for await (const chunk of request) {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    const payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    mcpRequests.push({ method: payload.method, authorization: request.headers.authorization });
    if (payload.id === undefined) {
      response.writeHead(202).end();
      return;
    }
    const result = payload.method === "initialize"
      ? {
        protocolVersion: payload.params.protocolVersion,
        capabilities: { tools: {} },
        serverInfo: { name: "fabric-test-mcp", version: "1.0.0" },
      }
      : payload.method === "tools/list"
        ? { tools: [] }
        : {};
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({ jsonrpc: "2.0", id: payload.id, result }));
  });
  const mcpEndpointUrl = await listen(mcpEndpoint);
  try {
    const workspaceConfigInstruction = join(workspace, "workspace-opencode-instructions.md");
    const homeConfigInstruction = join(fakeHome, "home-opencode-instructions.md");
    const interpolationFile = join(workspace, "private-instruction.txt");
    const skillDirectory = join(workspace, "skills", "{env:FABRIC_SKILL_SEGMENT}-{file:skill-segment}");
    await writeFile(join(workspace, "AGENTS.md"), "FABRIC_WORKSPACE_CONFIG_SENTINEL", "utf8");
    await writeFile(workspaceConfigInstruction, "FABRIC_WORKSPACE_OPENCODE_CONFIG_SENTINEL", "utf8");
    await writeFile(
      join(workspace, "opencode.json"),
      JSON.stringify({ instructions: [workspaceConfigInstruction] }),
      "utf8",
    );
    await mkdir(join(fakeHome, ".agents"), { recursive: true });
    await writeFile(join(fakeHome, ".agents", "AGENTS.md"), "FABRIC_HOME_CONFIG_SENTINEL", "utf8");
    await writeFile(homeConfigInstruction, "FABRIC_HOME_OPENCODE_CONFIG_SENTINEL", "utf8");
    await writeFile(interpolationFile, "FABRIC_PRIVATE_FILE_CONTENT", "utf8");
    await writeFile(join(workspace, "skill-segment"), "expanded-skill-segment", "utf8");
    await mkdir(join(fakeHome, ".config", "opencode"), { recursive: true });
    await writeFile(
      join(fakeHome, ".config", "opencode", "opencode.json"),
      JSON.stringify({ instructions: [homeConfigInstruction] }),
      "utf8",
    );
    const systemInstruction = [
      "FABRIC_SYSTEM_INSTRUCTION_SENTINEL",
      "Keep {env:TEST_API_KEY} and {file:" + interpolationFile + "} literal.",
    ].join(" ");
    await mkdir(skillDirectory, { recursive: true });
    await writeFile(
      join(skillDirectory, "SKILL.md"),
      "---\nname: review\ndescription: FABRIC_SKILL_SENTINEL\n---\nReview the change.\n",
      "utf8",
    );
    const start = {
      operation: "start",
      payload: {
        agent_name: "opencode-process-test",
        base_dir: workspace,
        config: {
          models: {
            default: {
              api_key_env: "TEST_API_KEY",
              base_url: `${endpointUrl}/v1`,
              model: "nvidia/nemotron-3.5-nano-30b-a3b",
              provider: "nvidia",
              temperature: 0.25,
              top_p: 0.8,
            },
          },
          instructions: {
            system: {
              content: systemInstruction,
              mode: "replace",
            },
          },
          skills: { paths: ["skills/{env:FABRIC_SKILL_SEGMENT}-{file:skill-segment}"] },
          mcp: {
            servers: {
              fabric_test: {
                transport: "streamable-http",
                url: `${mcpEndpointUrl}/mcp`,
                custom_headers: { Authorization: "Bearer ${MCP_ACCESS_TOKEN}" },
              },
            },
          },
        },
        runtime_context: context(workspace, "start", { FABRIC_SKILL_SEGMENT: "expanded-skill-segment" }),
      },
    };
    const invoke = (input, invocationId) => ({
      operation: "invoke",
      payload: {
        request: { input },
        runtime_context: context(workspace, invocationId),
      },
    });
    const stop = { operation: "stop", payload: { runtime_id: "runtime-1" } };
    const { exitCode, responses, stderr } = await exchange(
      workspace,
      [start, invoke("first", "first"), invoke("second", "second"), stop],
      {
        HOME: fakeHome,
        XDG_CONFIG_HOME: join(fakeHome, ".config"),
        MCP_ACCESS_TOKEN: "mcp-test-token",
      },
    );

    assert.equal(exitCode, 0, stderr);
    assert.deepEqual(responses.map((response) => response.operation), ["start", "invoke", "invoke", "stop"]);
    assert.equal(responses[0].outcome.status, "succeeded");
    assert.equal(responses[1].outcome.output.status, "succeeded");
    assert.equal(responses[1].outcome.output.output.response, "reply:user_count=1 latest=first");
    assert.equal(responses[2].outcome.output.status, "succeeded");
    assert.equal(responses[2].outcome.output.output.response, "reply:user_count=2 latest=second");
    assert.equal(responses[3].outcome.status, "succeeded");
    assert.ok(prompts >= 2);
    assert.ok(providerRequests.length >= 2);
    assert.ok(providerRequests.every((request) => !("prompt_cache_key" in request)));
    const promptRequests = providerRequests.filter((request) => {
      const userMessages = Array.isArray(request.messages)
        ? request.messages.filter((message) => message?.role === "user")
        : [];
      const latest = userMessages.at(-1)?.content;
      return latest === "first" || latest === "second";
    });
    assert.ok(promptRequests.length >= 2);
    assert.ok(promptRequests.every((request) => request.temperature === 0.25));
    assert.ok(promptRequests.every((request) => request.top_p === 0.8));
    const instructionPromptRequests = promptRequests.filter((request) =>
      request.messages?.some(
        (message) =>
          message?.role === "system" &&
          typeof message?.content === "string" &&
          message.content.includes("FABRIC_SYSTEM_INSTRUCTION_SENTINEL"),
      ),
    );
    assert.ok(instructionPromptRequests.length >= 2);
    const promptContents = promptRequests
      .flatMap((request) => request.messages ?? [])
      .map((message) => message?.content)
      .filter((content) => typeof content === "string");
    assert.ok(promptContents.some((content) => content.includes("{env:TEST_API_KEY}")));
    assert.ok(promptContents.some((content) => content.includes(`{file:${interpolationFile}}`)));
    assert.ok(promptContents.every((content) => !content.includes("not-a-real-key")));
    assert.ok(promptContents.every((content) => !content.includes("FABRIC_PRIVATE_FILE_CONTENT")));
    const skillPromptRequests = promptRequests.filter((request) =>
      request.messages?.some(
        (message) => typeof message?.content === "string" && message.content.includes("FABRIC_SKILL_SENTINEL"),
      ),
    );
    assert.ok(skillPromptRequests.length >= 2);
    assert.ok(mcpRequests.some((request) => request.method === "initialize"));
    assert.ok(mcpRequests.some((request) => request.method === "tools/list"));
    assert.ok(mcpRequests.every((request) => request.authorization === "Bearer mcp-test-token"));
    const ambientInstructions = providerRequests
      .flatMap((request) => (Array.isArray(request.messages) ? request.messages : []))
      .map((message) => message?.content)
      .filter((content) => typeof content === "string" && /FABRIC_(?:WORKSPACE|HOME)_(?:CONFIG|OPENCODE_CONFIG)_SENTINEL/u.test(content))
      .map((content) => content.match(/FABRIC_(?:WORKSPACE|HOME)_(?:CONFIG|OPENCODE_CONFIG)_SENTINEL/u)?.[0]);
    assert.deepEqual(ambientInstructions, []);
  } finally {
    await close(endpoint);
    await close(mcpEndpoint);
    await rm(workspace, { recursive: true, force: true });
    await rm(fakeHome, { recursive: true, force: true });
  }
});

test("rejects malformed and duplicate OpenCode skills during process startup", async () => {
  for (const scenario of [
    {
      directories: ["malformed"],
      files: [["malformed", "---\nname: [not a string]\n---\nMalformed skill.\n"]],
      paths: ["skills/malformed"],
      code: "opencode_skill_load_failed",
    },
    {
      directories: ["first", "second"],
      files: [
        ["first", "---\nname: duplicate\n---\nFirst skill.\n"],
        ["second", "---\nname: duplicate\n---\nSecond skill.\n"],
      ],
      paths: ["skills/first", "skills/second"],
      code: "opencode_skill_name_duplicate",
    },
  ]) {
    const workspace = await mkdtemp(join(tmpdir(), "fabric-opencode-invalid-skill-"));
    const fakeHome = await mkdtemp(join(tmpdir(), "fabric-opencode-invalid-skill-home-"));
    try {
      await Promise.all(scenario.directories.map((name) => mkdir(join(workspace, "skills", name), { recursive: true })));
      await Promise.all(
        scenario.files.map(([name, content]) => writeFile(join(workspace, "skills", name, "SKILL.md"), content, "utf8")),
      );
      const start = {
        operation: "start",
        payload: {
          agent_name: "opencode-invalid-skill-test",
          base_dir: workspace,
          config: {
            models: {
              default: {
                api_key_env: "TEST_API_KEY",
                base_url: "http://127.0.0.1:1/v1",
                model: "nvidia/nemotron-3.5-nano-30b-a3b",
                provider: "nvidia",
              },
            },
            skills: { paths: scenario.paths },
          },
          runtime_context: context(workspace, "start"),
        },
      };
      const { exitCode, responses, stderr } = await exchange(workspace, [start], {
        HOME: fakeHome,
        XDG_CONFIG_HOME: join(fakeHome, ".config"),
      });

      assert.equal(exitCode, 0, stderr);
      assert.equal(responses.length, 1);
      assert.equal(responses[0].outcome.status, "failed");
      assert.equal(responses[0].outcome.error.code, scenario.code);
    } finally {
      await rm(workspace, { recursive: true, force: true });
      await rm(fakeHome, { recursive: true, force: true });
    }
  }
});

test("fails process startup when an HTTP or stdio MCP server cannot connect", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "fabric-opencode-mcp-startup-"));
  const endpoint = createServer((_request, response) => response.writeHead(500).end());
  const endpointUrl = await listen(endpoint);
  const start = (server) => ({
    operation: "start",
    payload: {
      agent_name: "opencode-mcp-startup-test",
      base_dir: workspace,
      config: {
        models: {
          default: {
            api_key_env: "TEST_API_KEY",
            base_url: `${endpointUrl}/v1`,
            model: "nvidia/nemotron-3.5-nano-30b-a3b",
            provider: "nvidia",
          },
        },
        mcp: { servers: { unavailable: server } },
      },
      runtime_context: context(workspace, "start"),
    },
  });
  try {
    for (const server of [
      { transport: "streamable-http", url: "http://127.0.0.1:1/mcp" },
      { transport: "stdio", url: "not-a-real-mcp-command" },
    ]) {
      const { exitCode, responses, stderr } = await exchange(workspace, [start(server)], {}, 5_000);

      assert.equal(exitCode, 0, stderr);
      assert.equal(responses.length, 1);
      assert.equal(responses[0].outcome.status, "failed");
      assert.equal(responses[0].outcome.error.code, "opencode_mcp_connection_failed");
    }
  } finally {
    await close(endpoint);
    await rm(workspace, { recursive: true, force: true });
  }
});

test("denies interactive OpenCode operations instead of waiting for a reply", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "fabric-opencode-permissions-"));
  let providerRequests = 0;
  const endpoint = createServer(async (request, response) => {
    if (request.method !== "POST" || request.url !== "/v1/chat/completions") {
      response.writeHead(404).end();
      return;
    }
    for await (const _chunk of request) {
      // Consume the request body before writing the deterministic response.
    }
    providerRequests += 1;
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.end(
      providerRequests === 1
        ? openAiToolCall("read", { path: ".env" })
        : openAiStream("permission denial completed"),
    );
  });
  const endpointUrl = await listen(endpoint);
  try {
    await writeFile(join(workspace, ".env"), "SECRET=do-not-read\n", "utf8");
    const start = {
      operation: "start",
      payload: {
        agent_name: "opencode-permission-test",
        base_dir: workspace,
        config: {
          models: {
            default: {
              api_key_env: "TEST_API_KEY",
              base_url: `${endpointUrl}/v1`,
              model: "nvidia/nemotron-3.5-nano-30b-a3b",
              provider: "nvidia",
            },
          },
        },
        runtime_context: context(workspace, "start"),
      },
    };
    const invoke = {
      operation: "invoke",
      payload: {
        request: { input: "Read .env, then report whether it was available." },
        runtime_context: context(workspace, "permission"),
      },
    };
    const stop = { operation: "stop", payload: { runtime_id: "runtime-1" } };
    const { exitCode, responses, stderr } = await exchange(workspace, [start, invoke, stop], {}, 10_000);

    assert.equal(exitCode, 0, stderr);
    assert.deepEqual(responses.map((response) => response.operation), ["start", "invoke", "stop"]);
    assert.equal(responses[0].outcome.status, "succeeded");
    assert.equal(responses[1].outcome.output.status, "succeeded");
    assert.equal(responses[1].outcome.output.output.response, "permission denial completed");
    assert.equal(responses[2].outcome.status, "succeeded");
    assert.equal(providerRequests, 2);
  } finally {
    await close(endpoint);
    await rm(workspace, { recursive: true, force: true });
  }
});

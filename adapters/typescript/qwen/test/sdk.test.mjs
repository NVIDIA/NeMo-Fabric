// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { QwenSdkSessionFactory } from "../dist/qwen-sdk.js";

const mcpServerPath = fileURLToPath(new URL("./fixtures/mcp-server.mjs", import.meta.url));

test("bundled Qwen CLI keeps context across two SDK prompts", { timeout: 45_000 }, async () => {
  const requests = [];
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    requests.push(body);
    const isSecondTurn = JSON.stringify(body.messages).includes("Which token did I ask");
    response.writeHead(200, { "content-type": "text/event-stream" });
    const common = { id: `chatcmpl-${requests.length}`, object: "chat.completion.chunk", created: 0, model: "fabric-test-model" };
    response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: { role: "assistant", content: isSecondTurn ? "turn-2" : "turn-1" }, finish_reason: null }] })}\n\n`);
    response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } })}\n\n`);
    response.end("data: [DONE]\n\n");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  let session;
  try {
    const port = server.address().port;
    session = await new QwenSdkSessionFactory().create({
      agentName: "qwen-test", baseDir: process.cwd(),
      config: {
        models: { default: { provider: "openai", model: "fabric-test-model", api_key_env: "FABRIC_QWEN_TEST_KEY", base_url: `http://127.0.0.1:${port}/v1` } },
        instructions: { system: { content: "Fabric Qwen instruction probe", mode: "replace" } },
      },
      runtimeContext: {
        artifacts: {}, environment: { control_location: "external_control", env: { FABRIC_QWEN_TEST_KEY: "local-test-key" }, environment_id: "env-1", ownership: "caller_owned", provider: "local", workspace: process.cwd() },
        invocation_id: "start", request_id: "start", runtime_id: "runtime-1",
      },
    });
    const first = await session.prompt("Remember token cobalt");
    const second = await session.prompt("Which token did I ask you to remember?");
    assert.equal(first.text, "turn-1", JSON.stringify(first));
    assert.equal(second.text, "turn-2", JSON.stringify(second));
    assert.deepEqual(first.usage, { input_tokens: 10, output_tokens: 4, total_tokens: 14 });
    assert.deepEqual(second.usage, { input_tokens: 10, output_tokens: 4, total_tokens: 14 });
    assert.ok(requests.length >= 2);
    assert.match(String(requests[0].messages[0].content), /Fabric Qwen instruction probe/);
    assert.match(JSON.stringify(requests.at(-1).messages), /Remember token cobalt/);
  } finally {
    await session?.close();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }
});

test("Qwen SDK exposes an explicit skill", { timeout: 45_000 }, async () => {
  const workspace = await mkdtemp(join(tmpdir(), "fabric-qwen-capabilities-"));
  const skill = join(workspace, "fabric-probe-skill");
  await mkdir(skill);
  await writeFile(join(skill, "SKILL.md"), "---\nname: fabric-probe-skill\ndescription: A test skill\n---\n\nUse the probe tool.\n");
  const ambientSkill = join(workspace, ".qwen", "skills", "ambient-probe-skill");
  await mkdir(ambientSkill, { recursive: true });
  await writeFile(join(ambientSkill, "SKILL.md"), "---\nname: ambient-probe-skill\ndescription: Must not load\n---\n\nDo not load me.\n");
  const requests = [];
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    requests.push(body);
    response.writeHead(200, { "content-type": "text/event-stream" });
    const common = { id: "chatcmpl-capabilities", object: "chat.completion.chunk", created: 0, model: "fabric-test-model" };
    response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: { role: "assistant", content: "ready" }, finish_reason: null }] })}\n\n`);
    response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } })}\n\n`);
    response.end("data: [DONE]\n\n");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  let session;
  try {
    const port = server.address().port;
    session = await new QwenSdkSessionFactory().create({
      agentName: "qwen-capabilities", baseDir: workspace,
      config: {
        models: { default: { provider: "openai", model: "fabric-test-model", api_key_env: "FABRIC_QWEN_TEST_KEY", base_url: `http://127.0.0.1:${port}/v1` } },
        skills: { paths: [skill] },
      },
      runtimeContext: {
        artifacts: {}, environment: { control_location: "external_control", env: { FABRIC_QWEN_TEST_KEY: "local-test-key" }, environment_id: "env-1", ownership: "caller_owned", provider: "local", workspace },
        invocation_id: "start", request_id: "start", runtime_id: "runtime-capabilities",
      },
    });
    assert.equal((await session.prompt("List capabilities")).text, "ready");
    assert.match(JSON.stringify(requests), /fabric-probe-skill/);
    // Qwen lists workspace files in a separate system reminder, so test the
    // skill registry, not an arbitrary occurrence of the directory name.
    const system = String(requests.at(-1).messages[0].content);
    const registry = system.split("Here is the folder structure")[0];
    assert.doesNotMatch(registry, /ambient-probe-skill/);
  } finally {
    await session?.close();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    await rm(workspace, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  }
});

test("Qwen SDK discovers and executes normalized MCP servers", { timeout: 45_000 }, async () => {
  const requests = [];
  const mcpRequests = [];
  const mcpEndpoint = createServer(async (request, response) => {
    if (request.method !== "POST") {
      response.writeHead(405).end();
      return;
    }
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    mcpRequests.push({ method: payload.method, authorization: request.headers.authorization });
    if (payload.id === undefined) {
      response.writeHead(202).end();
      return;
    }
    let result;
    if (payload.method === "initialize") {
      result = {
        protocolVersion: payload.params.protocolVersion,
        capabilities: { tools: {} },
        serverInfo: { name: "fabric-qwen-http-test", version: "1.0.0" },
      };
    } else if (payload.method === "tools/list") {
      result = {
        tools: [{
          name: "remote_echo",
          description: "Remote echo for the Fabric Qwen MCP test",
          inputSchema: { type: "object", properties: {}, additionalProperties: false },
        }],
      };
    } else {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ jsonrpc: "2.0", id: payload.id, error: { code: -32601, message: "Method not found" } }));
      return;
    }
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({ jsonrpc: "2.0", id: payload.id, result }));
  });
  await new Promise((resolve) => mcpEndpoint.listen(0, "127.0.0.1", resolve));
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    requests.push(body);
    response.writeHead(200, { "content-type": "text/event-stream" });
    const common = { id: "chatcmpl-mcp", object: "chat.completion.chunk", created: 0, model: "fabric-test-model" };
    if (requests.length <= 2) {
      const functionCall = requests.length === 1
        ? { name: "tool_search", arguments: JSON.stringify({ query: "select:mcp__probe__echo" }) }
        : { name: "tool_call", arguments: JSON.stringify({ name: "mcp__probe__echo", arguments: { text: "hello" } }) };
      response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: { role: "assistant", tool_calls: [{ index: 0, id: `call-${requests.length}`, type: "function", function: functionCall }] }, finish_reason: null }] })}\n\n`);
      response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }], usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } })}\n\n`);
    } else {
      response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: { role: "assistant", content: "mcp-ready" }, finish_reason: null }] })}\n\n`);
      response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } })}\n\n`);
    }
    response.end("data: [DONE]\n\n");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  let session;
  try {
    const port = server.address().port;
    const mcpPort = mcpEndpoint.address().port;
    session = await new QwenSdkSessionFactory().create({
      agentName: "qwen-mcp", baseDir: process.cwd(),
      config: {
        models: { default: { provider: "openai", model: "fabric-test-model", api_key_env: "FABRIC_QWEN_TEST_KEY", base_url: `http://127.0.0.1:${port}/v1` } },
        harness: { settings: { permission_mode: "yolo" } },
        mcp: { servers: {
          probe: { transport: "stdio", url: process.execPath, args: [mcpServerPath] },
          remote: {
            transport: "streamable-http",
            url: `http://127.0.0.1:${mcpPort}/mcp`,
            custom_headers: { Authorization: "Bearer ${MCP_TOKEN}" },
          },
        } },
      },
      runtimeContext: {
        artifacts: {}, environment: { control_location: "external_control", env: { FABRIC_QWEN_TEST_KEY: "local-test-key", MCP_TOKEN: "mcp-test-token" }, environment_id: "env-1", ownership: "caller_owned", provider: "local", workspace: process.cwd() },
        invocation_id: "start", request_id: "start", runtime_id: "runtime-mcp",
      },
    });
    assert.equal((await session.prompt("Use the MCP server if needed")).text, "mcp-ready");
    const serialized = JSON.stringify(requests);
    assert.match(serialized, /mcp__probe__echo/);
    assert.match(serialized, /echo:hello/);
    assert.match(serialized, /mcp__remote__remote_echo/);
    assert.ok(mcpRequests.some((request) => request.method === "initialize"));
    assert.ok(mcpRequests.some((request) => request.method === "tools\/list"));
    assert.ok(mcpRequests.every((request) => request.authorization === "Bearer mcp-test-token"));
  } finally {
    await session?.close();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    mcpEndpoint.closeAllConnections();
    await new Promise((resolve) => mcpEndpoint.close(resolve));
  }
});

test("Qwen SDK reports an unavailable configured MCP server", { timeout: 45_000 }, async () => {
  const server = createServer(async (request, response) => {
    for await (const _chunk of request) { /* Drain the request. */ }
    response.writeHead(200, { "content-type": "text/event-stream" });
    const common = { id: "chatcmpl-mcp-failure", object: "chat.completion.chunk", created: 0, model: "fabric-test-model" };
    response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: { role: "assistant", content: "must-not-succeed" }, finish_reason: null }] })}\n\n`);
    response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 } })}\n\n`);
    response.end("data: [DONE]\n\n");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  let session;
  try {
    const port = server.address().port;
    session = await new QwenSdkSessionFactory().create({
      agentName: "qwen-mcp-failure", baseDir: process.cwd(),
      config: {
        models: { default: { provider: "openai", model: "fabric-test-model", api_key_env: "FABRIC_QWEN_TEST_KEY", base_url: `http://127.0.0.1:${port}/v1` } },
        mcp: { servers: {
          unavailable: { transport: "stdio", url: process.execPath, args: ["-e", "process.exit(1)"] },
        } },
      },
      runtimeContext: {
        artifacts: {}, environment: { control_location: "external_control", env: { FABRIC_QWEN_TEST_KEY: "local-test-key" }, environment_id: "env-1", ownership: "caller_owned", provider: "local", workspace: process.cwd() },
        invocation_id: "start", request_id: "start", runtime_id: "runtime-mcp-failure",
      },
    });
    assert.equal((await session.prompt("Use the unavailable MCP server")).error, "error_mcp_unavailable");
  } finally {
    await session?.close();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }
});

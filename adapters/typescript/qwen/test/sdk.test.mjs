// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { QwenSdkSessionFactory, QwenUsageTracker } from "../dist/qwen-sdk.js";

const mcpServerPath = fileURLToPath(new URL("./fixtures/mcp-server.mjs", import.meta.url));

function qwenResult(usage, session = "session-1") {
  // Slash commands report zero turns and API duration even when they call the
  // model, so the tracker must not depend on those fields.
  return { session_id: session, num_turns: 0, duration_api_ms: 0, usage };
}

test("Qwen usage reports growth and leaves counters that did not grow unknown", () => {
  const tracker = new QwenUsageTracker();
  assert.deepEqual(
    tracker.next(qwenResult({ input_tokens: 20, output_tokens: 8, cache_read_input_tokens: 6, total_tokens: 28 })),
    { input_tokens: 20, input_tokens_include_cache: true, cached_input_tokens: 6, output_tokens: 8, total_tokens: 28 },
  );
  // Qwen Code records an omitted count as 0, so no growth is not a measurement.
  assert.equal(tracker.next(qwenResult({ input_tokens: 20, output_tokens: 8, cache_read_input_tokens: 6, total_tokens: 28 })), undefined);
  assert.deepEqual(
    tracker.next(qwenResult({ input_tokens: 20, output_tokens: 8, cache_read_input_tokens: 6, total_tokens: 40 })),
    { total_tokens: 12 },
  );
  assert.deepEqual(
    tracker.next(qwenResult({ input_tokens: 30, output_tokens: 10, cache_read_input_tokens: 6, total_tokens: 52 })),
    { input_tokens: 10, input_tokens_include_cache: true, output_tokens: 2, total_tokens: 12 },
  );
  assert.equal(tracker.next(qwenResult({ input_tokens: 5, output_tokens: 9, cache_read_input_tokens: 6, total_tokens: 60 })), undefined);
});

test("Qwen usage is unknown when the session ID changes", () => {
  const tracker = new QwenUsageTracker();
  tracker.next(qwenResult({ input_tokens: 10, output_tokens: 4, total_tokens: 14 }));
  // A resumed session replays its history into the counters, so they grow.
  assert.equal(tracker.next(qwenResult({ input_tokens: 90, output_tokens: 30, total_tokens: 120 }, "session-2")), undefined);
  assert.deepEqual(
    tracker.next(qwenResult({ input_tokens: 95, output_tokens: 32, total_tokens: 127 }, "session-2")),
    { input_tokens: 5, input_tokens_include_cache: true, output_tokens: 2, total_tokens: 7 },
  );
});

test("Qwen usage leaves counters that Qwen Code corrupted unknown", () => {
  // Raw result shapes from the bundled CLI when a provider sends a string, a
  // negative, or an unsafe prompt_tokens value on each of two requests.
  for (const input_tokens of ["0055", -10, 18014398509481984]) {
    const tracker = new QwenUsageTracker();
    assert.deepEqual(
      tracker.next(qwenResult({ input_tokens, output_tokens: 4, cache_read_input_tokens: 0, total_tokens: 14 })),
      { output_tokens: 4, total_tokens: 14 },
    );
  }
});

test("Qwen usage leaves a derived total above the safe integer range unknown", () => {
  const tracker = new QwenUsageTracker();
  assert.deepEqual(
    tracker.next(qwenResult({ input_tokens: Number.MAX_SAFE_INTEGER, output_tokens: 2, cache_read_input_tokens: 0 })),
    { input_tokens: Number.MAX_SAFE_INTEGER, input_tokens_include_cache: true, output_tokens: 2 },
  );
});

test("Qwen usage derives a total only from known parts", () => {
  const tracker = new QwenUsageTracker();
  assert.equal(tracker.next(qwenResult({ input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0 })), undefined);
  assert.deepEqual(
    tracker.next(qwenResult({ input_tokens: 7, output_tokens: 0, cache_read_input_tokens: 0 })),
    { input_tokens: 7, input_tokens_include_cache: true },
  );
});

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
    response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7, prompt_tokens_details: { cached_tokens: 3 } } })}\n\n`);
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
    const expectedUsage = { input_tokens: 10, cached_input_tokens: 6, input_tokens_include_cache: true, output_tokens: 4, total_tokens: 14 };
    assert.deepEqual(first.usage, expectedUsage);
    assert.deepEqual(second.usage, expectedUsage);
    assert.ok(requests.length >= 2);
    assert.match(String(requests[0].messages[0].content), /Fabric Qwen instruction probe/);
    assert.match(JSON.stringify(requests.at(-1).messages), /Remember token cobalt/);
  } finally {
    await session?.close();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }
});

async function withFakeQwen(usageForRequest, runtimeId, run) {
  const requests = [];
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    requests.push(JSON.parse(Buffer.concat(chunks).toString("utf8")));
    const usage = usageForRequest(requests.length);
    response.writeHead(200, { "content-type": "text/event-stream" });
    const common = { id: "chatcmpl-usage", object: "chat.completion.chunk", created: 0, model: "fabric-test-model" };
    response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: { role: "assistant", content: "done" }, finish_reason: null }] })}\n\n`);
    response.write(`data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], ...(usage === undefined ? {} : { usage }) })}\n\n`);
    response.end("data: [DONE]\n\n");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  let session;
  try {
    const port = server.address().port;
    session = await new QwenSdkSessionFactory().create({
      agentName: runtimeId, baseDir: process.cwd(),
      config: { models: { default: { provider: "openai", model: "fabric-test-model", api_key_env: "FABRIC_QWEN_TEST_KEY", base_url: `http://127.0.0.1:${port}/v1` } } },
      runtimeContext: {
        artifacts: {}, environment: { control_location: "external_control", env: { FABRIC_QWEN_TEST_KEY: "local-test-key" }, environment_id: "env-1", ownership: "caller_owned", provider: "local", workspace: process.cwd() },
        invocation_id: "start", request_id: "start", runtime_id: runtimeId,
      },
    });
    await run(session, requests);
  } finally {
    await session?.close();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }
}

const reportedUsage = { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7, prompt_tokens_details: { cached_tokens: 3 } };

function compressionRequests(requests) {
  return requests.filter((body) => JSON.stringify(body.messages).includes("summarizes a conversation"));
}

// A normal run: every response reports usage, and only the second prompt
// hits the prompt cache.
test("Qwen SDK reports known counts for every prompt of a normal run", { timeout: 60_000 }, async () => {
  let cached = 0;
  const usageForRequest = () => ({ prompt_tokens: 5, completion_tokens: 2, total_tokens: 7, prompt_tokens_details: { cached_tokens: cached } });
  await withFakeQwen(usageForRequest, "runtime-normal-run", async (session, requests) => {
    const calls = [];
    const turns = [];
    for (const [prompt, cachedPerCall] of [["First question", 0], ["Second question", 3], ["Third question", 0]]) {
      cached = cachedPerCall;
      const before = requests.length;
      turns.push(await session.prompt(prompt));
      calls.push(requests.length - before);
    }
    const counts = (n) => ({ input_tokens: 5 * n, input_tokens_include_cache: true, output_tokens: 2 * n, total_tokens: 7 * n });
    // Qwen Code cannot tell a reported cache 0 from an omitted one, so cache
    // reads are known only when they grew.
    assert.deepEqual(turns[0].usage, counts(calls[0]), JSON.stringify(turns[0]));
    assert.deepEqual(turns[1].usage, { ...counts(calls[1]), cached_input_tokens: 3 * calls[1] }, JSON.stringify(turns[1]));
    assert.deepEqual(turns[2].usage, counts(calls[2]), JSON.stringify(turns[2]));
  });
});

// A provider can report cache reads once and omit them later, and Qwen Code
// records the omission as 0, so earlier cache reads prove nothing.
for (const clear of [false, true]) {
  test(`Qwen SDK leaves cache reads unknown when a later prompt omits them${clear ? " after /clear" : ""}`, { timeout: 60_000 }, async () => {
    let details = { prompt_tokens_details: { cached_tokens: 3 } };
    const usageForRequest = () => ({ prompt_tokens: 5, completion_tokens: 2, total_tokens: 7, ...details });
    await withFakeQwen(usageForRequest, `runtime-cache-omitted-${clear}`, async (session, requests) => {
      const first = await session.prompt("First question");
      assert.ok(first.usage.cached_input_tokens > 0, JSON.stringify(first));
      if (clear) assert.equal((await session.prompt("/clear")).usage, undefined);
      details = {};
      const before = requests.length;
      const later = await session.prompt("Second question");
      const calls = requests.length - before;
      assert.deepEqual(
        later.usage,
        { input_tokens: 5 * calls, input_tokens_include_cache: true, output_tokens: 2 * calls, total_tokens: 7 * calls },
        JSON.stringify(later),
      );
    });
  });
}

test("Qwen SDK leaves usage unknown for the invocation that resets its counters", { timeout: 45_000 }, async () => {
  await withFakeQwen(() => ({ prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 }), "runtime-reset", async (session) => {
    const first = await session.prompt("First question");
    // Qwen Code's /clear starts a new session and resets its usage counters.
    const cleared = await session.prompt("/clear");
    const afterClear = await session.prompt("Second question");
    // The provider sent no cache count, so cached input stays unknown.
    const expectedUsage = { input_tokens: 10, input_tokens_include_cache: true, output_tokens: 4, total_tokens: 14 };
    assert.deepEqual(first.usage, expectedUsage);
    assert.equal(cleared.usage, undefined, JSON.stringify(cleared));
    assert.equal(afterClear.text, "done", JSON.stringify(afterClear));
    assert.deepEqual(afterClear.usage, expectedUsage);
  });
});

test("Qwen SDK reports no usage when the provider reports none", { timeout: 45_000 }, async () => {
  await withFakeQwen(() => undefined, "runtime-no-usage", async (session) => {
    const turn = await session.prompt("Question without usage");
    assert.equal(turn.text, "done", JSON.stringify(turn));
    assert.equal(turn.usage, undefined, JSON.stringify(turn));
  });
});

// Qwen Code reports /compress with zero turns and zero API duration, although
// it sends the conversation to the model for a summary.
test("Qwen SDK reports usage from /compress model calls", { timeout: 45_000 }, async () => {
  await withFakeQwen(() => reportedUsage, "runtime-compress", async (session, requests) => {
    await session.prompt("First question");
    const compressed = await session.prompt("/compress");
    // Qwen Code may retry the summary, so expect the counts of every request.
    const calls = compressionRequests(requests).length;
    assert.ok(calls >= 1, "expected /compress to call the model");
    assert.deepEqual(
      compressed.usage,
      { input_tokens: 5 * calls, input_tokens_include_cache: true, cached_input_tokens: 3 * calls, output_tokens: 2 * calls, total_tokens: 7 * calls },
      JSON.stringify(compressed),
    );
  });
});

test("Qwen SDK leaves /compress usage unknown when the provider reports none", { timeout: 45_000 }, async () => {
  await withFakeQwen(() => undefined, "runtime-compress-no-usage", async (session, requests) => {
    await session.prompt("First question");
    const compressed = await session.prompt("/compress");
    assert.ok(compressionRequests(requests).length >= 1, "expected /compress to call the model");
    assert.equal(compressed.usage, undefined, JSON.stringify(compressed));
  });
});

// Qwen Code's cumulative counters do not show which responses lacked usage,
// so an invocation where only some responses report tokens is undercounted.
test("Qwen SDK reports only the counts of responses that sent usage", { timeout: 45_000 }, async () => {
  await withFakeQwen((request) => (request === 1 ? reportedUsage : undefined), "runtime-mixed-usage", async (session, requests) => {
    const turn = await session.prompt("Question with partial usage");
    assert.ok(requests.length >= 2, `expected more than one model request, got ${requests.length}`);
    assert.deepEqual(
      turn.usage,
      { input_tokens: 5, input_tokens_include_cache: true, cached_input_tokens: 3, output_tokens: 2, total_tokens: 7 },
      JSON.stringify(turn),
    );
  });
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

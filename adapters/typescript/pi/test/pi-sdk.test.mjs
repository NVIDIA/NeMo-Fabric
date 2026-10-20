// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { access, mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createHash, randomUUID } from "node:crypto";
import { createServer } from "node:http";

import {
  addEventUsage,
  addUsage,
  noteReportedFields,
  buildCatalogModel,
  endedCallCounts,
  modelAwareCompactionReserveTokens,
  PiSdkSessionFactory,
  recordReportedCounts,
  reportedCountsFrom,
  resolveCustomTools,
  summaryCallCount,
  selectPiMcpServers,
  withCustomBaseUrl,
} from "../dist/pi-sdk.js";
import { PiAdapterRuntime } from "../dist/runtime.js";

const [major, minor] = process.versions.node.split(".").map(Number);
const supportsPi = major > 22 || (major === 22 && minor >= 19);

async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  assert.notEqual(typeof address, "string");
  return `http://127.0.0.1:${address.port}`;
}

async function close(server) {
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
}

test("uses standard content when replaying reasoning through a custom model proxy", () => {
  const catalogModel = {
    api: "openai-completions",
    baseUrl: "https://integrate.api.nvidia.com/v1",
    compat: { supportsStore: false },
  };

  assert.deepEqual(withCustomBaseUrl(catalogModel, "http://model-proxy:10240"), {
    api: "openai-completions",
    baseUrl: "http://model-proxy:10240",
    compat: { supportsStore: false, requiresThinkingAsText: true },
  });
  assert.strictEqual(withCustomBaseUrl(catalogModel, undefined), catalogModel);
  assert.strictEqual(withCustomBaseUrl(catalogModel, ""), catalogModel);
  assert.deepEqual(withCustomBaseUrl(catalogModel, "http://model-proxy:10240", false), {
    api: "openai-completions",
    baseUrl: "https://integrate.api.nvidia.com/v1",
    compat: { supportsStore: false, requiresThinkingAsText: true },
  });
});

test("reserves output capacity without consuming more than half the context window", () => {
  assert.equal(modelAwareCompactionReserveTokens(16_384, 65_536, 262_144), 65_536);
  assert.equal(modelAwareCompactionReserveTokens(65_536, 32_768, 262_144), 65_536);
  assert.equal(modelAwareCompactionReserveTokens(16_384, 131_072, 131_072), 65_536);
  assert.equal(modelAwareCompactionReserveTokens(16_384, 65_536, 0), 65_536);
});

// A model Pi already ships, as getModel() would resolve it: every native field populated.
const NATIVE_MODEL = {
  id: "gpt-4.1-mini",
  name: "GPT-4.1 mini",
  api: "openai-responses",
  provider: "openai",
  baseUrl: "https://api.openai.com/v1",
  reasoning: true,
  input: ["text", "image"],
  cost: { input: 0.4, output: 1.6, cacheRead: 0.1, cacheWrite: 0 },
  contextWindow: 1_000_000,
  maxTokens: 32_768,
};

test("buildCatalogModel preserves every native field of a known model when nothing is overridden", () => {
  // Selecting a known model with no extensions and no base_url must not change any property.
  const entry = buildCatalogModel({ provider: "openai", model: "gpt-4.1-mini", api_key_env: "K" }, NATIVE_MODEL);
  assert.equal(entry.api, NATIVE_MODEL.api);
  assert.equal(entry.baseUrl, NATIVE_MODEL.baseUrl);
  assert.equal(entry.name, NATIVE_MODEL.name);
  assert.equal(entry.reasoning, NATIVE_MODEL.reasoning);
  assert.deepEqual(entry.input, NATIVE_MODEL.input);
  assert.deepEqual(entry.cost, NATIVE_MODEL.cost);
  assert.equal(entry.contextWindow, NATIVE_MODEL.contextWindow);
  assert.equal(entry.maxTokens, NATIVE_MODEL.maxTokens);
});

test("buildCatalogModel overrides only the explicitly supplied fields of a known model", () => {
  // Supply base_url + one extension; every other native field must survive unchanged.
  const entry = buildCatalogModel(
    {
      provider: "openai",
      model: "gpt-4.1-mini",
      api_key_env: "K",
      base_url: "https://proxy.example.test/v1",
      extensions: { max_tokens: 4096 },
    },
    NATIVE_MODEL,
  );
  assert.equal(entry.baseUrl, "https://proxy.example.test/v1"); // overridden
  assert.equal(entry.maxTokens, 4096); // overridden
  // Everything else preserved from the native entry.
  assert.equal(entry.api, NATIVE_MODEL.api);
  assert.equal(entry.name, NATIVE_MODEL.name);
  assert.equal(entry.reasoning, NATIVE_MODEL.reasoning);
  assert.deepEqual(entry.input, NATIVE_MODEL.input);
  assert.deepEqual(entry.cost, NATIVE_MODEL.cost);
  assert.equal(entry.contextWindow, NATIVE_MODEL.contextWindow);
});

test("buildCatalogModel requires context_window and max_tokens for an unknown model", () => {
  // No base (Pi does not know the model) and only api/base_url supplied: context_window and
  // max_tokens have no usable Pi default, so a complete definition is required rather than
  // silently registering a model with undefined (compaction-breaking) limits.
  assert.throws(
    () =>
      buildCatalogModel(
        { provider: "gw", model: "gw/model", api_key_env: "K", base_url: "https://gw/v1", extensions: { api: "openai-completions" } },
        undefined,
      ),
    (error) =>
      error.code === "pi_model_extensions_invalid" &&
      error.message.includes("context_window") &&
      error.message.includes("max_tokens"),
  );
});

test("buildCatalogModel accepts a complete unknown-model definition and invents no optional defaults", () => {
  // api + base_url + context_window + max_tokens is complete; optional fields (cost, reasoning,
  // input) stay unset so Pi defaults them — the adapter adds none.
  const entry = buildCatalogModel(
    {
      provider: "gw",
      model: "gw/model",
      api_key_env: "K",
      base_url: "https://gw/v1",
      extensions: { api: "openai-completions", context_window: 32_000, max_tokens: 4_096 },
    },
    undefined,
  );
  assert.equal(entry.api, "openai-completions");
  assert.equal(entry.baseUrl, "https://gw/v1");
  assert.equal(entry.contextWindow, 32_000);
  assert.equal(entry.maxTokens, 4_096);
  assert.equal(entry.cost, undefined);
  assert.equal(entry.reasoning, undefined);
  assert.equal(entry.input, undefined);
});

test("maps normalized MCP servers and tool filters to native Pi MCP configuration", () => {
  const servers = selectPiMcpServers(
    {
      mcp: {
        servers: {
          local: {
            transport: "stdio",
            url: "node",
            args: ["server.mjs"],
            env: { LITERAL: "$VALUE", COMMAND: "!do-not-run" },
            allowed_tools: ["read_issue", "edit_issue"],
            blocked_tools: ["edit_issue"],
          },
          remote: {
            transport: "streamable-http",
            url: "https://mcp.example.com/api",
            custom_headers: { Authorization: "Bearer ${TOKEN}" },
            blocked_tools: ["delete_issue"],
          },
        },
      },
    },
    { TOKEN: "secret$value" },
    {},
  );

  assert.deepEqual({ ...servers.local }, {
    type: "stdio",
    command: "node",
    args: ["server.mjs"],
    env: { LITERAL: "$VALUE", COMMAND: "!do-not-run" },
    exposure: "hidden",
    toolExposure: { read_issue: "direct", edit_issue: "hidden" },
  });
  assert.deepEqual({ ...servers.remote }, {
    type: "http",
    url: "https://mcp.example.com/api",
    headers: { Authorization: "Bearer secret$value" },
    exposure: "direct",
    toolExposure: { delete_issue: "hidden" },
  });
});

test("rejects normalized MCP fields that Pi cannot apply", () => {
  assert.throws(
    () => selectPiMcpServers({ mcp: { extensions: {} } }, {}, {}),
    (error) => error.code === "pi_mcp_extensions_unsupported",
  );
  assert.throws(
    () => selectPiMcpServers({ mcp: { servers: { remote: {
      transport: "streamable-http",
      url: "https://mcp.example.com",
      args: ["ignored"],
    } } } }, {}, {}),
    (error) => error.code === "pi_mcp_invalid_server",
  );
  assert.throws(
    () => selectPiMcpServers({ mcp: { servers: { authenticated: {
      transport: "streamable-http",
      url: "https://mcp.example.com",
      authentication: { type: "oauth" },
    } } } }, {}, {}),
    (error) => error.code === "pi_mcp_authentication_unsupported",
  );
  assert.throws(
    () => selectPiMcpServers({ mcp: { servers: { legacy: {
      transport: "sse",
      url: "https://mcp.example.com",
    } } } }, {}, {}),
    (error) => error.code === "pi_mcp_transport_unsupported",
  );
  assert.throws(
    () => selectPiMcpServers({ mcp: { servers: { "invalid name": {
      transport: "stdio",
      url: "server",
    } } } }, {}, {}),
    (error) => error.code === "pi_mcp_invalid_server",
  );
  assert.throws(
    () => selectPiMcpServers({ mcp: { servers: { patterned: {
      transport: "stdio",
      url: "server",
      allowed_tools: ["read_*"],
    } } } }, {}, {}),
    (error) => error.code === "pi_mcp_tool_pattern_unsupported",
  );
  assert.throws(
    () => selectPiMcpServers({ mcp: { servers: { headers: {
      transport: "streamable-http",
      url: "https://mcp.example.com",
      custom_headers: { Authorization: "first", authorization: "second" },
    } } } }, {}, {}),
    (error) => error.code === "pi_mcp_invalid_header",
  );
});

async function loadFabricConfiguredStdioServer() {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-mcp-")));
  const marker = join(workspace, "mcp-connected.txt");
  const serverPath = join(workspace, "mcp-server.mjs");
  const providerRequests = [];
  const provider = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    providerRequests.push(JSON.parse(Buffer.concat(chunks).toString("utf8")));
    const common = { id: "chatcmpl-pi-mcp", object: "chat.completion.chunk", created: 0, model: "openai/gpt-oss-20b" };
    const chunk = (delta, finishReason = null) =>
      `data: ${JSON.stringify({ ...common, choices: [{ index: 0, delta, finish_reason: finishReason }] })}\n\n`;
    response.writeHead(200, { "content-type": "text/event-stream" });
    if (providerRequests.length === 1) {
      response.end(
        `${chunk({ role: "assistant", tool_calls: [{ index: 0, id: "call-echo", type: "function", function: { name: "mcp__local__echo", arguments: JSON.stringify({ text: "nonce" }) } }] })}${chunk({}, "tool_calls")}data: [DONE]\n\n`,
      );
    } else {
      response.end(`${chunk({ role: "assistant", content: "echo:nonce" })}${chunk({}, "stop")}data: [DONE]\n\n`);
    }
  });
  const providerUrl = await listen(provider);
  await writeFile(
    serverPath,
    `import { writeFile } from "node:fs/promises";
let buffer = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  buffer += chunk;
  let newline;
  while ((newline = buffer.indexOf("\\n")) >= 0) {
    const line = buffer.slice(0, newline);
    buffer = buffer.slice(newline + 1);
    if (!line) continue;
    const message = JSON.parse(line);
    if (message.id === undefined) continue;
    let result;
    if (message.method === "initialize") {
      result = {
        protocolVersion: message.params.protocolVersion,
        capabilities: { tools: {}, resources: {} },
        serverInfo: { name: "fabric-test", version: "1.0.0" }
      };
    } else if (message.method === "tools/list") {
      result = { tools: [{ name: "echo", description: "Echo text", inputSchema: { type: "object", properties: { text: { type: "string" } } } }] };
      void writeFile(process.env.MARKER_FILE, "connected", "utf8");
    } else if (message.method === "tools/call") {
      result = { content: [{ type: "text", text: "echo:" + message.params.arguments.text }] };
    } else if (message.method === "resources/list") {
      result = { resources: [] };
    } else if (message.method === "resources/templates/list") {
      result = { resourceTemplates: [] };
    } else {
      result = {};
    }
    process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: message.id, result }) + "\\n");
  }
});
`,
    "utf8",
  );

  let handle;
  try {
    handle = await new PiSdkSessionFactory().create({
      agentName: "pi-mcp-test",
      baseDir: workspace,
      config: {
        models: {
          default: {
            api_key_env: "TEST_API_KEY",
            base_url: `${providerUrl}/v1`,
            model: "openai/gpt-oss-20b",
            provider: "nvidia",
          },
        },
        mcp: {
          servers: {
            local: {
              transport: "stdio",
              url: process.execPath,
              args: [serverPath],
              env: { MARKER_FILE: marker },
            },
          },
        },
        tools: { enabled: null },
      },
      runtimeContext: {
        artifacts: {},
        environment: {
          control_location: "external_control",
          env: { TEST_API_KEY: "not-a-real-key" },
          environment_id: "environment-1",
          ownership: "caller_owned",
          provider: "local",
          workspace,
        },
        invocation_id: "start",
        request_id: "request-start",
        runtime_id: "runtime-1",
      },
    });
    const deadline = Date.now() + 5_000;
    while (true) {
      try {
        assert.equal(await readFile(marker, "utf8"), "connected");
        break;
      } catch (error) {
        if (Date.now() >= deadline) throw error;
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
    }
    const outcome = await handle.prompt("Call the echo tool with nonce.");
    assert.equal(outcome.text, "echo:nonce");
    assert.equal(providerRequests.length, 2);
    assert.match(JSON.stringify(providerRequests[1]), /echo:nonce/);
  } finally {
    await handle?.stop();
    await close(provider);
    await rm(workspace, { recursive: true, force: true });
  }
}

test(
  "loads a Fabric-configured stdio server through Pi's native MCP extension",
  { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
  loadFabricConfiguredStdioServer,
);

test(
  "fails startup when a configured Pi MCP server cannot connect",
  { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
  async () => {
    const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-mcp-failure-")));
    try {
      await assert.rejects(
        new PiSdkSessionFactory().create({
          agentName: "pi-mcp-failure-test",
          baseDir: workspace,
          config: {
            models: { default: { api_key_env: "TEST_API_KEY", model: "gpt-4.1-mini", provider: "openai" } },
            mcp: { servers: { unavailable: { transport: "stdio", url: "not-a-real-mcp-command" } } },
            tools: { enabled: [] },
          },
          runtimeContext: {
            artifacts: {},
            environment: {
              control_location: "external_control",
              env: { TEST_API_KEY: "not-a-real-key" },
              environment_id: "environment-1",
              ownership: "caller_owned",
              provider: "local",
              workspace,
            },
            invocation_id: "start",
            request_id: "request-start",
            runtime_id: "runtime-mcp-failure",
          },
        }),
        (error) => error.code === "pi_mcp_connection_failed",
      );
    } finally {
      await rm(workspace, { recursive: true, force: true });
    }
  },
);

test(
  "does not send ambient Pi OAuth credentials to Fabric-configured MCP servers",
  { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
  async () => {
    const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-mcp-oauth-")));
    const ambientAgentDir = await realpath(await mkdtemp(join(tmpdir(), "ambient-pi-profile-")));
    const authorizations = [];
    const endpoint = createServer(async (request, response) => {
      if (request.method === "GET") {
        response.writeHead(405).end();
        return;
      }
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      const message = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      authorizations.push(request.headers.authorization);
      if (message.id === undefined) {
        response.writeHead(202).end();
        return;
      }
      const result = message.method === "initialize"
        ? {
            protocolVersion: message.params.protocolVersion,
            capabilities: { tools: {} },
            serverInfo: { name: "fabric-http-test", version: "1.0.0" },
          }
        : message.method === "tools/list"
          ? { tools: [] }
          : message.method === "resources/list"
            ? { resources: [] }
            : { resourceTemplates: [] };
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ jsonrpc: "2.0", id: message.id, result }));
    });
    const endpointUrl = `${await listen(endpoint)}/mcp`;
    const previousAgentDir = process.env.PI_CODING_AGENT_DIR;
    let handle;
    try {
      await writeFile(
        join(ambientAgentDir, "mcp-auth.json"),
        `${JSON.stringify({
          [endpointUrl]: {
            serverUrl: endpointUrl,
            tokens: { access_token: "ambient-token", token_type: "Bearer" },
          },
        })}\n`,
        "utf8",
      );
      process.env.PI_CODING_AGENT_DIR = ambientAgentDir;
      handle = await new PiSdkSessionFactory().create({
        agentName: "pi-mcp-oauth-test",
        baseDir: workspace,
        config: {
          models: { default: { api_key_env: "TEST_API_KEY", model: "gpt-4.1-mini", provider: "openai" } },
          mcp: { servers: { remote: { transport: "streamable-http", url: endpointUrl } } },
          tools: { enabled: [] },
        },
        runtimeContext: {
          artifacts: {},
          environment: {
            control_location: "external_control",
            env: { TEST_API_KEY: "not-a-real-key" },
            environment_id: "environment-1",
            ownership: "caller_owned",
            provider: "local",
            workspace,
          },
          invocation_id: "start",
          request_id: "request-start",
          runtime_id: "runtime-mcp-oauth",
        },
      });
      assert.ok(authorizations.length > 0);
      assert.ok(authorizations.every((value) => value === undefined));
      await assert.rejects(access(join(ambientAgentDir, "mcp.log")));
    } finally {
      if (previousAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
      else process.env.PI_CODING_AGENT_DIR = previousAgentDir;
      await handle?.stop();
      await close(endpoint);
      await rm(ambientAgentDir, { recursive: true, force: true });
      await rm(workspace, { recursive: true, force: true });
    }
  },
);

test("rejects append system instructions before loading the Pi harness", async () => {
  const factory = new PiSdkSessionFactory();

  await assert.rejects(
    factory.create({
      agentName: "pi-test",
      baseDir: "/tmp",
      config: {
        instructions: {
          system: {
            content: "Follow repository policy.",
            mode: "append",
          },
        },
      },
      runtimeContext: {},
    }),
    (error) =>
      error.code === "unsupported_system_instruction_mode" && error.metadata.field === "instructions.system.mode",
  );
});

test("resolves and executes a workspace TypeScript tool factory", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-tool-")));
  try {
    await writeFile(
      join(workspace, "echo-tool.ts"),
      `export default function ({ name, settings }: { name: string; settings: { prefix?: string } }) {
  return {
    name,
    label: "Echo",
    description: "Echo configured text",
    parameters: {
      type: "object",
      properties: { text: { type: "string" } },
      required: ["text"],
      additionalProperties: false
    },
    async execute(_toolCallId: string, params: { text: string }) {
      return {
        content: [{ type: "text", text: (settings.prefix ?? "") + params.text }],
        details: {}
      };
    }
  };
}
`,
      "utf8",
    );

    const [tool] = await resolveCustomTools(workspace, {
      echo: {
        kind: "module",
        ref: "echo-tool.ts",
        settings: { prefix: "configured: " },
      },
    });

    assert.equal(tool.name, "echo");
    const result = await tool.execute("call-1", { text: "hello" }, undefined, undefined, undefined);
    assert.equal(result.content[0].text, "configured: hello");
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("rejects custom tools that collide with Pi built-ins", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-tool-collision-")));
  try {
    await assert.rejects(
      resolveCustomTools(workspace, {
        read: { kind: "module", ref: "unused.js" },
      }),
      (error) => error.code === "pi_tool_collision",
    );
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("requires the factory result name to match the normalized definition name", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-tool-name-")));
  try {
    await writeFile(
      join(workspace, "wrong-name.js"),
      `export default function () {
  return {
    name: "other",
    label: "Other",
    description: "Wrong name",
    parameters: { type: "object", properties: {} },
    async execute() { return { content: [], details: {} }; }
  };
}
`,
      "utf8",
    );

    await assert.rejects(
      resolveCustomTools(workspace, {
        expected: { kind: "module", ref: "wrong-name.js" },
      }),
      (error) => error.code === "pi_tool_factory_invalid",
    );
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("rejects unsupported custom tool definition kinds", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-tool-kind-")));
  try {
    await assert.rejects(
      resolveCustomTools(workspace, {
        unsupported: { kind: "inline", ref: "unused.js" },
      }),
      (error) => error.code === "pi_tool_kind_unsupported",
    );
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("reports a missing custom tool module", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-tool-missing-")));
  try {
    await assert.rejects(
      resolveCustomTools(workspace, {
        missing: { kind: "module", ref: "missing.js" },
      }),
      (error) => error.code === "pi_tool_module_not_found",
    );
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("rejects custom tool modules outside the workspace", async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-tool-outside-")));
  const workspace = join(root, "workspace");
  try {
    await mkdir(workspace);
    await writeFile(join(root, "outside.js"), "export default function () {}\n", "utf8");

    await assert.rejects(
      resolveCustomTools(workspace, {
        outside: { kind: "module", ref: "../outside.js" },
      }),
      (error) => error.code === "pi_tool_module_outside_workspace",
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("reports a missing named factory export", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-tool-export-")));
  try {
    await writeFile(join(workspace, "exports.js"), "export function available() {}\n", "utf8");

    await assert.rejects(
      resolveCustomTools(workspace, {
        missing: { kind: "module", ref: "exports.js#missing" },
      }),
      (error) => error.code === "pi_tool_factory_missing",
    );
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("resolves a custom tool through a named export fragment", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-tool-fragment-")));
  try {
    await writeFile(
      join(workspace, "named-tool.js"),
      `export function createTool({ name }) {
  return {
    name,
    label: "Named Tool",
    description: "Loaded through a named export",
    parameters: { type: "object", properties: {} },
    async execute() { return { content: [], details: {} }; }
  };
}
`,
      "utf8",
    );

    const [tool] = await resolveCustomTools(workspace, {
      named: { kind: "module", ref: "named-tool.js#createTool" },
    });

    assert.equal(tool.name, "named");
    assert.equal(tool.label, "Named Tool");
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("loads the Relay extension explicitly and drains session shutdown before gateway stop", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-hooks-")));
  const requests = [];
  const server = createServer((request, response) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => {
      body += chunk;
    });
    request.on("end", () => {
      requests.push({
        body: JSON.parse(body),
        sessionId: request.headers["x-nemo-relay-session-id"],
        url: request.url,
      });
      response.writeHead(204).end();
    });
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  assert.notEqual(typeof address, "string");
  const gatewayUrl = `http://127.0.0.1:${address.port}`;
  const extensionPath = join(workspace, "relay-extension.js");
  const previousGatewayUrl = process.env.NEMO_RELAY_PI_GATEWAY_URL;
  try {
    await writeFile(
      extensionPath,
      `export default function (pi) {
  const post = (hook_event_name) => fetch(process.env.NEMO_RELAY_PI_GATEWAY_URL + "/hooks/pi", {
    method: "POST",
    headers: { "content-type": "application/json", "x-nemo-relay-session-id": "fabric-test-session" },
    body: JSON.stringify({ hook_event_name })
  });
  pi.on("session_start", async () => { await post("session_start"); });
  pi.on("session_shutdown", async () => { await post("session_shutdown"); });
}
`,
      "utf8",
    );
    process.env.NEMO_RELAY_PI_GATEWAY_URL = gatewayUrl;
    let selectedModel;
    let gatewayStopped = false;
    const relay = {
      extensionPath,
      pluginConfig: { version: 1, components: [] },
      async output() {
        return {};
      },
      async stop() {
        assert.deepEqual(
          requests.map((entry) => entry.body.hook_event_name),
          ["session_start", "session_shutdown"],
        );
        gatewayStopped = true;
      },
    };
    const factory = new PiSdkSessionFactory({
      async start(_input, model) {
        selectedModel = model;
        return relay;
      },
    });
    const runtime = new PiAdapterRuntime(factory);
    await runtime.start({
      agentName: "pi-relay-test",
      baseDir: workspace,
      config: {
        harness: { settings: { relay_extension_path: extensionPath } },
        models: {
          default: {
            api_key_env: "TEST_API_KEY",
            base_url: "https://proxy.example.test/v1",
            model: "gpt-4.1-mini",
            provider: "openai",
          },
        },
        tools: { enabled: [] },
      },
      runtimeContext: {
        artifacts: {},
        environment: {
          control_location: "external_control",
          env: { TEST_API_KEY: "not-a-real-key" },
          environment_id: "environment-1",
          ownership: "caller_owned",
          provider: "local",
          workspace,
        },
        invocation_id: "start",
        request_id: "request-start",
        runtime_id: "runtime-1",
        telemetry: { relay_enabled: true },
      },
    });
    await runtime.stop();

    assert.equal(selectedModel.api, "openai-responses");
    assert.equal(selectedModel.baseUrl, "https://proxy.example.test/v1");
    assert.equal(gatewayStopped, true);
    assert.deepEqual(requests, [
      {
        body: { hook_event_name: "session_start" },
        sessionId: "fabric-test-session",
        url: "/hooks/pi",
      },
      {
        body: { hook_event_name: "session_shutdown" },
        sessionId: "fabric-test-session",
        url: "/hooks/pi",
      },
    ]);
  } finally {
    if (previousGatewayUrl === undefined) {
      delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
    } else {
      process.env.NEMO_RELAY_PI_GATEWAY_URL = previousGatewayUrl;
    }
    await new Promise((resolve) => server.close(resolve));
    await rm(workspace, { recursive: true, force: true });
  }
});

test("reports an adapter-injected Relay extension load failure separately", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-extension-error-")));
  const extensionPath = join(workspace, "relay-extension.js");
  await writeFile(extensionPath, "export default {\n", "utf8");
  let relayStopped = false;
  const factory = new PiSdkSessionFactory({
    async start() {
      return {
        extensionPath,
        pluginConfig: { version: 1, components: [] },
        atifMatchers: [],
        async output() {
          return {};
        },
        async stop() {
          relayStopped = true;
        },
      };
    },
  });
  try {
    await assert.rejects(
      factory.create({
        agentName: "pi-relay-test",
        baseDir: workspace,
        config: {
          harness: { settings: { relay_extension_path: extensionPath } },
          models: {
            default: {
              api_key_env: "TEST_API_KEY",
              model: "gpt-4.1-mini",
              provider: "openai",
            },
          },
          tools: { enabled: [] },
        },
        runtimeContext: {
          artifacts: {},
          environment: {
            control_location: "external_control",
            env: { TEST_API_KEY: "not-a-real-key" },
            environment_id: "environment-1",
            ownership: "caller_owned",
            provider: "local",
            workspace,
          },
          invocation_id: "start",
          request_id: "request-start",
          runtime_id: "runtime-1",
          telemetry: { relay_enabled: true },
        },
      }),
      (error) =>
        error.code === "pi_relay_extension_load_failed" &&
        error.message.includes("NeMo Relay 0.9 or 0.10 release") &&
        error.metadata.relay_error.length > 0,
    );
    assert.equal(relayStopped, true);
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("reports Relay extension tool conflicts as Pi tool collisions", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-tool-collision-")));
  const userExtensionPath = join(workspace, "user-extension.js");
  const relayExtensionPath = join(workspace, "relay-extension.js");
  const extensionSource = `export default function (pi) {
  pi.registerTool({
    name: "duplicate_tool",
    label: "Duplicate",
    description: "A duplicate test tool",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    async execute() { return { content: [], details: {} }; }
  });
}
`;
  await Promise.all([
    writeFile(userExtensionPath, extensionSource, "utf8"),
    writeFile(relayExtensionPath, extensionSource, "utf8"),
  ]);
  let relayStopped = false;
  const factory = new PiSdkSessionFactory({
    async start() {
      return {
        extensionPath: relayExtensionPath,
        pluginConfig: { version: 1, components: [] },
        atifMatchers: [],
        async output() {
          return {};
        },
        async stop() {
          relayStopped = true;
        },
      };
    },
  });
  try {
    await assert.rejects(
      factory.create({
        agentName: "pi-relay-test",
        baseDir: workspace,
        config: {
          harness: { settings: { extensions: ["user-extension.js"] } },
          models: {
            default: {
              api_key_env: "TEST_API_KEY",
              model: "gpt-4.1-mini",
              provider: "openai",
            },
          },
          tools: { enabled: [] },
        },
        runtimeContext: {
          artifacts: {},
          environment: {
            control_location: "external_control",
            env: { TEST_API_KEY: "not-a-real-key" },
            environment_id: "environment-1",
            ownership: "caller_owned",
            provider: "local",
            workspace,
          },
          invocation_id: "start",
          request_id: "request-start",
          runtime_id: "runtime-1",
          telemetry: { relay_enabled: true },
        },
      }),
      (error) => error.code === "pi_tool_collision" && error.metadata.tool === "duplicate_tool",
    );
    assert.equal(relayStopped, true);
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("includes Pi flag conflict diagnostics in extension errors", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-flag-collision-")));
  const userExtensionPath = join(workspace, "user-extension.js");
  const relayExtensionPath = join(workspace, "relay-extension.js");
  const extensionSource = `export default function (pi) {
  pi.registerFlag("duplicate-flag", {
    description: "A duplicate test flag",
    type: "boolean",
    default: false
  });
}
`;
  await Promise.all([
    writeFile(userExtensionPath, extensionSource, "utf8"),
    writeFile(relayExtensionPath, extensionSource, "utf8"),
  ]);
  let relayStopped = false;
  const factory = new PiSdkSessionFactory({
    async start() {
      return {
        extensionPath: relayExtensionPath,
        pluginConfig: { version: 1, components: [] },
        atifMatchers: [],
        async output() {
          return {};
        },
        async stop() {
          relayStopped = true;
        },
      };
    },
  });
  try {
    await assert.rejects(
      factory.create({
        agentName: "pi-relay-test",
        baseDir: workspace,
        config: {
          harness: { settings: { extensions: ["user-extension.js"] } },
          models: {
            default: {
              api_key_env: "TEST_API_KEY",
              model: "gpt-4.1-mini",
              provider: "openai",
            },
          },
          tools: { enabled: [] },
        },
        runtimeContext: {
          artifacts: {},
          environment: {
            control_location: "external_control",
            env: { TEST_API_KEY: "not-a-real-key" },
            environment_id: "environment-1",
            ownership: "caller_owned",
            provider: "local",
            workspace,
          },
          invocation_id: "start",
          request_id: "request-start",
          runtime_id: "runtime-1",
          telemetry: { relay_enabled: true },
        },
      }),
      (error) =>
        error.code === "pi_extension_load_failed" &&
        error.message.includes("conflict") &&
        error.metadata.extension_error.includes('Flag "--duplicate-flag" conflicts with') &&
        error.metadata.extension_paths.includes(relayExtensionPath),
    );
    assert.equal(relayStopped, true);
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

// --- Model catalog registration (gateway-served models) ---------------------

// A complete Pi-metadata block for a model Pi does not ship. The adapter invents no defaults, so
// every field Pi requires must be supplied for an unknown gateway model; tests reuse this and
// omit one field when exercising a required-field rejection.
const GATEWAY_EXTENSIONS = {
  api: "openai-completions",
  context_window: 32000,
  max_tokens: 4096,
  cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
  reasoning: false,
  input: ["text"],
};

function makeRuntimeContext(workspace) {
  return {
    artifacts: {},
    environment: {
      control_location: "external_control",
      env: { TEST_API_KEY: "not-a-real-key" },
      environment_id: "environment-1",
      ownership: "caller_owned",
      provider: "local",
      workspace,
    },
    invocation_id: "start",
    request_id: "request-start",
    runtime_id: "runtime-1",
    telemetry: { relay_enabled: true },
  };
}

function captureModelFactory(capture) {
  return new PiSdkSessionFactory({
    async start(input, model) {
      capture.model = model;
      return {
        extensionPath: input.config.harness?.settings?.relay_extension_path ?? "relay-extension.js",
        pluginConfig: { version: 1, components: [] },
        async output() {
          return {};
        },
        async stop() {},
      };
    },
  });
}

test("inherits a built-in model's native api when extensions.api is omitted", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-catalog-native-")));
  const extensionPath = join(workspace, "relay-extension.js");
  await writeFile(extensionPath, "export default function () {}\n", "utf8");
  const capture = {};
  const factory = captureModelFactory(capture);
  const runtime = new PiAdapterRuntime(factory);
  try {
    await runtime.start({
      agentName: "pi-catalog-native",
      baseDir: workspace,
      config: {
        harness: { settings: { relay_extension_path: extensionPath } },
        models: {
          default: {
            api_key_env: "TEST_API_KEY",
            model: "gpt-4.1-mini",
            provider: "openai",
          },
        },
        tools: { enabled: [] },
      },
      runtimeContext: makeRuntimeContext(workspace),
    });
    await runtime.stop();
    // gpt-4.1-mini is a Pi built-in whose native api is openai-responses; with no settings.api
    // the adapter must NOT force a value — Pi inherits the built-in's api.
    assert.equal(capture.model.api, "openai-responses");
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("resolves a gateway model's api from extensions.api", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-catalog-gateway-")));
  const extensionPath = join(workspace, "relay-extension.js");
  await writeFile(extensionPath, "export default function () {}\n", "utf8");
  const capture = {};
  const factory = captureModelFactory(capture);
  const runtime = new PiAdapterRuntime(factory);
  try {
    await runtime.start({
      agentName: "pi-catalog-gateway",
      baseDir: workspace,
      config: {
        harness: { settings: { relay_extension_path: extensionPath } },
        models: {
          default: {
            api_key_env: "TEST_API_KEY",
            base_url: "https://gateway.example.test/v1",
            model: "nvidia/some-gateway-model",
            provider: "nvidia",
            extensions: { ...GATEWAY_EXTENSIONS, context_window: 32000, max_tokens: 4096 },
          },
        },
        tools: { enabled: [] },
      },
      runtimeContext: makeRuntimeContext(workspace),
    });
    await runtime.stop();
    // The relay factory receives the resolved model's wire api; settings.api must win.
    assert.equal(capture.model.api, "openai-completions");
    assert.equal(capture.model.baseUrl, "https://gateway.example.test/v1");
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("rejects an unknown gateway model that omits context_window/max_tokens", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-catalog-minimal-")));
  const extensionPath = join(workspace, "relay-extension.js");
  await writeFile(extensionPath, "export default function () {}\n", "utf8");
  const factory = captureModelFactory({});
  try {
    await assert.rejects(
      factory.create({
        agentName: "pi-catalog-minimal",
        baseDir: workspace,
        config: {
          harness: { settings: { relay_extension_path: extensionPath } },
          models: {
            default: {
              api_key_env: "TEST_API_KEY",
              base_url: "https://gateway.example.test/v1",
              model: "custom-gateway-model",
              provider: "custom-gateway-provider",
              // Only the wire protocol is supplied. context_window/max_tokens have no usable Pi
              // default, so an incomplete gateway definition must be rejected (not registered
              // with undefined limits that break compaction).
              extensions: { api: "openai-completions" },
            },
          },
          tools: { enabled: [] },
        },
        runtimeContext: makeRuntimeContext(workspace),
      }),
      (error) =>
        error.code === "pi_model_extensions_invalid" &&
        error.message.includes("context_window") &&
        error.message.includes("max_tokens"),
    );
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("rejects an unknown gateway model with no extensions.api", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-catalog-noapi-")));
  const extensionPath = join(workspace, "relay-extension.js");
  await writeFile(extensionPath, "export default function () {}\n", "utf8");
  const factory = captureModelFactory({});
  try {
    await assert.rejects(
      factory.create({
        agentName: "pi-catalog-noapi",
        baseDir: workspace,
        config: {
          harness: { settings: { relay_extension_path: extensionPath } },
          models: {
            default: {
              api_key_env: "TEST_API_KEY",
              base_url: "https://gateway.example.test/v1",
              model: "custom-gateway-model",
              provider: "custom-gateway-provider",
              // Full metadata EXCEPT api, so the only missing required field is the wire protocol.
              extensions: { context_window: 32000, max_tokens: 4096, cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 }, reasoning: false, input: ["text"] },
            },
          },
          tools: { enabled: [] },
        },
        runtimeContext: makeRuntimeContext(workspace),
      }),
      (error) => error.code === "pi_model_api_required" && error.message.includes("extensions.api"),
    );
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("rejects a model whose extensions.api is not a supported protocol", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-catalog-badapi-")));
  const extensionPath = join(workspace, "relay-extension.js");
  await writeFile(extensionPath, "export default function () {}\n", "utf8");
  const factory = captureModelFactory({});
  try {
    await assert.rejects(
      factory.create({
        agentName: "pi-catalog-badapi",
        baseDir: workspace,
        config: {
          harness: { settings: { relay_extension_path: extensionPath } },
          models: {
            default: {
              api_key_env: "TEST_API_KEY",
              base_url: "https://gateway.example.test/v1",
              model: "nvidia/some-gateway-model",
              provider: "nvidia",
              extensions: { api: "not-a-real-api" },
            },
          },
          tools: { enabled: [] },
        },
        runtimeContext: makeRuntimeContext(workspace),
      }),
      (error) => error.code === "pi_model_api_invalid" && error.message.includes("not-a-real-api"),
    );
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("rejects a model whose extensions.cost is incomplete", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-catalog-cost-")));
  const extensionPath = join(workspace, "relay-extension.js");
  await writeFile(extensionPath, "export default function () {}\n", "utf8");
  const factory = captureModelFactory({});
  try {
    await assert.rejects(
      factory.create({
        agentName: "pi-catalog-cost",
        baseDir: workspace,
        config: {
          harness: { settings: { relay_extension_path: extensionPath } },
          models: {
            default: {
              api_key_env: "TEST_API_KEY",
              base_url: "https://gateway.example.test/v1",
              model: "custom-gateway-model",
              provider: "custom-gateway-provider",
              extensions: { ...GATEWAY_EXTENSIONS, cost: {} },
            },
          },
          tools: { enabled: [] },
        },
        runtimeContext: makeRuntimeContext(workspace),
      }),
      (error) => error.code === "pi_model_extensions_invalid" && error.message.includes("cost"),
    );
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("rejects a model whose extensions.max_tokens is not positive", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-catalog-maxtok-")));
  const extensionPath = join(workspace, "relay-extension.js");
  await writeFile(extensionPath, "export default function () {}\n", "utf8");
  const factory = captureModelFactory({});
  try {
    await assert.rejects(
      factory.create({
        agentName: "pi-catalog-maxtok",
        baseDir: workspace,
        config: {
          harness: { settings: { relay_extension_path: extensionPath } },
          models: {
            default: {
              api_key_env: "TEST_API_KEY",
              base_url: "https://gateway.example.test/v1",
              model: "custom-gateway-model",
              provider: "custom-gateway-provider",
              extensions: { ...GATEWAY_EXTENSIONS, max_tokens: 0 },
            },
          },
          tools: { enabled: [] },
        },
        runtimeContext: makeRuntimeContext(workspace),
      }),
      (error) => error.code === "pi_model_extensions_invalid" && error.message.includes("max_tokens"),
    );
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("registers only the selected role's model when multiple roles are configured", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-catalog-selected-")));
  const extensionPath = join(workspace, "relay-extension.js");
  await writeFile(extensionPath, "export default function () {}\n", "utf8");
  const capture = {};
  const factory = captureModelFactory(capture);
  const runtime = new PiAdapterRuntime(factory);
  try {
    await runtime.start({
      agentName: "pi-catalog-selected",
      baseDir: workspace,
      config: {
        harness: { settings: { relay_extension_path: extensionPath } },
        models: {
          // Only the `default` role is registered/used; the sibling never reaches Pi, so there
          // is no cross-role credential/base-URL contamination.
          analysis: {
            api_key_env: "TEST_API_KEY",
            base_url: "https://other.example.test/v1",
            model: "custom-analysis-model",
            provider: "custom-gateway-provider",
            extensions: { ...GATEWAY_EXTENSIONS },
          },
          default: {
            api_key_env: "TEST_API_KEY",
            base_url: "https://selected.example.test/v1",
            model: "custom-default-model",
            provider: "custom-gateway-provider",
            extensions: { ...GATEWAY_EXTENSIONS },
          },
        },
        tools: { enabled: [] },
      },
      runtimeContext: makeRuntimeContext(workspace),
    });
    await runtime.stop();
    assert.equal(capture.model.baseUrl, "https://selected.example.test/v1");
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("rejects multiple model roles with no default", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-catalog-ambiguous-")));
  const extensionPath = join(workspace, "relay-extension.js");
  await writeFile(extensionPath, "export default function () {}\n", "utf8");
  const factory = captureModelFactory({});
  try {
    await assert.rejects(
      factory.create({
        agentName: "pi-catalog-ambiguous",
        baseDir: workspace,
        config: {
          harness: { settings: { relay_extension_path: extensionPath } },
          models: {
            reviewer_a: {
              api_key_env: "TEST_API_KEY",
              base_url: "https://a.example.test/v1",
              model: "model-a",
              provider: "custom-gateway-provider",
              extensions: { api: "openai-completions" },
            },
            reviewer_b: {
              api_key_env: "TEST_API_KEY",
              base_url: "https://b.example.test/v1",
              model: "model-b",
              provider: "custom-gateway-provider",
              extensions: { api: "openai-completions" },
            },
          },
          tools: { enabled: [] },
        },
        runtimeContext: makeRuntimeContext(workspace),
      }),
      (error) => error.code === "pi_model_ambiguous",
    );
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("accepts a well-formed extensions.cost", async () => {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-catalog-cost-ok-")));
  const extensionPath = join(workspace, "relay-extension.js");
  await writeFile(extensionPath, "export default function () {}\n", "utf8");
  const capture = {};
  const factory = captureModelFactory(capture);
  const runtime = new PiAdapterRuntime(factory);
  try {
    await runtime.start({
      agentName: "pi-catalog-cost-ok",
      baseDir: workspace,
      config: {
        harness: { settings: { relay_extension_path: extensionPath } },
        models: {
          default: {
            api_key_env: "TEST_API_KEY",
            base_url: "https://gateway.example.test/v1",
            model: "custom-gateway-model",
            provider: "custom-gateway-provider",
            extensions: {
              ...GATEWAY_EXTENSIONS,
              cost: { input: 1.5, output: 6, cacheRead: 0.3, cacheWrite: 0 },
            },
          },
        },
        tools: { enabled: [] },
      },
      runtimeContext: makeRuntimeContext(workspace),
    });
    await runtime.stop();
    assert.equal(capture.model.api, "openai-completions");
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

// Lookups for what a model call's provider reported: every count, or a call
// whose raw stream events the adapter never saw.
const allReported = () => ({ input: true, cacheRead: true, output: true });
const unseen = () => undefined;
// A prompt's running total before any call, with input counting the cache.
const NO_TOKENS = { input: 0, cacheRead: 0, output: 0 };

test("sums each call's own usage report", () => {
  const first = addUsage(NO_TOKENS, { input: 10, output: 2, cacheRead: 5, cacheWrite: 1, totalTokens: 18 });
  const second = addUsage(first, { input: 3, output: 4, cacheRead: 0, cacheWrite: 0, reasoning: 2 });

  assert.deepEqual(second, { input: 19, cacheRead: 5, output: 6 });
  assert.equal(addUsage(undefined, { input: 1, output: 1, cacheRead: 0, cacheWrite: 0 }), undefined);
});

// A tool result that reports usage, as a custom tool can.
const toolResult = (usage) => ({ type: "message_end", message: { role: "toolResult", usage } });

test("a tool usage report without a cache count keeps its other counts known", () => {
  const total = addEventUsage(NO_TOKENS, toolResult({ input: 10, cacheRead: 2, output: 3 }), allReported);

  // Input counts cache writes, so without cacheWrite the input and the total
  // are unknown; the cache reads and output stay known.
  assert.deepEqual(total, { cacheRead: 2, output: 3 });
  // A report with only cache reads keeps them.
  assert.deepEqual(addEventUsage(NO_TOKENS, toolResult({ cacheRead: 4 }), allReported), { cacheRead: 4 });
});

test("a tool usage report with only an output count adds that count", () => {
  const total = addEventUsage({ input: 40, cacheRead: 0, output: 1 }, toolResult({ output: 4 }), allReported);

  assert.deepEqual(total, { output: 5 });
});

test("a tool result without usage leaves the prompt's tokens unchanged", () => {
  const total = { input: 40, cacheRead: 0, output: 1 };

  assert.deepEqual(addEventUsage(total, { type: "message_end", message: { role: "toolResult" } }, unseen), total);
});

test("counts the tokens Pi bills outside assistant messages", () => {
  const usage = (tokens) => ({ input: tokens, output: 1, cacheRead: 0, cacheWrite: 0 });
  const events = [
    { type: "message_end", message: { role: "assistant", usage: usage(10) } },
    { type: "compaction_end", aborted: false, result: { summary: "s", usage: usage(100) } },
    { type: "message_end", message: { role: "toolResult", usage: usage(20) } },
    { type: "message_end", message: { role: "toolResult" } },
    { type: "entry_appended", entry: { type: "usage", usage: usage(5) } },
    { type: "entry_appended", entry: { type: "custom" } },
    { type: "turn_start" },
  ];

  const total = events.reduce((sum, event) => addEventUsage(sum, event, allReported), NO_TOKENS);

  assert.deepEqual(total, { input: 135, cacheRead: 0, output: 4 });
});

test("a compaction without a usage report leaves the prompt's tokens unknown", () => {
  assert.equal(
    addEventUsage(NO_TOKENS, { type: "compaction_end", aborted: false, result: { summary: "s" } }, allReported),
    undefined,
  );
  assert.equal(addEventUsage(NO_TOKENS, { type: "compaction_end", aborted: true }, allReported), undefined);
});

test("an aborted assistant message leaves the prompt's tokens unknown", () => {
  const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

  assert.equal(
    addEventUsage(
      NO_TOKENS,
      { type: "message_end", message: { role: "assistant", stopReason: "aborted", usage } },
      allReported,
    ),
    undefined,
  );
});

test("a model call's counts are known only as far as its provider reported them", () => {
  const zeroFilled = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0 };
  const assistant = { type: "message_end", message: { role: "assistant", usage: zeroFilled } };
  const reported = (counts) => () => ({ input: false, cacheRead: false, output: false, ...counts });

  // Pi fills an omitted count with 0, so a call the adapter did not see is unknown.
  assert.equal(addEventUsage(NO_TOKENS, assistant, unseen), undefined);
  assert.equal(
    addEventUsage(NO_TOKENS, { type: "compaction_end", aborted: false, result: { usage: zeroFilled } }, unseen),
    undefined,
  );
  assert.equal(
    addEventUsage(NO_TOKENS, { type: "entry_appended", entry: { type: "usage", usage: zeroFilled } }, unseen),
    undefined,
  );
  // A reported zero is a measured zero; an omitted count stays unknown.
  assert.deepEqual(addEventUsage(NO_TOKENS, assistant, allReported), NO_TOKENS);
  assert.deepEqual(addEventUsage(NO_TOKENS, assistant, reported({ input: true, cacheRead: true })), {
    input: 0,
    cacheRead: 0,
  });
  assert.deepEqual(addEventUsage(NO_TOKENS, assistant, reported({ input: true, output: true })), {
    input: 0,
    output: 0,
  });
  // Cache reads stay known without the input that includes them.
  assert.deepEqual(addEventUsage(NO_TOKENS, assistant, reported({ cacheRead: true })), { cacheRead: 0 });
  // A tool reports its own usage, so a zero there is a measured zero.
  assert.deepEqual(
    addEventUsage(NO_TOKENS, { type: "message_end", message: { role: "toolResult", usage: zeroFilled } }, unseen),
    NO_TOKENS,
  );
});

test("reads which counts each wire protocol's raw usage reports", () => {
  const counts = (api, events) => {
    const seen = new Set();
    for (const event of events) {
      noteReportedFields(seen, api, event);
    }
    return reportedCountsFrom(api, seen);
  };
  const anthropicStart = (usage) => ({ type: "message_start", message: { usage } });
  const anthropicUsage = { input_tokens: 10, cache_read_input_tokens: 4, cache_creation_input_tokens: 0 };

  // Anthropic input excludes the cache, so it needs both cache fields, and only
  // a message_delta reports the output count.
  assert.deepEqual(
    counts("anthropic-messages", [
      anthropicStart({ ...anthropicUsage, output_tokens: 1 }),
      { type: "message_delta", usage: { output_tokens: 5 } },
    ]),
    { input: true, cacheRead: true, output: true },
  );
  assert.deepEqual(
    counts("anthropic-messages", [
      anthropicStart({ input_tokens: 10, output_tokens: 1 }),
      { type: "message_delta", usage: { output_tokens: 5 } },
    ]),
    { input: false, cacheRead: false, output: true },
  );
  assert.deepEqual(
    counts("anthropic-messages", [anthropicStart({ ...anthropicUsage, output_tokens: 1 }), { type: "message_delta" }]),
    { input: true, cacheRead: true, output: false },
  );
  // The other protocols replace a call's usage with each report, and their
  // input includes the cache, so a missing cache detail leaves only it unknown.
  assert.deepEqual(
    counts("openai-completions", [
      { choices: [], usage: { prompt_tokens: 7, prompt_tokens_details: { cached_tokens: 2 } } },
      { choices: [], usage: { completion_tokens: 2 } },
    ]),
    { input: false, cacheRead: false, output: true },
  );
  assert.deepEqual(
    counts("openai-completions", [{ choices: [], usage: { prompt_tokens: 7, prompt_tokens_details: { cached_tokens: 2 } } }]),
    { input: true, cacheRead: true, output: false },
  );
  assert.deepEqual(
    counts("openai-responses", [{ type: "response.completed", response: { usage: { input_tokens: 3 } } }]),
    { input: true, cacheRead: false, output: false },
  );
  assert.deepEqual(
    counts("google-generative-ai", [{ usageMetadata: { promptTokenCount: 4, thoughtsTokenCount: 3 } }]),
    { input: true, cacheRead: false, output: true },
  );
  assert.deepEqual(
    counts("bedrock-converse-stream", [{ metadata: { usage: { outputTokens: 6 } } }]),
    { input: false, cacheRead: false, output: true },
  );
  // A protocol the adapter does not know reports nothing it can trust.
  assert.deepEqual(counts("custom-api", [{ usage: { prompt_tokens: 1, completion_tokens: 1 } }]), {
    input: false,
    cacheRead: false,
    output: false,
  });
});

test("a split-turn compaction counts its summary calls only when they add up", () => {
  const call = (tokens) => ({ input: tokens, output: 1, cacheRead: 0, cacheWrite: 0 });
  const history = call(20);
  const prefix = call(30);
  const combined = { input: 50, output: 2, cacheRead: 0, cacheWrite: 0 };
  const ended = { type: "compaction_end", aborted: false, result: { summary: "s", usage: combined } };
  // Only the summary calls have records; Pi reports their combined usage.
  const recorded = (key) => (key === history || key === prefix ? allReported() : undefined);
  const split = { isSplitTurn: true, messagesToSummarize: [{}], turnPrefixMessages: [{}] };

  assert.equal(summaryCallCount(split), 2);
  assert.equal(summaryCallCount({ ...split, messagesToSummarize: [] }), 1);
  assert.equal(summaryCallCount({ isSplitTurn: false, messagesToSummarize: [{}], turnPrefixMessages: [] }), 1);
  assert.equal(summaryCallCount(undefined), undefined);
  assert.deepEqual(addEventUsage(NO_TOKENS, ended, recorded, { expected: 2, calls: [history, prefix] }), {
    input: 50,
    cacheRead: 0,
    output: 2,
  });
  // A summary call the session did not see, or calls that do not add up to the
  // combined usage, leave the prompt's tokens unknown.
  assert.equal(addEventUsage(NO_TOKENS, ended, recorded, { expected: 2, calls: [prefix] }), undefined);
  assert.equal(addEventUsage(NO_TOKENS, ended, recorded, { expected: 2, calls: [history, call(29)] }), undefined);
  assert.equal(addEventUsage(NO_TOKENS, ended, recorded), undefined);
});

test("a retried compaction summary leaves the prompt's tokens unknown", () => {
  const usage = { input: 20, output: 1, cacheRead: 0, cacheWrite: 0 };
  const events = [
    { type: "summarization_retry_scheduled", attempt: 1, maxAttempts: 3, delayMs: 0, errorMessage: "stream dropped" },
    { type: "summarization_retry_attempt_start" },
    { type: "summarization_retry_finished" },
    { type: "compaction_end", aborted: false, result: { summary: "s", usage } },
  ];

  assert.equal(
    events.reduce((sum, event) => addEventUsage(sum, event, allReported), NO_TOKENS),
    undefined,
  );
});

test("counts a failed request as 0 tokens only after rate limits", () => {
  const rejected = { stopReason: "error", usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  const none = { input: false, cacheRead: false, output: false };
  const counts = (api, message, observation) =>
    endedCallCounts(api, new Set(), message, { accepted: false, streamed: false, statuses: [429], unanswered: false, ...observation });

  assert.deepEqual(counts("openai-completions", rejected, {}), allReported());
  assert.deepEqual(counts("anthropic-messages", rejected, { statuses: [429, 429] }), allReported());
  // A timeout, a server error, another client error, or an attempt without
  // any response may have been billed.
  for (const statuses of [[400], [408], [503], [504], [500], [429, 503]]) {
    assert.deepEqual(counts("openai-completions", rejected, { statuses }), none, String(statuses));
  }
  assert.deepEqual(counts("openai-completions", rejected, { statuses: [], unanswered: true }), none);
  assert.deepEqual(counts("openai-completions", rejected, { statuses: [] }), none);
  // Protocols whose requests do not go through the observed fetch.
  assert.deepEqual(counts("google-generative-ai", rejected, {}), none);
  assert.deepEqual(counts("bedrock-converse-stream", rejected, {}), none);
  assert.deepEqual(counts("custom-api", rejected, {}), none);
  // An accepted or streamed request, or one with counts, may have billed tokens.
  assert.deepEqual(counts("openai-completions", rejected, { accepted: true }), none);
  assert.deepEqual(counts("openai-completions", rejected, { streamed: true }), none);
  assert.deepEqual(counts("openai-completions", { ...rejected, usage: { ...rejected.usage, input: 3 } }, {}), none);
  assert.deepEqual(counts("openai-completions", { ...rejected, stopReason: "aborted" }, {}), none);
  // A successful attempt keeps its reported counts after a rate limit, but not
  // after an ambiguous attempt.
  const seen = new Set(["prompt_tokens", "completion_tokens"]);
  const succeeded = { stopReason: "stop", usage: rejected.usage };
  const afterRetry = (observation) =>
    endedCallCounts("openai-completions", seen, succeeded, { accepted: true, streamed: true, unanswered: false, ...observation });
  assert.deepEqual(afterRetry({ statuses: [429, 200] }), { input: true, cacheRead: false, output: true });
  for (const statuses of [[408, 200], [503, 200], [504, 200]]) {
    assert.deepEqual(afterRetry({ statuses }), none, String(statuses));
  }
  assert.deepEqual(afterRetry({ statuses: [200], unanswered: true }), none);
});

/**
 * Run prompts through the real Pi SDK against a local chat-completions endpoint
 * that answers each request from `replies` in order. A reply's `usage` is sent
 * as the stream's final usage chunk; a reply without one sends no usage. A
 * reply with `status` rejects the request with that HTTP status, one with
 * `dropStream` closes the connection after the first chunk, and one with
 * `closeBeforeHeaders` closes it before sending any response.
 * `modelExtensions` overrides the catalog entry, such as its context window,
 * and `extension` is the source of a Pi extension the session loads.
 */
async function promptsAgainstScriptedEndpoint(replies, prompts, modelExtensions = undefined, extension = undefined) {
  const workspace = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-usage-")));
  await writeFile(join(workspace, "note.txt"), "nonce\n", "utf8");
  if (extension !== undefined) {
    await writeFile(join(workspace, "usage-extension.js"), extension, "utf8");
  }
  let served = 0;
  const provider = createServer(async (request, response) => {
    for await (const _chunk of request);
    const reply = replies[served++];
    if (reply.closeBeforeHeaders) {
      request.socket.destroy();
      return;
    }
    if (reply.status !== undefined) {
      response.writeHead(reply.status, { "content-type": "application/json" });
      response.end(JSON.stringify({ error: { message: "rate limited", type: "rate_limit_error" } }));
      return;
    }
    const common = { id: `chatcmpl-usage-${served}`, object: "chat.completion.chunk", created: 0, model: "openai/gpt-oss-20b" };
    const chunk = (fields) => `data: ${JSON.stringify({ ...common, ...fields })}\n\n`;
    const delta = reply.readNote
      ? { role: "assistant", tool_calls: [{ index: 0, id: `call-${served}`, type: "function", function: { name: "read", arguments: JSON.stringify({ path: "note.txt" }) } }] }
      : { role: "assistant", content: reply.content };
    response.writeHead(200, { "content-type": "text/event-stream" });
    if (reply.dropStream) {
      response.write(chunk({ choices: [{ index: 0, delta, finish_reason: null }] }));
      // Let the client read the first chunk before the connection drops.
      setTimeout(() => response.destroy(), 100);
      return;
    }
    response.end(
      chunk({ choices: [{ index: 0, delta, finish_reason: null }] }) +
        chunk({ choices: [{ index: 0, delta: {}, finish_reason: reply.readNote ? "tool_calls" : "stop" }] }) +
        (reply.usage === undefined ? "" : chunk({ choices: [], usage: reply.usage })) +
        "data: [DONE]\n\n",
    );
  });
  const providerUrl = await listen(provider);
  let handle;
  try {
    handle = await new PiSdkSessionFactory().create({
      agentName: "pi-usage-test",
      baseDir: workspace,
      config: {
        ...(extension === undefined ? {} : { harness: { settings: { extensions: ["usage-extension.js"] } } }),
        models: {
          default: {
            api_key_env: "TEST_API_KEY",
            base_url: `${providerUrl}/v1`,
            model: "openai/gpt-oss-20b",
            provider: "nvidia",
            ...(modelExtensions === undefined ? {} : { extensions: modelExtensions }),
          },
        },
        tools: { enabled: null },
      },
      runtimeContext: { ...makeRuntimeContext(workspace), telemetry: undefined },
    });
    const outcomes = [];
    for (const prompt of prompts) {
      outcomes.push(await handle.prompt(prompt));
    }
    assert.equal(served, replies.length);
    return outcomes;
  } finally {
    await handle?.stop();
    await close(provider);
    await rm(workspace, { recursive: true, force: true });
  }
}

test(
  "reports each prompt's own tokens through the Pi SDK",
  { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
  async () => {
    const [first, second] = await promptsAgainstScriptedEndpoint(
      [
        { readNote: true, usage: { prompt_tokens: 100, completion_tokens: 10, prompt_tokens_details: { cached_tokens: 40 } } },
        { content: "first", usage: { prompt_tokens: 150, completion_tokens: 0 } },
        { content: "second", usage: { prompt_tokens: 300, completion_tokens: 25 } },
      ],
      ["read the note", "again"],
    );

    // A reported zero output count stays a measured zero. The second call sent
    // no cache details, so the prompt's cache reads are unknown.
    assert.deepEqual(first.usage, { input: 250, output: 10 });
    assert.deepEqual(second.usage, { input: 300, output: 25 });
  },
);

test(
  "leaves a prompt's tokens unknown when any provider response omits usage",
  { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
  async () => {
    const [omitted, mixed, reported] = await promptsAgainstScriptedEndpoint(
      [
        { content: "no usage" },
        { readNote: true, usage: { prompt_tokens: 100, completion_tokens: 10 } },
        { content: "partial" },
        { content: "reported", usage: { prompt_tokens: 50, completion_tokens: 5 } },
      ],
      ["one", "two", "three"],
    );

    assert.equal(omitted.text, "no usage");
    assert.equal(omitted.usage, undefined);
    assert.equal(mixed.text, "partial");
    assert.equal(mixed.usage, undefined);
    assert.deepEqual(reported.usage, { input: 50, output: 5 });
  },
);

test(
  "leaves a count the provider omitted unknown and keeps reported zeros",
  { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
  async () => {
    const [inputOnly, outputOnly, zeros, afterPartial] = await promptsAgainstScriptedEndpoint(
      [
        { content: "input only", usage: { prompt_tokens: 100 } },
        { content: "output only", usage: { completion_tokens: 5 } },
        { content: "zeros", usage: { prompt_tokens: 0, completion_tokens: 0, prompt_tokens_details: { cached_tokens: 0 } } },
        { readNote: true, usage: { prompt_tokens: 60 } },
        { content: "both", usage: { prompt_tokens: 70, completion_tokens: 7 } },
      ],
      ["one", "two", "three", "four"],
    );

    // Pi fills an omitted count with 0; the provider's raw report decides.
    assert.deepEqual(inputOnly.usage, { input: 100 });
    assert.deepEqual(outputOnly.usage, { output: 5 });
    assert.deepEqual(zeros.usage, { input: 0, cacheRead: 0, output: 0 });
    // One call without an output count leaves the prompt's output unknown.
    assert.deepEqual(afterPartial.usage, { input: 130 });
  },
);

test(
  "counts a compaction that Pi summarizes in one call",
  { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
  async () => {
    // The second prompt is longer than Pi's 20,000 recent tokens, so Pi cuts at
    // that prompt and summarizes only the earlier history, in one call.
    const [, compacted] = await promptsAgainstScriptedEndpoint(
      [
        { content: "first", usage: { prompt_tokens: 100, completion_tokens: 5 } },
        { readNote: true, usage: { prompt_tokens: 200, completion_tokens: 5 } },
        { content: "done", usage: { prompt_tokens: 90_000, completion_tokens: 5 } },
        { content: "history summary", usage: { prompt_tokens: 20, completion_tokens: 1 } },
      ],
      ["one", "y".repeat(100_000)],
      { context_window: 100_000, max_tokens: 1_000 },
    );

    assert.deepEqual(compacted.usage, { input: 90_220, output: 11 });
  },
);

for (const [name, historyUsage, expected] of [
  ["counts both summary calls of a split-turn compaction", { prompt_tokens: 20, completion_tokens: 1 }, { input: 90_240, output: 12 }],
  ["leaves input unknown when a split-turn summary call omits it", { completion_tokens: 1 }, { output: 12 }],
]) {
  test(name, { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" }, async () => {
    // The second prompt's final reply reports a context above the compaction
    // threshold and is longer than Pi's 20,000 recent tokens, so Pi cuts inside
    // that turn and summarizes the history and the turn prefix in two calls,
    // then reports only their combined usage.
    const [, compacted] = await promptsAgainstScriptedEndpoint(
      [
        { content: "first", usage: { prompt_tokens: 100, completion_tokens: 5 } },
        { readNote: true, usage: { prompt_tokens: 200, completion_tokens: 5 } },
        { content: "x".repeat(100_000), usage: { prompt_tokens: 90_000, completion_tokens: 5 } },
        { content: "history summary", usage: historyUsage },
        { content: "turn prefix summary", usage: { prompt_tokens: 20, completion_tokens: 1 } },
      ],
      ["one", "read the note"],
      { context_window: 100_000, max_tokens: 1_000 },
    );

    assert.deepEqual(compacted.usage, expected);
  });
}

test(
  "counts a rate-limited request as 0 tokens when Pi's retry succeeds",
  { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
  async () => {
    const [retried] = await promptsAgainstScriptedEndpoint(
      [{ status: 429 }, { content: "after retry", usage: { prompt_tokens: 50, completion_tokens: 5 } }],
      ["one"],
    );

    assert.equal(retried.text, "after retry");
    assert.deepEqual(retried.usage, { input: 50, output: 5 });
  },
);

for (const status of [408, 503, 504]) {
  test(
    `leaves a prompt's tokens unknown when a ${status} precedes a successful retry`,
    { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
    async () => {
      // The provider may have processed the prompt before answering with a
      // timeout or a server error.
      const [retried] = await promptsAgainstScriptedEndpoint(
        [{ status }, { content: "after retry", usage: { prompt_tokens: 50, completion_tokens: 5 } }],
        ["one"],
      );

      assert.equal(retried.text, "after retry");
      assert.equal(retried.usage, undefined);
    },
  );
}

test(
  "leaves a prompt's tokens unknown when a connection fails before any HTTP response",
  { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
  async () => {
    // The provider may have received and billed the prompt before the
    // connection failed, so the failed attempt is not a known 0.
    const [retried] = await promptsAgainstScriptedEndpoint(
      [{ closeBeforeHeaders: true }, { content: "after retry", usage: { prompt_tokens: 50, completion_tokens: 5 } }],
      ["one"],
    );

    assert.equal(retried.text, "after retry");
    assert.equal(retried.usage, undefined);
  },
);

/** One unmasked WebSocket text frame, as a server sends it. */
function webSocketTextFrame(text) {
  const payload = Buffer.from(text);
  let header;
  if (payload.length < 126) {
    header = Buffer.from([0x81, payload.length]);
  } else if (payload.length < 65_536) {
    header = Buffer.from([0x81, 126, payload.length >> 8, payload.length & 0xff]);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x81;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(payload.length), 2);
  }
  return Buffer.concat([header, payload]);
}

/** The Responses events of a completed reply that reports `usage`. */
function codexReplyEvents(usage) {
  const part = { type: "output_text", text: "done", annotations: [] };
  const item = { id: "msg_usage", type: "message", role: "assistant", status: "completed", content: [part] };
  const response = { id: "resp_usage", object: "response", created_at: 0, status: "completed", model: "gpt-test", output: [item], usage };
  return [
    { type: "response.created", response: { ...response, status: "in_progress", output: [] } },
    { type: "response.output_item.added", output_index: 0, item: { ...item, status: "in_progress", content: [] } },
    { type: "response.content_part.added", item_id: item.id, output_index: 0, content_index: 0, part: { ...part, text: "" } },
    { type: "response.output_text.delta", item_id: item.id, output_index: 0, content_index: 0, delta: "done" },
    { type: "response.output_text.done", item_id: item.id, output_index: 0, content_index: 0, text: "done" },
    { type: "response.content_part.done", item_id: item.id, output_index: 0, content_index: 0, part },
    { type: "response.output_item.done", output_index: 0, item },
    { type: "response.completed", response },
  ];
}

/**
 * Run one model call through Pi's Codex transport, wrapped by the adapter's
 * recorder, against a local endpoint. `webSocket` is "reply" to answer the
 * WebSocket request or "drop" to close the connection after receiving it
 * and before any event; `http` is the HTTP fallback's status, with 200
 * replying over server-sent events. Returns the counts the adapter recorded
 * for the call and the requests each transport received.
 */
async function codexCallCounts({ webSocket, http }) {
  const usage = { input_tokens: 50, input_tokens_details: { cached_tokens: 0 }, output_tokens: 5, total_tokens: 55 };
  const received = { webSocket: 0, http: 0 };
  const sockets = new Set();
  const server = createServer(async (request, response) => {
    for await (const _chunk of request);
    received.http += 1;
    if (http !== 200) {
      response.writeHead(http, { "content-type": "application/json" });
      response.end(JSON.stringify({ error: { message: "rate limited", type: "rate_limit_error" } }));
      return;
    }
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.end(codexReplyEvents(usage).map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""));
  });
  server.on("upgrade", (request, socket) => {
    sockets.add(socket);
    const accept = createHash("sha1")
      .update(`${request.headers["sec-websocket-key"]}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
      .digest("base64");
    socket.write(
      `HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`,
    );
    // The first client frame is Pi's response.create request.
    socket.once("data", () => {
      received.webSocket += 1;
      if (webSocket === "drop") {
        socket.destroy();
        return;
      }
      for (const event of codexReplyEvents(usage)) {
        socket.write(webSocketTextFrame(JSON.stringify(event)));
      }
    });
  });
  const baseUrl = await listen(server);
  const sessionId = randomUUID();
  const { closeOpenAICodexWebSocketSessions, stream } = await import("@earendil-works/pi-ai/api/openai-codex-responses");
  try {
    const records = new WeakMap();
    const agent = { streamFunction: stream };
    recordReportedCounts(agent, records, () => undefined);
    const claims = Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "account" } })).toString("base64url");
    const model = {
      id: "gpt-test",
      name: "gpt-test",
      api: "openai-codex-responses",
      provider: "openai-codex",
      baseUrl,
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 100_000,
      maxTokens: 1_000,
    };
    const response = await agent.streamFunction(
      model,
      { messages: [{ role: "user", content: "one", timestamp: 0 }] },
      { apiKey: `header.${claims}.signature`, sessionId, transport: "auto" },
    );
    const message = await response.result();
    await new Promise((resolve) => setImmediate(resolve));
    return { counts: records.get(message), message, received };
  } finally {
    // Pi keeps a session's WebSocket open for reuse until an idle timer ends.
    closeOpenAICodexWebSocketSessions(sessionId);
    for (const socket of sockets) {
      socket.destroy();
    }
    await close(server);
  }
}

test(
  "keeps a Codex WebSocket reply's counts known",
  { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
  async () => {
    const { counts, message, received } = await codexCallCounts({ webSocket: "reply", http: 200 });

    assert.deepEqual(received, { webSocket: 1, http: 0 });
    assert.equal(message.stopReason, "stop");
    assert.deepEqual(counts, { input: true, cacheRead: true, output: true });
  },
);

for (const [name, http, stopReason] of [
  ["a successful HTTP fallback", 200, "stop"],
  ["a rate-limited HTTP fallback", 429, "error"],
]) {
  test(
    `leaves a Codex call's counts unknown when its WebSocket request failed before ${name}`,
    { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
    async () => {
      // The provider received the WebSocket request and may have billed it,
      // but the fetch only sees the HTTP fallback.
      const { counts, message, received } = await codexCallCounts({ webSocket: "drop", http });

      assert.deepEqual(received, { webSocket: 1, http: 1 });
      assert.equal(message.stopReason, stopReason);
      assert.ok(message.diagnostics.some((diagnostic) => diagnostic.type === "provider_transport_failure"));
      assert.deepEqual(counts, { input: false, cacheRead: false, output: false });
    },
  );
}

test(
  "leaves a prompt's tokens unknown when a stream fails after it started",
  { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
  async () => {
    // The provider accepted the request and streamed, so it may have billed
    // tokens that the dropped stream never reported.
    const [dropped] = await promptsAgainstScriptedEndpoint(
      [{ content: "partial", dropStream: true }, { content: "after retry", usage: { prompt_tokens: 50, completion_tokens: 5 } }],
      ["one"],
    );

    assert.equal(dropped.text, "after retry");
    assert.equal(dropped.usage, undefined);
  },
);

test(
  "keeps usage known when an extension replaces an assistant message's usage object",
  { skip: supportsPi ? false : "Pi 1.0 requires Node 22.19 or newer" },
  async () => {
    const extension = `export default function (pi) {
  pi.on("message_end", (event) =>
    event.message.role === "assistant" ? { message: { ...event.message, usage: { ...event.message.usage } } } : undefined,
  );
}
`;
    const [replaced] = await promptsAgainstScriptedEndpoint(
      [{ content: "done", usage: { prompt_tokens: 50, completion_tokens: 5 } }],
      ["one"],
      undefined,
      extension,
    );

    assert.deepEqual(replaced.usage, { input: 50, output: 5 });
  },
);

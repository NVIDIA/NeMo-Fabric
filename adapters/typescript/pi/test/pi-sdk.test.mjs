// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { access, mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createServer } from "node:http";

import {
  buildCatalogModel,
  modelAwareCompactionReserveTokens,
  PiSdkSessionFactory,
  resolveCustomTools,
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
        error.message.includes("NeMo Relay 0.9 release") &&
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

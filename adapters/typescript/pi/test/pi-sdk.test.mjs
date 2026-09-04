// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdir, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createServer } from "node:http";

import { PiSdkSessionFactory, resolveCustomTools } from "../dist/pi-sdk.js";
import { PiAdapterRuntime } from "../dist/runtime.js";

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
        error.message.includes("matching NeMo Relay 0.9 release") &&
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

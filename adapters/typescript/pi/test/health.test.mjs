// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PiAdapterRuntime } from "../dist/runtime.js";
import { PiSdkSessionFactory } from "../dist/pi-sdk.js";
import {
  createAgentSession,
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { InMemoryCredentialStore } from "@earendil-works/pi-ai";

const input = {
  agentName: "pi-health",
  baseDir: "/tmp",
  config: {},
  runtimeContext: {
    artifacts: {},
    environment: {
      environment_id: "env",
      provider: "local",
      control_location: "external_control",
      ownership: "caller_owned",
    },
    runtime_id: "runtime-1",
    invocation_id: "start",
    request_id: "start",
  },
};

test("health dispatch reaches a busy session without another prompt", async () => {
  const entered = Promise.withResolvers();
  const release = Promise.withResolvers();
  const observations = [];
  let prompts = 0;
  const runtime = new PiAdapterRuntime({
    async create() {
      return {
        async prompt() {
          prompts += 1;
          entered.resolve();
          await release.promise;
          return { accepted: true, turnStarted: true, turnCount: 1, text: "done" };
        },
        async check(level) {
          observations.push(level);
          return {
            outcome: "passed",
            reason_code: null,
            activity: "busy",
            source: "fabric.adapter.liveness",
            observed_at: new Date().toISOString(),
          };
        },
        async stop() {},
      };
    },
  });
  await runtime.start(input);
  const invocation = runtime.invoke({ input: "PRIVATE" }, input.runtimeContext);
  try {
    await entered.promise;
    const observation = await runtime.check("live");
    assert.equal(observation.outcome, "passed");
    assert.equal(observation.activity, "busy");
    assert.deepEqual(observations, ["live"]);
    assert.equal(prompts, 1);
    assert.equal(JSON.stringify(observation).includes("PRIVATE"), false);
  } finally {
    release.resolve();
    await invocation;
    await runtime.stop();
  }
});

test("checks reject an absent session without creating one", async () => {
  let created = 0;
  const runtime = new PiAdapterRuntime({
    async create() {
      created += 1;
    },
  });
  const observation = await runtime.check("live");
  assert.equal(observation.outcome, "failed");
  assert.equal(observation.reason_code, "session_closed");
  assert.equal(created, 0);
});

test("stopping invalidates an observation and prevents fresh probes", async () => {
  const entered = Promise.withResolvers();
  const release = Promise.withResolvers();
  let probes = 0;
  const runtime = new PiAdapterRuntime({
    async create() {
      return {
        async prompt() {
          throw new Error("unused");
        },
        async check() {
          probes += 1;
          entered.resolve();
          await release.promise;
          return {
            outcome: "passed",
            reason_code: null,
            activity: "idle",
            source: "fabric.adapter.liveness",
            observed_at: new Date().toISOString(),
          };
        },
        async stop() {},
      };
    },
  });
  await runtime.start(input);
  const checking = runtime.check("live");
  await entered.promise;
  await runtime.stop();
  release.resolve();
  assert.equal((await checking).reason_code, "observation_changed");
  assert.equal((await runtime.check("live")).reason_code, "session_closed");
  assert.equal(probes, 1);
});

async function nativeSession(t, config = {}, directory) {
  const workspace = directory ?? await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-health-")));
  t.after(() => rm(workspace, { recursive: true, force: true }));
  const handle = await new PiSdkSessionFactory().create({
    ...input,
    baseDir: workspace,
    config: {
      models: { default: { provider: "openai", model: "gpt-4.1-mini", api_key_env: "TEST_ONLY_API_KEY" } },
      tools: { enabled: ["read"] },
      ...config,
    },
    runtimeContext: {
      ...input.runtimeContext,
      environment: {
        ...input.runtimeContext.environment,
        workspace,
        env: { TEST_ONLY_API_KEY: "PRIVATE-test-credential" },
      },
    },
  });
  t.after(() => handle.stop());
  return { handle, workspace };
}

test("real Pi health observations preserve the session and make no model calls", async (t) => {
  const { handle, workspace } = await nativeSession(t);
  const retained = join(workspace, "retained");
  await writeFile(retained, "KEEP");
  const observations = [];
  for (const level of ["live", "active", "ready", "live"]) {
    const result = await handle.check(level);
    assert.equal(result.outcome, "passed");
    assert.equal(result.activity, "idle");
    assert.ok(Number.isFinite(Date.parse(result.observed_at)));
    observations.push(result);
  }
  assert.equal(handle.turnCount, 0);
  assert.equal(await readFile(retained, "utf8"), "KEEP");
  assert.equal(JSON.stringify(observations).includes("PRIVATE"), false);
  await handle.stop();
  assert.equal((await handle.check("live")).reason_code, "session_closed");
});

for (const exposure of ["direct", "model-only", "deferred", "codemode"]) {
  test(`ready observes required ${exposure} tools after loadout changes`, async (t) => {
    const directory = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-tool-health-")));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const extension = join(directory, "tools.js");
    await writeFile(extension, `export default function (pi) {
  pi.registerTool({
    name: "health_tool", label: "Health Tool", description: "Test tool availability",
    exposure: ${JSON.stringify(exposure)},
    parameters: { type: "object", properties: {} },
    async execute() { return { content: [{ type: "text", text: "ok" }], details: {} }; }
  });
  pi.registerCommand("disable-tools", { description: "Disable tools", handler: async () => pi.setActiveTools([]) });
  pi.registerCommand("enable-tools", { description: "Enable tools", handler: async () => pi.setActiveTools(["health_tool"]) });
}
`);
    const { handle } = await nativeSession(t, {
      tools: { enabled: ["health_tool"] },
      harness: { settings: { extensions: ["tools.js"] } },
    }, directory);
    assert.equal((await handle.check("ready")).outcome, "passed");
    await handle.prompt("/disable-tools");
    assert.equal((await handle.check("live")).outcome, "passed");
    assert.equal((await handle.check("active")).outcome, "passed");
    const observation = await handle.check("ready");
    const available = exposure === "deferred" || exposure === "codemode";
    assert.equal(observation.outcome, available ? "passed" : "failed");
    assert.equal(observation.reason_code, available ? null : "health_failed");
    await handle.prompt("/enable-tools");
    assert.equal((await handle.check("ready")).outcome, "passed");
    assert.equal(handle.turnCount, 0);
  });
}

test("ready observes a required MCP server closing after successful startup", async (t) => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-health-mcp-")));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const script = join(directory, "server.mjs");
  const pidPath = join(directory, "pid");
  await writeFile(
    script,
    `import { writeFileSync } from "node:fs";
import { createInterface } from "node:readline";
const lines = createInterface({ input: process.stdin });
lines.on("line", (line) => {
  const request = JSON.parse(line);
  if (request.id === undefined) return;
  const result = request.method === "initialize"
    ? { protocolVersion: request.params.protocolVersion, capabilities: { tools: {} }, serverInfo: { name: "health", version: "1" } }
    : request.method === "tools/list" ? { tools: [] } : {};
  if (request.method === "tools/list") writeFileSync(process.env.PID_PATH, String(process.pid));
  process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: request.id, result }) + "\\n");
});

`,
  );
  const { handle } = await nativeSession(t, {
    tools: { enabled: [] },
    mcp: {
      servers: {
        local: { transport: "stdio", url: process.execPath, args: [script], env: { PID_PATH: pidPath } },
      },
    },
  });
  assert.equal((await handle.check("ready")).outcome, "passed");
  process.kill(Number(await readFile(pidPath, "utf8")), "SIGTERM");
  const deadline = Date.now() + 2000;
  while ((await handle.check("ready")).outcome === "passed") {
    assert.ok(Date.now() < deadline, "MCP closure was not observed");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.equal((await handle.check("live")).outcome, "passed");
  assert.equal((await handle.check("ready")).reason_code, "health_failed");
  assert.equal(handle.turnCount, 0);
});

test("native context liveness rejects disposed sessions while stored state remains readable", async (t) => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-native-health-")));
  const settings = SettingsManager.inMemory({}, { projectTrusted: false });
  const modelRuntime = await ModelRuntime.create({
    credentials: new InMemoryCredentialStore(),
    modelsPath: null,
    allowModelNetwork: false,
    refreshOnCreate: false,
  });
  await modelRuntime.setRuntimeApiKey("openai", "test-only");
  const loader = new DefaultResourceLoader({
    cwd: directory,
    agentDir: directory,
    settingsManager: settings,
    noExtensions: true,
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
    noContextFiles: true,
  });
  await loader.reload();
  const { session } = await createAgentSession({
    cwd: directory,
    agentDir: directory,
    modelRuntime,
    model: modelRuntime.getModel("openai", "gpt-4.1-mini"),
    resourceLoader: loader,
    settingsManager: settings,
    sessionManager: SessionManager.inMemory(directory),
    tools: ["read"],
  });
  t.after(async () => {
    session.dispose();
    await rm(directory, { recursive: true, force: true });
  });
  // An idle context exists before extension initialization has completed.
  assert.equal(session.extensionRunner.createContext().isIdle(), true);
  await session.bindExtensions({ mode: "print" });
  const context = session.extensionRunner.createContext();
  const sessionId = session.sessionId;
  assert.equal(context.isIdle(), true);
  assert.ok(session.getAllTools().some((tool) => tool.name === "read"));
  session.dispose();
  assert.equal(session.sessionId, sessionId);
  assert.equal(session.isIdle, true);
  assert.ok(session.getAllTools().some((tool) => tool.name === "read"));
  assert.throws(() => context.isIdle());
  assert.throws(() => session.extensionRunner.createContext().isIdle());
});

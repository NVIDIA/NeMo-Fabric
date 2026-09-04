// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  collectRelayArtifacts,
  encodeToml,
  loadRelayPluginConfig,
  validateRelayObservabilityV3,
  writeRelayConfigs,
} from "../dist/relay-config.js";
import { expectsLocalAtif, snapshotAtifFiles, waitForFinalizedAtif } from "../dist/relay-artifacts.js";
import { relayCliContract, startRelayGateway, stopRelayGateway } from "../dist/relay-gateway.js";
import { PiRelayFactory, resolveRelayExtensionPath } from "../dist/relay.js";

function startInput(baseDir, options = {}) {
  return {
    agentName: "pi-relay-test",
    baseDir,
    config: {
      harness: {
        settings: {
          ...(options.extensionPath === undefined ? {} : { relay_extension_path: options.extensionPath }),
        },
      },
      models: {
        default: {
          api_key_env: "TEST_API_KEY",
          model: "gpt-4.1-mini",
          provider: "openai",
        },
      },
    },
    runtimeContext: {
      artifacts: {},
      environment: {
        control_location: "external_control",
        environment_id: "environment-1",
        ownership: "caller_owned",
        provider: "local",
        workspace: baseDir,
      },
      invocation_id: "start",
      request_id: "request-start",
      runtime_id: "runtime-1",
      ...(options.relay === false ? {} : { telemetry: { relay_enabled: true } }),
    },
  };
}

function observability(config) {
  return {
    version: 1,
    components: [
      {
        kind: "observability",
        enabled: true,
        config: { version: 3, ...config },
      },
    ],
  };
}

class MockChild extends EventEmitter {
  exitCode = null;
  signalCode = null;
  signals = [];

  kill(signal) {
    this.signals.push(signal);
    return true;
  }
}

test("accepts only stable NeMo Relay 0.9 CLI versions", async () => {
  for (const version of ["0.9.0", "0.9.7", "0.9.0+build.4"]) {
    const contract = await relayCliContract("nemo-relay", async () => ({
      stdout: `nemo-relay ${version}\n`,
      exitCode: 0,
    }));
    assert.deepEqual(contract.version, [0, 9, Number(version.split(".")[2].split("+")[0])]);
  }
  for (const version of ["0.8.9", "1.0.0", "0.9.0-rc.1"]) {
    await assert.rejects(
      relayCliContract("nemo-relay", async () => ({
        stdout: `nemo-relay ${version}\n`,
        exitCode: 0,
      })),
      />=0\.9\.0,<0\.10\.0/,
    );
  }
  await assert.rejects(
    relayCliContract("nemo-relay", async () => ({
      stdout: "unknown\n",
      exitCode: 0,
    })),
    /version could not be determined/,
  );
});

test("normalizes file outputs without mutating a Relay stream sink", async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-config-")));
  const previousConfigPath = process.env.FABRIC_RELAY_CONFIG_PATH;
  try {
    const runtimeConfigPath = join(root, "relay-config.json");
    const streamSink = {
      type: "stream",
      name: "nemo-fabric-stream",
      url: "http://127.0.0.1:4319/atof",
      transport: "ndjson",
      timeout_millis: 3000,
      headers: {},
      header_env: { authorization: "RELAY_AUTHORIZATION" },
    };
    await writeFile(
      runtimeConfigPath,
      JSON.stringify({
        relay: {
          config: {
            version: 3,
            atof: {
              enabled: true,
              sinks: [{ type: "file", output_directory: "relay-atof" }, streamSink],
            },
            atif: { enabled: true },
          },
        },
      }),
      "utf8",
    );
    process.env.FABRIC_RELAY_CONFIG_PATH = runtimeConfigPath;

    const pluginConfig = await loadRelayPluginConfig(startInput(root));
    const config = pluginConfig.components[0].config;
    assert.deepEqual(config.atof.sinks[1], streamSink);
    assert.equal(config.atof.sinks[0].output_directory, join(root, "relay-atof", "runtime-1"));
    assert.equal(config.atof.sinks[0].filename, "events.atof.jsonl");
    assert.equal(config.atif.output_directory, join(root, "artifacts", "relay", "runtime-1"));
    assert.equal(config.atif.filename_template, "trajectory-{session_id}.atif.json");
    assert.equal(config.atif.agent_name, "pi-relay-test");
    assert.equal(config.atif.model_name, "gpt-4.1-mini");

    const paths = await writeRelayConfigs(pluginConfig);
    assert.equal(paths.configPath, join(root, "relay-config", "config.toml"));
    assert.equal(paths.pluginConfigPath, join(root, "relay-config", "plugins.toml"));
    const toml = await readFile(paths.pluginConfigPath, "utf8");
    assert.match(toml, /\[\[components\.config\.atof\.sinks\]\]\ntype = "stream"/);
    assert.match(toml, /\[components\.config\.atof\.sinks\.header_env\]\nauthorization = "RELAY_AUTHORIZATION"/);
    assert.equal(await readFile(paths.configPath, "utf8"), "\n");
  } finally {
    if (previousConfigPath === undefined) {
      delete process.env.FABRIC_RELAY_CONFIG_PATH;
    } else {
      process.env.FABRIC_RELAY_CONFIG_PATH = previousConfigPath;
    }
    await rm(root, { recursive: true, force: true });
  }
});

test("encodes ATOF, ATIF, OTEL, and OpenInference Relay plugin snapshots", () => {
  const snapshots = [
    [
      {
        atof: {
          enabled: true,
          sinks: [
            {
              type: "file",
              output_directory: "/tmp/atof",
              filename: "events.atof.jsonl",
            },
          ],
        },
      },
      `version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 3

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = "/tmp/atof"
filename = "events.atof.jsonl"
`,
    ],
    [
      {
        atif: {
          enabled: true,
          output_directory: "/tmp/atif",
          filename_template: "trajectory-{session_id}.atif.json",
        },
      },
      `version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 3

[components.config.atif]
enabled = true
output_directory = "/tmp/atif"
filename_template = "trajectory-{session_id}.atif.json"
`,
    ],
    [
      {
        opentelemetry: {
          enabled: true,
          endpoints: [{ type: "full", endpoint: "http://127.0.0.1:4318/v1/traces" }],
        },
      },
      `version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 3

[components.config.opentelemetry]
enabled = true

[[components.config.opentelemetry.endpoints]]
type = "full"
endpoint = "http://127.0.0.1:4318/v1/traces"
`,
    ],
    [
      {
        opentelemetry: {
          enabled: true,
          endpoints: [
            {
              type: "openinference",
              endpoint: "http://127.0.0.1:4320/v1/traces",
            },
          ],
        },
      },
      `version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 3

[components.config.opentelemetry]
enabled = true

[[components.config.opentelemetry.endpoints]]
type = "openinference"
endpoint = "http://127.0.0.1:4320/v1/traces"
`,
    ],
  ];

  for (const [config, expected] of snapshots) {
    assert.equal(encodeToml(observability(config)), expected);
  }
  assert.throws(() => encodeToml({ unsupported: 1.5 }), /unsupported by the local TOML encoder/);
  assert.throws(() => encodeToml({ unsupported: null }), /unsupported by the local TOML encoder/);
});

test("rejects removed and malformed Relay OpenTelemetry shapes", () => {
  assert.throws(
    () => validateRelayObservabilityV3(observability({ openinference: {} })),
    /removed the standalone openinference section/,
  );
  assert.throws(
    () => validateRelayObservabilityV3(observability({ opentelemetry: { enabled: true } })),
    /requires at least one endpoint/,
  );
  assert.throws(
    () =>
      validateRelayObservabilityV3(
        observability({
          opentelemetry: {
            enabled: true,
            endpoints: [{ type: "full", endpoint: "" }],
          },
        }),
      ),
    /non-empty string/,
  );
});

test("waits for a complete changed ATIF artifact and collects Relay files", async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-artifacts-")));
  try {
    const atifDir = join(root, "atif");
    const atofDir = join(root, "atof");
    await mkdir(atifDir);
    await mkdir(atofDir);
    await writeFile(join(atofDir, "events.atof.jsonl"), "{}\n", "utf8");
    const pluginConfig = observability({
      atof: {
        enabled: true,
        sinks: [
          {
            type: "file",
            output_directory: atofDir,
            filename: "events.atof.jsonl",
          },
        ],
      },
      atif: {
        enabled: true,
        output_directory: atifDir,
        filename_template: "trajectory-{session_id}.atif.json",
      },
    });
    const before = await snapshotAtifFiles(pluginConfig);
    const path = join(atifDir, "trajectory-session-1.atif.json");
    await writeFile(path, "{", "utf8");
    setTimeout(() => void writeFile(path, '{"session_id":"session-1"}', "utf8"), 10);

    assert.equal(
      await waitForFinalizedAtif(pluginConfig, before, {
        timeoutMs: 500,
        pollIntervalMs: 5,
      }),
      path,
    );
    assert.deepEqual(await collectRelayArtifacts(pluginConfig), [
      { kind: "atif", path },
      { kind: "atof", path: join(atofDir, "events.atof.jsonl") },
    ]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("collects default ATOF files and tolerates disappearing artifact directories", async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-atof-")));
  try {
    await writeFile(join(root, "first.jsonl"), "{}\n", "utf8");
    await writeFile(join(root, "ignored.json"), "{}\n", "utf8");
    const pluginConfig = observability({
      atof: { enabled: true, sinks: [{ type: "file", output_directory: root }] },
    });
    assert.deepEqual(await collectRelayArtifacts(pluginConfig), [
      { kind: "atof", path: join(root, "first.jsonl") },
    ]);

    await rm(root, { recursive: true, force: true });
    assert.deepEqual(await collectRelayArtifacts(pluginConfig), []);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("skips local ATIF waiting for remote-storage configurations", () => {
  assert.equal(expectsLocalAtif(observability({ atif: { enabled: true } })), true);
  assert.equal(expectsLocalAtif(observability({ atif: { enabled: true, storage: [{ type: "http" }] } })), false);
  assert.equal(expectsLocalAtif(observability({ atif: { enabled: false } })), false);
});

test("launches a detached gateway with an isolated log and exact upstream arguments", async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-gateway-")));
  try {
    const configPath = join(root, "config.toml");
    const logPath = join(root, "gateway.log");
    await writeFile(configPath, "\n", "utf8");
    const mockChild = new MockChild();
    let spawnCall;
    let healthUrl;
    const child = await startRelayGateway(
      {
        executable: "/opt/bin/nemo-relay",
        configPath,
        bind: "127.0.0.1:41000",
        url: "http://127.0.0.1:41000",
        logPath,
        openaiBaseUrl: "https://api.openai.com/v1",
      },
      root,
      {
        spawn(command, args, options) {
          spawnCall = { command, args, options };
          queueMicrotask(() => mockChild.emit("spawn"));
          return mockChild;
        },
        async fetch(url) {
          healthUrl = url;
          return new Response(null, { status: 200 });
        },
      },
    );

    assert.equal(child, mockChild);
    assert.equal(healthUrl, "http://127.0.0.1:41000/healthz");
    assert.equal(spawnCall.command, "/opt/bin/nemo-relay");
    assert.deepEqual(spawnCall.args, [
      "--config",
      configPath,
      "--bind",
      "127.0.0.1:41000",
      "--openai-base-url",
      "https://api.openai.com/v1",
    ]);
    assert.equal(spawnCall.options.cwd, root);
    assert.equal(spawnCall.options.detached, true);
    assert.equal(spawnCall.options.stdio[0], "ignore");
    assert.equal(spawnCall.options.stdio[1], spawnCall.options.stdio[2]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("escalates gateway shutdown from terminate to kill and is safe after exit", async () => {
  const mockChild = new MockChild();
  mockChild.kill = function (signal) {
    this.signals.push(signal);
    if (signal === "SIGKILL") {
      this.signalCode = signal;
      this.emit("exit", null, signal);
    }
    return true;
  };

  await stopRelayGateway(mockChild, 1);
  assert.deepEqual(mockChild.signals, ["SIGTERM", "SIGKILL"]);
  await stopRelayGateway(mockChild, 1);
  assert.deepEqual(mockChild.signals, ["SIGTERM", "SIGKILL"]);
});

test("validates Relay extension paths without workspace containment", async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-extension-")));
  try {
    const extensionDir = join(root, "outside-workspace", "nemo-relay");
    await mkdir(extensionDir, { recursive: true });
    assert.equal(await resolveRelayExtensionPath(startInput(root, { extensionPath: extensionDir })), extensionDir);
    await assert.rejects(
      resolveRelayExtensionPath(startInput(root, { extensionPath: "missing" })),
      (error) => error.code === "pi_relay_extension_not_found" && error.message.includes("relay_extension_path"),
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("keeps non-Relay startup inert and restores Relay extension environment on stop", async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-runtime-")));
  const extensionPath = join(root, "relay-extension.js");
  await writeFile(extensionPath, "export default function () {}\n", "utf8");
  const previous = {
    gateway: process.env.NEMO_RELAY_PI_GATEWAY_URL,
    openai: process.env.NEMO_RELAY_PI_OPENAI_UPSTREAM,
    anthropic: process.env.NEMO_RELAY_PI_ANTHROPIC_UPSTREAM,
  };
  process.env.NEMO_RELAY_PI_GATEWAY_URL = "http://ambient.invalid";
  process.env.NEMO_RELAY_PI_ANTHROPIC_UPSTREAM = "https://ambient.invalid";
  try {
    const unexpected = () => {
      throw new Error("Relay dependency called for non-Relay start");
    };
    const inert = new PiRelayFactory({ resolveCommand: unexpected });
    assert.equal(
      await inert.start(startInput(root, { relay: false }), {
        api: "openai-responses",
        baseUrl: "https://api.openai.com/v1",
      }),
      undefined,
    );
    assert.equal(process.env.NEMO_RELAY_PI_GATEWAY_URL, "http://ambient.invalid");

    const mockChild = new MockChild();
    let stopped = false;
    const factory = new PiRelayFactory({
      async resolveCommand() {
        return "/opt/bin/nemo-relay";
      },
      async checkContract() {
        return { version: [0, 9, 0] };
      },
      async loadPluginConfig() {
        return { version: 1, components: [] };
      },
      async writeConfigs() {
        return {
          configPath: join(root, "relay-config", "config.toml"),
          pluginConfigPath: join(root, "relay-config", "plugins.toml"),
        };
      },
      async findPort() {
        return 41001;
      },
      async startGateway() {
        return mockChild;
      },
      async stopGateway(child) {
        assert.equal(child, mockChild);
        stopped = true;
      },
    });
    const runtime = await factory.start(startInput(root, { extensionPath }), {
      api: "openai-responses",
      baseUrl: "https://api.openai.com/v1",
    });
    assert.equal(process.env.NEMO_RELAY_PI_GATEWAY_URL, "http://127.0.0.1:41001");
    assert.equal(process.env.NEMO_RELAY_PI_OPENAI_UPSTREAM, "https://api.openai.com/v1");
    assert.equal(process.env.NEMO_RELAY_PI_ANTHROPIC_UPSTREAM, undefined);
    await runtime.stop();
    await runtime.stop();
    assert.equal(stopped, true);
    assert.equal(process.env.NEMO_RELAY_PI_GATEWAY_URL, "http://ambient.invalid");
    assert.equal(process.env.NEMO_RELAY_PI_OPENAI_UPSTREAM, undefined);
    assert.equal(process.env.NEMO_RELAY_PI_ANTHROPIC_UPSTREAM, "https://ambient.invalid");
  } finally {
    for (const [name, value] of [
      ["NEMO_RELAY_PI_GATEWAY_URL", previous.gateway],
      ["NEMO_RELAY_PI_OPENAI_UPSTREAM", previous.openai],
      ["NEMO_RELAY_PI_ANTHROPIC_UPSTREAM", previous.anthropic],
    ]) {
      if (value === undefined) {
        delete process.env[name];
      } else {
        process.env[name] = value;
      }
    }
    await rm(root, { recursive: true, force: true });
  }
});

test("maps Relay setup failures to stable Pi adapter errors", async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-errors-")));
  const extensionPath = join(root, "relay-extension.js");
  await writeFile(extensionPath, "export default function () {}\n", "utf8");
  try {
    const input = startInput(root, { extensionPath });
    const model = {
      api: "openai-responses",
      baseUrl: "https://api.openai.com/v1",
    };
    await assert.rejects(
      new PiRelayFactory({
        async resolveCommand() {
          throw new Error("missing");
        },
      }).start(input, model),
      (error) => error.code === "pi_relay_unavailable" && error.message.includes("nemo-relay-cli-bin>=0.9.0"),
    );
    await assert.rejects(
      new PiRelayFactory({
        async resolveCommand() {
          return "/opt/bin/nemo-relay";
        },
        async checkContract() {
          throw new Error("old");
        },
      }).start(input, model),
      (error) => error.code === "pi_relay_incompatible",
    );
    await assert.rejects(
      new PiRelayFactory({
        async resolveCommand() {
          return "/opt/bin/nemo-relay";
        },
        async checkContract() {
          return { version: [0, 9, 0] };
        },
        async loadPluginConfig() {
          throw new Error("FABRIC_RELAY_CONFIG_PATH is required");
        },
      }).start(input, model),
      (error) => error.code === "pi_relay_configuration_failed" && error.message.includes("artifact root"),
    );
    await assert.rejects(
      new PiRelayFactory({
        async resolveCommand() {
          return "/opt/bin/nemo-relay";
        },
        async checkContract() {
          return { version: [0, 9, 0] };
        },
        async loadPluginConfig() {
          return { version: 1, components: [] };
        },
        async writeConfigs() {
          return {
            configPath: join(root, "relay-config", "config.toml"),
            pluginConfigPath: join(root, "relay-config", "plugins.toml"),
          };
        },
        async findPort() {
          return 41002;
        },
        async startGateway() {
          throw new Error("not ready");
        },
      }).start(input, model),
      (error) => error.code === "pi_relay_start_failed" && error.metadata.gateway_log_path.endsWith("gateway.log"),
    );
    await assert.rejects(
      new PiRelayFactory({
        async resolveCommand() {
          return "/opt/bin/nemo-relay";
        },
        async checkContract() {
          return { version: [0, 9, 0] };
        },
        async loadPluginConfig() {
          return { version: 1, components: [] };
        },
        async writeConfigs() {
          return {
            configPath: join(root, "relay-config", "config.toml"),
            pluginConfigPath: join(root, "relay-config", "plugins.toml"),
          };
        },
        async findPort() {
          throw new Error("cannot bind");
        },
      }).start(input, model),
      (error) => error.code === "pi_relay_start_failed" && error.metadata.gateway_log_path.endsWith("gateway.log"),
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

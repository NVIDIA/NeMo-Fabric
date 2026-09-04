// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ChildProcess } from "node:child_process";
import { realpath, stat } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";

import { LifecycleError, type AdapterStartInput } from "nemo-fabric-adapters-common";
import type { JsonObject } from "nemo-fabric-adapter-contract";

import {
  collectRelayArtifacts,
  loadRelayPluginConfig,
  prepareRelayAtifMatchers,
  type RelayArtifact,
  type RelayAtifMatcher,
  type RelayPluginConfig,
  writeRelayConfigs,
} from "./relay-config.js";
import {
  findAvailableTcpPort,
  relayCliContract,
  resolveRelayCommand,
  startRelayGateway,
  stopRelayGateway,
  type RelayGatewayLaunch,
} from "./relay-gateway.js";

const RELAY_INSTALL_COMMAND =
  "Build and install the NeMo Relay 0.9 CLI from source revision " +
  "30b684dbb09231ee956d40abad9af253596a81ad with " +
  "cargo install --path crates/cli --locked, then ensure nemo-relay is on PATH";
const RELAY_EXTENSION_REMEDIATION =
  "set harness.settings.relay_extension_path to the NeMo Relay 0.9 Pi extension file or package directory";
const RELAY_ENV_NAMES = [
  "NEMO_RELAY_PI_GATEWAY_URL",
  "NEMO_RELAY_PI_OPENAI_UPSTREAM",
  "NEMO_RELAY_PI_ANTHROPIC_UPSTREAM",
] as const;

export interface PiRelayModel {
  api: string;
  baseUrl: string;
}

export interface PiRelayControllerFactory {
  start(input: AdapterStartInput, model: PiRelayModel): Promise<PiRelayRuntime | undefined>;
}

export interface PiRelayDependencies {
  resolveCommand: typeof resolveRelayCommand;
  checkContract: typeof relayCliContract;
  loadPluginConfig: typeof loadRelayPluginConfig;
  writeConfigs: typeof writeRelayConfigs;
  findPort: typeof findAvailableTcpPort;
  startGateway: typeof startRelayGateway;
  stopGateway: typeof stopRelayGateway;
}

function relayEnabled(input: AdapterStartInput): boolean {
  return input.runtimeContext.telemetry?.relay_enabled === true;
}

function relayExtensionSetting(input: AdapterStartInput): string | undefined {
  const value = input.config.harness?.settings?.relay_extension_path;
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function relayErrorMetadata(error: unknown, metadata: JsonObject = {}): JsonObject {
  return {
    ...metadata,
    relay_error: error instanceof Error ? error.message : String(error),
  };
}

export async function resolveRelayExtensionPath(input: AdapterStartInput): Promise<string> {
  const configured = relayExtensionSetting(input);
  if (configured === undefined) {
    throw new LifecycleError(
      "pi_relay_extension_not_found",
      `The NeMo Relay Pi extension path is required; ${RELAY_EXTENSION_REMEDIATION}`,
    );
  }
  try {
    const workspace = resolve(input.runtimeContext.environment.workspace ?? input.baseDir);
    const candidate = await realpath(isAbsolute(configured) ? configured : resolve(workspace, configured));
    const info = await stat(candidate);
    if (!info.isFile() && !info.isDirectory()) {
      throw new Error("not a file or directory");
    }
    return candidate;
  } catch (error) {
    throw new LifecycleError(
      "pi_relay_extension_not_found",
      `The configured NeMo Relay Pi extension could not be read; ${RELAY_EXTENSION_REMEDIATION}`,
      { metadata: relayErrorMetadata(error, { configured_path: configured }) },
    );
  }
}

function relayWorkingDirectory(input: AdapterStartInput): string {
  return resolve(input.runtimeContext.environment.workspace ?? input.baseDir);
}

function setRelayEnvironment(gatewayUrl: string, model: PiRelayModel): () => void {
  const previous = new Map<string, string | undefined>();
  for (const name of RELAY_ENV_NAMES) {
    previous.set(name, process.env[name]);
    delete process.env[name];
  }
  process.env.NEMO_RELAY_PI_GATEWAY_URL = gatewayUrl;
  if (model.api === "openai-completions" || model.api === "openai-responses") {
    process.env.NEMO_RELAY_PI_OPENAI_UPSTREAM = model.baseUrl;
  } else if (model.api === "anthropic-messages") {
    process.env.NEMO_RELAY_PI_ANTHROPIC_UPSTREAM = model.baseUrl;
  }
  return () => {
    for (const [name, value] of previous) {
      if (value === undefined) {
        delete process.env[name];
      } else {
        process.env[name] = value;
      }
    }
  };
}

export class PiRelayRuntime {
  readonly extensionPath: string;
  readonly pluginConfig: RelayPluginConfig;
  readonly atifMatchers: RelayAtifMatcher[];
  private readonly child: ChildProcess;
  private readonly launch: RelayGatewayLaunch;
  private readonly pluginConfigPath: string;
  private readonly restoreEnvironment: () => void;
  private readonly stopGateway: typeof stopRelayGateway;
  private stopped = false;

  constructor(options: {
    extensionPath: string;
    pluginConfig: RelayPluginConfig;
    atifMatchers: RelayAtifMatcher[];
    child: ChildProcess;
    launch: RelayGatewayLaunch;
    pluginConfigPath: string;
    restoreEnvironment: () => void;
    stopGateway?: typeof stopRelayGateway;
  }) {
    this.extensionPath = options.extensionPath;
    this.pluginConfig = options.pluginConfig;
    this.atifMatchers = options.atifMatchers;
    this.child = options.child;
    this.launch = options.launch;
    this.pluginConfigPath = options.pluginConfigPath;
    this.restoreEnvironment = options.restoreEnvironment;
    this.stopGateway = options.stopGateway ?? stopRelayGateway;
  }

  async output(artifacts?: RelayArtifact[]): Promise<JsonObject> {
    return {
      relay_runtime: {
        enabled: true,
        emitter: "pi/nemo-relay",
        config_path: process.env.FABRIC_RELAY_CONFIG_PATH ?? null,
        gateway_config_path: this.launch.configPath,
        plugin_config_path: this.pluginConfigPath,
        gateway_url: this.launch.url,
        gateway_log_path: this.launch.logPath,
      },
      relay_artifacts: (artifacts ?? (await collectRelayArtifacts(this.pluginConfig, this.atifMatchers))).map(
        ({ kind, path }): JsonObject => ({ kind, path }),
      ),
    };
  }

  async stop(): Promise<void> {
    if (this.stopped) {
      return;
    }
    try {
      await this.stopGateway(this.child);
      this.stopped = true;
    } catch (error) {
      throw new LifecycleError("pi_relay_stop_failed", "NeMo Relay gateway failed to stop", {
        metadata: relayErrorMetadata(error, { gateway_log_path: this.launch.logPath }),
      });
    } finally {
      this.restoreEnvironment();
    }
  }
}

export class PiRelayFactory implements PiRelayControllerFactory {
  private readonly dependencies: PiRelayDependencies;

  constructor(dependencies: Partial<PiRelayDependencies> = {}) {
    this.dependencies = {
      resolveCommand: resolveRelayCommand,
      checkContract: relayCliContract,
      loadPluginConfig: loadRelayPluginConfig,
      writeConfigs: writeRelayConfigs,
      findPort: findAvailableTcpPort,
      startGateway: startRelayGateway,
      stopGateway: stopRelayGateway,
      ...dependencies,
    };
  }

  async start(input: AdapterStartInput, model: PiRelayModel): Promise<PiRelayRuntime | undefined> {
    if (!relayEnabled(input)) {
      return undefined;
    }

    const extensionPath = await resolveRelayExtensionPath(input);
    let executable: string;
    try {
      executable = await this.dependencies.resolveCommand(
        input.baseDir,
        process.env.FABRIC_TEST_NEMO_RELAY_COMMAND ?? "nemo-relay",
      );
    } catch (error) {
      throw new LifecycleError(
        "pi_relay_unavailable",
        `NeMo Relay CLI executable was not found. ${RELAY_INSTALL_COMMAND}`,
        { metadata: relayErrorMetadata(error) },
      );
    }
    try {
      await this.dependencies.checkContract(executable);
    } catch (error) {
      throw new LifecycleError(
        "pi_relay_incompatible",
        `The installed NeMo Relay CLI is incompatible. ${RELAY_INSTALL_COMMAND}`,
        { metadata: relayErrorMetadata(error) },
      );
    }

    let pluginConfig: RelayPluginConfig;
    let atifMatchers: RelayAtifMatcher[];
    let configPath: string;
    let pluginConfigPath: string;
    try {
      pluginConfig = await this.dependencies.loadPluginConfig(input);
      ({ configPath, pluginConfigPath } = await this.dependencies.writeConfigs(pluginConfig));
      atifMatchers = await prepareRelayAtifMatchers(pluginConfig);
    } catch (error) {
      throw new LifecycleError(
        "pi_relay_configuration_failed",
        "NeMo Relay runtime configuration could not be prepared",
        { metadata: relayErrorMetadata(error) },
      );
    }

    const logPath = join(dirname(configPath), "gateway.log");
    let port: number;
    try {
      port = await this.dependencies.findPort();
    } catch (error) {
      throw new LifecycleError("pi_relay_start_failed", "NeMo Relay gateway failed to start", {
        metadata: relayErrorMetadata(error, { gateway_log_path: logPath }),
      });
    }
    const bind = `127.0.0.1:${port}`;
    const launch: RelayGatewayLaunch = {
      executable,
      configPath,
      bind,
      url: `http://${bind}`,
      logPath,
      ...(model.api === "openai-completions" || model.api === "openai-responses"
        ? { openaiBaseUrl: model.baseUrl }
        : {}),
      ...(model.api === "anthropic-messages" ? { anthropicBaseUrl: model.baseUrl } : {}),
    };
    let child: ChildProcess;
    try {
      child = await this.dependencies.startGateway(launch, relayWorkingDirectory(input));
    } catch (error) {
      throw new LifecycleError("pi_relay_start_failed", "NeMo Relay gateway failed to start", {
        metadata: relayErrorMetadata(error, { gateway_log_path: launch.logPath }),
      });
    }
    const restoreEnvironment = setRelayEnvironment(launch.url, model);
    return new PiRelayRuntime({
      extensionPath,
      pluginConfig,
      atifMatchers,
      child,
      launch,
      pluginConfigPath,
      restoreEnvironment,
      stopGateway: this.dependencies.stopGateway,
    });
  }
}

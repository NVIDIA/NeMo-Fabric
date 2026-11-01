// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { constants } from "node:fs";
import { access, realpath, stat } from "node:fs/promises";
import { delimiter, isAbsolute, join, resolve } from "node:path";

import type { AgentConfig } from "nemo-fabric-adapter-contract";
import {
  LifecycleError,
  insecureHttpModelHosts,
  modelEndpointAllowed,
  type AdapterStartInput,
} from "nemo-fabric-adapters-common";

// Deep Agents Code's own switches for its update checks, automatic updates
// and price and plugin refreshes.
export const QUIET_ENVIRONMENT: Readonly<Record<string, string>> = {
  DEEPAGENTS_CODE_NO_UPDATE_CHECK: "1",
  DEEPAGENTS_CODE_AUTO_UPDATE: "0",
  DEEPAGENTS_CODE_PRICES_AUTO_UPDATE: "0",
  DEEPAGENTS_CODE_PLUGIN_AUTO_UPDATE: "0",
  // Configuration comes from FabricConfig, not a .env file in the workspace.
  DEEPAGENTS_CODE_READ_PROJECT_DOTENV: "0",
};

export interface DeepAgentsCodeSettings {
  approvalMode: "auto" | "yolo";
  terminal: boolean;
}

export interface DeepAgentsCodeModel {
  model: string;
  apiKeyEnv: string;
  baseUrl?: string;
}

const ENVIRONMENT_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/u;

/** Read the settings, which Fabric has already checked against the descriptor's settings_schema. */
export function selectSettings(config: AgentConfig): DeepAgentsCodeSettings {
  const settings: { approval_mode?: unknown; interfaces?: { terminal?: { enabled?: boolean } } } =
    config.harness?.settings ?? {};
  const terminal = settings.interfaces?.terminal;
  return {
    approvalMode: settings.approval_mode === "yolo" ? "yolo" : "auto",
    terminal: terminal !== undefined && terminal.enabled !== false,
  };
}

export function selectModel(config: AgentConfig): DeepAgentsCodeModel {
  const models = Object.entries(config.models ?? {});
  const selected = models.length === 1 ? models[0]?.[1] : config.models?.default;
  if (selected === undefined) {
    throw new LifecycleError("deepagents_code_model_missing", "Deep Agents Code requires a default model or exactly one model");
  }
  if (selected.provider !== "openai" || selected.model.trim().length === 0) {
    throw new LifecycleError("deepagents_code_model_invalid", "Deep Agents Code requires a non-empty model from provider openai");
  }
  const apiKeyEnv = selected.api_key_env ?? "OPENAI_API_KEY";
  if (!ENVIRONMENT_NAME.test(apiKeyEnv)) {
    throw new LifecycleError("deepagents_code_model_invalid", "Deep Agents Code api_key_env must be an environment variable name");
  }
  const baseUrl = selected.base_url ?? undefined;
  const insecureHosts = insecureHttpModelHosts(config.harness?.settings, "deepagents_code_settings_invalid");
  if (baseUrl !== undefined && !modelEndpointAllowed(baseUrl, insecureHosts)) {
    throw new LifecycleError(
      "deepagents_code_model_endpoint_invalid",
      "Deep Agents Code base_url must use HTTPS, loopback HTTP, or a host in insecure_http_model_hosts",
    );
  }
  return { model: selected.model, apiKeyEnv, ...(baseUrl === undefined ? {} : { baseUrl }) };
}

/** Return Deep Agents Code's config.toml for one OpenAI-compatible model. */
export function configToml(model: DeepAgentsCodeModel): string {
  const lines = [
    "[models]",
    `default = ${JSON.stringify(`openai:${model.model}`)}`,
    "",
    "[models.providers.openai]",
    `models = [${JSON.stringify(model.model)}]`,
    `api_key_env = ${JSON.stringify(model.apiKeyEnv)}`,
  ];
  if (model.baseUrl !== undefined) {
    // OpenAI-compatible endpoints serve Chat Completions, not the Responses API.
    lines.push(
      `base_url = ${JSON.stringify(model.baseUrl)}`,
      "",
      "[models.providers.openai.params]",
      "use_responses_api = false",
    );
  }
  return `${lines.join("\n")}\n`;
}

export async function workspacePath(input: AdapterStartInput): Promise<string> {
  const configured = input.runtimeContext.environment.workspace ?? input.baseDir;
  try {
    const workspace = await realpath(isAbsolute(configured) ? configured : resolve(input.baseDir, configured));
    if (!(await stat(workspace)).isDirectory()) throw new Error("not a directory");
    return workspace;
  } catch {
    throw new LifecycleError("deepagents_code_workspace_invalid", "The Deep Agents Code workspace must be an existing directory");
  }
}

/** Return `<artifact root>/.fabric/deepagents-code/runtimes/<runtime-id>`. */
export function runtimeDirectory(input: AdapterStartInput): string {
  const root = input.runtimeContext.artifacts.root;
  const artifactRoot =
    root === undefined || root === null
      ? resolve(input.baseDir, "artifacts")
      : resolve(input.baseDir, String(root));
  return join(artifactRoot, ".fabric", "deepagents-code", "runtimes", input.runtimeContext.runtime_id);
}

/** Find dcode on PATH, as a shell would, for the launcher's absolute path. */
export async function resolveCommand(path: string | undefined): Promise<string> {
  for (const directory of (path ?? "").split(delimiter)) {
    if (directory.length === 0) continue;
    const candidate = join(directory, "dcode");
    try {
      await access(candidate, constants.X_OK);
      if ((await stat(candidate)).isFile()) return candidate;
    } catch {
      // Try the next directory.
    }
  }
  throw new LifecycleError("deepagents_code_harness_unavailable", "dcode was not found on PATH or is not executable");
}

function shellQuote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

/** Return the terminal launcher, which holds paths and switches but no secrets. */
export function launcherScript(options: { home: string; workspace: string; command: string; marker: string }): string {
  const exports = Object.entries({ DEEPAGENTS_HOME: options.home, ...QUIET_ENVIRONMENT })
    .map(([name, value]) => `export ${name}=${shellQuote(value)}`);
  return [
    "#!/bin/sh",
    "# Opens Deep Agents Code's terminal UI in a separate session with this NeMo Fabric",
    "# runtime's configuration. The runtime holds its own thread while it runs, so",
    "# `threads list` shows that thread but `-r` can't open it.",
    `if [ ! -e ${shellQuote(options.marker)} ]; then`,
    "  echo 'This NeMo Fabric runtime is not running.' >&2",
    "  exit 1",
    "fi",
    ...exports,
    `cd ${shellQuote(options.workspace)} || exit 1`,
    `exec ${shellQuote(options.command)} --no-tracing "$@"`,
    "",
  ].join("\n");
}

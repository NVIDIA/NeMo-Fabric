// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { realpath, stat } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";

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
}

export interface DeepAgentsCodeModel {
  model: string;
  apiKeyEnv: string;
  baseUrl?: string;
}

const ENVIRONMENT_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/u;

/** Read the settings, which Fabric has already checked against the descriptor's settings_schema. */
export function selectSettings(config: AgentConfig): DeepAgentsCodeSettings {
  const settings = config.harness?.settings ?? {};
  return { approvalMode: settings.approval_mode === "yolo" ? "yolo" : "auto" };
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

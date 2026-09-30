// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentConfig, AgentMcpServerConfig, AgentModelConfig } from "nemo-fabric-adapter-contract";
import { LifecycleError } from "nemo-fabric-adapters-common";
import { realpath, stat } from "node:fs/promises";
import { join, resolve } from "node:path";

export interface QwenModel {
  id: string;
  apiKeyEnv: string;
  baseUrl?: string;
  temperature?: number;
  topP?: number;
}

export interface QwenMcpServer {
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  httpUrl?: string;
  headers?: Record<string, string>;
  includeTools?: string[];
  excludeTools?: string[];
}

const ENV_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/u;
const ENV_REFERENCE = /\$\{([A-Za-z_][A-Za-z0-9_]*)\}/gu;

function endpoint(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === "https:" ||
      (url.protocol === "http:" && (url.hostname === "localhost" || url.hostname === "[::1]" || /^127(?:\.\d{1,3}){3}$/u.test(url.hostname)));
  } catch {
    return false;
  }
}

function loopbackHostname(hostname: string): boolean {
  return hostname === "localhost" || hostname === "[::1]" || /^127(?:\.\d{1,3}){3}$/u.test(hostname);
}

function expandHeader(
  value: string,
  configuredEnvironment: Record<string, string>,
  parentEnvironment: NodeJS.ProcessEnv,
): string {
  return value.replaceAll(ENV_REFERENCE, (_match, name: string) => {
    const resolved = configuredEnvironment[name] ?? parentEnvironment[name];
    if (resolved === undefined) {
      throw new LifecycleError("qwen_mcp_header_variable_missing", "A Qwen MCP header references an unset variable");
    }
    return resolved;
  });
}

function validateHeader(name: string, value: string): void {
  if (!/^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/u.test(name) || /[\r\n\0]/u.test(value)) {
    throw new LifecycleError("qwen_mcp_invalid_header", "A configured Qwen MCP HTTP header is invalid");
  }
}

function validateMcpServerCommon(name: string, server: AgentMcpServerConfig): void {
  if (name.trim().length === 0 || server.extensions !== undefined ||
      (server.authentication !== undefined && server.authentication !== null)) {
    throw new LifecycleError("qwen_mcp_invalid_server", "Qwen MCP server configuration is unsupported or invalid");
  }
}

export function selectMcpServers(
  config: AgentConfig,
  configuredEnvironment: Record<string, string>,
  parentEnvironment: NodeJS.ProcessEnv,
): Record<string, QwenMcpServer> {
  if (config.mcp?.extensions !== undefined) {
    throw new LifecycleError("qwen_mcp_unsupported", "Qwen does not support MCP extensions");
  }
  const selected = Object.create(null) as Record<string, QwenMcpServer>;
  for (const [name, server] of Object.entries(config.mcp?.servers ?? {})) {
    validateMcpServerCommon(name, server);
    const filters = {
      ...(server.allowed_tools == null ? {} : { includeTools: server.allowed_tools }),
      ...(server.blocked_tools === undefined || server.blocked_tools.length === 0
        ? {}
        : { excludeTools: server.blocked_tools }),
    };
    if (server.transport === "stdio") {
      if (Object.keys(server.custom_headers ?? {}).length > 0 || server.url.trim().length === 0) {
        throw new LifecycleError("qwen_mcp_invalid_server", "Qwen stdio MCP servers require a command and do not accept HTTP headers");
      }
      selected[name] = {
        command: server.url,
        ...(server.args === undefined ? {} : { args: server.args }),
        ...(server.env === undefined ? {} : { env: server.env }),
        ...filters,
      };
      continue;
    }
    if (server.transport === "streamable-http") {
      if ((server.args?.length ?? 0) > 0 || Object.keys(server.env ?? {}).length > 0) {
        throw new LifecycleError("qwen_mcp_invalid_server", "Qwen streamable-HTTP MCP servers do not accept command arguments or environment variables");
      }
      let url: URL;
      try {
        url = new URL(server.url);
      } catch {
        throw new LifecycleError("qwen_mcp_invalid_server", "Qwen streamable-HTTP MCP servers require an HTTP or HTTPS URL");
      }
      if ((url.protocol !== "http:" && url.protocol !== "https:") ||
          (url.protocol === "http:" && !loopbackHostname(url.hostname))) {
        throw new LifecycleError("qwen_mcp_invalid_server", "Qwen streamable-HTTP MCP servers require HTTPS unless the endpoint is loopback");
      }
      const headers = Object.fromEntries(
        Object.entries(server.custom_headers ?? {}).map(([headerName, value]) => {
          const expanded = expandHeader(value, configuredEnvironment, parentEnvironment);
          validateHeader(headerName, expanded);
          return [headerName, expanded];
        }),
      );
      selected[name] = {
        httpUrl: server.url,
        ...(Object.keys(headers).length === 0 ? {} : { headers }),
        ...filters,
      };
      continue;
    }
    throw new LifecycleError("qwen_mcp_transport_unsupported", "Qwen supports stdio and streamable-http MCP transports", {
      metadata: {
        field: `mcp.servers.${name}.transport`,
        transport: server.transport,
        supported_transports: ["stdio", "streamable-http"],
      },
    });
  }
  return selected;
}

function validateModel(model: AgentModelConfig): QwenModel {
  if (model.provider !== "openai" || typeof model.model !== "string" || model.model.trim().length === 0 ||
      typeof model.api_key_env !== "string" || !ENV_NAME.test(model.api_key_env) ||
      (model.base_url != null && (typeof model.base_url !== "string" || !endpoint(model.base_url))) ||
      (model.temperature != null && (typeof model.temperature !== "number" || !Number.isFinite(model.temperature))) ||
      (model.top_p != null && (typeof model.top_p !== "number" || !Number.isFinite(model.top_p) || model.top_p < 0 || model.top_p > 1))) {
    throw new LifecycleError("qwen_invalid_model", "Qwen model configuration does not match the adapter schema");
  }
  if (model.max_tokens != null || model.settings !== undefined || model.extensions !== undefined) {
    throw new LifecycleError("qwen_unsupported_model_field", "Qwen received an undeclared model field");
  }
  return {
    id: model.model,
    apiKeyEnv: model.api_key_env,
    ...(model.base_url == null ? {} : { baseUrl: model.base_url }),
    ...(model.temperature == null ? {} : { temperature: model.temperature }),
    ...(model.top_p == null ? {} : { topP: model.top_p }),
  };
}

export function selectModel(config: AgentConfig): QwenModel {
  const models = Object.entries(config.models ?? {});
  const selected = config.models?.default ?? (models.length === 1 ? models[0]?.[1] : undefined);
  if (selected === undefined) {
    throw new LifecycleError(
      models.length === 0 ? "qwen_model_required" : "qwen_model_ambiguous",
      "Qwen requires one model or a default model role",
    );
  }
  return validateModel(selected);
}

export function selectSystemPrompt(config: AgentConfig): string | { type: "preset"; preset: "qwen_code"; append: string } | undefined {
  const instruction = config.instructions?.system;
  if (instruction == null) return undefined;
  if (typeof instruction.content !== "string" || instruction.content.trim().length === 0) {
    throw new LifecycleError("qwen_invalid_system_instruction", "Qwen system instructions must be non-empty text");
  }
  if (instruction.extensions !== undefined) {
    throw new LifecycleError("qwen_unsupported_instruction_field", "Qwen received an undeclared instruction field");
  }
  if (instruction.mode === undefined || instruction.mode === "replace") return instruction.content;
  if (instruction.mode === "append") return { type: "preset", preset: "qwen_code", append: instruction.content };
  throw new LifecycleError("unsupported_system_instruction_mode", "Qwen supports replace and append system instructions", {
    metadata: { field: "instructions.system.mode", supported_modes: ["replace", "append"] },
  });
}

export type QwenPermissionMode = "default" | "plan" | "auto-edit" | "auto" | "yolo";

export function selectPermissionMode(config: AgentConfig): QwenPermissionMode {
  const settings = config.harness?.settings ?? {};
  const mode = settings.permission_mode ?? "default";
  if (Object.keys(settings).some((name) => name !== "permission_mode") ||
      !["default", "plan", "auto-edit", "auto", "yolo"].includes(String(mode))) {
    throw new LifecycleError("qwen_invalid_settings", "Qwen harness settings do not match its descriptor schema");
  }
  return mode as QwenPermissionMode;
}

export function selectBlockedTools(config: AgentConfig): string[] {
  if (config.tools?.definitions !== undefined || config.tools?.enabled !== undefined || config.tools?.extensions !== undefined) {
    throw new LifecycleError("qwen_unsupported_tools", "Qwen supports only normalized tools.blocked in this adapter");
  }
  const blocked = config.tools?.blocked ?? [];
  if (!blocked.every((name) => typeof name === "string" && name.trim().length > 0)) {
    throw new LifecycleError("qwen_invalid_tools", "Qwen blocked tool names must be non-empty strings");
  }
  return blocked;
}

export async function selectSkillPaths(config: AgentConfig, baseDir: string): Promise<string[]> {
  if (config.skills?.extensions !== undefined) {
    throw new LifecycleError("qwen_unsupported_skills", "Qwen does not support skill extensions");
  }
  const paths: string[] = [];
  const seen = new Set<string>();
  for (const entry of config.skills?.paths ?? []) {
    if (typeof entry !== "string" || entry.trim().length === 0) {
      throw new LifecycleError("qwen_invalid_skill", "Qwen skill paths must be non-empty strings");
    }
    let directory: string;
    try {
      directory = await realpath(resolve(baseDir, entry));
      if (!(await stat(directory)).isDirectory() || !(await stat(join(directory, "SKILL.md"))).isFile()) {
        throw new Error("not a skill directory");
      }
    } catch {
      throw new LifecycleError("qwen_skill_not_found", "A configured Qwen skill directory or SKILL.md is missing");
    }
    if (seen.has(directory)) throw new LifecycleError("qwen_skill_duplicate", "Qwen skill paths must be unique");
    seen.add(directory);
    paths.push(directory);
  }
  return paths;
}

export function rejectUnclaimedConfig(config: AgentConfig): void {
  if (config.harness?.extensions !== undefined ||
      config.runtime?.max_turns != null || config.workflow !== undefined || config.extensions !== undefined) {
    throw new LifecycleError("qwen_unsupported_config", "Qwen received configuration not claimed by its descriptor");
  }
}

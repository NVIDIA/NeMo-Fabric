// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { realpath, stat } from "node:fs/promises";
import { isAbsolute, resolve } from "node:path";

import type { AgentConfig } from "nemo-fabric-adapter-contract";
import {
  LifecycleError,
  type AcpMcpServer,
  type AdapterStartInput,
} from "nemo-fabric-adapters-common";

export interface KiroSettings {
  modeId?: string;
  effort?: "low" | "medium" | "high" | "xhigh" | "max";
  permissionMode: "deny" | "allow_once";
}

const ENV_REFERENCE = /\$\{([A-Za-z_][A-Za-z0-9_]*)\}/gu;
const HEADER_NAME = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/u;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function selectSettings(config: AgentConfig): KiroSettings {
  const value = config.harness?.settings ?? {};
  if (!isRecord(value)) {
    throw new LifecycleError("kiro_settings_invalid", "Kiro harness settings must be an object");
  }
  const modeId = value.mode_id;
  const effort = value.effort;
  const permissionMode = value.permission_mode ?? "deny";
  if (modeId !== undefined && (typeof modeId !== "string" || modeId.length === 0)) {
    throw new LifecycleError("kiro_settings_invalid", "Kiro mode_id must be a non-empty string");
  }
  if (effort !== undefined && !["low", "medium", "high", "xhigh", "max"].includes(String(effort))) {
    throw new LifecycleError("kiro_settings_invalid", "Kiro effort is not supported");
  }
  if (permissionMode !== "deny" && permissionMode !== "allow_once") {
    throw new LifecycleError("kiro_settings_invalid", "Kiro permission_mode is not supported");
  }
  return {
    ...(modeId === undefined ? {} : { modeId: modeId as string }),
    ...(effort === undefined ? {} : { effort: effort as KiroSettings["effort"] }),
    permissionMode,
  };
}

export function selectModel(config: AgentConfig): string | undefined {
  const models = Object.entries(config.models ?? {});
  if (models.length === 0) return undefined;
  if (models.length !== 1 || models[0]?.[0] !== "default") {
    throw new LifecycleError("kiro_model_roles_unsupported", "Kiro accepts only the default model role");
  }
  const model = models[0][1];
  if (model.provider !== "kiro" || model.model.trim().length === 0) {
    throw new LifecycleError("kiro_model_invalid", "Kiro requires a non-empty model from provider kiro");
  }
  if (model.api_key_env !== undefined && model.api_key_env !== null && model.api_key_env !== "KIRO_API_KEY") {
    throw new LifecycleError("kiro_credential_unsupported", "Kiro accepts only KIRO_API_KEY as api_key_env");
  }
  return model.model;
}

export async function workspacePath(input: AdapterStartInput): Promise<string> {
  const configured = input.runtimeContext.environment.workspace ?? input.baseDir;
  try {
    const workspace = await realpath(isAbsolute(configured) ? configured : resolve(input.baseDir, configured));
    if (!(await stat(workspace)).isDirectory()) throw new Error("not a directory");
    return workspace;
  } catch {
    throw new LifecycleError("kiro_workspace_invalid", "The Kiro workspace must be an existing directory");
  }
}

function expandHeader(value: string, environment: Record<string, string>, parent: NodeJS.ProcessEnv): string {
  return value.replaceAll(ENV_REFERENCE, (_reference, name: string) => {
    const configured = environment[name];
    if (Object.hasOwn(environment, name) && configured !== undefined) return configured;
    const inherited = parent[name];
    if (Object.hasOwn(parent, name) && inherited !== undefined) return inherited;
    throw new LifecycleError(
      "kiro_mcp_header_variable_missing",
      "A configured Kiro MCP header references an unavailable environment variable",
    );
  });
}

function validateHeader(name: string, value: string): void {
  if (!HEADER_NAME.test(name) || value.length === 0 || value.trim() !== value || /[\r\n\0]/u.test(value)) {
    throw new LifecycleError("kiro_mcp_invalid_header", "A configured Kiro MCP HTTP header is invalid");
  }
}

function loopback(hostname: string): boolean {
  return hostname === "localhost" || hostname === "[::1]" || /^127(?:\.\d{1,3}){3}$/u.test(hostname);
}

export function selectMcpServers(
  config: AgentConfig,
  environment: Record<string, string>,
  parent: NodeJS.ProcessEnv,
): AcpMcpServer[] {
  if (config.mcp?.extensions !== undefined) {
    throw new LifecycleError("kiro_mcp_extensions_unsupported", "Kiro does not support Fabric MCP extensions");
  }
  const selected: AcpMcpServer[] = [];
  for (const [name, server] of Object.entries(config.mcp?.servers ?? {})) {
    if (name.trim().length === 0 || server.extensions !== undefined) {
      throw new LifecycleError("kiro_mcp_invalid_server", "Kiro MCP server configuration is invalid");
    }
    if (server.authentication !== undefined && server.authentication !== null) {
      throw new LifecycleError(
        "kiro_mcp_authentication_unsupported",
        "Kiro MCP authentication is unsupported; configure an explicit authorization header",
      );
    }
    if (
      (server.allowed_tools !== undefined && server.allowed_tools !== null) ||
      (server.blocked_tools?.length ?? 0) > 0
    ) {
      throw new LifecycleError(
        "kiro_mcp_tool_filters_unsupported",
        "Kiro ACP does not support Fabric MCP per-server tool filters",
      );
    }
    if (server.transport === "stdio") {
      if (
        !isAbsolute(server.url) ||
        server.url.trim().length === 0 ||
        Object.keys(server.custom_headers ?? {}).length > 0
      ) {
        throw new LifecycleError(
          "kiro_mcp_invalid_server",
          "Kiro stdio MCP servers require an absolute executable and do not accept HTTP headers",
        );
      }
      selected.push({
        name,
        command: server.url,
        args: server.args ?? [],
        env: Object.entries(server.env ?? {}).map(([envName, value]) => ({ name: envName, value })),
      });
      continue;
    }
    if (server.transport === "streamable-http" || server.transport === "sse") {
      if ((server.args?.length ?? 0) > 0 || Object.keys(server.env ?? {}).length > 0) {
        throw new LifecycleError(
          "kiro_mcp_invalid_server",
          "Kiro remote MCP servers do not accept command arguments or process environment variables",
        );
      }
      let endpoint: URL;
      try {
        endpoint = new URL(server.url);
      } catch {
        throw new LifecycleError("kiro_mcp_invalid_server", "Kiro remote MCP servers require an HTTP or HTTPS URL");
      }
      if (
        (endpoint.protocol !== "http:" && endpoint.protocol !== "https:") ||
        (endpoint.protocol === "http:" && !loopback(endpoint.hostname))
      ) {
        throw new LifecycleError(
          "kiro_mcp_invalid_server",
          "Kiro remote MCP servers require HTTPS unless the endpoint is loopback",
        );
      }
      const headerNames = new Set<string>();
      const headers = Object.entries(server.custom_headers ?? {}).map(([headerName, rawValue]) => {
        const normalized = headerName.toLowerCase();
        if (headerNames.has(normalized)) {
          throw new LifecycleError("kiro_mcp_invalid_header", "A configured Kiro MCP HTTP header is invalid");
        }
        headerNames.add(normalized);
        const value = expandHeader(rawValue, environment, parent);
        validateHeader(headerName, value);
        return { name: headerName, value };
      });
      selected.push({
        type: server.transport === "streamable-http" ? "http" : "sse",
        name,
        url: server.url,
        headers,
      });
      continue;
    }
    throw new LifecycleError(
      "kiro_mcp_transport_unsupported",
      "Kiro supports stdio, streamable-http, and sse MCP transports",
      {
        metadata: {
          field: `mcp.servers.${name}.transport`,
          transport: server.transport,
          supported_transports: ["stdio", "streamable-http", "sse"],
        },
      },
    );
  }
  return selected;
}

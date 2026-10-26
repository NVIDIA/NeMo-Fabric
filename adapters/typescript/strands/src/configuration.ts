// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { realpath, stat } from "node:fs/promises";
import { isAbsolute, resolve } from "node:path";

import type { AgentConfig } from "nemo-fabric-adapter-contract";
import {
  LifecycleError,
  insecureHttpModelHosts,
  modelEndpointAllowed,
  type AcpMcpServer,
  type AdapterStartInput,
} from "nemo-fabric-adapters-common";

export interface StrandsSettings {
  builtinTools: string[];
  maxInvocations: number;
  insecureHosts: string[];
}

const ENV_REFERENCE = /\$\{([A-Za-z_][A-Za-z0-9_]*)\}/gu;
const HEADER_NAME = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/u;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function selectSettings(config: AgentConfig): StrandsSettings {
  const settings = config.harness?.settings ?? {};
  const supported = ["shell", "read", "write", "edit", "web_fetch"];
  if (
    !isRecord(settings) ||
    Object.keys(settings).some(
      (name) =>
        ![
          "builtin_tools",
          "max_invocations",
          "insecure_http_model_hosts",
        ].includes(name),
    )
  ) {
    throw new LifecycleError(
      "strands_settings_invalid",
      "Strands harness settings are invalid",
    );
  }
  const tools = settings.builtin_tools ?? ["shell", "read", "write", "edit"];
  const max = settings.max_invocations ?? 100;
  if (
    !Array.isArray(tools) ||
    tools.some(
      (tool) => typeof tool !== "string" || !supported.includes(tool),
    ) ||
    new Set(tools).size !== tools.length ||
    !Number.isSafeInteger(max) ||
    Number(max) < 1 ||
    Number(max) > 10000
  ) {
    throw new LifecycleError(
      "strands_settings_invalid",
      "Strands harness settings are invalid",
    );
  }
  return {
    builtinTools: tools as string[],
    maxInvocations: Number(max),
    insecureHosts: insecureHttpModelHosts(settings, "strands_settings_invalid"),
  };
}

export function selectModel(
  config: AgentConfig,
  settings: StrandsSettings,
):
  | {
      name: string;
      endpoint?: string;
      keyEnvironment?: string;
    }
  | undefined {
  const models = Object.entries(config.models ?? {});
  if (models.length === 0) return undefined;
  if (models.length !== 1 || models[0]?.[0] !== "default") {
    throw new LifecycleError(
      "strands_model_roles_unsupported",
      "Strands accepts only the default model role",
    );
  }
  const model = models[0][1];
  if (
    typeof model.model !== "string" ||
    !model.model.trim() ||
    model.model.startsWith("-")
  ) {
    throw new LifecycleError(
      "strands_model_invalid",
      "Strands requires a non-empty model identifier",
    );
  }
  if (model.provider === "bedrock") {
    if (model.base_url != null || model.api_key_env != null) {
      throw new LifecycleError(
        "strands_model_invalid",
        "Bedrock uses the AWS SDK credential chain and endpoint",
      );
    }
    return { name: `bedrock/${model.model}` };
  }
  if (
    !["openai", "nvidia"].includes(model.provider) ||
    !model.base_url ||
    !modelEndpointAllowed(model.base_url, settings.insecureHosts)
  ) {
    throw new LifecycleError(
      "strands_model_invalid",
      "Strands requires an allowed OpenAI-compatible model endpoint",
    );
  }
  const endpoint = new URL(model.base_url);
  if (
    endpoint.username ||
    endpoint.password ||
    endpoint.search ||
    endpoint.hash ||
    (model.api_key_env != null &&
      !/^[A-Za-z_][A-Za-z0-9_]*$/u.test(model.api_key_env))
  ) {
    throw new LifecycleError(
      "strands_model_invalid",
      "Strands model endpoint or credential reference is invalid",
    );
  }
  // Strands' litellm profile is a direct Chat Completions client; no proxy is required.
  return {
    name: `litellm/${model.model}`,
    endpoint: model.base_url,
    ...(model.api_key_env == null ? {} : { keyEnvironment: model.api_key_env }),
  };
}

export function systemInstructions(config: AgentConfig): string | undefined {
  const instruction = config.instructions?.system;
  if (instruction == null) return undefined;
  if (instruction.mode !== "append") {
    throw new LifecycleError(
      "strands_instruction_mode_unsupported",
      "Strands supports only append system instructions",
    );
  }
  return instruction.content;
}

export async function workspacePath(input: AdapterStartInput): Promise<string> {
  const configured =
    input.runtimeContext.environment.workspace ?? input.baseDir;
  try {
    const workspace = await realpath(
      isAbsolute(configured) ? configured : resolve(input.baseDir, configured),
    );
    if (!(await stat(workspace)).isDirectory())
      throw new Error("not a directory");
    return workspace;
  } catch {
    throw new LifecycleError(
      "strands_workspace_invalid",
      "The Strands workspace must be an existing directory",
    );
  }
}

function expandHeader(
  value: string,
  environment: Record<string, string>,
  parent: NodeJS.ProcessEnv,
): string {
  return value.replaceAll(ENV_REFERENCE, (_reference, name: string) => {
    const configured = environment[name];
    if (Object.hasOwn(environment, name) && configured !== undefined)
      return configured;
    const inherited = parent[name];
    if (Object.hasOwn(parent, name) && inherited !== undefined)
      return inherited;
    throw new LifecycleError(
      "strands_mcp_header_variable_missing",
      "A configured Strands MCP header references an unavailable environment variable",
    );
  });
}

function validateHeader(name: string, value: string): void {
  if (
    !HEADER_NAME.test(name) ||
    value.length === 0 ||
    value.trim() !== value ||
    /[\r\n\0]/u.test(value)
  ) {
    throw new LifecycleError(
      "strands_mcp_invalid_header",
      "A configured Strands MCP HTTP header is invalid",
    );
  }
}

function loopback(hostname: string): boolean {
  return (
    hostname === "localhost" ||
    hostname === "[::1]" ||
    /^127(?:\.\d{1,3}){3}$/u.test(hostname)
  );
}

export function selectMcpServers(
  config: AgentConfig,
  environment: Record<string, string>,
  parent: NodeJS.ProcessEnv,
): AcpMcpServer[] {
  if (config.mcp?.extensions !== undefined) {
    throw new LifecycleError(
      "strands_mcp_extensions_unsupported",
      "Strands does not support Fabric MCP extensions",
    );
  }
  const selected: AcpMcpServer[] = [];
  for (const [name, server] of Object.entries(config.mcp?.servers ?? {})) {
    if (name.trim().length === 0 || server.extensions !== undefined) {
      throw new LifecycleError(
        "strands_mcp_invalid_server",
        "Strands MCP server configuration is invalid",
      );
    }
    if (server.authentication !== undefined && server.authentication !== null) {
      throw new LifecycleError(
        "strands_mcp_authentication_unsupported",
        "Strands MCP authentication is unsupported; configure an explicit authorization header",
      );
    }
    if (
      (server.allowed_tools !== undefined && server.allowed_tools !== null) ||
      (server.blocked_tools?.length ?? 0) > 0
    ) {
      throw new LifecycleError(
        "strands_mcp_tool_filters_unsupported",
        "Strands ACP does not support Fabric MCP per-server tool filters",
      );
    }
    if (server.transport === "stdio") {
      if (
        !isAbsolute(server.url) ||
        server.url.trim().length === 0 ||
        Object.keys(server.custom_headers ?? {}).length > 0
      ) {
        throw new LifecycleError(
          "strands_mcp_invalid_server",
          "Strands stdio MCP servers require an absolute executable and do not accept HTTP headers",
        );
      }
      selected.push({
        name,
        command: server.url,
        args: server.args ?? [],
        env: Object.entries(server.env ?? {}).map(([envName, value]) => ({
          name: envName,
          value,
        })),
      });
      continue;
    }
    if (server.transport === "streamable-http" || server.transport === "sse") {
      if (
        (server.args?.length ?? 0) > 0 ||
        Object.keys(server.env ?? {}).length > 0
      ) {
        throw new LifecycleError(
          "strands_mcp_invalid_server",
          "Strands remote MCP servers do not accept command arguments or process environment variables",
        );
      }
      let endpoint: URL;
      try {
        endpoint = new URL(server.url);
      } catch {
        throw new LifecycleError(
          "strands_mcp_invalid_server",
          "Strands remote MCP servers require an HTTP or HTTPS URL",
        );
      }
      if (
        (endpoint.protocol !== "http:" && endpoint.protocol !== "https:") ||
        (endpoint.protocol === "http:" && !loopback(endpoint.hostname))
      ) {
        throw new LifecycleError(
          "strands_mcp_invalid_server",
          "Strands remote MCP servers require HTTPS unless the endpoint is loopback",
        );
      }
      const headerNames = new Set<string>();
      const headers = Object.entries(server.custom_headers ?? {}).map(
        ([headerName, rawValue]) => {
          const normalized = headerName.toLowerCase();
          if (headerNames.has(normalized)) {
            throw new LifecycleError(
              "strands_mcp_invalid_header",
              "A configured Strands MCP HTTP header is invalid",
            );
          }
          headerNames.add(normalized);
          const value = expandHeader(rawValue, environment, parent);
          validateHeader(headerName, value);
          return { name: headerName, value };
        },
      );
      selected.push({
        type: server.transport === "streamable-http" ? "http" : "sse",
        name,
        url: server.url,
        headers,
      });
      continue;
    }
    throw new LifecycleError(
      "strands_mcp_transport_unsupported",
      "Strands supports stdio, streamable-http, and sse MCP transports",
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

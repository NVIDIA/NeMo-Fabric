// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pi SDK integration boundary. It translates normalized adapter configuration
// into a controlled in-memory Pi session, including model credentials, skills,
// extensions, custom tools, and workspace containment.

import { mkdtemp, realpath, rm, stat } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { extname, isAbsolute, join, relative, resolve, sep } from "node:path";

import type { AuthProvider, JsonRpcMessage, McpTransport } from "@earendil-works/pi-mcp";
import type {
  AgentSession,
  ExtensionCommandContextActions,
  ExtensionFactory,
  McpExtensionOptions,
  McpServerConfig,
  McpTransportFactory,
  ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { createJiti } from "jiti/static";
import type { AgentConfig, AgentModelConfig, AgentToolDefinition, JsonObject } from "nemo-fabric-adapter-contract";
import { LifecycleError, type AdapterStartInput } from "nemo-fabric-adapters-common";

import {
  PiRelayFactory,
  type PiRelayControllerFactory,
  type PiRelayRuntime,
} from "./relay.js";
import type { PiPromptOutcome, PiSessionFactory, PiSessionHandle } from "./runtime.js";

// ProviderConfigInput isn't re-exported from pi's public entrypoint; derive it from
// registerProvider's param type (ModelRuntime's constructor is private, so no InstanceType).
type PiModelRuntime = Awaited<ReturnType<typeof import("@earendil-works/pi-coding-agent").ModelRuntime.create>>;
type PiProviderConfigInput = Parameters<PiModelRuntime["registerProvider"]>[1];
type PiCatalogModel = NonNullable<PiProviderConfigInput["models"]>[number];
// A model already resolved from Pi's catalog (getModel), used as the base to overlay onto.
type PiResolvedModel = NonNullable<ReturnType<PiModelRuntime["getModel"]>>;

// Per-model Pi metadata (api, context_window, max_tokens, cost, reasoning, input) is adapter-
// owned data carried in the model `extensions` block. It is only used to DEFINE a model Pi does
// not already know (a gateway model); a model already in Pi's catalog is used as-is and never
// rebuilt, so none of its native properties change. The adapter supplies no defaults of its own
// — for an unknown model every field Pi requires must be configured, mirroring how defining a
// custom model in a local Pi models.json requires those fields.

// The wire protocols Pi understands; a configured model extensions.api must be one of these.
const SUPPORTED_MODEL_APIS = [
  "openai-completions",
  "openai-responses",
  "azure-openai-responses",
  "openai-codex-responses",
  "anthropic-messages",
  "bedrock-converse-stream",
  "google-generative-ai",
  "google-vertex",
  "mistral-conversations",
  "pi-messages",
] as const;

interface PiHarnessSettings {
  extensions: string[];
}

interface PiToolFactoryContext {
  name: string;
  settings: JsonObject;
  workspace: string;
}

export function withCustomBaseUrl<T extends { api: string; baseUrl: string; compat?: object }>(
  catalogModel: T,
  baseUrl: string | null | undefined,
  overrideBaseUrl = true,
): T {
  if (!baseUrl) {
    return catalogModel;
  }
  const model = overrideBaseUrl ? { ...catalogModel, baseUrl } : catalogModel;
  if (catalogModel.api !== "openai-completions") {
    return model;
  }
  return {
    ...model,
    // Generic OpenAI-compatible proxies may reject provider-specific
    // reasoning_content fields when Pi replays an assistant tool call.
    compat: { ...catalogModel.compat, requiresThinkingAsText: true },
  };
}

export function modelAwareCompactionReserveTokens(
  configuredReserveTokens: number,
  maxOutputTokens: number,
  contextWindow: number,
): number {
  const reserveTokens = Math.max(configuredReserveTokens, maxOutputTokens);
  if (contextWindow <= 0) {
    return reserveTokens;
  }
  return Math.min(reserveTokens, Math.floor(contextWindow / 2));
}

type PiToolFactory = (context: PiToolFactoryContext) => ToolDefinition | Promise<ToolDefinition>;

const PI_BUILTIN_TOOL_NAMES = new Set(["read", "bash", "edit", "write", "grep", "find", "ls"]);
const TOOL_MODULE_EXTENSIONS = new Set([".js", ".mjs", ".cjs", ".ts", ".mts", ".cts"]);
const PI_HARNESS_INSTALL_COMMAND =
  "npm install @earendil-works/pi-ai@^1.0.0 @earendil-works/pi-coding-agent@^1.0.0 @earendil-works/pi-mcp@^1.0.0";
const MCP_STARTUP_TIMEOUT_MS = 10_000;

interface PiSdkModules {
  InMemoryCredentialStore: typeof import("@earendil-works/pi-ai").InMemoryCredentialStore;
  StdioTransport: typeof import("@earendil-works/pi-mcp").StdioTransport;
  StreamableHttpTransport: typeof import("@earendil-works/pi-mcp").StreamableHttpTransport;
  createAgentSession: typeof import("@earendil-works/pi-coding-agent").createAgentSession;
  createMcpExtension: typeof import("@earendil-works/pi-coding-agent").createMcpExtension;
  DefaultResourceLoader: typeof import("@earendil-works/pi-coding-agent").DefaultResourceLoader;
  ModelRuntime: typeof import("@earendil-works/pi-coding-agent").ModelRuntime;
  SessionManager: typeof import("@earendil-works/pi-coding-agent").SessionManager;
  SettingsManager: typeof import("@earendil-works/pi-coding-agent").SettingsManager;
}

function isMissingModuleError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error.code === "ERR_MODULE_NOT_FOUND" || error.code === "MODULE_NOT_FOUND")
  );
}

async function loadPiSdk(): Promise<PiSdkModules> {
  let ai: typeof import("@earendil-works/pi-ai");
  let codingAgent: typeof import("@earendil-works/pi-coding-agent");
  let mcp: typeof import("@earendil-works/pi-mcp");
  try {
    [ai, codingAgent, mcp] = await Promise.all([
      import("@earendil-works/pi-ai"),
      import("@earendil-works/pi-coding-agent"),
      import("@earendil-works/pi-mcp"),
    ]);
  } catch (error) {
    if (isMissingModuleError(error)) {
      throw new LifecycleError(
        "pi_harness_unavailable",
        `The Pi SDK harness is not installed. Install a compatible harness with: ${PI_HARNESS_INSTALL_COMMAND}`,
      );
    }
    throw new LifecycleError("pi_harness_load_failed", "The installed Pi SDK harness could not be loaded");
  }

  if (
    typeof ai.InMemoryCredentialStore !== "function" ||
    typeof mcp.StdioTransport !== "function" ||
    typeof mcp.StreamableHttpTransport !== "function" ||
    typeof codingAgent.createAgentSession !== "function" ||
    typeof codingAgent.createMcpExtension !== "function" ||
    typeof codingAgent.DefaultResourceLoader !== "function" ||
    typeof codingAgent.ModelRuntime !== "function" ||
    typeof codingAgent.SessionManager !== "function" ||
    typeof codingAgent.SettingsManager !== "function"
  ) {
    throw new LifecycleError(
      "pi_harness_incompatible",
      "The installed Pi SDK harness does not expose the APIs required by this adapter",
    );
  }

  return {
    InMemoryCredentialStore: ai.InMemoryCredentialStore,
    StdioTransport: mcp.StdioTransport,
    StreamableHttpTransport: mcp.StreamableHttpTransport,
    createAgentSession: codingAgent.createAgentSession,
    createMcpExtension: codingAgent.createMcpExtension,
    DefaultResourceLoader: codingAgent.DefaultResourceLoader,
    ModelRuntime: codingAgent.ModelRuntime,
    SessionManager: codingAgent.SessionManager,
    SettingsManager: codingAgent.SettingsManager,
  };
}

const MCP_HEADER_NAME = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/u;
const MCP_ENV_REFERENCE = /\$\{([A-Za-z_][A-Za-z0-9_]*)\}/gu;
const MCP_SERVER_NAME = /^[A-Za-z0-9_-]+$/u;

function loopbackHostname(hostname: string): boolean {
  return hostname === "localhost" || hostname === "[::1]" || /^127(?:\.\d{1,3}){3}$/u.test(hostname);
}

function expandMcpHeader(
  value: string,
  configuredEnvironment: Record<string, string>,
  parentEnvironment: NodeJS.ProcessEnv,
): string {
  return value.replaceAll(MCP_ENV_REFERENCE, (_reference, variable: string) => {
    const configured = configuredEnvironment[variable];
    if (Object.hasOwn(configuredEnvironment, variable) && configured !== undefined) {
      return configured;
    }
    const inherited = parentEnvironment[variable];
    if (Object.hasOwn(parentEnvironment, variable) && inherited !== undefined) {
      return inherited;
    }
    throw new LifecycleError(
      "pi_mcp_header_variable_missing",
      "A configured Pi MCP header references an environment variable that is not available",
    );
  });
}

function validateMcpHeader(name: string, value: string): void {
  if (!MCP_HEADER_NAME.test(name) || value.length === 0 || value.trim() !== value) {
    throw new LifecycleError("pi_mcp_invalid_header", "A configured Pi MCP HTTP header is invalid");
  }
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    if (codePoint === undefined || codePoint > 0xFF || (codePoint < 0x20 && codePoint !== 0x09) || codePoint === 0x7F) {
      throw new LifecycleError("pi_mcp_invalid_header", "A configured Pi MCP HTTP header is invalid");
    }
  }
}

function validateMcpToolNames(names: readonly string[] | null | undefined): void {
  if (names?.some((name) => name.includes("*")) === true) {
    throw new LifecycleError(
      "pi_mcp_tool_pattern_unsupported",
      "Pi MCP tool filters require exact tool names and do not accept patterns",
    );
  }
}

function mcpToolExposure(
  allowed: string[] | null | undefined,
  blocked: string[] | undefined,
): Pick<McpServerConfig, "exposure" | "toolExposure"> {
  validateMcpToolNames(allowed);
  validateMcpToolNames(blocked);
  const toolExposure: Record<string, "direct" | "hidden"> = {};
  if (allowed !== undefined && allowed !== null) {
    for (const name of allowed) {
      toolExposure[name] = "direct";
    }
  }
  for (const name of blocked ?? []) {
    toolExposure[name] = "hidden";
  }
  return {
    exposure: allowed === undefined || allowed === null ? "direct" : "hidden",
    ...(Object.keys(toolExposure).length === 0 ? {} : { toolExposure }),
  };
}

export function selectPiMcpServers(
  config: AgentConfig,
  configuredEnvironment: Record<string, string>,
  parentEnvironment: NodeJS.ProcessEnv,
): Record<string, McpServerConfig> {
  if (config.mcp?.extensions !== undefined) {
    throw new LifecycleError("pi_mcp_extensions_unsupported", "Pi does not support Fabric MCP extensions");
  }
  const selected = Object.create(null) as Record<string, McpServerConfig>;
  for (const [name, server] of Object.entries(config.mcp?.servers ?? {})) {
    if (!MCP_SERVER_NAME.test(name) || server.extensions !== undefined) {
      throw new LifecycleError("pi_mcp_invalid_server", "Pi MCP server configuration is unsupported or invalid");
    }
    if (server.authentication !== undefined && server.authentication !== null) {
      throw new LifecycleError(
        "pi_mcp_authentication_unsupported",
        "Pi MCP authentication is unsupported; configure a custom authorization header instead",
      );
    }
    const exposure = mcpToolExposure(server.allowed_tools, server.blocked_tools);
    if (server.transport === "stdio") {
      if (server.url.trim().length === 0 || Object.keys(server.custom_headers ?? {}).length > 0) {
        throw new LifecycleError(
          "pi_mcp_invalid_server",
          "Pi stdio MCP servers require a command and do not accept HTTP headers",
        );
      }
      selected[name] = {
        type: "stdio",
        command: server.url,
        ...(server.args === undefined ? {} : { args: server.args }),
        ...(server.env === undefined ? {} : { env: server.env }),
        ...exposure,
      };
      continue;
    }
    if (server.transport === "streamable-http") {
      if ((server.args?.length ?? 0) > 0 || Object.keys(server.env ?? {}).length > 0) {
        throw new LifecycleError(
          "pi_mcp_invalid_server",
          "Pi streamable-HTTP MCP servers do not accept command arguments or environment variables",
        );
      }
      let endpoint: URL;
      try {
        endpoint = new URL(server.url);
      } catch {
        throw new LifecycleError(
          "pi_mcp_invalid_server",
          "Pi streamable-HTTP MCP servers require an HTTP or HTTPS URL",
        );
      }
      if (
        (endpoint.protocol !== "http:" && endpoint.protocol !== "https:") ||
        (endpoint.protocol === "http:" && !loopbackHostname(endpoint.hostname))
      ) {
        throw new LifecycleError(
          "pi_mcp_invalid_server",
          "Pi streamable-HTTP MCP servers require HTTPS unless the endpoint is loopback",
        );
      }
      const headerNames = new Set<string>();
      const headers = Object.fromEntries(
        Object.entries(server.custom_headers ?? {}).map(([headerName, value]) => {
          const normalizedName = headerName.toLowerCase();
          if (headerNames.has(normalizedName)) {
            throw new LifecycleError("pi_mcp_invalid_header", "A configured Pi MCP HTTP header is invalid");
          }
          headerNames.add(normalizedName);
          const expanded = expandMcpHeader(value, configuredEnvironment, parentEnvironment);
          validateMcpHeader(headerName, expanded);
          return [headerName, expanded];
        }),
      );
      selected[name] = {
        type: "http",
        url: server.url,
        ...(Object.keys(headers).length === 0 ? {} : { headers }),
        ...exposure,
      };
      continue;
    }
    throw new LifecycleError(
      "pi_mcp_transport_unsupported",
      `Pi does not support MCP transport ${JSON.stringify(server.transport)}; supported transports: stdio, streamable-http`,
      {
        metadata: {
          field: `mcp.servers.${name}.transport`,
          transport: server.transport,
          supported_transports: ["stdio", "streamable-http"],
        },
      },
    );
  }
  return selected;
}

type McpCredentials = NonNullable<McpExtensionOptions["credentials"]>;

function isolatedMcpCredentials(): McpCredentials {
  const credentials = {
    forServer: () => ({
      load: () => undefined,
      save: () => undefined,
      withRefreshLock: (operation: () => Promise<unknown>) => operation(),
    }),
    tokens: () => undefined,
    remove: () => false,
  };
  // SAFETY: Pi types this option as its concrete credential-store class even though
  // the extension uses only this public method surface.
  return credentials as unknown as McpCredentials;
}

interface McpReadiness {
  fail(): void;
  ready(): void;
  result: Promise<boolean>;
}

function mcpReadiness(): McpReadiness {
  let settle: (ready: boolean) => void = () => undefined;
  let settled = false;
  const result = new Promise<boolean>((resolve) => {
    settle = resolve;
  });
  const finish = (ready: boolean) => {
    if (!settled) {
      settled = true;
      settle(ready);
    }
  };
  return { fail: () => finish(false), ready: () => finish(true), result };
}

function trackedMcpTransport(transport: McpTransport, readiness: McpReadiness): McpTransport {
  const toolListRequests = new Set<unknown>();
  return {
    async start() {
      try {
        await transport.start();
      } catch (error) {
        readiness.fail();
        throw error;
      }
    },
    async send(message: JsonRpcMessage) {
      if ("method" in message && message.method === "tools/list" && "id" in message) {
        toolListRequests.add(message.id);
      }
      try {
        await transport.send(message);
      } catch (error) {
        readiness.fail();
        throw error;
      }
    },
    close: () => transport.close(),
    onMessage: (listener) => transport.onMessage((message) => {
      if ("id" in message && toolListRequests.has(message.id)) {
        toolListRequests.delete(message.id);
        if ("error" in message) {
          readiness.fail();
        } else {
          readiness.ready();
        }
      }
      listener(message);
    }),
    onError: (listener) => transport.onError((error) => {
      readiness.fail();
      listener(error);
    }),
    onClose: (listener) => transport.onClose(() => {
      readiness.fail();
      listener();
    }),
    ...(transport.setProtocolVersion === undefined
      ? {}
      : { setProtocolVersion: (version: string) => transport.setProtocolVersion?.(version) }),
  };
}

function expandHome(value: string): string {
  if (value === "~") {
    return homedir();
  }
  if (value.startsWith("~/") || (process.platform === "win32" && value.startsWith("~\\"))) {
    return join(homedir(), value.slice(2));
  }
  return value;
}

function mcpTransportFactory(pi: PiSdkModules, readiness: Map<string, McpReadiness>): McpTransportFactory {
  return (entry, cwd, authProvider: AuthProvider | undefined) => {
    const status = readiness.get(entry.name);
    if (status === undefined) {
      throw new Error("Unrecognized Pi MCP server");
    }
    const transport = "url" in entry.config
      ? new pi.StreamableHttpTransport({
          url: entry.config.url,
          headers: entry.config.headers,
          authProvider,
        })
      : new pi.StdioTransport({
          command: expandHome(entry.config.command),
          args: entry.config.args?.map(expandHome),
          cwd: resolve(cwd, expandHome(entry.config.cwd ?? ".")),
          env: entry.config.env,
          stderr: "pipe",
        });
    return trackedMcpTransport(transport, status);
  };
}

async function requireMcpConnections(readiness: Map<string, McpReadiness>): Promise<void> {
  if (readiness.size === 0) {
    return;
  }
  let cancelTimer: () => void = () => undefined;
  const results = await Promise.race([
    Promise.all(Array.from(readiness.values(), (status) => status.result)),
    new Promise<undefined>((resolve) => {
      const timer = setTimeout(resolve, MCP_STARTUP_TIMEOUT_MS);
      cancelTimer = () => clearTimeout(timer);
    }),
  ]);
  cancelTimer();
  if (results === undefined || results.some((ready) => !ready)) {
    throw new LifecycleError(
      "pi_mcp_connection_failed",
      "One or more configured Pi MCP servers could not connect",
    );
  }
}

function mcpRegistrationExtension(servers: Record<string, McpServerConfig>): ExtensionFactory {
  return (api) => {
    for (const [name, server] of Object.entries(servers)) {
      api.registerMcpServer(name, server);
    }
  };
}

function selectModel(config: AgentConfig): AgentModelConfig {
  const entries = Object.entries(config.models ?? {});
  if (entries.length === 0) {
    throw new LifecycleError("pi_model_required", "The Pi adapter requires one configured model");
  }
  const selected = config.models?.default ?? (entries.length === 1 ? entries[0]?.[1] : undefined);
  if (selected === undefined) {
    throw new LifecycleError(
      "pi_model_ambiguous",
      "Configure a default model role when the Pi adapter receives multiple models",
    );
  }
  return selected;
}

/**
 * Build the Pi catalog entry to register for the selected model, as an OVERLAY on `base` (the
 * model's existing Pi catalog entry, or undefined when Pi doesn't know it). The rule is a thin
 * pass-through: a field the config SUPPLIES (via the model `extensions` block, or `base_url` for
 * the endpoint) is used; a field it OMITS falls back to `base`'s value, and if there is no base
 * (an unknown gateway model) an optional field is left UNSET so Pi applies its own default. The
 * adapter invents no defaults of its own.
 *
 * It does, however, require what Pi has no usable default for. For an unknown model that means a
 * complete-enough definition: `api` (enforced at registration — Pi can't resolve the wire
 * protocol otherwise), `base_url`, and `context_window`/`max_tokens` (absent, these resolve to
 * undefined and break compaction/limit math rather than getting a sane default). Supplied values
 * are validated (`api` a {@link SUPPORTED_MODEL_APIS} value, token limits positive, cost
 * well-formed) so a malformed override fails fast instead of reaching Pi.
 */
export function buildCatalogModel(model: AgentModelConfig, base: PiResolvedModel | undefined): PiCatalogModel {
  const meta = (model.extensions ?? {}) as Record<string, unknown>;
  const fail = (reason: string): never => {
    throw new LifecycleError("pi_model_extensions_invalid", reason, { metadata: { model: model.model } });
  };
  if (meta.api !== undefined && !SUPPORTED_MODEL_APIS.includes(meta.api as (typeof SUPPORTED_MODEL_APIS)[number])) {
    throw new LifecycleError(
      "pi_model_api_invalid",
      `Unsupported model api '${String(meta.api)}'; model extensions.api must be one of: ${SUPPORTED_MODEL_APIS.join(", ")}`,
      { metadata: { model: model.model } },
    );
  }
  // Validate a SUPPLIED token limit (positive number); when absent, inherit base's value or leave
  // it unset for Pi to default. Never fabricate a value.
  const positive = (value: unknown, inherited: number | undefined, field: string): number | undefined => {
    if (value === undefined) {
      return inherited;
    }
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      return fail(`model extensions.${field} must be a positive number`);
    }
    return value;
  };
  const suppliedInput = Array.isArray(meta.input)
    ? (meta.input.filter((item) => item === "text" || item === "image") as ("text" | "image")[])
    : undefined;
  const api = typeof meta.api === "string" ? (meta.api as PiCatalogModel["api"]) : base?.api;
  const baseUrl = model.base_url ?? base?.baseUrl;
  const reasoning = meta.reasoning !== undefined ? meta.reasoning === true : base?.reasoning;
  const input = suppliedInput !== undefined && suppliedInput.length > 0 ? suppliedInput : base?.input;
  const cost = parseCost(meta.cost, fail) ?? base?.cost;
  const contextWindow = positive(meta.context_window, base?.contextWindow, "context_window");
  const maxTokens = positive(model.max_tokens ?? meta.max_tokens, base?.maxTokens, "max_tokens");
  // For an unknown model (not in Pi's catalog), require the fields Pi has no usable default for:
  // base_url (Pi can't reach it otherwise) and context_window/max_tokens (absent, they resolve to
  // undefined and break compaction/limit math rather than getting a sane Pi default). `api` is
  // enforced separately at registration. The adapter still invents nothing — it requires the
  // config to supply what Pi genuinely needs, failing fast with a clear error.
  if (base === undefined) {
    const missing = [
      baseUrl === undefined ? "base_url" : undefined,
      contextWindow === undefined ? "context_window" : undefined,
      maxTokens === undefined ? "max_tokens" : undefined,
    ].filter((field): field is string => field !== undefined);
    if (missing.length > 0) {
      fail(
        `model '${model.model}' is not known to Pi, so a complete definition is required; ` +
          `missing: ${missing.join(", ")} (set base_url and extensions.context_window/max_tokens)`,
      );
    }
  }
  // Build a sparse entry: include only resolved fields. Omitted optional ones (cost, reasoning,
  // input) are left unset so Pi applies its own defaults — the adapter adds none. The SDK input
  // type marks some fields non-optional but tolerates their absence at runtime.
  const entry: Record<string, unknown> = { id: model.model };
  if (typeof meta.name === "string" || base?.name !== undefined) {
    entry.name = typeof meta.name === "string" ? meta.name : base?.name;
  }
  if (api !== undefined) entry.api = api;
  if (baseUrl !== undefined) entry.baseUrl = baseUrl;
  if (reasoning !== undefined) entry.reasoning = reasoning;
  if (input !== undefined) entry.input = input;
  if (cost !== undefined) entry.cost = cost;
  if (contextWindow !== undefined) entry.contextWindow = contextWindow;
  if (maxTokens !== undefined) entry.maxTokens = maxTokens;
  return entry as unknown as PiCatalogModel;
}

// Validate a supplied cost (all four numeric rates) and reject a partial object (e.g. `{}`),
// which Pi would turn into NaN costs. Returns undefined when no cost is configured, so the
// caller keeps the base/default cost instead.
function parseCost(value: unknown, fail: (reason: string) => never): PiCatalogModel["cost"] | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "object" || value === null) {
    return fail("model extensions.cost must be an object with numeric input/output/cacheRead/cacheWrite rates");
  }
  const record = value as Record<string, unknown>;
  const rates = ["input", "output", "cacheRead", "cacheWrite"] as const;
  const parsed: Record<string, number> = {};
  for (const rate of rates) {
    const amount = record[rate];
    if (typeof amount !== "number" || !Number.isFinite(amount) || amount < 0) {
      return fail(`model extensions.cost.${rate} must be a non-negative number`);
    }
    parsed[rate] = amount;
  }
  // SAFETY: every rate was just validated as a finite non-negative number above.
  return parsed as unknown as PiCatalogModel["cost"];
}

function harnessSettings(config: AgentConfig): PiHarnessSettings {
  const raw = config.harness?.settings;
  const extensions = raw?.extensions;
  if (extensions === undefined) {
    return { extensions: [] };
  }
  if (!Array.isArray(extensions)) {
    throw new LifecycleError("pi_invalid_settings", "Pi extension settings do not match the adapter schema");
  }
  const values: string[] = [];
  for (const entry of extensions) {
    if (typeof entry !== "string") {
      throw new LifecycleError("pi_invalid_settings", "Pi extension settings do not match the adapter schema");
    }
    values.push(entry);
  }
  return { extensions: values };
}

function containedBy(root: string, candidate: string): boolean {
  const path = relative(root, candidate);
  return path === "" || (!path.startsWith(`..${sep}`) && path !== ".." && !isAbsolute(path));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function resolveToolModule(workspace: string, reference: string): Promise<{ path: string; exportName: string }> {
  const match = /^(?<path>[^#]+?)(?:#(?<export>[A-Za-z_$][\w$]*))?$/.exec(reference);
  const configuredPath = match?.groups?.path;
  if (configuredPath === undefined || isAbsolute(configuredPath)) {
    throw new LifecycleError(
      "pi_tool_ref_invalid",
      "Pi tool module references must use a workspace-relative path with an optional export fragment",
    );
  }
  let candidate: string;
  try {
    candidate = await realpath(resolve(workspace, configuredPath));
  } catch {
    throw new LifecycleError("pi_tool_module_not_found", "A configured Pi tool module does not exist");
  }
  if (!containedBy(workspace, candidate)) {
    throw new LifecycleError("pi_tool_module_outside_workspace", "Pi tool modules must be contained by the workspace");
  }
  if (!(await stat(candidate)).isFile() || !TOOL_MODULE_EXTENSIONS.has(extname(candidate))) {
    throw new LifecycleError(
      "pi_tool_module_invalid",
      "Pi tool modules must be JavaScript or TypeScript files",
    );
  }
  return { path: candidate, exportName: match?.groups?.export ?? "default" };
}

function validateToolDefinition(name: string, value: unknown): ToolDefinition {
  if (
    !isRecord(value) ||
    value.name !== name ||
    typeof value.label !== "string" ||
    value.label.length === 0 ||
    typeof value.description !== "string" ||
    value.description.length === 0 ||
    !isRecord(value.parameters) ||
    typeof value.execute !== "function"
  ) {
    throw new LifecycleError(
      "pi_tool_factory_invalid",
      "A Pi tool factory returned an invalid tool definition or a mismatched tool name",
      { metadata: { tool: name } },
    );
  }
  // SAFETY: the checks above verify every field of the ToolDefinition contract at runtime.
  return value as unknown as ToolDefinition;
}

export async function resolveCustomTools(
  workspace: string,
  definitions: Record<string, AgentToolDefinition>,
): Promise<ToolDefinition[]> {
  const jiti = createJiti(import.meta.url, { interopDefault: false });
  const tools: ToolDefinition[] = [];
  for (const [name, definition] of Object.entries(definitions)) {
    if (PI_BUILTIN_TOOL_NAMES.has(name)) {
      throw new LifecycleError("pi_tool_collision", "A Fabric tool definition collides with a Pi built-in tool", {
        metadata: { tool: name },
      });
    }
    if (definition.kind !== "module") {
      throw new LifecycleError("pi_tool_kind_unsupported", "Pi supports only module tool definitions");
    }
    const moduleReference = await resolveToolModule(workspace, definition.ref);
    let loaded: unknown;
    try {
      loaded = await jiti.import(moduleReference.path);
    } catch {
      throw new LifecycleError("pi_tool_module_load_failed", "A configured Pi tool module could not be loaded", {
        metadata: { tool: name },
      });
    }
    const factory = isRecord(loaded) ? loaded[moduleReference.exportName] : undefined;
    if (typeof factory !== "function") {
      throw new LifecycleError("pi_tool_factory_missing", "A configured Pi tool module export is not a factory", {
        metadata: { tool: name },
      });
    }
    let tool: unknown;
    try {
      tool = await (factory as PiToolFactory)({
        name,
        settings: definition.settings ?? {},
        workspace,
      });
    } catch {
      throw new LifecycleError("pi_tool_factory_failed", "A configured Pi tool factory failed", {
        metadata: { tool: name },
      });
    }
    tools.push(validateToolDefinition(name, tool));
  }
  return tools;
}

function rejectToolCollisions(customTools: ToolDefinition[], configuredExtensionTools: string[]): void {
  const customNames = new Set(customTools.map((tool) => tool.name));
  const seenExtensionNames = new Set<string>();
  for (const name of configuredExtensionTools) {
    if (PI_BUILTIN_TOOL_NAMES.has(name) || customNames.has(name) || seenExtensionNames.has(name)) {
      throw new LifecycleError("pi_tool_collision", "Two configured Pi tool sources use the same tool name", {
        metadata: { tool: name },
      });
    }
    seenExtensionNames.add(name);
  }
}

function relayExtensionLoadErrors(
  errors: Array<{ path: string; error: string }>,
  extensions: Array<{ path: string; resolvedPath?: string }>,
  relay: PiRelayRuntime | undefined,
  workspace: string,
): Array<{ path: string; error: string }> {
  if (relay === undefined) {
    return [];
  }
  const loaded = extensions.some((extension) =>
    [extension.path, extension.resolvedPath].some(
      (path) => path !== undefined && resolve(workspace, path) === relay.extensionPath,
    ),
  );
  if (loaded) {
    return [];
  }
  return errors.filter((error) => {
    const path = resolve(workspace, error.path);
    return path === relay.extensionPath || containedBy(relay.extensionPath, path);
  });
}

async function resolveExtensionPaths(workspace: string, configured: string[]): Promise<string[]> {
  const resolved: string[] = [];
  for (const entry of configured) {
    if (isAbsolute(entry)) {
      throw new LifecycleError("pi_extension_outside_workspace", "Pi extension paths must be workspace-relative");
    }
    let candidate: string;
    try {
      candidate = await realpath(resolve(workspace, entry));
    } catch {
      throw new LifecycleError("pi_extension_not_found", "A configured Pi extension path does not exist");
    }
    if (!containedBy(workspace, candidate)) {
      throw new LifecycleError("pi_extension_outside_workspace", "Pi extension path resolves outside the workspace");
    }
    const info = await stat(candidate);
    if (!info.isFile() || (!candidate.endsWith(".ts") && !candidate.endsWith(".js"))) {
      throw new LifecycleError("pi_unsupported_extension", "Pi extensions must be .ts or .js files");
    }
    resolved.push(candidate);
  }
  return resolved;
}

async function resolveSkillPaths(baseDir: string, configured: string[]): Promise<string[]> {
  const resolved: string[] = [];
  for (const entry of configured) {
    let candidate: string;
    try {
      candidate = await realpath(resolve(baseDir, entry));
    } catch {
      throw new LifecycleError("pi_skill_not_found", "A configured NeMo Fabric skill path does not exist");
    }
    let info;
    try {
      info = await stat(candidate);
    } catch {
      throw new LifecycleError("pi_skill_not_found", "A configured NeMo Fabric skill path does not exist");
    }
    if (!info.isDirectory()) {
      throw new LifecycleError("pi_skill_invalid", "NeMo Fabric skill paths must be directories");
    }
    try {
      if (!(await stat(join(candidate, "SKILL.md"))).isFile()) {
        throw new Error("not a file");
      }
    } catch {
      throw new LifecycleError(
        "pi_skill_invalid",
        "NeMo Fabric skill directories must contain a SKILL.md file",
      );
    }
    resolved.push(candidate);
  }
  return resolved;
}

function credentialValue(input: AdapterStartInput, name: string): string | undefined {
  return input.runtimeContext.environment.env?.[name] ?? process.env[name];
}

function promptText(message: { content?: unknown }): string {
  if (!Array.isArray(message.content)) {
    return "";
  }
  return message.content
    .filter((block): block is { type: "text"; text: string } => {
      return (
        typeof block === "object" &&
        block !== null &&
        "type" in block &&
        block.type === "text" &&
        "text" in block &&
        typeof block.text === "string"
      );
    })
    .map((block) => block.text)
    .join("");
}

function unsupportedSessionAction(name: string): never {
  throw new LifecycleError("pi_unsupported_session_operation", `Pi session operation ${name} is not supported`);
}

class PiSdkSessionHandle implements PiSessionHandle {
  readonly relay?: PiRelayRuntime;
  private readonly session: AgentSession;
  private readonly state: { shutdownRequested: boolean };
  private readonly unsubscribeTurnCounter: () => void;
  private stopped = false;
  private cumulativeTurnCount = 0;
  private readonly runtimeDir: string;

  constructor(session: AgentSession, state: { shutdownRequested: boolean }, runtimeDir: string, relay?: PiRelayRuntime) {
    this.session = session;
    this.state = state;
    this.runtimeDir = runtimeDir;
    this.relay = relay;
    this.unsubscribeTurnCounter = this.session.subscribe((event) => {
      if (event.type === "turn_start") {
        this.cumulativeTurnCount += 1;
      }
    });
  }

  get turnCount(): number {
    return this.cumulativeTurnCount;
  }

  async prompt(text: string): Promise<PiPromptOutcome> {
    let accepted = false;
    let turnStarted = false;
    let finalAssistant:
      | { role: "assistant"; content: unknown; stopReason: string; errorMessage?: string }
      | undefined;
    const unsubscribe = this.session.subscribe((event) => {
      if (event.type === "turn_start") {
        turnStarted = true;
      }
      if (event.type === "message_end" && event.message.role === "assistant") {
        finalAssistant = event.message;
      }
    });
    try {
      await this.session.prompt(text, {
        expandPromptTemplates: true,
        source: "interactive",
        preflightResult: () => {
          accepted = true;
        },
      });
    } finally {
      unsubscribe();
    }
    return {
      accepted,
      turnStarted,
      turnCount: this.cumulativeTurnCount,
      text: finalAssistant === undefined ? undefined : promptText(finalAssistant),
      stopReason: finalAssistant?.stopReason,
      errorMessage: finalAssistant?.errorMessage,
      shutdownRequested: this.state.shutdownRequested,
    };
  }

  async stop(): Promise<void> {
    if (this.stopped) {
      return;
    }
    this.stopped = true;
    let failure: unknown;
    try {
      try {
        await this.session.abort();
      } catch (error) {
        failure = error;
      }
      try {
        await this.session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
      } catch (error) {
        failure ??= error;
      }
      try {
        this.unsubscribeTurnCounter();
      } catch (error) {
        failure ??= error;
      }
      try {
        this.session.dispose();
      } catch (error) {
        failure ??= error;
      }
    } finally {
      await rm(this.runtimeDir, { recursive: true, force: true });
    }
    if (failure !== undefined) {
      throw failure;
    }
  }
}

export class PiSdkSessionFactory implements PiSessionFactory {
  private readonly relayFactory: PiRelayControllerFactory;

  constructor(relayFactory: PiRelayControllerFactory = new PiRelayFactory()) {
    this.relayFactory = relayFactory;
  }

  async create(input: AdapterStartInput): Promise<PiSessionHandle> {
    const systemInstruction = input.config.instructions?.system;
    if (systemInstruction?.mode === "append") {
      throw new LifecycleError(
        "unsupported_system_instruction_mode",
        "Pi does not support instructions.system.mode='append'; supported modes: replace",
        {
          metadata: {
            field: "instructions.system.mode",
            mode: systemInstruction.mode,
            supported_modes: ["replace"],
          },
        },
      );
    }
    const mcpServers = selectPiMcpServers(
      input.config,
      input.runtimeContext.environment.env ?? {},
      process.env,
    );
    const pi = await loadPiSdk();
    let workspace: string;
    try {
      workspace = await realpath(resolve(input.runtimeContext.environment.workspace ?? input.baseDir));
      if (!(await stat(workspace)).isDirectory()) {
        throw new Error("not a directory");
      }
    } catch {
      throw new LifecycleError("pi_workspace_invalid", "The Fabric runtime workspace must be a directory");
    }
    const selected = selectModel(input.config);
    const apiKeyEnv = selected.api_key_env;
    if (apiKeyEnv === undefined || apiKeyEnv === null || apiKeyEnv.length === 0) {
      throw new LifecycleError("pi_api_key_env_required", "The selected Pi model requires api_key_env");
    }
    const apiKey = credentialValue(input, apiKeyEnv);
    if (apiKey === undefined || apiKey.length === 0) {
      throw new LifecycleError("pi_credential_missing", `Credential environment variable ${apiKeyEnv} is not set`);
    }

    const settings = pi.SettingsManager.inMemory({}, { projectTrusted: false });
    const extensionPaths = await resolveExtensionPaths(workspace, harnessSettings(input.config).extensions);
    const skillPaths = await resolveSkillPaths(input.baseDir, input.config.skills?.paths ?? []);
    const customTools = await resolveCustomTools(workspace, input.config.tools?.definitions ?? {});
    const credentials = new pi.InMemoryCredentialStore();
    const modelRuntime = await pi.ModelRuntime.create({
      credentials,
      modelsPath: null,
      allowModelNetwork: false,
      refreshOnCreate: false,
    });
    await modelRuntime.setRuntimeApiKey(selected.provider, apiKey);

    // Register the selected model into Pi's catalog as an overlay on its existing entry (if any):
    // supplied config fields override, omitted ones keep Pi's native value, and for an unknown
    // (gateway) model the overlay base is empty so defaults + a required `api` apply. This way
    // selecting a known provider/model and changing nothing preserves all its native properties,
    // while an explicit base_url / extensions field is still honored. A Fabric agent runs ONE
    // model role per session (selectModel enforces default-or-sole), mirroring the other
    // single-model adapters. The selected provider's credential was set above.
    const existing = modelRuntime.getModel(selected.provider, selected.model);
    const catalogEntry = buildCatalogModel(selected, existing);
    try {
      modelRuntime.registerProvider(selected.provider, { models: [catalogEntry] });
    } catch (error) {
      const cause = error instanceof Error ? error.message : String(error);
      // Pi throws here with a 'no "api" specified' message when it can't resolve the model's
      // wire protocol (not a built-in and no extensions.api). Only THAT cause maps to the
      // api-required remediation; any other registration failure keeps a generic code so the
      // message isn't misleading.
      if (/no "api" specified/i.test(cause)) {
        throw new LifecycleError(
          "pi_model_api_required",
          `Set extensions.api for a gateway-served model (one of: ${SUPPORTED_MODEL_APIS.join(", ")}); the selected model has no api Pi can resolve`,
          { metadata: { provider: selected.provider, model: selected.model, cause } },
        );
      }
      throw new LifecycleError(
        "pi_provider_registration_failed",
        `Pi rejected the configuration for provider '${selected.provider}'`,
        { metadata: { provider: selected.provider, cause } },
      );
    }
    const catalogModel = modelRuntime.getModel(selected.provider, selected.model);
    if (catalogModel === undefined) {
      throw new LifecycleError("pi_model_unknown", "The selected provider and model are not present in Pi's catalog");
    }
    // base_url was already applied during registration; withCustomBaseUrl only adds the
    // OpenAI-compatible-proxy compat shim here (never re-overriding the resolved baseUrl).
    const model = withCustomBaseUrl(catalogModel, selected.base_url, false);
    const compactionReserveTokens = modelAwareCompactionReserveTokens(
      settings.getCompactionReserveTokens(),
      model.maxTokens ?? 0,
      model.contextWindow ?? 0,
    );
    settings.applyOverrides({
      compaction: {
        reserveTokens: compactionReserveTokens,
      },
    });
    let relay: PiRelayRuntime | undefined;
    let handle: PiSdkSessionHandle | undefined;
    let runtimeDir: string | undefined;
    try {
      relay = await this.relayFactory.start(input, {
        api: model.api,
        baseUrl: model.baseUrl,
      });
      if (relay !== undefined && !extensionPaths.includes(relay.extensionPath)) {
        extensionPaths.push(relay.extensionPath);
      }
      runtimeDir = await mkdtemp(join(tmpdir(), "nemo-fabric-pi-"));
      const agentDir = runtimeDir;
      const mcpConnections = new Map(Object.keys(mcpServers).map((name) => [name, mcpReadiness()]));
      const extensionFactories = Object.keys(mcpServers).length === 0
        ? []
        : [
            { name: "nemo-fabric-mcp-servers", hidden: true, factory: mcpRegistrationExtension(mcpServers) },
            {
              name: "nemo-fabric-mcp",
              hidden: true,
              factory: pi.createMcpExtension({
                loadConfig: () => ({ servers: [], errors: [], autoEnableCodemode: false }),
                credentials: isolatedMcpCredentials(),
                logPath: join(agentDir, "mcp.log"),
                createTransport: mcpTransportFactory(pi, mcpConnections),
                startupWaitMs: MCP_STARTUP_TIMEOUT_MS,
              }),
            },
          ];
      const resourceLoader = new pi.DefaultResourceLoader({
        cwd: workspace,
        agentDir,
        settingsManager: settings,
        additionalExtensionPaths: extensionPaths,
        additionalSkillPaths: skillPaths,
        extensionFactories,
        noExtensions: true,
        noSkills: true,
        noPromptTemplates: true,
        noThemes: true,
        noContextFiles: true,
        systemPrompt: systemInstruction?.content,
      });
      await resourceLoader.reload();
      const extensionResult = resourceLoader.getExtensions();
      rejectToolCollisions(
        customTools,
        extensionResult.extensions.flatMap((extension) => [...extension.tools.keys()]),
      );
      const extensionErrors = extensionResult.errors;
      const relayErrors = relayExtensionLoadErrors(extensionErrors, extensionResult.extensions, relay, workspace);
      if (relayErrors.length > 0) {
        throw new LifecycleError(
          "pi_relay_extension_load_failed",
          "The configured NeMo Relay Pi extension failed to load; use the Pi extension from the NeMo Relay 0.9 release",
          {
            metadata: {
              count: relayErrors.length,
              relay_error: relayErrors.map((error) => error.error).join("; "),
            },
          },
        );
      }
      if (extensionErrors.length > 0) {
        throw new LifecycleError(
          "pi_extension_load_failed",
          "One or more configured Pi extensions failed to load or conflict",
          {
            metadata: {
              count: extensionErrors.length,
              extension_error: extensionErrors.map((error) => error.error).join("; "),
              extension_paths: extensionErrors.map((error) => error.path),
            },
          },
        );
      }
      const skillDiagnostics = resourceLoader.getSkills().diagnostics;
      const blockingSkillDiagnostics = skillDiagnostics.filter(
        (diagnostic) => diagnostic.type === "error" || diagnostic.type === "collision",
      );
      for (const diagnostic of skillDiagnostics.filter((entry) => entry.type === "warning")) {
        process.stderr.write(`Pi skill warning: ${diagnostic.message}\n`);
      }
      if (blockingSkillDiagnostics.length > 0) {
        throw new LifecycleError("pi_skill_load_failed", "One or more configured NeMo Fabric skills failed to load", {
          metadata: { count: blockingSkillDiagnostics.length },
        });
      }

      const enabled = input.config.tools?.enabled;
      const blocked = input.config.tools?.blocked ?? [];
      const state = { shutdownRequested: false };
      const { session } = await pi.createAgentSession({
        cwd: workspace,
        agentDir,
        model,
        modelRuntime,
        resourceLoader,
        sessionManager: pi.SessionManager.inMemory(workspace),
        settingsManager: settings,
        customTools,
        tools: enabled === null ? undefined : enabled,
        excludeTools: blocked,
      });
      handle = new PiSdkSessionHandle(session, state, runtimeDir, relay);
      const blockedNames = new Set(blocked);
      const availableNames = new Set(session.getAllTools().map((tool) => tool.name));
      const missing = (enabled ?? []).filter((name) => !blockedNames.has(name) && !availableNames.has(name));
      if (missing.length > 0) {
        throw new LifecycleError("pi_tool_missing", "One or more enabled tools are not registered", {
          metadata: { tools: missing },
        });
      }

      const commandContextActions: ExtensionCommandContextActions = {
        waitForIdle: () => session.waitForIdle(),
        newSession: async () => unsupportedSessionAction("newSession"),
        fork: async () => unsupportedSessionAction("fork"),
        navigateTree: async () => unsupportedSessionAction("navigateTree"),
        switchSession: async () => unsupportedSessionAction("switchSession"),
        reload: async () => unsupportedSessionAction("reload"),
      };
      await session.bindExtensions({
        mode: "print",
        commandContextActions,
        abortHandler: () => {
          void session.abort();
        },
        shutdownHandler: () => {
          state.shutdownRequested = true;
          void session.abort();
        },
        onError: () => {
          process.stderr.write("Pi extension handler failed\n");
        },
      });
      await requireMcpConnections(mcpConnections);
      // bindExtensions emits session_start; emitting it here would create a
      // duplicate Relay session scope.
      return handle;
    } catch (error) {
      await handle?.stop().catch(() => {
        process.stderr.write("Pi session cleanup failed after adapter startup error\n");
      });
      await relay?.stop().catch(() => {
        process.stderr.write("NeMo Relay cleanup failed after Pi adapter startup error\n");
      });
      if (handle === undefined && runtimeDir !== undefined) {
        await rm(runtimeDir, { recursive: true, force: true });
      }
      throw error;
    }
  }
}

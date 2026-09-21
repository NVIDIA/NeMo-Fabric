// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// OpenCode v2 SDK boundary. It owns dynamic harness loading, the embedded host,
// one native session, and the temporary credential environment required by the
// upstream SDK.

import type { AgentUsage } from "nemo-fabric-adapter-contract";
import type { AdapterStartInput } from "nemo-fabric-adapters-common";
import { LifecycleError } from "nemo-fabric-adapters-common";
import { randomUUID } from "node:crypto";
import { realpath, stat } from "node:fs/promises";
import { join, resolve } from "node:path";

import { selectModel, selectSystemInstruction } from "./configuration.js";
import { ModelEndpointProxy } from "./model-endpoint-proxy.js";
import type { OpenCodePromptOutcome, OpenCodeSessionFactory, OpenCodeSessionHandle } from "./runtime.js";

type OpenCodeClient = Awaited<ReturnType<(typeof import("@opencode/sdk"))["OpenCode"]["create"]>>;
type OpenCodeEmbedOptions = { overrides?: unknown[] };
type OpenCodeEmbedOptionsLoader = (configContent: string) => Promise<OpenCodeEmbedOptions>;
type ModuleResolver = (specifier: string) => string;
type ModuleLoader = (specifier: string) => Promise<unknown>;
interface EndpointProxy {
  readonly url: string;
  close(): Promise<void>;
}
type EndpointProxyFactory = (baseUrl: string) => Promise<EndpointProxy>;
type OpenCodeMcpServer =
  | { type: "local"; command: string[]; environment?: Record<string, string> }
  | { type: "remote"; url: string; headers?: Record<string, string>; oauth: false };
type OpenCodeMcpServers = Record<string, OpenCodeMcpServer>;
interface OpenCodeSkill {
  directory: string;
  location: string;
}
type EmbeddedOpenCodeCreate = (
  options: Parameters<(typeof import("@opencode/sdk"))["OpenCode"]["create"]>[0],
  embedOptions: OpenCodeEmbedOptions,
) => Promise<OpenCodeClient>;

// Fabric invokes OpenCode non-interactively. Session rules are evaluated after
// OpenCode's agent defaults, so they override upstream "ask" rules without
// changing the permissions of other OpenCode sessions in the process.
const NONINTERACTIVE_SESSION_PERMISSIONS = [
  { action: "external_directory", resource: "*", effect: "deny" },
  { action: "read", resource: "*.env", effect: "deny" },
  { action: "read", resource: "*.env.*", effect: "deny" },
  { action: "read", resource: "*.env.example", effect: "allow" },
  { action: "question", resource: "*", effect: "deny" },
] as const;
const MCP_CONNECTION_TIMEOUT_MS = 35_000;
const MCP_CONNECTION_POLL_INTERVAL_MS = 50;
const HTTP_HEADER_NAME = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/u;

function loopbackHostname(hostname: string): boolean {
  return hostname === "localhost" || hostname === "::1" || hostname === "[::1]" || /^127(?:\.\d{1,3}){3}$/u.test(hostname);
}

/**
 * OpenCode's Promise SDK accepts embedding overrides. They set the
 * location-specific configuration and instruction services to OpenCode's
 * documented `project: false, global: false` profile while leaving the session
 * location at the Fabric workspace.
 */
async function loadIsolatedEmbedOptions(configContent: string): Promise<OpenCodeEmbedOptions> {
  const [{ Config }, { InstructionDiscovery }] = await Promise.all([
    import("@opencode/core/config"),
    import("@opencode/core/instruction-discovery"),
  ]);
  return {
    overrides: [
      Config.node.replace(Config.configured({ project: false, global: false, content: configContent })),
      InstructionDiscovery.node.replace(InstructionDiscovery.configured({ project: false, global: false })),
    ],
  };
}

function hostConfigContent(
  model: ReturnType<typeof selectModel>,
  baseUrl = model.baseUrl,
  systemInstruction?: string,
  skillPaths: string[] = [],
  mcpServers: OpenCodeMcpServers = {},
): string {
  const literals = new Map<string, string>();
  const literal = (value: string): string => {
    const placeholder = `__NEMO_FABRIC_OPENCODE_LITERAL_${randomUUID()}__`;
    literals.set(placeholder, value);
    return placeholder;
  };
  const provider = {
    settings: { apiKey: `{env:${model.apiKeyEnv}}` },
  };
  if (baseUrl !== undefined) {
    Object.assign(provider, {
      package: "aisdk:@ai-sdk/openai-compatible",
      settings: { ...provider.settings, baseURL: baseUrl },
      models: {
        [model.model]:
          model.sampling === undefined
            ? {}
            : {
                body: {
                  ...(model.sampling.temperature === undefined ? {} : { temperature: model.sampling.temperature }),
                  ...(model.sampling.topP === undefined ? {} : { top_p: model.sampling.topP }),
                },
              },
      },
    });
  }
  const configuredMcpServers = Object.fromEntries(
    Object.entries(mcpServers).map(([name, server]) => [
      literal(name),
      server.type === "local"
        ? {
            type: "local",
            command: server.command.map(literal),
            ...(server.environment === undefined
              ? {}
              : {
                  environment: Object.fromEntries(
                    Object.entries(server.environment).map(([key, value]) => [literal(key), literal(value)]),
                  ),
                }),
          }
        : {
            type: "remote",
            url: literal(server.url),
            ...(server.headers === undefined
              ? {}
              : {
                  headers: Object.fromEntries(
                    Object.entries(server.headers).map(([key, value]) => [literal(key), literal(value)]),
                  ),
                }),
            oauth: false,
          },
    ]),
  );
  let content = JSON.stringify({
    providers: {
      [model.provider]: provider,
    },
    ...(systemInstruction === undefined ? {} : { agents: { build: { system: literal(systemInstruction) } } }),
    ...(skillPaths.length === 0 ? {} : { skills: skillPaths.map(literal) }),
    ...(Object.keys(configuredMcpServers).length === 0 ? {} : { mcp: { servers: configuredMcpServers } }),
  });
  for (const [placeholder, value] of literals) {
    const encoded = JSON.stringify(value).replaceAll("{", "\\u007b");
    content = content.replace(JSON.stringify(placeholder), () => encoded);
  }
  return content;
}

async function resolveSkills(baseDir: string, configured: string[]): Promise<OpenCodeSkill[]> {
  const skills: OpenCodeSkill[] = [];
  for (const entry of configured) {
    if (typeof entry !== "string") {
      throw new LifecycleError("opencode_invalid_skill", "OpenCode skill paths must be strings");
    }
    let candidate: string;
    try {
      candidate = await realpath(resolve(baseDir, entry));
    } catch {
      throw new LifecycleError("opencode_skill_not_found", "A configured NeMo Fabric skill path does not exist");
    }
    try {
      if (!(await stat(candidate)).isDirectory()) {
        throw new Error("not a directory");
      }
    } catch {
      throw new LifecycleError("opencode_skill_invalid", "NeMo Fabric skill paths must be directories");
    }
    let location: string;
    try {
      location = await realpath(join(candidate, "SKILL.md"));
      if (!(await stat(location)).isFile()) {
        throw new Error("not a file");
      }
    } catch {
      throw new LifecycleError(
        "opencode_skill_invalid",
        "NeMo Fabric skill directories must contain a SKILL.md file",
      );
    }
    skills.push({ directory: candidate, location });
  }
  return skills;
}

async function verifyLoadedSkills(
  client: OpenCodeClient,
  workspace: string,
  configured: OpenCodeSkill[],
): Promise<void> {
  if (configured.length === 0) {
    return;
  }
  let loaded: Array<{ name?: unknown; location?: unknown }>;
  try {
    await client.plugin.awaitActivation({ location: { directory: workspace } });
    loaded = (await client.skill.list({ location: { directory: workspace } })).data;
  } catch {
    throw new LifecycleError("opencode_skill_status_unavailable", "OpenCode could not determine the configured skill status");
  }
  const byLocation = new Map(
    loaded
      .filter((skill): skill is { name: string; location: string } =>
        typeof skill.name === "string" && typeof skill.location === "string",
      )
      .map((skill) => [skill.location, skill]),
  );
  const configuredSkills = configured.map((skill) => byLocation.get(skill.location));
  if (configuredSkills.some((skill) => skill === undefined)) {
    throw new LifecycleError(
      "opencode_skill_load_failed",
      "OpenCode did not load every configured NeMo Fabric skill",
    );
  }
  const names = new Set<string>();
  for (const skill of configuredSkills) {
    if (skill === undefined) {
      continue;
    }
    if (names.has(skill.name)) {
      throw new LifecycleError(
        "opencode_skill_name_duplicate",
        "Configured NeMo Fabric skills must have distinct names",
      );
    }
    names.add(skill.name);
  }
}

function expandMcpHeaders(
  headers: Record<string, string> | undefined,
  configuredEnvironment: Record<string, string>,
  parentEnvironment: NodeJS.ProcessEnv,
): Record<string, string> | undefined {
  if (headers === undefined) {
    return undefined;
  }
  return Object.fromEntries(
    Object.entries(headers).map(([name, value]) => {
      const expanded = value.replace(/\$\{([A-Za-z_][A-Za-z0-9_]*)\}/gu, (_reference, variable: string) => {
        const configured = configuredEnvironment[variable];
        if (Object.hasOwn(configuredEnvironment, variable) && configured !== undefined) {
          return configured;
        }
        const inherited = parentEnvironment[variable];
        if (Object.hasOwn(parentEnvironment, variable) && inherited !== undefined) {
          return inherited;
        }
        throw new LifecycleError(
          "opencode_mcp_header_variable_missing",
          "A configured OpenCode MCP header references an environment variable that is not available",
        );
      });
      validateMcpHttpHeader(name, expanded);
      return [name, expanded];
    }),
  );
}

function validateMcpHttpHeader(name: string, value: string): void {
  if (
    !HTTP_HEADER_NAME.test(name) ||
    value.length === 0 ||
    value.trim().length === 0 ||
    value.startsWith(" ") ||
    value.startsWith("\t") ||
    value.endsWith(" ") ||
    value.endsWith("\t")
  ) {
    throw new LifecycleError("opencode_mcp_invalid_header", "A configured OpenCode MCP HTTP header is invalid");
  }
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    if (
      codePoint === undefined ||
      codePoint > 0xFF ||
      (codePoint < 0x20 && codePoint !== 0x09) ||
      codePoint === 0x7F
    ) {
      throw new LifecycleError("opencode_mcp_invalid_header", "A configured OpenCode MCP HTTP header is invalid");
    }
  }
}

function selectMcpServers(
  config: AdapterStartInput["config"],
  configuredEnvironment: Record<string, string>,
  parentEnvironment: NodeJS.ProcessEnv,
): OpenCodeMcpServers {
  const servers = Object.create(null) as OpenCodeMcpServers;
  for (const [name, server] of Object.entries(config.mcp?.servers ?? {})) {
    if (server.authentication !== undefined && server.authentication !== null) {
      throw new LifecycleError(
        "opencode_mcp_authentication_unsupported",
        "OpenCode MCP authentication is unsupported; configure a custom authorization header instead",
      );
    }
    if (
      (server.allowed_tools !== undefined && server.allowed_tools !== null) ||
      (server.blocked_tools?.length ?? 0) > 0
    ) {
      throw new LifecycleError(
        "opencode_mcp_tool_filters_unsupported",
        "OpenCode MCP per-server tool filters are unsupported",
      );
    }
    if (server.transport === "stdio") {
      if (Object.keys(server.custom_headers ?? {}).length > 0) {
        throw new LifecycleError(
          "opencode_mcp_invalid_server",
          "OpenCode stdio MCP servers do not accept custom HTTP headers",
        );
      }
      servers[name] = {
        type: "local",
        command: [server.url, ...(server.args ?? [])],
        ...(Object.keys(server.env ?? {}).length === 0 ? {} : { environment: server.env }),
      };
      continue;
    }
    if (server.transport === "streamable-http") {
      if ((server.args?.length ?? 0) > 0) {
        throw new LifecycleError(
          "opencode_mcp_invalid_server",
          "OpenCode streamable-HTTP MCP servers do not accept command arguments",
        );
      }
      if (Object.keys(server.env ?? {}).length > 0) {
        throw new LifecycleError(
          "opencode_mcp_invalid_server",
          "OpenCode streamable-HTTP MCP servers do not accept environment variables",
        );
      }
      let endpoint: URL;
      try {
        endpoint = new URL(server.url);
      } catch {
        throw new LifecycleError(
          "opencode_mcp_invalid_server",
          "OpenCode streamable-HTTP MCP servers require an HTTP or HTTPS URL",
        );
      }
      if (endpoint.protocol !== "http:" && endpoint.protocol !== "https:") {
        throw new LifecycleError(
          "opencode_mcp_invalid_server",
          "OpenCode streamable-HTTP MCP servers require an HTTP or HTTPS URL",
        );
      }
      if (endpoint.protocol === "http:" && !loopbackHostname(endpoint.hostname)) {
        throw new LifecycleError(
          "opencode_mcp_invalid_server",
          "OpenCode streamable-HTTP MCP servers require HTTPS unless the endpoint is loopback",
        );
      }
      servers[name] = {
        type: "remote",
        url: server.url,
        ...(Object.keys(server.custom_headers ?? {}).length === 0
          ? {}
          : { headers: expandMcpHeaders(server.custom_headers, configuredEnvironment, parentEnvironment) }),
        oauth: false,
      };
      continue;
    }
    throw new LifecycleError(
      "opencode_mcp_transport_unsupported",
      `OpenCode does not support MCP transport ${JSON.stringify(server.transport)}; supported transports: stdio, streamable-http`,
      {
        metadata: {
          field: `mcp.servers.${name}.transport`,
          transport: server.transport,
          supported_transports: ["stdio", "streamable-http"],
        },
      },
    );
  }
  return servers;
}

async function waitForMcpConnections(
  client: OpenCodeClient,
  workspace: string,
  servers: string[],
): Promise<void> {
  if (servers.length === 0) {
    return;
  }
  const deadline = Date.now() + MCP_CONNECTION_TIMEOUT_MS;
  while (true) {
    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) {
      throw new LifecycleError(
        "opencode_mcp_connection_timeout",
        "Timed out waiting for configured OpenCode MCP servers to connect",
        { metadata: { servers } },
      );
    }
    const signal = AbortSignal.timeout(remainingMs);
    let statusByServer: Map<string, string>;
    try {
      const listed = await client.mcp.list(
        { location: { directory: workspace } },
        { signal },
      );
      statusByServer = new Map(
        listed.data.map((server: { name: string; status: { status: string } }) => [server.name, server.status.status]),
      );
    } catch {
      if (signal.aborted) {
        throw new LifecycleError(
          "opencode_mcp_connection_timeout",
          "Timed out waiting for configured OpenCode MCP servers to connect",
          { metadata: { servers } },
        );
      }
      throw new LifecycleError(
        "opencode_mcp_status_unavailable",
        "OpenCode could not determine the configured MCP connection status",
      );
    }
    for (const name of servers) {
      const status = statusByServer.get(name);
      if (status !== undefined && status !== "pending" && status !== "connected") {
        throw new LifecycleError(
          "opencode_mcp_connection_failed",
          "A configured OpenCode MCP server did not connect",
          { metadata: { server: name, status } },
        );
      }
    }
    if (servers.every((name) => statusByServer.get(name) === "connected")) {
      return;
    }
    const pollDelayMs = Math.min(MCP_CONNECTION_POLL_INTERVAL_MS, deadline - Date.now());
    if (pollDelayMs <= 0) {
      throw new LifecycleError(
        "opencode_mcp_connection_timeout",
        "Timed out waiting for configured OpenCode MCP servers to connect",
        { metadata: { servers } },
      );
    }
    await new Promise((resolve) => setTimeout(resolve, pollDelayMs));
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function usage(message: Record<string, unknown>): AgentUsage | undefined {
  const tokens = isRecord(message.tokens) ? message.tokens : undefined;
  const input = tokens?.input;
  const output = tokens?.output;
  const cost = message.cost;
  if (typeof input !== "number" && typeof output !== "number" && typeof cost !== "number") {
    return undefined;
  }
  const result: AgentUsage = {};
  if (typeof input === "number") {
    result.input_tokens = input;
  }
  if (typeof output === "number") {
    result.output_tokens = output;
  }
  if (typeof input === "number" && typeof output === "number") {
    result.total_tokens = input + output;
  }
  if (typeof cost === "number") {
    result.cost_usd = cost;
  }
  return result;
}

function totalUsage(messages: Record<string, unknown>[]): AgentUsage | undefined {
  let inputTokens = 0;
  let outputTokens = 0;
  let costUsd = 0;
  let hasInputTokens = false;
  let hasOutputTokens = false;
  let hasCostUsd = false;
  for (const message of messages) {
    const current = usage(message);
    if (current?.input_tokens !== undefined && current.input_tokens !== null) {
      inputTokens += current.input_tokens;
      hasInputTokens = true;
    }
    if (current?.output_tokens !== undefined && current.output_tokens !== null) {
      outputTokens += current.output_tokens;
      hasOutputTokens = true;
    }
    if (current?.cost_usd !== undefined && current.cost_usd !== null) {
      costUsd += current.cost_usd;
      hasCostUsd = true;
    }
  }
  if (!hasInputTokens && !hasOutputTokens && !hasCostUsd) {
    return undefined;
  }
  return {
    ...(hasInputTokens ? { input_tokens: inputTokens } : {}),
    ...(hasOutputTokens ? { output_tokens: outputTokens } : {}),
    ...(hasInputTokens && hasOutputTokens ? { total_tokens: inputTokens + outputTokens } : {}),
    ...(hasCostUsd ? { cost_usd: costUsd } : {}),
  };
}

function patch(diffs: unknown): string | undefined {
  if (!Array.isArray(diffs)) {
    return undefined;
  }
  const patches = diffs
    .filter((diff): diff is Record<string, unknown> => isRecord(diff))
    .map((diff) => diff.patch)
    .filter((value): value is string => typeof value === "string" && value.length > 0);
  return patches.length === 0 ? undefined : patches.join("\n");
}

function messageIds(messages: unknown[]): Set<string> {
  return new Set(
    messages
      .filter((message): message is Record<string, unknown> => isRecord(message))
      .map((message) => message.id)
      .filter((id): id is string => typeof id === "string"),
  );
}

/** Extract the terminal assistant outcome after OpenCode reports a session idle. */
export function extractOpenCodePromptOutcome(
  messages: unknown[],
  existingMessageIds?: ReadonlySet<string>,
): OpenCodePromptOutcome {
  const assistantMessages: Record<string, unknown>[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.type !== "assistant") {
      continue;
    }
    if (
      existingMessageIds !== undefined &&
      (typeof message.id !== "string" || existingMessageIds.has(message.id))
    ) {
      continue;
    }
    assistantMessages.push(message);
  }
  const terminal = assistantMessages.at(-1);
  if (terminal === undefined) {
    return {};
  }
  const invocationUsage = totalUsage(assistantMessages);
  if (terminal.finish === "error" || terminal.error !== undefined) {
    return {
      errorMessage: "OpenCode model invocation failed",
      ...(invocationUsage === undefined ? {} : { usage: invocationUsage }),
    };
  }
  if (!Array.isArray(terminal.content)) {
    return {};
  }
  const text = terminal.content
    .filter((part): part is Record<string, unknown> => isRecord(part) && part.type === "text")
    .map((part) => part.text)
    .filter((part): part is string => typeof part === "string")
    .join("");
  return {
    ...(text.length === 0 ? {} : { text }),
    ...(invocationUsage === undefined ? {} : { usage: invocationUsage }),
  };
}

export async function loadOpenCodeSdk(
  resolveModule: ModuleResolver = (specifier) => import.meta.resolve(specifier),
  loadModule: ModuleLoader = (specifier) => import(specifier),
): Promise<typeof import("@opencode/sdk")> {
  let coreConfig: string;
  let sdk: string;
  try {
    coreConfig = resolveModule("@opencode/core/config");
    sdk = resolveModule("@opencode/sdk");
  } catch {
    throw new LifecycleError(
      "opencode_harness_unavailable",
      "The OpenCode v2 harness is not installed. Install compatible packages with: npm install @opencode/core@2.0.3 @opencode/sdk@2.0.3",
    );
  }
  try {
    const [, loadedSdk] = await Promise.all([loadModule(coreConfig), loadModule(sdk)]);
    return loadedSdk as typeof import("@opencode/sdk");
  } catch {
    throw new LifecycleError("opencode_harness_load_failed", "The installed OpenCode v2 SDK could not be loaded");
  }
}

async function loadEmbeddedOpenCodeCreate(): Promise<EmbeddedOpenCodeCreate> {
  // @opencode/sdk@2.0.3 exposes its two-argument Promise SDK entry point as a
  // packaged file but does not export it from the package root. Resolve the
  // package root first so this remains valid when npm hoists dependencies. The
  // adapter exact-pins its Core and SDK peers while this private-file workaround
  // is required.
  const sdkRoot = import.meta.resolve("@opencode/sdk");
  const { create } = await import(new URL("./promise.js", sdkRoot).href);
  return create as EmbeddedOpenCodeCreate;
}

interface EnvironmentLease {
  release(): void;
}

const INHERITED_ENVIRONMENT_NAMES = new Set([
  "APPDATA",
  "BUN_INSTALL",
  "COMSPEC",
  "ComSpec",
  "DBUS_SESSION_BUS_ADDRESS",
  "HOME",
  "HTTP_PROXY",
  "HTTPS_PROXY",
  "LANG",
  "LC_ALL",
  "LC_CTYPE",
  "LOCALAPPDATA",
  "NO_PROXY",
  "PATH",
  "PATHEXT",
  "Path",
  "PathExt",
  "SHELL",
  "SSL_CERT_DIR",
  "SSL_CERT_FILE",
  "SYSTEMROOT",
  "SystemRoot",
  "TEMP",
  "TERM",
  "TERM_PROGRAM",
  "TMP",
  "TMPDIR",
  "USER",
  "USERPROFILE",
  "XDG_CACHE_HOME",
  "XDG_CONFIG_HOME",
  "XDG_DATA_HOME",
  "XDG_RUNTIME_DIR",
  "http_proxy",
  "https_proxy",
  "no_proxy",
]);

function replaceEnvironment(values: NodeJS.ProcessEnv): void {
  for (const name of Object.keys(process.env)) {
    delete process.env[name];
  }
  for (const [name, value] of Object.entries(values)) {
    process.env[name] = value;
  }
}

function leaseEnvironment(input: AdapterStartInput, credentialName: string): EnvironmentLease {
  const original = { ...process.env };
  const configured = input.runtimeContext.environment.env ?? {};
  const configuredCredential = Object.hasOwn(configured, credentialName) ? configured[credentialName] : undefined;
  const inheritedCredential = Object.hasOwn(original, credentialName) ? original[credentialName] : undefined;
  const credential = configuredCredential ?? inheritedCredential;
  const values = Object.create(null) as NodeJS.ProcessEnv;
  for (const name of INHERITED_ENVIRONMENT_NAMES) {
    const value = original[name];
    if (value !== undefined) {
      values[name] = value;
    }
  }
  for (const [name, value] of Object.entries(configured)) {
    values[name] = value;
  }
  if (credential === undefined || credential.length === 0) {
    throw new LifecycleError("opencode_credential_missing", "The configured OpenCode credential is not available");
  }
  values[credentialName] = credential;
  replaceEnvironment(values);
  return { release: () => replaceEnvironment(original) };
}

class OpenCodeSdkSessionHandle implements OpenCodeSessionHandle {
  readonly id: string;
  private readonly client: OpenCodeClient;
  private readonly environment: EnvironmentLease;
  private readonly endpointProxy?: EndpointProxy;
  private stopped = false;

  constructor(id: string, client: OpenCodeClient, environment: EnvironmentLease, endpointProxy?: EndpointProxy) {
    this.id = id;
    this.client = client;
    this.environment = environment;
    this.endpointProxy = endpointProxy;
  }

  async prompt(text: string): Promise<OpenCodePromptOutcome> {
    let promptMayHaveBeenSubmitted = false;
    try {
      const historyBeforePrompt = await this.client.sessions.context({ sessionID: this.id });
      const existingMessageIds = messageIds(historyBeforePrompt);
      promptMayHaveBeenSubmitted = true;
      await this.client.sessions.prompt({ sessionID: this.id, text });
      await this.client.sessions.wait({ sessionID: this.id });
      const historyAfterPrompt = await this.client.sessions.context({ sessionID: this.id });
      const outcome = extractOpenCodePromptOutcome(historyAfterPrompt, existingMessageIds);
      try {
        const sessionPatch = patch(await this.client.sessions.diff({ sessionID: this.id }));
        return { ...outcome, ...(sessionPatch === undefined ? {} : { patch: sessionPatch }) };
      } catch {
        return outcome;
      }
    } catch {
      throw new LifecycleError("opencode_session_failed", "OpenCode session communication failed", {
        retryable: !promptMayHaveBeenSubmitted,
      });
    }
  }

  async stop(): Promise<void> {
    if (this.stopped) {
      return;
    }
    this.stopped = true;
    try {
      await this.client.sessions.remove({ sessionID: this.id });
    } finally {
      try {
        await this.client.close();
      } finally {
        try {
          await this.endpointProxy?.close();
        } finally {
          this.environment.release();
        }
      }
    }
  }
}

export class OpenCodeSdkSessionFactory implements OpenCodeSessionFactory {
  private readonly sdkLoader: typeof loadOpenCodeSdk;
  private readonly embedOptionsLoader: OpenCodeEmbedOptionsLoader;
  private readonly endpointProxyFactory: EndpointProxyFactory;

  constructor(
    sdkLoader: typeof loadOpenCodeSdk = loadOpenCodeSdk,
    embedOptionsLoader: OpenCodeEmbedOptionsLoader =
      sdkLoader === loadOpenCodeSdk ? loadIsolatedEmbedOptions : async () => ({}),
    endpointProxyFactory: EndpointProxyFactory = ModelEndpointProxy.create,
  ) {
    this.sdkLoader = sdkLoader;
    this.embedOptionsLoader = embedOptionsLoader;
    this.endpointProxyFactory = endpointProxyFactory;
  }

  async create(input: AdapterStartInput): Promise<OpenCodeSessionHandle> {
    const model = selectModel(input.config);
    const systemInstruction = selectSystemInstruction(input.config);
    const skills = await resolveSkills(input.baseDir, input.config.skills?.paths ?? []);
    const configuredEnvironment = input.runtimeContext.environment.env ?? {};
    const mcpServers = selectMcpServers(input.config, configuredEnvironment, process.env);
    const workspace = input.runtimeContext.environment.workspace ?? input.baseDir;
    const environment = leaseEnvironment(input, model.apiKeyEnv);
    let client: OpenCodeClient | undefined;
    let endpointProxy: EndpointProxy | undefined;
    let sessionId: string | undefined;
    try {
      const sdk = await this.sdkLoader();
      if (model.baseUrl !== undefined) {
        endpointProxy = await this.endpointProxyFactory(model.baseUrl);
      }
      const configContent = hostConfigContent(
        model,
        endpointProxy?.url,
        systemInstruction,
        skills.map((skill) => skill.directory),
        mcpServers,
      );
      const embedOptions = await this.embedOptionsLoader(configContent);
      // The OpenCode convenience namespace silently ignores the embed argument.
      // Use the v2.0.3 Promise SDK entry point for production and retain the
      // public convenience API only for injected unit-test doubles.
      const createEmbeddedClient =
        this.sdkLoader === loadOpenCodeSdk
          ? await loadEmbeddedOpenCodeCreate()
          : (sdk.OpenCode.create as EmbeddedOpenCodeCreate);
      client = await createEmbeddedClient({
        config: { directory: workspace, project: false, content: configContent },
        fs: { filewatcher: false },
        models: { fetch: false },
      }, embedOptions);
      await verifyLoadedSkills(client, workspace, skills);
      const session = await client.sessions.create({
        location: { directory: workspace },
        model: { providerID: model.provider, id: model.model },
        permissions: NONINTERACTIVE_SESSION_PERMISSIONS,
      });
      sessionId = session.id;
      await waitForMcpConnections(client, workspace, Object.keys(mcpServers));
      return new OpenCodeSdkSessionHandle(session.id, client, environment, endpointProxy);
    } catch (error) {
      if (client !== undefined) {
        if (sessionId !== undefined) {
          try {
            await client.sessions.remove({ sessionID: sessionId });
          } catch {
            // Preserve the startup failure while still closing the embedded client.
          }
        }
        try {
          await client.close();
        } catch {
          // Preserve the startup failure while releasing local adapter resources.
        }
      }
      try {
        await endpointProxy?.close();
      } catch {
        // Preserve the startup failure while releasing local adapter resources.
      } finally {
        try {
          environment.release();
        } catch {
          // Preserve the startup failure when restoring the environment fails.
        }
      }
      if (error instanceof LifecycleError) {
        throw error;
      }
      throw new LifecycleError("opencode_start_failed", "OpenCode could not start an embedded session");
    }
  }
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// OpenCode v2 SDK boundary. It owns dynamic harness loading, the embedded host,
// one native session, and the temporary credential environment required by the
// upstream SDK.

import type { AgentUsage } from "nemo-fabric-adapter-contract";
import type { AdapterStartInput } from "nemo-fabric-adapters-common";
import { LifecycleError } from "nemo-fabric-adapters-common";
import { randomUUID } from "node:crypto";
import { cp, mkdtemp, readdir, realpath, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { isAbsolute, join, relative, resolve, sep } from "node:path";

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
interface RuntimeSkills {
  root?: string;
  skills: OpenCodeSkill[];
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
// OpenCode starts a background model call to name an untitled session, and that
// call can finish after the invocation ends. A fixed title skips it.
const SESSION_TITLE = "NeMo Fabric";
// OpenCode allows 30 seconds each for MCP transport startup and initial tool
// catalog discovery. Keep the Fabric readiness deadline aligned with both.
const MCP_CONNECTION_TIMEOUT_MS = 65_000;
const MCP_CONNECTION_POLL_INTERVAL_MS = 50;
const SKILL_DISCOVERY_TIMEOUT_MS = 5_000;
const SKILL_DISCOVERY_POLL_INTERVAL_MS = 50;
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

async function rejectAdditionalDiscoverableSkills(directory: string, root = directory): Promise<void> {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const location = join(directory, entry.name);
    if (entry.isDirectory()) {
      await rejectAdditionalDiscoverableSkills(location, root);
      continue;
    }
    const atRoot = directory === root;
    if ((atRoot && entry.name.endsWith(".md") && entry.name !== "SKILL.md") || (!atRoot && entry.name === "SKILL.md")) {
      throw new LifecycleError(
        "opencode_skill_unexpected",
        "Configured OpenCode skill directories must not contain additional discoverable skills",
      );
    }
  }
}

async function prepareRuntimeSkills(baseDir: string, configured: string[]): Promise<RuntimeSkills> {
  if (configured.length === 0) {
    return { skills: [] };
  }
  // OpenCode reports canonical skill paths. Canonicalize the temporary root as
  // well so platforms where /tmp is a symlink do not turn a configured skill
  // into a false "unexpected" catalog entry.
  const root = await realpath(await mkdtemp(join(tmpdir(), "nemo-fabric-opencode-skills-")));
  const skills: OpenCodeSkill[] = [];
  try {
    for (const [index, entry] of configured.entries()) {
      if (typeof entry !== "string") {
        throw new LifecycleError("opencode_invalid_skill", "OpenCode skill paths must be strings");
      }
      let source: string;
      try {
        source = await realpath(resolve(baseDir, entry));
      } catch {
        throw new LifecycleError("opencode_skill_not_found", "A configured NeMo Fabric skill path does not exist");
      }
      try {
        if (!(await stat(source)).isDirectory()) {
          throw new Error("not a directory");
        }
      } catch {
        throw new LifecycleError("opencode_skill_invalid", "NeMo Fabric skill paths must be directories");
      }
      try {
        const sourceLocation = await realpath(join(source, "SKILL.md"));
        if (!(await stat(sourceLocation)).isFile()) {
          throw new Error("not a file");
        }
      } catch {
        throw new LifecycleError(
          "opencode_skill_invalid",
          "NeMo Fabric skill directories must contain a SKILL.md file",
        );
      }
      const directory = join(root, String(index));
      try {
        await cp(source, directory, { recursive: true, dereference: true, errorOnExist: true });
      } catch {
        throw new LifecycleError("opencode_skill_invalid", "OpenCode could not snapshot a configured skill directory");
      }
      await rejectAdditionalDiscoverableSkills(directory);
      skills.push({ directory, location: await realpath(join(directory, "SKILL.md")) });
    }
    return { root, skills };
  } catch (error) {
    await rm(root, { recursive: true, force: true });
    throw error;
  }
}

async function verifyLoadedSkills(
  client: OpenCodeClient,
  workspace: string,
  configured: OpenCodeSkill[],
): Promise<void> {
  if (configured.length === 0) {
    return;
  }
  const configuredLocations = new Set(configured.map((skill) => skill.location));
  // OpenCode catalogs are nonblocking as of 2.0.23. Poll until every configured
  // skill appears or the bounded discovery window expires. Configured directories
  // are immutable private snapshots, so no additional skill can appear afterward.
  const deadline = Date.now() + SKILL_DISCOVERY_TIMEOUT_MS;
  while (true) {
    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) {
      throw new LifecycleError(
        "opencode_skill_load_failed",
        "OpenCode did not load every configured NeMo Fabric skill",
      );
    }
    const signal = AbortSignal.timeout(remainingMs);
    let loaded: Array<{ name?: unknown; path?: unknown }>;
    try {
      loaded = (
        await client.skill.list(
          { location: { directory: workspace } },
          { signal },
        )
      ).data;
    } catch {
      if (signal.aborted) {
        throw new LifecycleError(
          "opencode_skill_load_failed",
          "OpenCode did not load every configured NeMo Fabric skill",
        );
      }
      throw new LifecycleError("opencode_skill_status_unavailable", "OpenCode could not determine the configured skill status");
    }
    const byLocation = new Map(
      loaded
        .filter((skill): skill is { name: string; path: string } =>
          typeof skill.name === "string" && typeof skill.path === "string",
        )
        .map((skill) => [skill.path, skill]),
    );
    const hasUnexpectedConfiguredSkill = Array.from(byLocation.keys()).some(
      (location) =>
        !configuredLocations.has(location) &&
        configured.some((skill) => {
          const relativeLocation = relative(skill.directory, location);
          return relativeLocation !== ".." && !relativeLocation.startsWith(`..${sep}`) && !isAbsolute(relativeLocation);
        }),
    );
    if (hasUnexpectedConfiguredSkill) {
      throw new LifecycleError(
        "opencode_skill_unexpected",
        "OpenCode loaded a skill that was not configured by NeMo Fabric",
      );
    }
    const configuredSkills = configured.map((skill) => byLocation.get(skill.location));
    if (configuredSkills.every((skill) => skill !== undefined)) {
      const names = new Set<string>();
      for (const skill of configuredSkills) {
        if (names.has(skill.name)) {
          throw new LifecycleError(
            "opencode_skill_name_duplicate",
            "Configured NeMo Fabric skills must have distinct names",
          );
        }
        names.add(skill.name);
      }
      return;
    }
    if (Date.now() >= deadline) {
      throw new LifecycleError(
        "opencode_skill_load_failed",
        "OpenCode did not load every configured NeMo Fabric skill",
      );
    }
    const pollDelayMs = Math.min(SKILL_DISCOVERY_POLL_INTERVAL_MS, deadline - Date.now());
    if (pollDelayMs <= 0) {
      throw new LifecycleError(
        "opencode_skill_load_failed",
        "OpenCode did not load every configured NeMo Fabric skill",
      );
    }
    await new Promise((resolve) => setTimeout(resolve, pollDelayMs));
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

/** OpenCode's five token counts, on a session or on one model-call record. A count is absent when it was not returned. */
interface TokenCounters {
  input?: number;
  cacheRead?: number;
  cacheWrite?: number;
  output?: number;
  reasoning?: number;
}

function tokenCounter(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

// OpenCode 2.0.23 always returns all five session counters. Reading each one
// defensively keeps a malformed session from turning into a measured count.
function tokenCounters(holder: unknown): TokenCounters {
  const tokens = isRecord(holder) && isRecord(holder.tokens) ? holder.tokens : {};
  const cache = isRecord(tokens.cache) ? tokens.cache : {};
  return {
    input: tokenCounter(tokens.input),
    cacheRead: tokenCounter(cache.read),
    cacheWrite: tokenCounter(cache.write),
    output: tokenCounter(tokens.output),
    reasoning: tokenCounter(tokens.reasoning),
  };
}

/** The growth of one cumulative counter, unknown when either side is missing or the counter went down. */
function counterDelta(before: number | undefined, after: number | undefined): number | undefined {
  return before === undefined || after === undefined || after < before ? undefined : after - before;
}

/** The sum, unknown when a value is unknown or the sum is past the safe integer range. */
function sumDefined(values: (number | undefined)[]): number | undefined {
  if (!values.every((value) => value !== undefined)) {
    return undefined;
  }
  const sum = values.reduce((total: number, value) => total + value, 0);
  return Number.isSafeInteger(sum) ? sum : undefined;
}

/** Which sides of the counter growth every new model call measured. */
interface CallCoverage {
  input: boolean;
  output: boolean;
}

function isCallRecord(message: unknown): message is Record<string, unknown> {
  return isRecord(message) && (message.type === "assistant" || message.type === "compaction");
}

/**
 * Which sides the session counters measure for the model calls in `messages`,
 * or undefined when a call has no complete token report. OpenCode adds nothing
 * for a failed call or compaction without usage, and it stores a count the
 * provider left out as 0. A real call has input, so a call with no input
 * leaves input unknown; one with no output, which is rare, leaves output
 * unknown.
 */
function callCoverage(messages: unknown[]): CallCoverage | undefined {
  const coverage = { input: true, output: true };
  for (const record of messages.filter(isCallRecord)) {
    const tokens = tokenCounters(record);
    const input = sumDefined([tokens.input, tokens.cacheRead, tokens.cacheWrite]);
    const output = sumDefined([tokens.output, tokens.reasoning]);
    if (input === undefined || output === undefined) {
      return undefined;
    }
    coverage.input &&= input > 0;
    coverage.output &&= output > 0;
  }
  return coverage;
}

/** The session IDs that subagent calls in `messages` started or continued. */
function subagentSessionIDs(messages: unknown[]): Set<string> {
  const sessionIDs = new Set<string>();
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) {
      continue;
    }
    for (const part of message.content) {
      if (!isRecord(part) || part.type !== "tool" || part.name !== "subagent" || !isRecord(part.state)) {
        continue;
      }
      for (const holder of [part.state.input, part.state.metadata]) {
        if (isRecord(holder) && typeof holder.sessionID === "string") {
          sessionIDs.add(holder.sessionID);
        }
      }
    }
  }
  return sessionIDs;
}

/**
 * Whether a model call in one session's new `messages` follows a completion
 * notice for one of `childSessionIDs`. The call may answer the notice.
 */
function answersCompletionNotice(messages: unknown[], childSessionIDs: ReadonlySet<string>): boolean {
  // Newest first, so a notice at index i precedes every call at a lower index.
  return messages.some(
    (message, index) =>
      isRecord(message) &&
      message.type === "synthetic" &&
      isRecord(message.metadata) &&
      typeof message.metadata.childID === "string" &&
      childSessionIDs.has(message.metadata.childID) &&
      messages.slice(0, index).some(isCallRecord),
  );
}

/** One session's place in the tree, cumulative counters, newest message and run state at a point in time. */
interface SessionUsageSnapshot {
  parentID?: string;
  counters: TokenCounters;
  newestMessageId?: string;
  running: boolean;
}

// A new session starts with every counter at zero.
const NEW_SESSION: SessionUsageSnapshot = {
  counters: { input: 0, cacheRead: 0, cacheWrite: 0, output: 0, reasoning: 0 },
  running: false,
};
// OpenCode accepts at most 200 messages per page.
const PAGE_SIZE = 100;

/**
 * The items of an OpenCode cursor-paginated listing. OpenCode returns a next
 * cursor with every non-empty page, so the walk ends on an empty page, and a
 * malformed page or a non-empty page without a cursor fails the read.
 */
async function* listedItems(fetchPage: (cursor?: string) => Promise<unknown>): AsyncGenerator<unknown> {
  let cursor: string | undefined;
  for (;;) {
    const page = await fetchPage(cursor);
    const items = isRecord(page) ? page.data : undefined;
    if (!Array.isArray(items)) {
      throw new Error("OpenCode returned a malformed page");
    }
    if (items.length === 0) {
      return;
    }
    yield* items;
    const next = isRecord(page) && isRecord(page.cursor) ? page.cursor.next : undefined;
    if (typeof next !== "string") {
      throw new Error("OpenCode returned a page without a cursor");
    }
    cursor = next;
  }
}

/**
 * Invocation usage from the summed counter growth of the prompted session and
 * its subagent sessions. OpenCode splits input into uncached input, cache reads
 * and cache writes, and output into visible output and reasoning; the result
 * counts the parts together, so input includes the cache and output includes
 * reasoning. A side that a call did not measure, or whose counter is missing or
 * went down, is unknown, and so is the total.
 */
function usageFromGrowth(growth: TokenCounters[], coverage: CallCoverage): AgentUsage | undefined {
  const part = (key: keyof TokenCounters) => sumDefined(growth.map((counters) => counters[key]));
  const cacheRead = part("cacheRead");
  const input = coverage.input ? sumDefined([part("input"), cacheRead, part("cacheWrite")]) : undefined;
  const output = coverage.output ? sumDefined([part("output"), part("reasoning")]) : undefined;
  if (input === undefined && output === undefined) {
    return undefined;
  }
  const total = sumDefined([input, output]);
  return {
    ...(input === undefined
      ? {}
      : { input_tokens: input, cached_input_tokens: cacheRead, input_tokens_include_cache: true }),
    ...(output === undefined ? {} : { output_tokens: output }),
    ...(total === undefined ? {} : { total_tokens: total }),
  };
}

function counterGrowth(before: TokenCounters, after: TokenCounters): TokenCounters {
  return {
    input: counterDelta(before.input, after.input),
    cacheRead: counterDelta(before.cacheRead, after.cacheRead),
    cacheWrite: counterDelta(before.cacheWrite, after.cacheWrite),
    output: counterDelta(before.output, after.output),
    reasoning: counterDelta(before.reasoning, after.reasoning),
  };
}

/** Usage is observational, so a failed read leaves usage unknown instead of failing the prompt. */
async function observeUsage<T>(read: () => Promise<T>): Promise<T | undefined> {
  try {
    return await read();
  } catch {
    return undefined;
  }
}

/**
 * Reads invocation usage from OpenCode's cumulative session counters, which
 * include compaction calls that leave no assistant message. A subagent runs in
 * a child session with its own counters, so the reader adds the growth of every
 * session below the prompted one. It checks each new model call in the full
 * message history of every session, which, unlike the session context, does
 * not start at the latest compaction.
 */
class OpenCodeUsageReader {
  private readonly client: OpenCodeClient;
  private readonly sessionID: string;

  constructor(client: OpenCodeClient, sessionID: string) {
    this.client = client;
    this.sessionID = sessionID;
  }

  async snapshot(): Promise<Map<string, SessionUsageSnapshot>> {
    const snapshots = new Map<string, SessionUsageSnapshot>();
    const active: unknown = await this.client.sessions.active();
    if (!isRecord(active)) {
      throw new Error("OpenCode returned malformed active sessions");
    }
    for (const { sessionID, parentID } of await this.sessionTree()) {
      const session = await this.client.sessions.get({ sessionID });
      const newest = (await this.client.message.list({ sessionID, limit: 1 })).data[0];
      snapshots.set(sessionID, {
        ...(parentID === undefined ? {} : { parentID }),
        counters: tokenCounters(session),
        running: Object.hasOwn(active, sessionID),
        ...(isRecord(newest) && typeof newest.id === "string" ? { newestMessageId: newest.id } : {}),
      });
    }
    return snapshots;
  }

  /**
   * Usage since `before`. A subagent session that was already running belongs
   * to an earlier invocation, so its counter growth, and that of every session
   * below it, is left out. A completion notice for a subagent session that
   * existed before the invocation is earlier work too. Usage is unknown when a model
   * call is not measured, when the invocation continues a running earlier
   * session or makes a call after an earlier session's notice, or when one of
   * its own sessions is still running.
   */
  async usageSince(before: Map<string, SessionUsageSnapshot>): Promise<AgentUsage | undefined> {
    const after = await this.snapshot();
    if (before.get(this.sessionID)?.running !== false || [...before.keys()].some((id) => !after.has(id))) {
      return undefined;
    }
    const excluded = new Set([...before].filter(([, snapshot]) => snapshot.running).map(([sessionID]) => sessionID));
    const growth: TokenCounters[] = [];
    const added: unknown[] = [];
    // The tree lists every parent before its children.
    for (const [sessionID, current] of after) {
      if (excluded.has(sessionID) || (current.parentID !== undefined && excluded.has(current.parentID))) {
        excluded.add(sessionID);
        continue;
      }
      const previous = before.get(sessionID) ?? NEW_SESSION;
      const messages = await this.messagesSince(sessionID, previous.newestMessageId);
      if (current.running || messages === undefined) {
        return undefined;
      }
      // A session ID cannot tell one run of a child from the next, so a notice
      // for a child that existed before the invocation always counts as
      // earlier work.
      const earlierChildren = new Set([...before.keys(), ...excluded].filter((childID) => childID !== sessionID));
      const continuesExcluded = [...subagentSessionIDs(messages)].some((childID) => excluded.has(childID));
      if (continuesExcluded || answersCompletionNotice(messages, earlierChildren)) {
        return undefined;
      }
      added.push(...messages);
      growth.push(counterGrowth(previous.counters, current.counters));
    }
    const coverage = callCoverage(added);
    return coverage === undefined ? undefined : usageFromGrowth(growth, coverage);
  }

  /** The prompted session followed by every session below it, each after its parent. */
  private async sessionTree(): Promise<{ sessionID: string; parentID?: string }[]> {
    const tree: { sessionID: string; parentID?: string }[] = [{ sessionID: this.sessionID }];
    // The array iterator also visits entries pushed during the loop.
    for (const { sessionID: parentID } of tree) {
      const children = listedItems((cursor) =>
        this.client.sessions.list(cursor === undefined ? { parentID, limit: PAGE_SIZE } : { cursor, limit: PAGE_SIZE }),
      );
      for await (const child of children) {
        if (!isRecord(child) || typeof child.id !== "string") {
          throw new Error("OpenCode returned a malformed session");
        }
        tree.push({ sessionID: child.id, parentID });
      }
    }
    return tree;
  }

  /** Messages newer than `boundaryID`, or undefined when that message is no longer in the history. */
  private async messagesSince(sessionID: string, boundaryID?: string): Promise<unknown[] | undefined> {
    const messages: unknown[] = [];
    // Newest first. OpenCode rejects an order together with a cursor.
    const history = listedItems((cursor) =>
      this.client.message.list(
        cursor === undefined ? { sessionID, limit: PAGE_SIZE } : { sessionID, cursor, limit: PAGE_SIZE },
      ),
    );
    for await (const message of history) {
      if (boundaryID !== undefined && isRecord(message) && message.id === boundaryID) {
        return messages;
      }
      messages.push(message);
    }
    return boundaryID === undefined ? messages : undefined;
  }
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

function newAssistantMessages(
  messages: unknown[],
  existingMessageIds?: ReadonlySet<string>,
): Record<string, unknown>[] {
  return messages.filter(
    (message): message is Record<string, unknown> =>
      isRecord(message) &&
      message.type === "assistant" &&
      (existingMessageIds === undefined ||
        (typeof message.id === "string" && !existingMessageIds.has(message.id))),
  );
}

/**
 * Extract the terminal assistant outcome after OpenCode reports a session idle.
 * The invocation's usage is attached whenever it is known, including when the
 * prompt produced no usable assistant message.
 */
export function extractOpenCodePromptOutcome(
  messages: unknown[],
  existingMessageIds?: ReadonlySet<string>,
  usage?: AgentUsage,
): OpenCodePromptOutcome {
  const withUsage = usage === undefined ? {} : { usage };
  const terminal = newAssistantMessages(messages, existingMessageIds).at(-1);
  if (terminal === undefined) {
    return withUsage;
  }
  if (terminal.finish === "error" || terminal.error !== undefined) {
    return { errorMessage: "OpenCode model invocation failed", ...withUsage };
  }
  if (!Array.isArray(terminal.content)) {
    return withUsage;
  }
  const text = terminal.content
    .filter((part): part is Record<string, unknown> => isRecord(part) && part.type === "text")
    .map((part) => part.text)
    .filter((part): part is string => typeof part === "string")
    .join("");
  return {
    ...(text.length === 0 ? {} : { text }),
    ...withUsage,
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
      "The OpenCode v2 harness is not installed. Install compatible packages with: npm install @opencode/core@2.0.23 @opencode/sdk@2.0.23",
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
  // The OpenCode SDK exposes its two-argument Promise SDK entry point as a
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
  private readonly skillsRoot?: string;
  private stopped = false;

  constructor(
    id: string,
    client: OpenCodeClient,
    environment: EnvironmentLease,
    endpointProxy?: EndpointProxy,
    skillsRoot?: string,
  ) {
    this.id = id;
    this.client = client;
    this.environment = environment;
    this.endpointProxy = endpointProxy;
    this.skillsRoot = skillsRoot;
  }

  async prompt(text: string): Promise<OpenCodePromptOutcome> {
    let promptMayHaveBeenSubmitted = false;
    try {
      const historyBeforePrompt = await this.client.sessions.context({ sessionID: this.id });
      const existingMessageIds = messageIds(historyBeforePrompt);
      const usageReader = new OpenCodeUsageReader(this.client, this.id);
      const usageBeforePrompt = await observeUsage(() => usageReader.snapshot());
      promptMayHaveBeenSubmitted = true;
      await this.client.sessions.prompt({ sessionID: this.id, text });
      await this.client.sessions.wait({ sessionID: this.id });
      const historyAfterPrompt = await this.client.sessions.context({ sessionID: this.id });
      const outcome = extractOpenCodePromptOutcome(
        historyAfterPrompt,
        existingMessageIds,
        usageBeforePrompt === undefined
          ? undefined
          : await observeUsage(() => usageReader.usageSince(usageBeforePrompt)),
      );
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
          try {
            if (this.skillsRoot !== undefined) {
              await rm(this.skillsRoot, { recursive: true, force: true });
            }
          } finally {
            this.environment.release();
          }
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
    const configuredEnvironment = input.runtimeContext.environment.env ?? {};
    const mcpServers = selectMcpServers(input.config, configuredEnvironment, process.env);
    const workspace = input.runtimeContext.environment.workspace ?? input.baseDir;
    const runtimeSkills = await prepareRuntimeSkills(input.baseDir, input.config.skills?.paths ?? []);
    const skills = runtimeSkills.skills;
    let environment: EnvironmentLease | undefined;
    let client: OpenCodeClient | undefined;
    let endpointProxy: EndpointProxy | undefined;
    let sessionId: string | undefined;
    try {
      environment = leaseEnvironment(input, model.apiKeyEnv);
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
      // Use the Promise SDK entry point for production and retain the
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
        title: SESSION_TITLE,
        location: { directory: workspace },
        model: { providerID: model.provider, id: model.model },
        permissions: NONINTERACTIVE_SESSION_PERMISSIONS,
      });
      sessionId = session.id;
      await waitForMcpConnections(client, workspace, Object.keys(mcpServers));
      return new OpenCodeSdkSessionHandle(session.id, client, environment, endpointProxy, runtimeSkills.root);
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
          if (runtimeSkills.root !== undefined) {
            await rm(runtimeSkills.root, { recursive: true, force: true });
          }
        } catch {
          // Preserve the startup failure while releasing local adapter resources.
        } finally {
          try {
            environment?.release();
          } catch {
            // Preserve the startup failure when restoring the environment fails.
          }
        }
      }
      if (error instanceof LifecycleError) {
        throw error;
      }
      throw new LifecycleError("opencode_start_failed", "OpenCode could not start an embedded session");
    }
  }
}

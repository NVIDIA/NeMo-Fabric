// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdir, mkdtemp, realpath, rm, stat } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { isAbsolute, join, resolve } from "node:path";

import { Agent, fetch as undiciFetch } from "undici";

import type {
  AdapterCheckResult,
  CheckLevel,
  AgentUsage,
} from "nemo-fabric-adapter-contract";
import type { AdapterStartInput } from "nemo-fabric-adapters-common";
import { unobservedHealth, LifecycleError } from "nemo-fabric-adapters-common";

import { KiloPromptError } from "./runtime.js";
import { KiloUsageEvents } from "./usage-events.js";
import { extractInvocationUsage, findPrompt, messageId, unknownUsage } from "./usage.js";

import { selectMaxTurns, selectModel, selectSystemInstruction } from "./configuration.js";
import type { KiloPromptOutcome, KiloSessionFactory, KiloSessionHandle } from "./runtime.js";

interface KiloRequestOptions { throwOnError: true; signal?: AbortSignal }

interface KiloClient {
  event?: { subscribe(parameters: { directory: string }, options: { signal: AbortSignal; sseMaxRetryAttempts: number; onSseError: () => void }): Promise<{ stream: AsyncIterable<unknown> }> };
  memory?: { status(parameters: { directory: string }, options: KiloRequestOptions): Promise<{ data: unknown }> };
  global: { health(options: KiloRequestOptions): Promise<{data: unknown}> };
  app: {
    skills(parameters: Record<string, unknown>, options: KiloRequestOptions): Promise<{data: unknown}>;
  };
  mcp: {
    status(parameters: Record<string, unknown>, options: KiloRequestOptions): Promise<{data: unknown}>;
  };
  session: {
    get(parameters: Record<string, unknown>, options: KiloRequestOptions): Promise<{data: unknown}>;
    create(parameters: Record<string, unknown>, options: KiloRequestOptions): Promise<{data: unknown}>;
    messages(parameters: Record<string, unknown>, options: KiloRequestOptions): Promise<{ data: unknown }>;
    prompt(parameters: Record<string, unknown>, options: KiloRequestOptions): Promise<{data: unknown}>;
    abort(parameters: Record<string, unknown>, options: KiloRequestOptions): Promise<unknown>;
    delete(parameters: Record<string, unknown>, options: KiloRequestOptions): Promise<unknown>;
  };
}
interface KiloServer { url: string; isAlive(): boolean; close(): Promise<void> }
type ClientFactory = (options: {baseUrl: string; directory: string; fetch: (request: Request) => Promise<Response>}) => KiloClient;
type ServerFactory = (workspace: string, environment: NodeJS.ProcessEnv, executable: string) => Promise<KiloServer>;
type SpawnFactory = typeof spawn;
type PortFactory = () => Promise<number>;

const START_TIMEOUT_MS = 30_000;
const STOP_TIMEOUT_MS = 5_000;
const PROMPT_TIMEOUT_MS = 30 * 60_000;
// Observation must not consume the caller's remaining execution deadline.
const USAGE_EVENT_TIMEOUT_MS = 500;
const USAGE_RETURN_MARGIN_MS = 100;

function observationDeadline(deadlineMillis?: number): number {
  const localDeadline = Date.now() + USAGE_EVENT_TIMEOUT_MS;
  return deadlineMillis !== undefined && Number.isFinite(deadlineMillis)
    ? Math.min(localDeadline, deadlineMillis - USAGE_RETURN_MARGIN_MS)
    : localDeadline;
}
// Kilo Code generates a title with a separate model call for a session that
// still has its default title, and that call never appears in the session's
// messages. An explicit title skips it.
const SESSION_TITLE = "NeMo Fabric";
// A prompt answers only when its turn ends, so each request's signal bounds it instead of Node's 300 s fetch timeouts.
const KILO_DISPATCHER = new Agent({headersTimeout: 0, bodyTimeout: 0});
const ENVIRONMENT_VARIABLE = /\$\{([A-Za-z_][A-Za-z0-9_]*)\}/gu;
const HTTP_HEADER_NAME = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/u;
const INHERITED_ENVIRONMENT_NAMES = new Set([
  "APPDATA", "COMSPEC", "ComSpec", "HOME", "HTTP_PROXY", "HTTPS_PROXY", "LANG", "LC_ALL", "LC_CTYPE",
  "LOCALAPPDATA", "NO_PROXY", "PATH", "PATHEXT", "Path", "PathExt", "SHELL", "SSL_CERT_DIR", "SSL_CERT_FILE",
  "SYSTEMROOT", "SystemRoot", "TEMP", "TERM", "TMP", "TMPDIR", "USER", "USERPROFILE", "http_proxy", "https_proxy", "no_proxy",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function workspacePath(input: AdapterStartInput): Promise<string> {
  const configured = input.runtimeContext.environment.workspace ?? input.baseDir;
  try {
    const path = await realpath(isAbsolute(configured) ? configured : resolve(input.baseDir, configured));
    if (!(await stat(path)).isDirectory()) throw new Error("not directory");
    return path;
  } catch {
    throw new LifecycleError("kilo_workspace_invalid", "The Kilo Code workspace must be an existing directory");
  }
}

async function skillPaths(input: AdapterStartInput): Promise<string[]> {
  const paths: string[] = [];
  for (const entry of input.config.skills?.paths ?? []) {
    if (typeof entry !== "string") throw new LifecycleError("kilo_invalid_skill", "Kilo Code skill paths must be strings");
    try {
      const directory = await realpath(resolve(input.baseDir, entry));
      if (!(await stat(directory)).isDirectory() || !(await stat(join(directory, "SKILL.md"))).isFile()) throw new Error("invalid");
      paths.push(directory);
    } catch {
      throw new LifecycleError("kilo_skill_invalid", "NeMo Fabric skill paths must be directories that contain SKILL.md");
    }
  }
  return paths;
}

/** Map normalized tool policy to non-interactive Kilo permissions, with blocked tools taking precedence. */
function toolPermission(input: AdapterStartInput): Record<string, unknown> {
  if (input.config.tools?.definitions !== undefined) {
    throw new LifecycleError("kilo_tool_definitions_unsupported", "Kilo Code does not support normalized tool definitions through this adapter");
  }
  const enabled = input.config.tools?.enabled;
  const blocked = input.config.tools?.blocked ?? [];
  if (enabled !== undefined && enabled !== null && !enabled.every((name) => typeof name === "string" && name.length > 0)) {
    throw new LifecycleError("kilo_invalid_tool_policy", "Kilo Code tool names must be non-empty strings");
  }
  if (!blocked.every((name) => typeof name === "string" && name.length > 0)) {
    throw new LifecycleError("kilo_invalid_tool_policy", "Kilo Code tool names must be non-empty strings");
  }
  if (enabled?.includes("question") === true) {
    throw new LifecycleError("kilo_interactive_tool_unsupported", "The Kilo Code adapter cannot enable the interactive question tool");
  }
  const permission: Record<string, unknown> = {
    question: "deny",
    external_directory: "deny",
    doom_loop: "deny",
  };
  const protectedRead = {"*": "allow", "*.env": "deny", "*.env.*": "deny", "*.env.example": "allow"};
  if (enabled !== undefined && enabled !== null) {
    permission["*"] = "deny";
    permission.read = enabled.includes("read") ? protectedRead : "deny";
    for (const name of enabled) {
      if (name !== "read") permission[name] = "allow";
    }
  } else {
    permission.read = protectedRead;
    // Kilo Code asks before shell commands outside its read-only allowlist, and nothing answers in a headless run.
    permission.bash = "allow";
  }
  for (const name of blocked) permission[name] = "deny";
  return permission;
}

function expandHeaders(headers: Record<string, string>, configured: Record<string, string>): Record<string, string> {
  const expanded: Record<string, string> = {};
  for (const [name, value] of Object.entries(headers)) {
    if (!HTTP_HEADER_NAME.test(name) || typeof value !== "string") {
      throw new LifecycleError("kilo_mcp_invalid_header", "Kilo Code MCP headers must use valid HTTP header names and string values");
    }
    expanded[name] = value.replace(ENVIRONMENT_VARIABLE, (_match, variable: string) => {
      const replacement = configured[variable] ?? process.env[variable];
      if (replacement === undefined) throw new LifecycleError("kilo_mcp_environment_missing", "A Kilo Code MCP header references an unavailable environment variable");
      return replacement;
    });
  }
  return expanded;
}

function mcpConfig(input: AdapterStartInput): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  const environment = input.runtimeContext.environment.env ?? {};
  for (const [name, server] of Object.entries(input.config.mcp?.servers ?? {})) {
    if (server.allowed_tools != null || (server.blocked_tools?.length ?? 0) > 0 || server.authentication != null || server.extensions !== undefined) {
      throw new LifecycleError("kilo_mcp_filter_unsupported", "Kilo Code does not support normalized MCP authentication or per-server tool filters through this adapter");
    }
    if (server.transport === "stdio") {
      if (Object.keys(server.custom_headers ?? {}).length > 0) {
        throw new LifecycleError("kilo_mcp_invalid_server", "Kilo Code stdio MCP servers do not accept HTTP headers");
      }
      result[name] = {type: "local", command: [server.url, ...(server.args ?? [])], ...(Object.keys(server.env ?? {}).length > 0 ? {environment: server.env} : {})};
      continue;
    }
    if (server.transport === "streamable-http" || server.transport === "sse") {
      if ((server.args?.length ?? 0) > 0 || Object.keys(server.env ?? {}).length > 0) {
        throw new LifecycleError("kilo_mcp_invalid_server", "Kilo Code remote MCP servers do not accept command arguments or process environment variables");
      }
      let endpoint: URL;
      try { endpoint = new URL(server.url); } catch { throw new LifecycleError("kilo_mcp_invalid_server", "Kilo Code remote MCP servers require an HTTP or HTTPS URL"); }
      const loopback = ["localhost", "::1", "[::1]"].includes(endpoint.hostname) || /^127(?:\.\d{1,3}){3}$/u.test(endpoint.hostname);
      if (endpoint.protocol !== "https:" && !(endpoint.protocol === "http:" && loopback)) {
        throw new LifecycleError("kilo_mcp_invalid_server", "Kilo Code remote MCP servers require HTTPS unless the endpoint is loopback");
      }
      result[name] = {type: "remote", url: server.url, ...(Object.keys(server.custom_headers ?? {}).length > 0 ? {headers: expandHeaders(server.custom_headers ?? {}, environment)} : {}), oauth: false};
      continue;
    }
    throw new LifecycleError("kilo_mcp_transport_unsupported", `Kilo Code does not support MCP transport ${JSON.stringify(server.transport)}`);
  }
  return result;
}

function buildConfig(input: AdapterStartInput, skills: string[]): {content: string; credential: string} {
  const model = selectModel(input.config);
  const credential = input.runtimeContext.environment.env?.[model.apiKeyEnv] ?? process.env[model.apiKeyEnv];
  if (credential === undefined || credential.length === 0) {
    throw new LifecycleError("kilo_credential_missing", "The configured Kilo Code credential is not available");
  }
  const system = selectSystemInstruction(input.config);
  const steps = selectMaxTurns(input.config);
  const provider: Record<string, unknown> = {options: {apiKey: `{env:${model.apiKeyEnv}}`}};
  if (model.baseUrl !== undefined) {
    provider.npm = "@ai-sdk/openai-compatible";
    provider.options = {apiKey: `{env:${model.apiKeyEnv}}`, baseURL: model.baseUrl};
    provider.models = {[model.model]: {}};
  }
  const agent: Record<string, unknown> = {
    model: `${model.provider}/${model.model}`,
    ...(system === undefined ? {} : {prompt: system}),
    ...(steps === undefined ? {} : {steps}),
    ...(model.temperature === undefined ? {} : {temperature: model.temperature}),
    ...(model.topP === undefined ? {} : {top_p: model.topP}),
  };
  const permission = toolPermission(input);
  const mcp = mcpConfig(input);
  return {
    credential,
    content: JSON.stringify({provider: {[model.provider]: provider}, permission, agent: {build: agent}, ...(skills.length === 0 ? {} : {skills: {paths: skills}}), ...(Object.keys(mcp).length === 0 ? {} : {mcp})}),
  };
}

/** Verify configured skills loaded, bounding the status request by the startup timeout. */
async function verifySkills(client: KiloClient, workspace: string, configured: string[]): Promise<void> {
  if (configured.length === 0) return;
  let data: unknown;
  try {
    data = (await client.app.skills({directory: workspace}, {throwOnError: true, signal: AbortSignal.timeout(START_TIMEOUT_MS)})).data;
  } catch {
    throw new LifecycleError("kilo_skill_status_unavailable", "Kilo Code could not determine configured skill status");
  }
  const locations = new Set(Array.isArray(data) ? data.filter(isRecord).map((skill) => skill.location).filter((value): value is string => typeof value === "string") : []);
  if (configured.some((directory) => !locations.has(join(directory, "SKILL.md")))) {
    throw new LifecycleError("kilo_skill_load_failed", "Kilo Code did not load every configured NeMo Fabric skill");
  }
}

/** Verify configured MCP servers connected within the status request's startup timeout. */
async function verifyMcp(client: KiloClient, workspace: string, configured: string[]): Promise<void> {
  if (configured.length === 0) return;
  let data: unknown;
  try {
    data = (await client.mcp.status({directory: workspace}, {throwOnError: true, signal: AbortSignal.timeout(START_TIMEOUT_MS)})).data;
  } catch {
    throw new LifecycleError("kilo_mcp_status_unavailable", "Kilo Code could not determine configured MCP server status");
  }
  if (!isRecord(data)) throw new LifecycleError("kilo_mcp_status_unavailable", "Kilo Code could not determine configured MCP server status");
  for (const name of configured) {
    const status = isRecord(data[name]) ? data[name].status : undefined;
    if (status !== "connected") {
      throw new LifecycleError("kilo_mcp_connection_failed", "A configured Kilo Code MCP server did not connect", {metadata: {server: name, status: typeof status === "string" ? status : "missing"}});
    }
  }
}

function childEnvironment(input: AdapterStartInput, profile: string, config: string, credentialName: string, credential: string): NodeJS.ProcessEnv {
  const environment: NodeJS.ProcessEnv = {};
  for (const name of INHERITED_ENVIRONMENT_NAMES) if (process.env[name] !== undefined) environment[name] = process.env[name];
  Object.assign(environment, input.runtimeContext.environment.env ?? {});
  environment[credentialName] = credential;
  environment.KILO_CONFIG_CONTENT = config;
  environment.KILO_CONFIG_DIR = join(profile, "config");
  environment.KILO_DISABLE_PROJECT_CONFIG = "1";
  environment.XDG_CONFIG_HOME = join(profile, "xdg-config");
  environment.XDG_DATA_HOME = join(profile, "xdg-data");
  environment.XDG_CACHE_HOME = join(profile, "xdg-cache");
  environment.XDG_STATE_HOME = join(profile, "xdg-state");
  return environment;
}

function childExited(child: ChildProcessWithoutNullStreams): boolean {
  return child.exitCode !== null || child.signalCode !== null;
}

async function waitForChildExit(child: ChildProcessWithoutNullStreams, timeoutMs: number): Promise<boolean> {
  if (childExited(child)) return true;
  return await new Promise<boolean>((resolveExit) => {
    const timer = setTimeout(() => {
      child.off("exit", onExit);
      resolveExit(false);
    }, timeoutMs);
    const onExit = (): void => {
      clearTimeout(timer);
      resolveExit(true);
    };
    child.once("exit", onExit);
  });
}

export async function closeChild(child: ChildProcessWithoutNullStreams, timeoutMs = STOP_TIMEOUT_MS): Promise<void> {
  if (child.pid === undefined || childExited(child)) return;
  child.kill("SIGTERM");
  if (await waitForChildExit(child, timeoutMs)) return;
  if (childExited(child)) return;
  child.kill("SIGKILL");
  if (!(await waitForChildExit(child, timeoutMs))) {
    throw new LifecycleError("kilo_stop_failed", "Kilo Code did not stop after being killed");
  }
}

async function availableLoopbackPort(): Promise<number> {
  return await new Promise<number>((resolvePort, reject) => {
    const probe = createServer();
    probe.unref();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const address = probe.address();
      if (address === null || typeof address === "string") {
        probe.close();
        reject(new Error("missing loopback port"));
        return;
      }
      probe.close((error) => error === undefined ? resolvePort(address.port) : reject(error));
    });
  });
}

export function classifySpawnError(error: NodeJS.ErrnoException): LifecycleError {
  return error.code === "ENOENT"
    ? new LifecycleError("kilo_harness_unavailable", "Kilo Code is not installed. Install compatible packages with: npm install @kilocode/cli@7.7.12 @kilocode/sdk@7.7.12")
    : new LifecycleError("kilo_start_failed", "Kilo Code could not be launched");
}

export async function startKiloServer(workspace: string, environment: NodeJS.ProcessEnv, executable: string, spawnChild: SpawnFactory = spawn, portFactory: PortFactory = availableLoopbackPort): Promise<KiloServer> {
  let port: number;
  try {
    port = await portFactory();
  } catch {
    throw new LifecycleError("kilo_start_failed", "Kilo Code could not reserve a loopback server port");
  }
  const child = spawnChild(executable, ["serve", "--hostname=127.0.0.1", `--port=${port}`], {cwd: workspace, env: environment});
  child.stderr.resume();
  return await new Promise<KiloServer>((resolveServer, reject) => {
    let output = "";
    let settled = false;
    const fail = async (error: LifecycleError): Promise<void> => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try { await closeChild(child); } catch { /* Preserve the startup failure. */ }
      reject(error);
    };
    const timer = setTimeout(() => void fail(new LifecycleError("kilo_start_timeout", "Timed out while starting the Kilo Code server")), START_TIMEOUT_MS);
    timer.unref();
    child.once("error", (error: NodeJS.ErrnoException) => void fail(classifySpawnError(error)));
    child.once("exit", () => void fail(new LifecycleError("kilo_start_failed", "Kilo Code exited before its server became ready")));
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      output = (output + chunk).slice(-4096);
      const match = output.match(/kilo server listening on (https?:\/\/\S+)/u);
      if (match?.[1] === undefined || settled) return;
      settled = true;
      clearTimeout(timer);
      resolveServer({url: match[1], isAlive: () => child.pid !== undefined && !childExited(child), close: () => closeChild(child)});
    });
  });
}

/** The terminal text and error state of the last assistant message that a Kilo Code prompt returns. */
export function extractPromptOutcome(data: unknown): KiloPromptOutcome {
  if (!isRecord(data)) return {};
  const info = isRecord(data.info) ? data.info : {};
  const text = Array.isArray(data.parts) ? data.parts.filter(isRecord).filter((part) => part.type === "text" && typeof part.text === "string").map((part) => part.text).join("") : "";
  return { ...(text.length === 0 ? {} : { text }), ...(info.error === undefined ? {} : { error: true }), ...(typeof info.id === "string" ? { messageId: info.id } : {}) };
}

/** Forward SDK requests without transport response timeouts, retaining the caller's abort signal. */
export async function kiloFetch(request: Request): Promise<Response> {
  const response = await undiciFetch(request.url, {
    method: request.method,
    headers: [...request.headers],
    body: request.body,
    duplex: "half",
    signal: request.signal,
    dispatcher: KILO_DISPATCHER,
  });
  return response as unknown as Response;
}

async function loadClientFactory(): Promise<ClientFactory> {
  try {
    const sdk = await import("@kilocode/sdk/v2");
    return sdk.createKiloClient as ClientFactory;
  } catch {
    throw new LifecycleError("kilo_harness_unavailable", "The Kilo Code SDK is not installed. Install compatible packages with: npm install @kilocode/cli@7.7.12 @kilocode/sdk@7.7.12");
  }
}

async function resolveKiloExecutable(): Promise<string> {
  try {
    const executable = fileURLToPath(import.meta.resolve("@kilocode/cli/bin/kilo"));
    if (!(await stat(executable)).isFile()) throw new Error("not a file");
    return executable;
  } catch {
    throw new LifecycleError("kilo_harness_unavailable", "The Kilo Code CLI is not installed. Install compatible packages with: npm install @kilocode/cli@7.7.12 @kilocode/sdk@7.7.12");
  }
}

class KiloSdkSessionHandle implements KiloSessionHandle {
  private stopped = false;
  private healthPending = false;
  private readonly usageEvents = new KiloUsageEvents();
  private readonly eventController = new AbortController();
  private eventTask?: Promise<void>;
  private memoryUnverified = false;
  /** The IDs of the user messages of the adapter's prompts, in this session. */
  private readonly prompts = new Set<string>();
  constructor(readonly id: string, private readonly workspace: string, private readonly model: ReturnType<typeof selectModel>, private readonly client: KiloClient, private readonly server: KiloServer, private readonly profile: string, private readonly promptTimeoutMs: number, private readonly usageScopeVerified: boolean) { }
  async startUsageEvents(): Promise<void> {
    if (this.client.event === undefined) { this.usageEvents.end(); return; }
    this.eventTask = (async () => {
      try {
        const subscription = await this.client.event!.subscribe({ directory: this.workspace }, {
          signal: this.eventController.signal,
          sseMaxRetryAttempts: 1,
          onSseError: () => this.usageEvents.end(),
        });
        for await (const value of subscription.stream) this.usageEvents.observe(value);
      } catch {
        // Event visibility must not change whether paid execution succeeds.
      } finally {
        if (!this.eventController.signal.aborted) this.usageEvents.end();
      }
    })();
    await this.usageEvents.waitFor(() => this.usageEvents.connected, USAGE_EVENT_TIMEOUT_MS);
  }
  private async stopUsageEvents(): Promise<void> {
    this.eventController.abort();
    let timer: NodeJS.Timeout | undefined;
    const deadline = new Promise<void>(resolve => { timer = setTimeout(resolve, STOP_TIMEOUT_MS); timer.unref(); });
    try { await Promise.race([this.eventTask, deadline]); }
    finally { if (timer !== undefined) clearTimeout(timer); }
  }
  async check(level: CheckLevel): Promise<AdapterCheckResult> {
    if (!["live", "active", "ready"].includes(level))
      return {
        ...unobservedHealth("unsupported_level"),
        outcome: "unsupported",
      };
    if (this.stopped)
      return {
        ...unobservedHealth("session_closed"),
        outcome: "failed",
        activity: "stopped",
      };
    if (!this.server.isAlive())
      return {
        outcome: "failed",
        reason_code: "agent_exited",
        activity: "stopped",
        source: "fabric.adapter.liveness",
        observed_at: new Date().toISOString(),
      };
    if (level === "live")
      return {
        outcome: "passed",
        reason_code: null,
        activity: "unknown",
        source: "fabric.adapter.liveness",
        observed_at: new Date().toISOString(),
      };
    if (this.healthPending) return unobservedHealth("probe_timeout");
    this.healthPending = true;
    const signal = AbortSignal.timeout(3000);
    try {
      const options = {
        throwOnError: true as const,
        signal,
      };
      const response =
        level === "active"
          ? await this.client.session.get(
              { sessionID: this.id, directory: this.workspace },
              options,
            )
          : await this.client.global.health(options);
      if (this.stopped || !this.server.isAlive())
        return unobservedHealth("observation_changed");
      if (!isRecord(response.data)) return unobservedHealth("probe_error");
      if (level === "active" && response.data.id !== this.id)
        return unobservedHealth("observation_changed");
      if (
        level === "ready" &&
        (typeof response.data.healthy !== "boolean" ||
          typeof response.data.version !== "string")
      )
        return unobservedHealth("probe_error");
      if (level === "ready" && response.data.healthy !== true)
        return {
          outcome: "failed",
          reason_code: "health_failed",
          activity: "unknown",
          source: "fabric.adapter.readiness",
          observed_at: new Date().toISOString(),
        };
      return {
        outcome: "passed",
        reason_code: null,
        activity: "unknown",
        source:
          level === "active"
            ? "fabric.adapter.interface"
            : "fabric.adapter.readiness",
        observed_at: new Date().toISOString(),
      };
    } catch {
      return unobservedHealth(
        this.stopped || !this.server.isAlive()
          ? "observation_changed"
          : signal.aborted
            ? "probe_timeout"
            : "probe_error",
      );
    } finally {
      this.healthPending = false;
    }
  }

  private async abortPrompt(): Promise<void> {
    try {
      await this.client.session.abort({sessionID: this.id, directory: this.workspace}, {throwOnError: true, signal: AbortSignal.timeout(STOP_TIMEOUT_MS)});
    } catch { /* Preserve the timeout failure. */ }
  }
  /** A session's messages, oldest first, or the newest `limit` of them; undefined when listing fails. */
  private async listMessages(sessionID: string, limit?: number, deadline = observationDeadline()): Promise<unknown[] | undefined> {
    const remaining = deadline - Date.now();
    if (remaining <= 0) return undefined;
    try {
      const { data } = await this.client.session.messages({ sessionID, directory: this.workspace, ...(limit === undefined ? {} : { limit }) }, { throwOnError: true, signal: AbortSignal.timeout(Math.max(1, Math.ceil(remaining))) });
      return Array.isArray(data) ? data : undefined;
    } catch {
      return undefined;
    }
  }
  /** Memory can make separate model calls without recording ordinary step usage. */
  private async memoryDisabled(deadline: number): Promise<boolean> {
    const remaining = deadline - Date.now();
    if (remaining <= 0 || this.client.memory === undefined) { this.memoryUnverified = true; return false; }
    try {
      const { data } = await this.client.memory.status({ directory: this.workspace }, { throwOnError: true, signal: AbortSignal.timeout(Math.max(1, Math.ceil(remaining))) });
      if (!isRecord(data) || !isRecord(data.state) || data.state.enabled !== false) this.memoryUnverified = true;
    } catch {
      // Unknown memory state cannot establish the absence of auxiliary model work.
      this.memoryUnverified = true;
    }
    return !this.memoryUnverified;
  }
  async prompt(text: string, deadlineMillis?: number): Promise<KiloPromptOutcome> {
    const deadline = observationDeadline(deadlineMillis);
    const [before, memoryBefore] = await Promise.all([this.listMessages(this.id, 1, deadline), this.memoryDisabled(deadline)]);
    // Connecting during execution cannot prove that its earliest retries were seen.
    // A later invocation can recover once the stream is ready before its call.
    const eventCoverageAtStart = this.usageScopeVerified && memoryBefore && this.usageEvents.connected && !this.usageEvents.failed;
    const retriesBefore = this.usageEvents.retryCount(this.id);
    let outcome: KiloPromptOutcome;
    try {
      outcome = await this.sendPrompt(text);
    } catch (error) {
      const failure = error instanceof LifecycleError ? error : new LifecycleError("kilo_session_failed", "Kilo Code session communication failed", { retryable: true });
      const usage = await this.collectUsage(before, { deadlineMillis, eventCoverageAtStart, retriesBefore, unfinished: true });
      throw new KiloPromptError(failure, usage);
    }
    const finalMessageId = outcome.messageId;
    delete outcome.messageId;
    return { ...outcome, usage: await this.collectUsage(before, { deadlineMillis, eventCoverageAtStart, retriesBefore, finalMessageId }) };
  }
  private async collectUsage(before: unknown[] | undefined, options: { deadlineMillis?: number; eventCoverageAtStart: boolean; retriesBefore: number; finalMessageId?: string; unfinished?: boolean }): Promise<AgentUsage> {
    const { deadlineMillis, eventCoverageAtStart, retriesBefore, finalMessageId, unfinished = false } = options;
    const deadline = observationDeadline(deadlineMillis);
    const unverified = (): AgentUsage => unknownUsage(unfinished ? "unfinished_work" : "scope_unverified");
    if (before === undefined) return unverified();
    const lastId = messageId(before[0]);
    if (before.length > 0 && lastId === undefined) return unverified();
    const [messages, memoryAfter] = await Promise.all([this.listMessages(this.id, undefined, deadline), this.memoryDisabled(deadline)]);
    const prompt = findPrompt(messages, lastId);
    if (messages === undefined || prompt === undefined) return unverified();
    this.prompts.add(prompt);
    const drained = eventCoverageAtStart && !this.usageEvents.failed && finalMessageId !== undefined &&
      await this.usageEvents.through(finalMessageId, Math.max(0, deadline - Date.now()));
    return extractInvocationUsage(messages, {
      sessionId: this.id, prompt, prompts: this.prompts,
      retried: this.usageEvents.retryCount(this.id) !== retriesBefore,
      eventCoverage: drained && !this.usageEvents.failed && memoryAfter && !this.usageEvents.memoryObserved, unfinished,
    });
  }
  private async sendPrompt(text: string): Promise<KiloPromptOutcome> {
    const controller = new AbortController();
    let timedOut = false;
    let abortPromise: Promise<void> | undefined;
    let timer: NodeJS.Timeout | undefined;
    const deadline = new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => {
        timedOut = true;
        controller.abort();
        abortPromise = this.abortPrompt();
        reject(new Error("Kilo Code prompt deadline exceeded"));
      }, this.promptTimeoutMs);
    });
    try {
      const response = await Promise.race([
        this.client.session.prompt({sessionID: this.id, directory: this.workspace, model: {providerID: this.model.provider, modelID: this.model.model}, agent: "build", parts: [{type: "text", text}]}, {throwOnError: true, signal: controller.signal}),
        deadline,
      ]);
      return extractPromptOutcome(response.data);
    } catch {
      if (timedOut) {
        await abortPromise;
        throw new LifecycleError("kilo_prompt_timeout", "Timed out while waiting for Kilo Code to complete the prompt", {retryable: false});
      }
      throw new LifecycleError("kilo_session_failed", "Kilo Code session communication failed", {retryable: false});
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  }
  async stop(): Promise<void> {
    if (this.stopped) return;
    this.stopped = true;
    await this.stopUsageEvents();
    try { await this.client.session.delete({sessionID: this.id, directory: this.workspace}, {throwOnError: true, signal: AbortSignal.timeout(STOP_TIMEOUT_MS)}); } catch { /* Continue local cleanup. */ }
    try { await this.server.close(); } finally { await rm(this.profile, {recursive: true, force: true}); }
  }
}

export class KiloSdkSessionFactory implements KiloSessionFactory {
  constructor(private readonly clientLoader: () => Promise<ClientFactory> = loadClientFactory, private readonly serverFactory: ServerFactory = startKiloServer, private readonly promptTimeoutMs: number = PROMPT_TIMEOUT_MS) {}
  /** Start an isolated server and session, with cancellation deadlines on startup requests. */
  async create(input: AdapterStartInput): Promise<KiloSessionHandle> {
    const model = selectModel(input.config);
    const workspace = await workspacePath(input);
    const skills = await skillPaths(input);
    const mcpNames = Object.keys(input.config.mcp?.servers ?? {});
    const configured = buildConfig(input, skills);
    const profile = await mkdtemp(join(tmpdir(), "nemo-fabric-kilo-"));
    let server: KiloServer | undefined;
    try {
      await Promise.all(["config", "xdg-config", "xdg-data", "xdg-cache", "xdg-state"].map((name) => mkdir(join(profile, name))));
      const clientFactory = await this.clientLoader();
      const executable = this.serverFactory === startKiloServer ? await resolveKiloExecutable() : "kilo";
      server = await this.serverFactory(workspace, childEnvironment(input, profile, configured.content, model.apiKeyEnv, configured.credential), executable);
      const client = clientFactory({baseUrl: server.url, directory: workspace, fetch: kiloFetch});
      await verifySkills(client, workspace, skills);
      const created = await client.session.create({ directory: workspace, title: SESSION_TITLE, agent: "build", model: { providerID: model.provider, id: model.model } }, { throwOnError: true, signal: AbortSignal.timeout(START_TIMEOUT_MS) });
      if (!isRecord(created.data) || typeof created.data.id !== "string") throw new Error("missing session id");
      await verifyMcp(client, workspace, mcpNames);
      const handle = new KiloSdkSessionHandle(created.data.id, workspace, model, client, server, profile, this.promptTimeoutMs,
        input.runtimeContext.environment.env?.KILO_EXPERIMENTAL_NATIVE_LLM === undefined &&
        input.runtimeContext.environment.env?.KILO_CONFIG === undefined);
      await handle.startUsageEvents();
      return handle;
    } catch (error) {
      try { await server?.close(); } catch { /* Preserve startup failure. */ }
      await rm(profile, {recursive: true, force: true});
      if (error instanceof LifecycleError) throw error;
      throw new LifecycleError("kilo_start_failed", "Kilo Code could not start a session");
    }
  }
}

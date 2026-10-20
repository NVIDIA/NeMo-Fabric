// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdir, mkdtemp, realpath, rm, stat } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { isAbsolute, join, resolve } from "node:path";

import type { AgentUsage } from "nemo-fabric-adapter-contract";
import type { AdapterStartInput } from "nemo-fabric-adapters-common";
import { LifecycleError } from "nemo-fabric-adapters-common";

import { selectMaxTurns, selectModel, selectSystemInstruction } from "./configuration.js";
import type { KiloPromptOutcome, KiloSessionFactory, KiloSessionHandle } from "./runtime.js";

interface KiloRequestOptions { throwOnError: true; signal?: AbortSignal }

interface KiloClient {
  app: {
    skills(parameters: Record<string, unknown>, options: {throwOnError: true}): Promise<{data: unknown}>;
  };
  mcp: {
    status(parameters: Record<string, unknown>, options: {throwOnError: true}): Promise<{data: unknown}>;
  };
  session: {
    create(parameters: Record<string, unknown>, options: KiloRequestOptions): Promise<{data: unknown}>;
    messages(parameters: Record<string, unknown>, options: KiloRequestOptions): Promise<{data: unknown}>;
    prompt(parameters: Record<string, unknown>, options: KiloRequestOptions): Promise<{data: unknown}>;
    abort(parameters: Record<string, unknown>, options: KiloRequestOptions): Promise<unknown>;
    delete(parameters: Record<string, unknown>, options: KiloRequestOptions): Promise<unknown>;
  };
}
interface KiloServer { url: string; close(): Promise<void> }
type ClientFactory = (options: {baseUrl: string; directory: string}) => KiloClient;
type ServerFactory = (workspace: string, environment: NodeJS.ProcessEnv, executable: string) => Promise<KiloServer>;
type SpawnFactory = typeof spawn;
type PortFactory = () => Promise<number>;

const START_TIMEOUT_MS = 30_000;
const STOP_TIMEOUT_MS = 5_000;
const PROMPT_TIMEOUT_MS = 30 * 60_000;
const LIST_TIMEOUT_MS = 30_000;
// Kilo Code generates a title with a separate model call for a session that
// still has its default title, and that call never appears in the session's
// messages. An explicit title skips it.
const SESSION_TITLE = "NeMo Fabric";
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

async function verifySkills(client: KiloClient, workspace: string, configured: string[]): Promise<void> {
  if (configured.length === 0) return;
  let data: unknown;
  try {
    data = (await client.app.skills({directory: workspace}, {throwOnError: true})).data;
  } catch {
    throw new LifecycleError("kilo_skill_status_unavailable", "Kilo Code could not determine configured skill status");
  }
  const locations = new Set(Array.isArray(data) ? data.filter(isRecord).map((skill) => skill.location).filter((value): value is string => typeof value === "string") : []);
  if (configured.some((directory) => !locations.has(join(directory, "SKILL.md")))) {
    throw new LifecycleError("kilo_skill_load_failed", "Kilo Code did not load every configured NeMo Fabric skill");
  }
}

async function verifyMcp(client: KiloClient, workspace: string, configured: string[]): Promise<void> {
  if (configured.length === 0) return;
  let data: unknown;
  try {
    data = (await client.mcp.status({directory: workspace}, {throwOnError: true})).data;
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
      resolveServer({url: match[1], close: () => closeChild(child)});
    });
  });
}

/** The terminal text and error state of the last assistant message that a Kilo Code prompt returns. */
export function extractPromptOutcome(data: unknown): KiloPromptOutcome {
  if (!isRecord(data)) return {};
  const info = isRecord(data.info) ? data.info : {};
  const text = Array.isArray(data.parts) ? data.parts.filter(isRecord).filter((part) => part.type === "text" && typeof part.text === "string").map((part) => part.text).join("") : "";
  return { ...(text.length === 0 ? {} : {text}), ...(info.error === undefined ? {} : {error: true}) };
}

/** Token counts of one or more model calls; a count is absent when any call left it unknown. */
interface CallUsage { input?: number; cacheRead?: number; output?: number }

function tokenCount(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

/** A sum of two counts, unknown when either is unknown or the sum is not a safe integer. */
function addCount(left: number | undefined, right: number | undefined): number | undefined {
  if (left === undefined || right === undefined) return undefined;
  const sum = left + right;
  return Number.isSafeInteger(sum) ? sum : undefined;
}

/** A count that Kilo Code may have zero-filled: 0 cannot be told apart from an omitted count. */
function reportedCount(value: number | undefined): number | undefined {
  return value === 0 ? undefined : value;
}

/**
 * One model call's counts from its Kilo Code step-finish tokens. Kilo Code
 * reports input without cache reads and writes and output without reasoning,
 * so both are added back. Kilo Code writes 0 for any count the provider
 * omitted, so a call's input or output that adds up to 0 is unknown, and so
 * is a 0 cache read: Kilo Code zero-fills missing cache details, and nothing
 * in the step shows whether that call reported its cache reads.
 */
function stepUsage(tokens: unknown): CallUsage {
  const counts = isRecord(tokens) ? tokens : {};
  const cache = isRecord(counts.cache) ? counts.cache : {};
  const cacheRead = tokenCount(cache.read);
  const input = reportedCount(addCount(addCount(tokenCount(counts.input), cacheRead), tokenCount(cache.write)));
  const output = reportedCount(addCount(tokenCount(counts.output), tokenCount(counts.reasoning)));
  return {input, cacheRead: reportedCount(cacheRead), output};
}

type Message = Record<string, unknown>;

function messageInfo(message: Message): Record<string, unknown> {
  return isRecord(message.info) ? message.info : {};
}

function messageId(message: unknown): string | undefined {
  const id = isRecord(message) ? messageInfo(message).id : undefined;
  return typeof id === "string" ? id : undefined;
}

function messageParts(message: Message): Message[] {
  return Array.isArray(message.parts) ? message.parts.filter(isRecord) : [];
}

/** Whether Kilo Code finished processing an assistant message, normally or with an error. */
function assistantDone(info: Record<string, unknown>): boolean {
  return (isRecord(info.time) && info.time.completed !== undefined) || info.error !== undefined;
}

/** A `task` tool call: its status, the child session it runs in, when it started, and whether it runs in the background. */
interface TaskCall { status?: string; sessionId?: string; start?: number; background: boolean }

function taskCalls(message: Message): TaskCall[] {
  return messageParts(message).filter((part) => part.type === "tool" && part.tool === "task").map((part) => {
    const state = isRecord(part.state) ? part.state : {};
    const metadata = isRecord(state.metadata) ? state.metadata : {};
    const input = isRecord(state.input) ? state.input : {};
    const time = isRecord(state.time) ? state.time : {};
    return {
      ...(typeof state.status === "string" ? {status: state.status} : {}),
      ...(typeof metadata.sessionId === "string" ? {sessionId: metadata.sessionId} : {}),
      ...(typeof time.start === "number" ? {start: time.start} : {}),
      background: metadata.background === true || input.background === true,
    };
  });
}

/**
 * The child session whose result a user message delivers, when Kilo Code's
 * task tool added it for a finished background subagent. Kilo Code marks its
 * text part as synthetic with `metadata.background` and renders the child
 * session ID as `<task id="...">`.
 */
function backgroundResultSession(message: Message): string | undefined {
  if (messageInfo(message).role !== "user") return undefined;
  for (const part of messageParts(message)) {
    if (part.type !== "text" || part.synthetic !== true || !isRecord(part.metadata) || part.metadata.background !== true) continue;
    const match = typeof part.text === "string" ? /<task id="([^"]+)"/u.exec(part.text) : null;
    return match?.[1] ?? "";
  }
  return undefined;
}

const AMBIGUOUS = Symbol("ambiguous");
type Owner = string | typeof AMBIGUOUS;

interface Turn { owner: Owner; adapterPrompt: boolean }

/**
 * Ownership of a session's messages: the adapter prompt whose turn produced
 * each one. This is the single place that decides which invocation a model
 * call belongs to.
 *
 * Kilo Code runs one turn at a time per session and sets each assistant
 * message's `parentID` to the user message of its turn. A user message is
 * owned by the adapter prompt it is (`prompts`), or, for a background
 * subagent's result, by the owner of the `task` call that started that
 * subagent. Kilo Code also adds user messages inside a running turn for
 * compaction and its follow-ups. Such a message belongs to the turn that was
 * running, which is the latest assistant message's turn or a turn whose
 * prompt or background result has no assistant message yet. When those
 * candidates share an owner, the message has that owner. When all of them
 * come from adapter prompts, it belongs to the newest prompt, because the
 * adapter submits a prompt only after its previous prompt returned. Any
 * other case, such as a background result waiting before a newer prompt, is
 * ambiguous.
 */
/** The turn running when Kilo Code added a user message of its own, from the latest assistant message's turn and the turns not started yet. */
function runningTurn(latestAssistant: Turn | undefined, openRoots: Turn[]): Turn {
  const candidates = [...(latestAssistant === undefined ? [] : [latestAssistant]), ...openRoots];
  const newest = candidates.at(-1);
  if (newest === undefined || newest.owner === AMBIGUOUS) return {owner: AMBIGUOUS, adapterPrompt: false};
  if (candidates.every((turn) => turn.owner === newest.owner) || candidates.every((turn) => turn.adapterPrompt)) return newest;
  return {owner: AMBIGUOUS, adapterPrompt: false};
}

function sessionOwners(messages: Message[], prompts: ReadonlySet<string>): Map<string, Owner> {
  const users = new Map<string, Turn>();
  const tasks = new Map<string, Owner>();
  let latestAssistant: Turn | undefined;
  // Prompts and background results whose turn has no assistant message yet.
  let openRoots: Turn[] = [];
  for (const message of messages) {
    const info = messageInfo(message);
    const id = messageId(message);
    if (info.role === "user" && id !== undefined) {
      let turn: Turn;
      const delivered = backgroundResultSession(message);
      if (prompts.has(id)) {
        turn = {owner: id, adapterPrompt: true};
        openRoots.push(turn);
      } else if (delivered !== undefined) {
        turn = {owner: tasks.get(delivered) ?? AMBIGUOUS, adapterPrompt: false};
        openRoots.push(turn);
      } else {
        turn = runningTurn(latestAssistant, openRoots);
      }
      users.set(id, turn);
    } else if (info.role === "assistant") {
      const turn = typeof info.parentID === "string" ? users.get(info.parentID) : undefined;
      latestAssistant = turn ?? {owner: AMBIGUOUS, adapterPrompt: false};
      openRoots = openRoots.filter((root) => root !== turn);
      for (const task of taskCalls(message)) {
        if (task.sessionId !== undefined) tasks.set(task.sessionId, latestAssistant.owner);
      }
    }
  }
  return new Map([...users].map(([id, turn]) => [id, turn.owner]));
}

/** The assistant messages of `prompt`'s turn, or undefined when a message after the prompt cannot be attributed. */
function ownedAssistants(messages: Message[], prompt: string, prompts: ReadonlySet<string>): Message[] | undefined {
  const owners = sessionOwners(messages, prompts);
  const promptIndex = messages.findIndex((message) => messageId(message) === prompt);
  if (promptIndex < 0) return undefined;
  const owned: Message[] = [];
  // The prompt's work starts at its message, so only later messages can be its own.
  for (const message of messages.slice(promptIndex + 1)) {
    const info = messageInfo(message);
    if (info.role === "user" && owners.get(messageId(message) ?? "") === AMBIGUOUS) return undefined;
    if (info.role !== "assistant") continue;
    const owner = typeof info.parentID === "string" ? owners.get(info.parentID) : undefined;
    if (owner === undefined || owner === AMBIGUOUS) return undefined;
    if (owner === prompt) owned.push(message);
  }
  return owned;
}

/**
 * The adapter's prompt among the messages after `lastMessageIdBefore`, the
 * newest message before the prompt (undefined when the session was empty):
 * the first user message with a text part that Kilo Code did not add itself.
 * Kilo Code marks the text of background results, compaction replays and
 * compaction follow-ups as synthetic.
 */
export function findPrompt(messages: unknown, lastMessageIdBefore: string | undefined): string | undefined {
  if (!Array.isArray(messages)) return undefined;
  const start = lastMessageIdBefore === undefined ? 0 : messages.findIndex((message) => messageId(message) === lastMessageIdBefore) + 1;
  if (start === 0 && lastMessageIdBefore !== undefined) return undefined;
  const prompt = messages.slice(start).filter(isRecord).find((message) => messageInfo(message).role === "user" && messageParts(message).some((part) => part.type === "text" && part.synthetic !== true));
  return prompt === undefined ? undefined : messageId(prompt);
}

/** The foreground `task` calls of `prompt`'s turn whose child sessions hold model calls to add. */
export function ownedChildSessions(messages: unknown, prompt: string, prompts: ReadonlySet<string>): string[] {
  if (!Array.isArray(messages)) return [];
  const owned = ownedAssistants(messages.filter(isRecord), prompt, prompts) ?? [];
  return [...new Set(owned.flatMap(taskCalls).filter((task) => !task.background).flatMap((task) => task.sessionId === undefined ? [] : [task.sessionId]))];
}

/**
 * The step-finish tokens of a turn's finished assistant messages, or
 * undefined when one of them is still running or finished without a
 * step-finish part, because Kilo Code keeps no counts for a call it did not
 * finish.
 */
function finishedSteps(assistants: Message[]): unknown[] | undefined {
  const steps: unknown[] = [];
  for (const message of assistants) {
    const parts = messageParts(message).filter((part) => part.type === "step-finish");
    if (!assistantDone(messageInfo(message)) || parts.length === 0) return undefined;
    steps.push(...parts.map((part) => part.tokens));
  }
  return steps;
}

/**
 * The model calls a foreground `task` call made in its child session: the
 * assistant messages created from the call's start until the next `task` call
 * that resumed the same session. Undefined when the child could not be read,
 * when another `task` call on the session started at the same time, so the
 * windows cannot be told apart, when a message has no creation time, or when
 * the child started its own subagent, which Kilo Code 7.7.12 allows only with
 * a raised subagent depth.
 */
function childSteps(task: TaskCall, resumes: number[], children: ReadonlyMap<string, unknown[]>): unknown[] | undefined {
  if (task.sessionId === undefined || task.start === undefined) return undefined;
  const child = children.get(task.sessionId);
  if (child === undefined) return undefined;
  const start = task.start;
  if (resumes.filter((value) => value === start).length > 1) return undefined;
  const end = Math.min(...resumes.filter((value) => value > start), Number.POSITIVE_INFINITY);
  const window: Message[] = [];
  for (const message of child.filter(isRecord)) {
    const time = messageInfo(message).time;
    const created = isRecord(time) && typeof time.created === "number" ? time.created : undefined;
    if (created === undefined) return undefined;
    if (created >= start && created < end && messageInfo(message).role === "assistant") window.push(message);
  }
  if (window.some((message) => taskCalls(message).length > 0)) return undefined;
  return finishedSteps(window);
}

/**
 * The usage of the model calls that `prompt`'s turn made, from the adapter's
 * session messages, oldest first, and the child sessions of its foreground
 * subagents. `prompts` holds every prompt the adapter submitted in the
 * session. Usage is undefined when a call cannot be attributed, when an owned
 * call is unfinished or reported neither input nor output, and when the turn
 * started a background subagent, because the subagent and Kilo Code's reply
 * to its result can run after the prompt returns. That reply belongs to this
 * prompt, so no later prompt counts it.
 *
 * Cached input is known only when every counted call reported a nonzero
 * cache read. Kilo Code computes most message costs from its model catalog and does not
 * mark the ones a provider reported, so the usage has no `cost_usd`.
 */
export function extractInvocationUsage(messages: unknown, prompt: string, prompts: ReadonlySet<string>, children: ReadonlyMap<string, unknown[]> = new Map()): AgentUsage | undefined {
  if (!Array.isArray(messages)) return undefined;
  const records = messages.filter(isRecord);
  const owned = ownedAssistants(records, prompt, prompts);
  if (owned === undefined || owned.length === 0) return undefined;
  const steps = finishedSteps(owned);
  if (steps === undefined) return undefined;
  const resumes = new Map<string, number[]>();
  for (const task of records.flatMap(taskCalls)) {
    if (task.sessionId !== undefined && task.start !== undefined) resumes.set(task.sessionId, [...(resumes.get(task.sessionId) ?? []), task.start]);
  }
  for (const task of owned.flatMap(taskCalls)) {
    // A task call that failed before Kilo Code created its child session made no subagent calls.
    if (task.sessionId === undefined && task.status === "error") continue;
    if (task.background) return undefined;
    const child = childSteps(task, resumes.get(task.sessionId ?? "") ?? [], children);
    if (child === undefined) return undefined;
    steps.push(...child);
  }
  let sum: CallUsage = {input: 0, cacheRead: 0, output: 0};
  for (const tokens of steps) {
    const call = stepUsage(tokens);
    if (call.input === undefined && call.output === undefined) return undefined;
    sum = {input: addCount(sum.input, call.input), cacheRead: addCount(sum.cacheRead, call.cacheRead), output: addCount(sum.output, call.output)};
  }
  const cachedInput = sum.input !== undefined ? sum.cacheRead : undefined;
  // Kilo Code's step total is the AI SDK's input plus output with an omitted
  // side counted as 0, not a provider total, so the total is derived here.
  const total = addCount(sum.input, sum.output);
  const usage: AgentUsage = {
    ...(sum.input === undefined ? {} : {input_tokens: sum.input, input_tokens_include_cache: true}),
    ...(cachedInput === undefined ? {} : {cached_input_tokens: cachedInput}),
    ...(sum.output === undefined ? {} : {output_tokens: sum.output}),
    ...(total === undefined ? {} : {total_tokens: total}),
  };
  return Object.keys(usage).length === 0 ? undefined : usage;
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
  /** The IDs of the user messages of the adapter's prompts, in this session. */
  private readonly prompts = new Set<string>();
  constructor(readonly id: string, private readonly workspace: string, private readonly model: ReturnType<typeof selectModel>, private readonly client: KiloClient, private readonly server: KiloServer, private readonly profile: string, private readonly promptTimeoutMs: number) {}
  private async abortPrompt(): Promise<void> {
    try {
      await this.client.session.abort({sessionID: this.id, directory: this.workspace}, {throwOnError: true, signal: AbortSignal.timeout(STOP_TIMEOUT_MS)});
    } catch { /* Preserve the timeout failure. */ }
  }
  /** A session's messages, oldest first, or the newest `limit` of them; undefined when listing fails. */
  private async listMessages(sessionID: string, limit?: number): Promise<unknown[] | undefined> {
    try {
      const {data} = await this.client.session.messages({sessionID, directory: this.workspace, ...(limit === undefined ? {} : {limit})}, {throwOnError: true, signal: AbortSignal.timeout(LIST_TIMEOUT_MS)});
      return Array.isArray(data) ? data : undefined;
    } catch {
      return undefined;
    }
  }
  async prompt(text: string): Promise<KiloPromptOutcome> {
    const before = await this.listMessages(this.id, 1);
    const outcome = await this.sendPrompt(text);
    if (before === undefined) return outcome;
    const lastId = messageId(before[0]);
    if (before.length > 0 && lastId === undefined) return outcome;
    const messages = await this.listMessages(this.id);
    const prompt = findPrompt(messages, lastId);
    if (prompt === undefined) return outcome;
    this.prompts.add(prompt);
    const children = new Map<string, unknown[]>();
    for (const sessionID of ownedChildSessions(messages, prompt, this.prompts)) {
      const child = await this.listMessages(sessionID);
      if (child === undefined) return outcome;
      children.set(sessionID, child);
    }
    const usage = extractInvocationUsage(messages, prompt, this.prompts, children);
    return usage === undefined ? outcome : {...outcome, usage};
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
    try { await this.client.session.delete({sessionID: this.id, directory: this.workspace}, {throwOnError: true, signal: AbortSignal.timeout(STOP_TIMEOUT_MS)}); } catch { /* Continue local cleanup. */ }
    try { await this.server.close(); } finally { await rm(this.profile, {recursive: true, force: true}); }
  }
}

export class KiloSdkSessionFactory implements KiloSessionFactory {
  constructor(private readonly clientLoader: () => Promise<ClientFactory> = loadClientFactory, private readonly serverFactory: ServerFactory = startKiloServer, private readonly promptTimeoutMs: number = PROMPT_TIMEOUT_MS) {}
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
      const client = clientFactory({baseUrl: server.url, directory: workspace});
      await verifySkills(client, workspace, skills);
      const created = await client.session.create({directory: workspace, title: SESSION_TITLE, agent: "build", model: {providerID: model.provider, id: model.model}}, {throwOnError: true});
      if (!isRecord(created.data) || typeof created.data.id !== "string") throw new Error("missing session id");
      await verifyMcp(client, workspace, mcpNames);
      return new KiloSdkSessionHandle(created.data.id, workspace, model, client, server, profile, this.promptTimeoutMs);
    } catch (error) {
      try { await server?.close(); } catch { /* Preserve startup failure. */ }
      await rm(profile, {recursive: true, force: true});
      if (error instanceof LifecycleError) throw error;
      throw new LifecycleError("kilo_start_failed", "Kilo Code could not start a session");
    }
  }
}

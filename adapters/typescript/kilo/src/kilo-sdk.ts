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

interface KiloClient {
  app: {
    skills(parameters: Record<string, unknown>, options: {throwOnError: true}): Promise<{data: unknown}>;
  };
  mcp: {
    status(parameters: Record<string, unknown>, options: {throwOnError: true}): Promise<{data: unknown}>;
  };
  session: {
    create(parameters: Record<string, unknown>, options: {throwOnError: true}): Promise<{data: unknown}>;
    prompt(parameters: Record<string, unknown>, options: {throwOnError: true}): Promise<{data: unknown}>;
    delete(parameters: Record<string, unknown>, options: {throwOnError: true}): Promise<unknown>;
  };
}
interface KiloServer { url: string; close(): Promise<void> }
type ClientFactory = (options: {baseUrl: string; directory: string}) => KiloClient;
type ServerFactory = (workspace: string, environment: NodeJS.ProcessEnv, executable: string) => Promise<KiloServer>;

const START_TIMEOUT_MS = 30_000;
const STOP_TIMEOUT_MS = 5_000;
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
    permission: toolPermission(input),
    ...(system === undefined ? {} : {prompt: system}),
    ...(steps === undefined ? {} : {steps}),
    ...(model.temperature === undefined ? {} : {temperature: model.temperature}),
    ...(model.topP === undefined ? {} : {top_p: model.topP}),
  };
  const mcp = mcpConfig(input);
  return {
    credential,
    content: JSON.stringify({provider: {[model.provider]: provider}, agent: {build: agent}, ...(skills.length === 0 ? {} : {skills: {paths: skills}}), ...(Object.keys(mcp).length === 0 ? {} : {mcp})}),
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

async function closeChild(child: ChildProcessWithoutNullStreams): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return;
  child.kill("SIGTERM");
  await new Promise<void>((resolveClose) => {
    const timer = setTimeout(() => { child.kill("SIGKILL"); resolveClose(); }, STOP_TIMEOUT_MS);
    timer.unref();
    child.once("exit", () => { clearTimeout(timer); resolveClose(); });
  });
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

export async function startKiloServer(workspace: string, environment: NodeJS.ProcessEnv, executable: string): Promise<KiloServer> {
  let port: number;
  try {
    port = await availableLoopbackPort();
  } catch {
    throw new LifecycleError("kilo_start_failed", "Kilo Code could not reserve a loopback server port");
  }
  const child = spawn(executable, ["serve", "--hostname=127.0.0.1", `--port=${port}`], {cwd: workspace, env: environment});
  child.stderr.resume();
  return await new Promise<KiloServer>((resolveServer, reject) => {
    let output = "";
    let settled = false;
    const fail = (error: LifecycleError): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      void closeChild(child);
      reject(error);
    };
    const timer = setTimeout(() => fail(new LifecycleError("kilo_start_timeout", "Timed out while starting the Kilo Code server")), START_TIMEOUT_MS);
    timer.unref();
    child.once("error", () => fail(new LifecycleError("kilo_harness_unavailable", "Kilo Code is not installed. Install compatible packages with: npm install @kilocode/cli@7.7.12 @kilocode/sdk@7.7.12")));
    child.once("exit", () => fail(new LifecycleError("kilo_start_failed", "Kilo Code exited before its server became ready")));
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

export function extractPromptOutcome(data: unknown): KiloPromptOutcome {
  if (!isRecord(data)) return {};
  const info = isRecord(data.info) ? data.info : {};
  const tokens = isRecord(info.tokens) ? info.tokens : {};
  const usage: AgentUsage = {};
  const inputTokens = typeof tokens.input === "number" ? tokens.input : undefined;
  const outputTokens = typeof tokens.output === "number" ? tokens.output : undefined;
  if (inputTokens !== undefined) usage.input_tokens = inputTokens;
  if (outputTokens !== undefined) usage.output_tokens = outputTokens;
  if (typeof tokens.total === "number") usage.total_tokens = tokens.total;
  else if (inputTokens !== undefined && outputTokens !== undefined) usage.total_tokens = inputTokens + outputTokens;
  if (typeof info.cost === "number") usage.cost_usd = info.cost;
  const text = Array.isArray(data.parts) ? data.parts.filter(isRecord).filter((part) => part.type === "text" && typeof part.text === "string").map((part) => part.text).join("") : "";
  return { ...(text.length === 0 ? {} : {text}), ...(info.error === undefined ? {} : {error: true}), ...(Object.keys(usage).length === 0 ? {} : {usage}) };
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
  constructor(readonly id: string, private readonly workspace: string, private readonly model: ReturnType<typeof selectModel>, private readonly client: KiloClient, private readonly server: KiloServer, private readonly profile: string) {}
  async prompt(text: string): Promise<KiloPromptOutcome> {
    try {
      const response = await this.client.session.prompt({sessionID: this.id, directory: this.workspace, model: {providerID: this.model.provider, modelID: this.model.model}, agent: "build", parts: [{type: "text", text}]}, {throwOnError: true});
      return extractPromptOutcome(response.data);
    } catch {
      throw new LifecycleError("kilo_session_failed", "Kilo Code session communication failed", {retryable: false});
    }
  }
  async stop(): Promise<void> {
    if (this.stopped) return;
    this.stopped = true;
    try { await this.client.session.delete({sessionID: this.id, directory: this.workspace}, {throwOnError: true}); } catch { /* Continue local cleanup. */ }
    try { await this.server.close(); } finally { await rm(this.profile, {recursive: true, force: true}); }
  }
}

export class KiloSdkSessionFactory implements KiloSessionFactory {
  constructor(private readonly clientLoader: () => Promise<ClientFactory> = loadClientFactory, private readonly serverFactory: ServerFactory = startKiloServer) {}
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
      const created = await client.session.create({directory: workspace, agent: "build", model: {providerID: model.provider, id: model.model}}, {throwOnError: true});
      if (!isRecord(created.data) || typeof created.data.id !== "string") throw new Error("missing session id");
      await verifyMcp(client, workspace, mcpNames);
      return new KiloSdkSessionHandle(created.data.id, workspace, model, client, server, profile);
    } catch (error) {
      try { await server?.close(); } catch { /* Preserve startup failure. */ }
      await rm(profile, {recursive: true, force: true});
      if (error instanceof LifecycleError) throw error;
      throw new LifecycleError("kilo_start_failed", "Kilo Code could not start a session");
    }
  }
}

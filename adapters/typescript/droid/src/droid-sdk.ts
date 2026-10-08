// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cp, mkdir, mkdtemp, realpath, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";

import type {
  AgentConfig,
  AgentMcpServerConfig,
  AgentModelConfig,
  AgentUsage,
  JsonObject,
} from "nemo-fabric-adapter-contract";
import { LifecycleError, type AdapterStartInput } from "nemo-fabric-adapters-common";

import type { DroidPromptOutcome, DroidSessionFactory, DroidSessionHandle } from "./runtime.js";

const DROID_SDK_INSTALL_COMMAND = "npm install @factory/droid-sdk@0.9.1";
const MCP_STARTUP_TIMEOUT_MS = 10_000;
const MCP_STATUS_POLL_INTERVAL_MS = 50;

interface DroidTokenUsage {
  inputTokens: number;
  outputTokens: number;
  cacheCreationTokens: number;
  cacheReadTokens: number;
  thinkingTokens: number;
  factoryCredits?: number;
}

interface DroidErrorEvent {
  message?: unknown;
}

interface DroidResult {
  type: "result";
  sessionId: unknown;
  durationMs: unknown;
  tokenUsage: unknown;
  text: unknown;
  turnCount: unknown;
  success: unknown;
  subtype: unknown;
  interrupted: unknown;
  error: DroidErrorEvent | null | unknown;
}

interface DroidToolInfo {
  id?: unknown;
}

interface DroidMcpStatus {
  name?: unknown;
  status?: unknown;
  error?: unknown;
}

interface DroidSkillInfo {
  name?: unknown;
  filePath?: unknown;
  enabled?: unknown;
}

interface DroidSdkSession {
  readonly id: string;
  stream(prompt: string): AsyncGenerator<unknown, void, undefined>;
  close(): Promise<void>;
  listTools(): Promise<DroidToolInfo[]>;
  updateSettings(settings: { disabledToolIds: string[] }): Promise<unknown>;
  listMcpServers(): Promise<{ servers: DroidMcpStatus[] }>;
  listSkills(): Promise<{ skills: DroidSkillInfo[] }>;
}

interface DroidSdkModule {
  createSession(options: Record<string, unknown>): Promise<DroidSdkSession>;
}

export type DroidSdkLoader = () => Promise<DroidSdkModule>;

function isMissingModuleError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error.code === "ERR_MODULE_NOT_FOUND" || error.code === "MODULE_NOT_FOUND")
  );
}

async function loadDroidSdk(): Promise<DroidSdkModule> {
  try {
    const moduleName: string = "@factory/droid-sdk/node";
    const sdk = (await import(moduleName)) as DroidSdkModule;
    if (typeof sdk.createSession !== "function") {
      throw new LifecycleError(
        "droid_harness_incompatible",
        "The installed Factory Droid SDK does not expose createSession",
      );
    }
    return sdk;
  } catch (error) {
    if (error instanceof LifecycleError) throw error;
    if (isMissingModuleError(error)) {
      throw new LifecycleError(
        "droid_harness_unavailable",
        `The Factory Droid SDK is not installed. Install the supported SDK with: ${DROID_SDK_INSTALL_COMMAND}`,
      );
    }
    throw new LifecycleError("droid_harness_load_failed", "The installed Factory Droid SDK could not be loaded");
  }
}

function selectModel(config: AgentConfig): AgentModelConfig {
  const entries = Object.entries(config.models ?? {});
  if (entries.length === 0) {
    throw new LifecycleError("droid_model_required", "The Droid adapter requires one configured model");
  }
  const selected = config.models?.default ?? (entries.length === 1 ? entries[0]?.[1] : undefined);
  if (selected === undefined) {
    throw new LifecycleError(
      "droid_model_ambiguous",
      "Configure a default model role when the Droid adapter receives multiple models",
    );
  }
  if (selected.provider !== "factory") {
    throw new LifecycleError("droid_provider_unsupported", "The Droid adapter supports Factory-managed model IDs only");
  }
  return selected;
}

function credentialValue(input: AdapterStartInput, name: string): string | undefined {
  return input.runtimeContext.environment.env?.[name] ?? process.env[name];
}

async function resolveWorkspace(input: AdapterStartInput): Promise<string> {
  try {
    const workspace = await realpath(resolve(input.runtimeContext.environment.workspace ?? input.baseDir));
    if (!(await stat(workspace)).isDirectory()) throw new Error("not a directory");
    return workspace;
  } catch {
    throw new LifecycleError("droid_workspace_invalid", "The NeMo Fabric runtime workspace must be a directory");
  }
}

function rejectMcpExtensions(name: string, server: AgentMcpServerConfig): void {
  if (server.extensions !== undefined && Object.keys(server.extensions).length > 0) {
    throw new LifecycleError("droid_mcp_extensions_unsupported", "Droid does not support MCP extensions", {
      metadata: { server: name },
    });
  }
  if (server.authentication !== undefined && server.authentication !== null) {
    throw new LifecycleError(
      "droid_mcp_authentication_unsupported",
      "Droid MCP authentication is not supported by this adapter",
      { metadata: { server: name } },
    );
  }
  if (server.allowed_tools !== undefined && server.allowed_tools !== null) {
    throw new LifecycleError(
      "droid_mcp_allowlist_unsupported",
      "Droid MCP allowed_tools filtering is not supported by this adapter",
      { metadata: { server: name } },
    );
  }
  if (server.blocked_tools !== undefined && server.blocked_tools.length > 0) {
    throw new LifecycleError(
      "droid_mcp_blocklist_unsupported",
      "Droid MCP blocked_tools filtering is not supported by this adapter",
      { metadata: { server: name } },
    );
  }
}

function validateRemoteUrl(name: string, value: string): void {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new LifecycleError("droid_mcp_url_invalid", "Droid network MCP servers require an absolute URL", {
      metadata: { server: name },
    });
  }
  const loopback = parsed.hostname === "localhost" || parsed.hostname === "127.0.0.1" || parsed.hostname === "[::1]";
  if (parsed.protocol !== "https:" && !(parsed.protocol === "http:" && loopback)) {
    throw new LifecycleError(
      "droid_mcp_url_insecure",
      "Droid network MCP servers require HTTPS except for loopback development endpoints",
      { metadata: { server: name } },
    );
  }
}

function normalizedMcpServer(name: string, server: AgentMcpServerConfig): JsonObject {
  rejectMcpExtensions(name, server);
  if (server.transport === "stdio") {
    if (server.custom_headers !== undefined && Object.keys(server.custom_headers).length > 0) {
      throw new LifecycleError("droid_mcp_headers_unsupported", "Droid stdio MCP servers do not support custom_headers", {
        metadata: { server: name },
      });
    }
    return {
      type: "stdio",
      command: server.url,
      args: server.args ?? [],
      env: server.env ?? {},
      disabled: false,
    };
  }
  if (server.transport === "streamable-http" || server.transport === "sse") {
    if ((server.args?.length ?? 0) > 0 || Object.keys(server.env ?? {}).length > 0) {
      throw new LifecycleError(
        "droid_mcp_process_fields_unsupported",
        "Droid network MCP servers do not support args or env",
        { metadata: { server: name } },
      );
    }
    validateRemoteUrl(name, server.url);
    return {
      type: server.transport === "sse" ? "sse" : "http",
      url: server.url,
      headers: server.custom_headers ?? {},
      disabled: false,
    };
  }
  throw new LifecycleError("droid_mcp_transport_unsupported", "Droid does not support the configured MCP transport", {
    metadata: { server: name, transport: server.transport },
  });
}

function normalizedMcp(input: AdapterStartInput): Record<string, JsonObject> {
  if (input.config.mcp?.extensions !== undefined && Object.keys(input.config.mcp.extensions).length > 0) {
    throw new LifecycleError("droid_mcp_extensions_unsupported", "Droid does not support top-level MCP extensions");
  }
  return Object.fromEntries(
    Object.entries(input.config.mcp?.servers ?? {}).map(([name, server]) => [name, normalizedMcpServer(name, server)]),
  );
}

interface RuntimeProfile {
  root?: string;
  home?: string;
  entrypoints: string[];
}

async function prepareRuntimeProfile(
  input: AdapterStartInput,
  mcpServers: Record<string, JsonObject>,
  parentEnvironment: NodeJS.ProcessEnv,
): Promise<RuntimeProfile> {
  if (input.config.skills?.extensions !== undefined) {
    throw new LifecycleError("droid_skill_extensions_unsupported", "Droid does not support skill extensions");
  }
  const configured = input.config.skills?.paths ?? [];
  if (configured.length === 0 && Object.keys(mcpServers).length === 0) return { entrypoints: [] };

  const root = await mkdtemp(join(tmpdir(), "nemo-fabric-droid-"));
  const home = join(root, "home");
  const factoryRoot = join(home, ".factory");
  const skillsRoot = join(home, ".agents", "skills");
  try {
    const runtimeEnvironment = input.runtimeContext.environment.env ?? {};
    const sourceHome =
      runtimeEnvironment.FACTORY_HOME_OVERRIDE ||
      runtimeEnvironment.HOME ||
      runtimeEnvironment.USERPROFILE ||
      parentEnvironment.FACTORY_HOME_OVERRIDE ||
      parentEnvironment.HOME ||
      parentEnvironment.USERPROFILE;
    if (sourceHome !== undefined) {
      const sourceSettings = join(sourceHome, ".factory", "settings.json");
      try {
        if ((await stat(sourceSettings)).isFile()) {
          await mkdir(factoryRoot, { recursive: true });
          await cp(sourceSettings, join(factoryRoot, "settings.json"), { errorOnExist: true });
        }
      } catch (error) {
        if (!(typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT")) {
          throw new LifecycleError(
            "droid_settings_unavailable",
            "Droid settings could not be copied into the isolated NeMo Fabric runtime",
          );
        }
      }
    }
    if (Object.keys(mcpServers).length > 0) {
      await mkdir(factoryRoot, { recursive: true });
      await writeFile(
        join(factoryRoot, "mcp.json"),
        `${JSON.stringify({ mcpServers }, null, 2)}\n`,
        { encoding: "utf8", mode: 0o600 },
      );
    }
    if (configured.length > 0) await mkdir(skillsRoot, { recursive: true });
    const names = new Set<string>();
    const entrypoints: string[] = [];
    for (const configuredPath of configured) {
      let source: string;
      try {
        source = await realpath(resolve(input.baseDir, configuredPath));
        if (!(await stat(source)).isDirectory() || !(await stat(join(source, "SKILL.md"))).isFile()) {
          throw new Error("invalid skill directory");
        }
      } catch {
        throw new LifecycleError(
          "droid_skill_invalid",
          "Droid skill paths must be directories containing a SKILL.md file",
          { metadata: { path: configuredPath } },
        );
      }
      const name = basename(source);
      if (name.length === 0 || names.has(name)) {
        throw new LifecycleError("droid_skill_collision", "Configured Droid skills must use distinct directory names", {
          metadata: { skill: name },
        });
      }
      names.add(name);
      const destination = join(skillsRoot, name);
      await cp(source, destination, { recursive: true, errorOnExist: true });
      entrypoints.push(await realpath(join(destination, "SKILL.md")));
    }
    return { root, home, entrypoints };
  } catch (error) {
    await rm(root, { recursive: true, force: true });
    throw error;
  }
}

async function validateSkills(session: DroidSdkSession, expectedEntrypoints: string[]): Promise<void> {
  if (expectedEntrypoints.length === 0) return;
  const report = await session.listSkills();
  if (!Array.isArray(report.skills)) {
    throw new LifecycleError("droid_skill_status_unavailable", "Droid could not determine configured skill status");
  }
  const loaded = new Map<string, DroidSkillInfo>();
  for (const skill of report.skills) {
    if (typeof skill.filePath !== "string") continue;
    try {
      loaded.set(await realpath(skill.filePath), skill);
    } catch {
      // A malformed or stale ambient skill cannot satisfy a configured path.
    }
  }
  const missing = expectedEntrypoints.filter((entrypoint) => !loaded.has(entrypoint));
  const disabled = expectedEntrypoints.filter((entrypoint) => loaded.get(entrypoint)?.enabled === false);
  if (missing.length > 0 || disabled.length > 0) {
    throw new LifecycleError("droid_skill_load_failed", "Droid did not load every configured NeMo Fabric skill", {
      metadata: { missing, disabled },
    });
  }
}

function validNonNegativeNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function validTokenCount(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0;
}

function normalizeUsage(value: unknown): AgentUsage | undefined {
  if (value === null || value === undefined) return undefined;
  if (typeof value !== "object") {
    throw new LifecycleError("droid_malformed_result", "Droid returned malformed usage metadata");
  }
  const usage = value as Partial<DroidTokenUsage>;
  const counts = [
    usage.inputTokens,
    usage.outputTokens,
    usage.cacheCreationTokens,
    usage.cacheReadTokens,
    usage.thinkingTokens,
  ];
  if (!counts.every(validTokenCount) || (usage.factoryCredits !== undefined && !validNonNegativeNumber(usage.factoryCredits))) {
    throw new LifecycleError("droid_malformed_result", "Droid returned malformed usage metadata");
  }
  const total = usage.inputTokens! + usage.outputTokens!;
  if (!Number.isSafeInteger(total)) {
    throw new LifecycleError("droid_malformed_result", "Droid returned malformed usage metadata");
  }
  return {
    input_tokens: usage.inputTokens!,
    output_tokens: usage.outputTokens!,
    total_tokens: total,
    extensions: {
      cache_read_tokens: usage.cacheReadTokens!,
      cache_write_tokens: usage.cacheCreationTokens!,
      thinking_tokens: usage.thinkingTokens!,
      ...(usage.factoryCredits === undefined ? {} : { factory_credits: usage.factoryCredits }),
    },
  };
}

function normalizeResult(value: unknown): DroidPromptOutcome {
  if (typeof value !== "object" || value === null) {
    throw new LifecycleError("droid_malformed_result", "Droid did not return a terminal result");
  }
  const result = value as DroidResult;
  if (
    result.type !== "result" ||
    typeof result.sessionId !== "string" ||
    result.sessionId.length === 0 ||
    !validTokenCount(result.durationMs) ||
    !validTokenCount(result.turnCount) ||
    typeof result.text !== "string" ||
    typeof result.success !== "boolean" ||
    typeof result.interrupted !== "boolean" ||
    !["success", "interrupted", "error_during_execution", "error_structured_output"].includes(String(result.subtype))
  ) {
    throw new LifecycleError("droid_malformed_result", "Droid returned a malformed terminal result");
  }
  const usage = normalizeUsage(result.tokenUsage);
  const extensions: JsonObject = {
    session_id: result.sessionId,
    finish_reason: String(result.subtype),
    duration_ms: Number(result.durationMs),
    turn_count: Number(result.turnCount),
  };
  if (result.success && result.subtype === "success" && !result.interrupted) {
    return { status: "completed", text: result.text, ...(usage === undefined ? {} : { usage }), extensions };
  }
  if (result.interrupted && result.subtype === "interrupted" && !result.success) {
    return { status: "interrupted", text: result.text, ...(usage === undefined ? {} : { usage }), extensions };
  }
  if (!result.success && !result.interrupted && String(result.subtype).startsWith("error_")) {
    const error = result.error as { message?: unknown } | null;
    const errorMessage =
      error !== null && typeof error === "object" && typeof error.message === "string"
        ? error.message
        : result.text;
    return {
      status: "failed",
      text: result.text,
      errorMessage: errorMessage.length === 0 ? "The Droid invocation failed" : errorMessage,
      ...(usage === undefined ? {} : { usage }),
      extensions,
    };
  }
  throw new LifecycleError("droid_malformed_result", "Droid returned inconsistent terminal result fields");
}

function sdkErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim().length > 0) return error.message;
  if (typeof error === "string" && error.trim().length > 0) return error;
  return "Droid session communication failed";
}

async function applyToolPolicy(
  session: DroidSdkSession,
  config: AgentConfig,
  mcpServers: Record<string, JsonObject>,
): Promise<void> {
  const enabled = config.tools?.enabled;
  const blocked = config.tools?.blocked ?? [];
  if (enabled === undefined && blocked.length === 0) return;

  const listed = await session.listTools();
  const mcpPrefixes = Object.keys(mcpServers).map((name) => `${name}___`);
  const available = listed
    .map((tool) => tool.id)
    .filter((id): id is string =>
      typeof id === "string" && id.length > 0 && !mcpPrefixes.some((prefix) => id.startsWith(prefix)),
    );
  const malformed = listed.some((tool) => typeof tool.id !== "string" || tool.id.length === 0);
  if (malformed || new Set(available).size !== available.length) {
    throw new LifecycleError("droid_tool_status_invalid", "Droid returned malformed built-in tool metadata");
  }
  const known = new Set(available);
  const requested = [...(enabled ?? []), ...blocked];
  const unknown = [...new Set(requested.filter((name) => !known.has(name)))].sort();
  if (unknown.length > 0) {
    throw new LifecycleError("droid_tool_unknown", "One or more configured tools are not Droid built-in tool IDs", {
      metadata: { tools: unknown },
    });
  }
  const disabled = new Set(blocked);
  if (enabled !== undefined && enabled !== null) {
    const selected = new Set(enabled);
    for (const name of available) {
      if (!selected.has(name)) disabled.add(name);
    }
  }
  await session.updateSettings({ disabledToolIds: [...disabled].sort() });
}

async function validateMcpStatus(
  session: DroidSdkSession,
  configured: Record<string, JsonObject>,
  timeoutMs: number,
): Promise<void> {
  const expected = new Set(Object.keys(configured));
  if (expected.size === 0) return;
  const deadline = Date.now() + timeoutMs;
  let missing = [...expected];
  let pending: string[] = [];
  while (true) {
    const remainingBeforeRequestMs = deadline - Date.now();
    if (remainingBeforeRequestMs <= 0) {
      throw new LifecycleError("droid_mcp_load_failed", "Droid did not connect every configured MCP server", {
        metadata: { missing, connecting: pending },
      });
    }
    let cancelTimer: () => void = () => undefined;
    let report: { servers: DroidMcpStatus[] } | undefined;
    try {
      report = await Promise.race([
        session.listMcpServers(),
        new Promise<undefined>((resolve) => {
          const timer = setTimeout(resolve, remainingBeforeRequestMs);
          cancelTimer = () => clearTimeout(timer);
        }),
      ]);
    } catch {
      throw new LifecycleError("droid_mcp_status_unavailable", "Droid could not determine configured MCP server status");
    } finally {
      cancelTimer();
    }
    if (report === undefined) {
      throw new LifecycleError("droid_mcp_load_failed", "Droid did not connect every configured MCP server", {
        metadata: { missing, connecting: pending },
      });
    }
    if (!Array.isArray(report.servers)) {
      throw new LifecycleError("droid_mcp_status_unavailable", "Droid could not determine configured MCP server status");
    }
    const byName = new Map(report.servers.map((server) => [server.name, server]));
    missing = [...expected].filter((name) => !byName.has(name));
    const failed = [...expected].filter((name) => {
      const status = byName.get(name)?.status;
      return status === "failed" || status === "disabled" || status === "disconnected";
    });
    if (failed.length > 0) {
      throw new LifecycleError("droid_mcp_load_failed", "Droid did not load every configured MCP server", {
        metadata: {
          missing,
          failed: failed.map((name) => ({
            name,
            ...(typeof byName.get(name)?.error === "string" ? { error: String(byName.get(name)?.error) } : {}),
          })),
        },
      });
    }
    pending = [...expected].filter((name) => byName.get(name)?.status === "connecting");
    const invalid = [...expected].filter((name) => {
      const status = byName.get(name)?.status;
      return status !== undefined && status !== "connected" && status !== "connecting";
    });
    if (invalid.length > 0) {
      throw new LifecycleError("droid_mcp_status_unavailable", "Droid returned an unknown MCP server status", {
        metadata: { servers: invalid },
      });
    }
    if (missing.length === 0 && pending.length === 0) return;
    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) {
      throw new LifecycleError("droid_mcp_load_failed", "Droid did not connect every configured MCP server", {
        metadata: { missing, connecting: pending },
      });
    }
    await new Promise((resolve) => setTimeout(resolve, Math.min(MCP_STATUS_POLL_INTERVAL_MS, remainingMs)));
  }
}

class DroidSdkSessionHandle implements DroidSessionHandle {
  private readonly session: DroidSdkSession;
  private readonly runtimeRoot?: string;
  private stopped = false;

  constructor(session: DroidSdkSession, runtimeRoot?: string) {
    this.session = session;
    this.runtimeRoot = runtimeRoot;
  }

  async prompt(text: string): Promise<DroidPromptOutcome> {
    if (this.stopped) {
      throw new LifecycleError("droid_session_stopped", "The Droid session is already stopped");
    }
    let terminal: unknown;
    try {
      for await (const message of this.session.stream(text)) {
        if (typeof message === "object" && message !== null && "type" in message && message.type === "result") {
          terminal = message;
        }
      }
    } catch (error) {
      throw new LifecycleError("droid_session_failed", sdkErrorMessage(error), { retryable: true });
    }
    if (terminal === undefined) {
      throw new LifecycleError("droid_session_failed", "Droid session ended without a terminal result", {
        retryable: true,
      });
    }
    return normalizeResult(terminal);
  }

  async stop(): Promise<void> {
    if (this.stopped) return;
    this.stopped = true;
    try {
      await this.session.close();
    } finally {
      if (this.runtimeRoot !== undefined) {
        await rm(this.runtimeRoot, { recursive: true, force: true });
      }
    }
  }
}

export class DroidSdkSessionFactory implements DroidSessionFactory {
  private readonly loader: DroidSdkLoader;
  private readonly mcpStartupTimeoutMs: number;
  private readonly parentEnvironment: NodeJS.ProcessEnv;

  constructor(
    loader: DroidSdkLoader = loadDroidSdk,
    mcpStartupTimeoutMs = MCP_STARTUP_TIMEOUT_MS,
    parentEnvironment: NodeJS.ProcessEnv = process.env,
  ) {
    this.loader = loader;
    this.mcpStartupTimeoutMs = mcpStartupTimeoutMs;
    this.parentEnvironment = parentEnvironment;
  }

  async create(input: AdapterStartInput): Promise<DroidSessionHandle> {
    const selected = selectModel(input.config);
    const apiKeyEnv = selected.api_key_env;
    if (apiKeyEnv === undefined || apiKeyEnv === null || apiKeyEnv.length === 0) {
      throw new LifecycleError("droid_api_key_env_required", "The selected Factory model requires api_key_env");
    }
    const apiKey = credentialValue(input, apiKeyEnv);
    if (apiKey === undefined || apiKey.length === 0) {
      throw new LifecycleError("droid_credential_missing", `Credential environment variable ${apiKeyEnv} is not set`);
    }
    const workspace = await resolveWorkspace(input);
    const mcpServers = normalizedMcp(input);
    const instruction = input.config.instructions?.system;
    if (instruction?.extensions !== undefined && Object.keys(instruction.extensions).length > 0) {
      throw new LifecycleError("droid_instruction_extensions_unsupported", "Droid does not support instruction extensions");
    }
    if (instruction?.mode !== undefined && instruction.mode !== "replace" && instruction.mode !== "append") {
      throw new LifecycleError(
        "unsupported_system_instruction_mode",
        `Droid does not support instructions.system.mode='${instruction.mode}'; supported modes: replace, append`,
      );
    }
    const runtimeProfile = await prepareRuntimeProfile(input, mcpServers, this.parentEnvironment);
    const runtimeEnvironment = input.runtimeContext.environment.env ?? {};

    let session: DroidSdkSession | undefined;
    try {
      const sdk = await this.loader();
      session = await sdk.createSession({
        cwd: workspace,
        modelId: selected.model,
        apiKey,
        autonomyLevel: "high",
        env:
          runtimeProfile.home === undefined
            ? runtimeEnvironment
            : {
                ...runtimeEnvironment,
                HOME: runtimeProfile.home,
                USERPROFILE: runtimeProfile.home,
                FACTORY_HOME_OVERRIDE: runtimeProfile.home,
              },
        ...(instruction === undefined || instruction === null
          ? {}
          : {
              systemPrompt:
                instruction.mode === "append"
                  ? { type: "preset", preset: "droid", append: instruction.content }
                  : instruction.content,
            }),
      });
      await applyToolPolicy(session, input.config, mcpServers);
      await validateMcpStatus(session, mcpServers, this.mcpStartupTimeoutMs);
      await validateSkills(session, runtimeProfile.entrypoints);
      return new DroidSdkSessionHandle(session, runtimeProfile.root);
    } catch (error) {
      await session?.close().catch(() => undefined);
      if (runtimeProfile.root !== undefined) {
        await rm(runtimeProfile.root, { recursive: true, force: true });
      }
      if (error instanceof LifecycleError) throw error;
      const message = sdkErrorMessage(error);
      if (/\b(?:ENOENT|not found|spawn droid)\b/i.test(message)) {
        throw new LifecycleError(
          "droid_cli_unavailable",
          "The Factory Droid CLI is not installed or is not available on PATH",
        );
      }
      throw new LifecycleError("droid_start_failed", "Factory Droid SDK initialization failed");
    }
  }
}

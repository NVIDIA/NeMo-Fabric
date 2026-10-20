// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { cp, mkdir, mkdtemp, readFile, realpath, rm, stat } from "node:fs/promises";
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
const DETAILED_TOKEN_USAGE_KEY = "nemo_fabric.detailed_token_usage.v1";
const MCP_STARTUP_TIMEOUT_MS = 10_000;
const MCP_STATUS_POLL_INTERVAL_MS = 50;
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

interface DroidMcpToolInfo {
  serverName?: unknown;
  name?: unknown;
}

interface DroidSkillInfo {
  name?: unknown;
  filePath?: unknown;
  enabled?: unknown;
}

interface DroidMcpStdioConfig {
  name: string;
  command: string;
  args: string[];
  env: Record<string, string>;
}

interface DroidMcpNetworkConfig {
  name: string;
  type: "http" | "sse";
  url: string;
  headers: Array<{ name: string; value: string }>;
}

type DroidMcpServerConfig = DroidMcpStdioConfig | DroidMcpNetworkConfig;

type DroidNotificationCallback = (notification: Record<string, unknown>) => void;

interface DroidSdkSession {
  readonly id: string;
  /** Kept current by the SDK from settings_updated notifications. */
  readonly settings?: { modelId?: unknown };
  stream(prompt: string): AsyncGenerator<unknown, void, undefined>;
  onNotification?(callback: DroidNotificationCallback, filter?: { type?: string }): () => void;
  close(): Promise<void>;
  listTools(): Promise<DroidToolInfo[]>;
  updateSettings(settings: { disabledToolIds: string[] }): Promise<unknown>;
  listMcpTools(): Promise<DroidMcpToolInfo[]>;
  listSkills(): Promise<{ skills: DroidSkillInfo[] }>;
}

interface DroidSdkModule {
  createSession(options: Record<string, unknown>): Promise<DroidSdkSession>;
}

interface EnvironmentLease {
  release(): void;
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
    throw new LifecycleError("droid_provider_unsupported", "The Droid adapter requires models to use provider='factory'");
  }
  return selected;
}

function credentialValue(
  input: AdapterStartInput,
  name: string,
  parentEnvironment: NodeJS.ProcessEnv,
): string | undefined {
  return input.runtimeContext.environment.env?.[name] ?? parentEnvironment[name];
}

function replaceEnvironment(values: NodeJS.ProcessEnv): void {
  for (const name of Object.keys(process.env)) {
    delete process.env[name];
  }
  for (const [name, value] of Object.entries(values)) {
    if (value !== undefined) process.env[name] = value;
  }
}

function leaseEnvironment(
  configured: Record<string, string>,
  credentialName: string,
  credential: string,
  parentEnvironment: NodeJS.ProcessEnv,
): EnvironmentLease {
  const original = { ...process.env };
  const values = Object.create(null) as NodeJS.ProcessEnv;
  for (const name of INHERITED_ENVIRONMENT_NAMES) {
    const value = parentEnvironment[name];
    if (value !== undefined) values[name] = value;
  }
  Object.assign(values, configured);
  values[credentialName] = credential;
  replaceEnvironment(values);
  return { release: () => replaceEnvironment(original) };
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

function normalizedMcpServer(name: string, server: AgentMcpServerConfig): DroidMcpServerConfig {
  rejectMcpExtensions(name, server);
  if (server.transport === "stdio") {
    if (server.custom_headers !== undefined && Object.keys(server.custom_headers).length > 0) {
      throw new LifecycleError("droid_mcp_headers_unsupported", "Droid stdio MCP servers do not support custom_headers", {
        metadata: { server: name },
      });
    }
    return {
      name,
      command: server.url,
      args: server.args ?? [],
      env: server.env ?? {},
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
      name,
      type: server.transport === "sse" ? "sse" : "http",
      url: server.url,
      headers: Object.entries(server.custom_headers ?? {}).map(([headerName, value]) => ({
        name: headerName,
        value,
      })),
    };
  }
  throw new LifecycleError("droid_mcp_transport_unsupported", "Droid does not support the configured MCP transport", {
    metadata: { server: name, transport: server.transport },
  });
}

function normalizedMcp(input: AdapterStartInput): DroidMcpServerConfig[] {
  if (input.config.mcp?.extensions !== undefined && Object.keys(input.config.mcp.extensions).length > 0) {
    throw new LifecycleError("droid_mcp_extensions_unsupported", "Droid does not support top-level MCP extensions");
  }
  return Object.entries(input.config.mcp?.servers ?? {}).map(([name, server]) => normalizedMcpServer(name, server));
}

interface RuntimeProfile {
  root?: string;
  home?: string;
  entrypoints: string[];
  settingsEnvironment: Record<string, string>;
  settingsFiles: DroidSettings[];
}

function settingsHome(input: AdapterStartInput, parentEnvironment: NodeJS.ProcessEnv): string | undefined {
  const runtimeEnvironment = input.runtimeContext.environment.env ?? {};
  return (
    runtimeEnvironment.FACTORY_HOME_OVERRIDE ||
    runtimeEnvironment.HOME ||
    runtimeEnvironment.USERPROFILE ||
    parentEnvironment.FACTORY_HOME_OVERRIDE ||
    parentEnvironment.HOME ||
    parentEnvironment.USERPROFILE
  );
}

type CustomModelSettings = Record<string, unknown>;

type DroidSettings = Record<string, unknown>;

/** Reads the Droid settings files in the order Droid layers them. */
async function readDroidSettings(sourceHome: string | undefined): Promise<DroidSettings[]> {
  if (sourceHome === undefined) return [];
  const files: DroidSettings[] = [];
  for (const filename of ["settings.json", "settings.local.json"]) {
    try {
      const settings: unknown = JSON.parse(await readFile(join(sourceHome, ".factory", filename), "utf8"));
      if (typeof settings === "object" && settings !== null && !Array.isArray(settings)) {
        files.push(settings as DroidSettings);
      }
    } catch (error) {
      if (typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT") continue;
      if (error instanceof SyntaxError) continue; // Droid reports malformed settings during session creation.
      throw new LifecycleError(
        "droid_settings_unavailable",
        "Droid settings could not be read by NeMo Fabric",
      );
    }
  }
  return files;
}

/** Returns the first custom model entry for `model` in each settings file. */
function customModelSettings(settingsFiles: DroidSettings[], model: string): CustomModelSettings[] {
  if (!model.startsWith("custom:")) return [];
  const modelId = model.slice("custom:".length);
  const entries: CustomModelSettings[] = [];
  for (const settings of settingsFiles) {
    const customModels = settings.customModels;
    if (!Array.isArray(customModels)) continue;
    const configured: unknown = customModels.find(
      (candidate) =>
        typeof candidate === "object" &&
        candidate !== null &&
        "model" in candidate &&
        candidate.model === modelId,
    );
    if (typeof configured === "object" && configured !== null) entries.push(configured as CustomModelSettings);
  }
  return entries;
}

function customModelSettingsEnvironment(
  input: AdapterStartInput,
  customModels: CustomModelSettings[],
  parentEnvironment: NodeJS.ProcessEnv,
): Record<string, string> {
  let apiKey: unknown;
  for (const configured of customModels) {
    if ("apiKey" in configured) apiKey = configured.apiKey;
  }
  if (typeof apiKey !== "string") return {};
  const match = /^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$/.exec(apiKey);
  if (match === null) return {};
  const name = match[1];
  if (name === undefined) return {};
  const value = credentialValue(input, name, parentEnvironment);
  if (value === undefined || value.length === 0) {
    throw new LifecycleError("droid_credential_missing", `Credential environment variable ${name} is not set`);
  }
  return { [name]: value };
}

/**
 * How Droid 0.233.0 fills a turn's inputTokens on the provider path it uses for
 * the configured model. Droid routes by model provider: Google models use the
 * Gemini handler, Anthropic and Factory-hosted models use the Anthropic
 * Messages or Chat Completions handler, OpenAI and xAI models use the Responses
 * handler, and Bedrock Converse custom models use the Converse handler.
 */
type DroidInputConvention =
  /** Gemini: Droid keeps promptTokenCount, which includes cached content. */
  | "includes_cache_reads"
  /** Responses and Chat Completions: Droid subtracts cache reads from the provider input. */
  | "excludes_cache_reads"
  /** Anthropic Messages and Bedrock Converse: the provider input excludes cache reads and writes. */
  | "excludes_cache";

interface DroidUsageConvention {
  input: DroidInputConvention;
  /**
   * The provider sends a cache read count, or omits it only when it is 0, on
   * every response. Droid's 0 then means no cache reads rather than no report.
   */
  reportsCacheReads: boolean;
}

// Anthropic and Bedrock define total input as input plus cache reads and
// writes, and the OpenAI Responses usage object requires cached_tokens. Gemini
// JSON omits a zero cachedContentTokenCount. Factory-hosted, xAI, and custom
// endpoints are not shown to report cache reads on every response.
const GEMINI: DroidUsageConvention = { input: "includes_cache_reads", reportsCacheReads: true };
const ANTHROPIC_OR_BEDROCK: DroidUsageConvention = { input: "excludes_cache", reportsCacheReads: true };
const OPENAI_RESPONSES: DroidUsageConvention = { input: "excludes_cache_reads", reportsCacheReads: true };
const EXCLUDES_CACHE: DroidUsageConvention = { input: "excludes_cache", reportsCacheReads: false };
const EXCLUDES_CACHE_READS: DroidUsageConvention = { input: "excludes_cache_reads", reportsCacheReads: false };

const BUILT_IN_MODEL_CONVENTIONS: ReadonlyArray<readonly [string, DroidUsageConvention]> = [
  ["claude-", ANTHROPIC_OR_BEDROCK],
  ["gemini-", GEMINI],
  ["gpt-", OPENAI_RESPONSES],
  ["grok-", EXCLUDES_CACHE_READS],
  // Factory-hosted models use the Anthropic Messages or Chat Completions handler;
  // the Chat Completions handler never records cache writes.
  ["deepseek-", EXCLUDES_CACHE],
  ["glm-", EXCLUDES_CACHE],
  ["kimi-", EXCLUDES_CACHE],
  ["minimax-", EXCLUDES_CACHE],
  ["mistral-", EXCLUDES_CACHE],
  ["nemotron-", EXCLUDES_CACHE],
  ["qwen", EXCLUDES_CACHE],
];

const GOOGLE_HINT = /gemini|garnet|generativelanguage\.googleapis\.com|aiplatform\.googleapis\.com/i;

function customModelConvention(settings: CustomModelSettings): DroidUsageConvention | undefined {
  switch (settings.provider) {
    case "bedrock-converse":
      return ANTHROPIC_OR_BEDROCK;
    case "anthropic":
      return EXCLUDES_CACHE;
    case "openai":
      return EXCLUDES_CACHE_READS;
    case "generic-chat-completion-api": {
      // Droid sends a chat-completions model to its Gemini handler when the
      // model, base model, or base URL resolves to Google.
      const hints = [settings.model, settings.baseModelId, settings.baseUrl].filter((value) => typeof value === "string");
      return hints.some((value) => GOOGLE_HINT.test(value)) ? undefined : EXCLUDES_CACHE_READS;
    }
    default:
      return undefined;
  }
}

function customConvention(customModels: CustomModelSettings[]): DroidUsageConvention | undefined {
  const conventions = new Set(customModels.map(customModelConvention));
  return conventions.size === 1 ? [...conventions][0] : undefined;
}

// Droid's router models pick a model per request, so their convention is unknown.
const ROUTER_MODELS = new Set(["auto", "auto-fast"]);

/** Selects the usage convention for whichever model the Droid session uses. */
class DroidUsageConventions {
  private readonly settingsFiles: DroidSettings[];
  private readonly resolved = new Map<string, DroidUsageConvention | undefined>();

  constructor(settingsFiles: DroidSettings[]) {
    this.settingsFiles = settingsFiles;
  }

  forModel(model: string): DroidUsageConvention | undefined {
    if (!this.resolved.has(model)) this.resolved.set(model, this.resolve(model));
    return this.resolved.get(model);
  }

  private resolve(model: string): DroidUsageConvention | undefined {
    if (ROUTER_MODELS.has(model)) return undefined;
    if (model.startsWith("custom:")) return customConvention(customModelSettings(this.settingsFiles, model));
    return BUILT_IN_MODEL_CONVENTIONS.find(([prefix]) => model.startsWith(prefix))?.[1];
  }
}

async function prepareRuntimeProfile(
  input: AdapterStartInput,
  hasMcpServers: boolean,
  parentEnvironment: NodeJS.ProcessEnv,
  model: string,
): Promise<RuntimeProfile> {
  if (input.config.skills?.extensions !== undefined) {
    throw new LifecycleError("droid_skill_extensions_unsupported", "Droid does not support skill extensions");
  }
  const configured = input.config.skills?.paths ?? [];
  const sourceHome = settingsHome(input, parentEnvironment);
  const settingsFiles = await readDroidSettings(sourceHome);
  const settingsEnvironment = customModelSettingsEnvironment(
    input,
    customModelSettings(settingsFiles, model),
    parentEnvironment,
  );
  if (configured.length === 0 && !hasMcpServers) {
    return { entrypoints: [], settingsEnvironment, settingsFiles };
  }

  const root = await mkdtemp(join(tmpdir(), "nemo-fabric-droid-"));
  const home = join(root, "home");
  const factoryRoot = join(home, ".factory");
  const skillsRoot = join(home, ".agents", "skills");
  try {
    if (sourceHome !== undefined) {
      for (const filename of ["settings.json", "settings.local.json"]) {
        const sourceSettings = join(sourceHome, ".factory", filename);
        try {
          if ((await stat(sourceSettings)).isFile()) {
            await mkdir(factoryRoot, { recursive: true });
            await cp(sourceSettings, join(factoryRoot, filename), { errorOnExist: true });
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
    return { root, home, entrypoints, settingsEnvironment, settingsFiles };
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

/**
 * Returns the turn's input including cache reads and writes, or undefined when
 * the provider path or Droid's input count does not establish it. Droid fills
 * a missing input count with 0, so 0 input stays unknown.
 */
function inputIncludingCache(
  convention: DroidInputConvention | undefined,
  inputTokens: number,
  cacheReadTokens: number,
  cacheWriteTokens: number,
): number | undefined {
  if (inputTokens === 0) return undefined;
  switch (convention) {
    case "excludes_cache":
      return inputTokens + cacheReadTokens + cacheWriteTokens;
    // The Gemini and Chat Completions handlers never record cache writes. The
    // Responses handler records a nonstandard write count without subtracting
    // it, so whether input already includes those writes is not established.
    case "excludes_cache_reads":
      return cacheWriteTokens === 0 ? inputTokens + cacheReadTokens : undefined;
    case "includes_cache_reads":
      return cacheWriteTokens === 0 ? inputTokens : undefined;
    default:
      return undefined;
  }
}

function normalizeUsage(value: unknown, convention: DroidUsageConvention | undefined): AgentUsage | undefined {
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
  const { inputTokens, outputTokens, cacheCreationTokens, cacheReadTokens, thinkingTokens, factoryCredits } =
    usage as DroidTokenUsage;
  // Droid stores a count the provider omitted as 0, so a 0 count is unknown.
  // Output includes reasoning on every provider path.
  const output = outputTokens > 0 ? outputTokens : undefined;
  // A 0 cache read count is a real 0 only on paths that report cache reads on
  // every response; elsewhere it may hide cache hits that belong in the input.
  const cacheReadsKnown = cacheReadTokens > 0 || convention?.reportsCacheReads === true;
  const inclusiveInput = cacheReadsKnown
    ? inputIncludingCache(convention?.input, inputTokens, cacheReadTokens, cacheCreationTokens)
    : undefined;
  // Without an inclusive count, report Droid's own input count, leave its
  // cache convention unset, and derive no total: the input may leave out cache.
  const input = inclusiveInput ?? (inputTokens > 0 ? inputTokens : undefined);
  const total = inclusiveInput === undefined || output === undefined ? undefined : inclusiveInput + output;
  if ((input !== undefined && !Number.isSafeInteger(input)) || (total !== undefined && !Number.isSafeInteger(total))) {
    throw new LifecycleError("droid_malformed_result", "Droid returned malformed usage metadata");
  }
  const inputUsage: JsonObject = {
    ...(input === undefined ? {} : { input_tokens: input }),
    ...(cacheReadTokens > 0 || inclusiveInput !== undefined ? { cached_input_tokens: cacheReadTokens } : {}),
    ...(inclusiveInput === undefined ? {} : { input_tokens_include_cache: true }),
  };
  const detailed: JsonObject = {
    ...inputUsage,
    ...(cacheCreationTokens > 0 ? { cache_write_input_tokens: cacheCreationTokens } : {}),
    ...(output === undefined ? {} : { output_tokens: output, output_tokens_include_reasoning: true }),
  };
  const extensions: JsonObject = {
    ...(Object.keys(detailed).length === 0 ? {} : { [DETAILED_TOKEN_USAGE_KEY]: detailed }),
    ...(thinkingTokens > 0 ? { thinking_tokens: thinkingTokens } : {}),
    ...(factoryCredits === undefined || factoryCredits === 0 ? {} : { factory_credits: factoryCredits }),
  };
  if (Object.keys(extensions).length === 0) return undefined;
  return {
    ...inputUsage,
    ...(output === undefined ? {} : { output_tokens: output }),
    ...(total === undefined ? {} : { total_tokens: total }),
    extensions,
  };
}

function normalizeResult(value: unknown, convention: DroidUsageConvention | undefined): DroidPromptOutcome {
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
  const usage = normalizeUsage(result.tokenUsage, convention);
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
  mcpServers: DroidMcpServerConfig[],
): Promise<void> {
  const enabled = config.tools?.enabled;
  const blocked = config.tools?.blocked ?? [];
  if (enabled === undefined && blocked.length === 0) return;

  const listed = await session.listTools();
  const mcpPrefixes = mcpServers.map((server) => `${server.name}___`);
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

async function validateMcpTools(
  session: DroidSdkSession,
  configured: DroidMcpServerConfig[],
  timeoutMs: number,
): Promise<void> {
  const expected = new Set(configured.map((server) => server.name));
  if (expected.size === 0) return;
  const deadline = Date.now() + timeoutMs;
  let missing = [...expected];
  while (true) {
    const remainingBeforeRequestMs = deadline - Date.now();
    if (remainingBeforeRequestMs <= 0) {
      throw new LifecycleError("droid_mcp_load_failed", "Droid did not load tools from every configured MCP server", {
        metadata: { missing },
      });
    }
    let cancelTimer: () => void = () => undefined;
    let tools: DroidMcpToolInfo[] | undefined;
    try {
      tools = await Promise.race([
        session.listMcpTools(),
        new Promise<undefined>((resolve) => {
          const timer = setTimeout(resolve, remainingBeforeRequestMs);
          cancelTimer = () => clearTimeout(timer);
        }),
      ]);
    } catch {
      throw new LifecycleError("droid_mcp_status_unavailable", "Droid could not determine configured MCP tool status");
    } finally {
      cancelTimer();
    }
    if (tools === undefined) {
      throw new LifecycleError("droid_mcp_load_failed", "Droid did not load tools from every configured MCP server", {
        metadata: { missing },
      });
    }
    if (
      !Array.isArray(tools) ||
      tools.some(
        (tool) =>
          typeof tool.serverName !== "string" ||
          tool.serverName.length === 0 ||
          typeof tool.name !== "string" ||
          tool.name.length === 0,
      )
    ) {
      throw new LifecycleError("droid_mcp_status_unavailable", "Droid returned malformed MCP tool metadata");
    }
    const discovered = new Set(tools.map((tool) => tool.serverName as string));
    missing = [...expected].filter((name) => !discovered.has(name));
    if (missing.length === 0) return;
    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) {
      throw new LifecycleError("droid_mcp_load_failed", "Droid did not load tools from every configured MCP server", {
        metadata: { missing },
      });
    }
    await new Promise((resolve) => setTimeout(resolve, Math.min(MCP_STATUS_POLL_INTERVAL_MS, remainingMs)));
  }
}

function hasTokenCounts(value: unknown): boolean {
  if (typeof value !== "object" || value === null) return false;
  const usage = value as Record<string, unknown>;
  return ["inputTokens", "outputTokens", "cacheCreationTokens", "cacheReadTokens", "thinkingTokens"].some(
    (key) => typeof usage[key] === "number" && Number(usage[key]) > 0,
  );
}

/** What the session reported during one invocation that decides its usage convention. */
interface TurnObservation {
  /** Models the session used during the turn, from settings_updated notifications. */
  models: Set<string>;
  /** The turn's completion reported usage from Task subagents. */
  childUsage: boolean;
}

class DroidSdkSessionHandle implements DroidSessionHandle {
  private readonly session: DroidSdkSession;
  private readonly runtimeRoot?: string;
  private readonly configuredModel: string;
  private readonly conventions: DroidUsageConventions;
  private stopped = false;

  constructor(
    session: DroidSdkSession,
    configuredModel: string,
    conventions: DroidUsageConventions,
    runtimeRoot?: string,
  ) {
    this.session = session;
    this.configuredModel = configuredModel;
    this.conventions = conventions;
    this.runtimeRoot = runtimeRoot;
  }

  private currentModel(): string {
    const modelId = this.session.settings?.modelId;
    return typeof modelId === "string" && modelId.length > 0 ? modelId : this.configuredModel;
  }

  /**
   * Watches this session's notifications for the whole invocation: every
   * model it switches to and whether the turn's completion carries child usage.
   * Without notifications the turn cannot be checked, so it counts as having
   * child usage.
   */
  private observeTurn(): { observation: TurnObservation; stop(): void } {
    const observation: TurnObservation = { models: new Set([this.currentModel()]), childUsage: false };
    if (typeof this.session.onNotification !== "function") {
      observation.childUsage = true;
      return { observation, stop: () => undefined };
    }
    const stop = this.session.onNotification((notification) => {
      const params = notification.params as { sessionId?: unknown; notification?: Record<string, unknown> } | undefined;
      const inner = params?.notification;
      if (inner === undefined || (params?.sessionId !== undefined && params.sessionId !== this.session.id)) return;
      if (inner.type === "settings_updated") {
        const modelId = (inner.settings as { modelId?: unknown } | undefined)?.modelId;
        if (typeof modelId === "string" && modelId.length > 0) observation.models.add(modelId);
      } else if (inner.type === "agent_turn_completed" && hasTokenCounts(inner.childTokenUsage)) {
        observation.childUsage = true;
      }
    });
    return { observation, stop };
  }

  async prompt(text: string): Promise<DroidPromptOutcome> {
    if (this.stopped) {
      throw new LifecycleError("droid_session_stopped", "The Droid session is already stopped");
    }
    const turn = this.observeTurn();
    let terminal: unknown;
    try {
      for await (const message of this.session.stream(text)) {
        if (typeof message === "object" && message !== null && "type" in message && message.type === "result") {
          terminal = message;
        }
      }
    } catch (error) {
      throw new LifecycleError("droid_session_failed", sdkErrorMessage(error), { retryable: true });
    } finally {
      turn.stop();
    }
    if (terminal === undefined) {
      throw new LifecycleError("droid_session_failed", "Droid session ended without a terminal result", {
        retryable: true,
      });
    }
    // Droid adds subagent usage to the turn's counters, and a subagent can use
    // another model; a model change mixes conventions too. Such a turn keeps
    // the reported counters without a cache convention.
    turn.observation.models.add(this.currentModel());
    const conventions = new Set([...turn.observation.models].map((model) => this.conventions.forModel(model)));
    const convention = conventions.size === 1 && !turn.observation.childUsage ? [...conventions][0] : undefined;
    return normalizeResult(terminal, convention);
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
    const apiKey = credentialValue(input, apiKeyEnv, this.parentEnvironment);
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
    const runtimeProfile = await prepareRuntimeProfile(
      input,
      mcpServers.length > 0,
      this.parentEnvironment,
      selected.model,
    );
    const runtimeEnvironment = input.runtimeContext.environment.env ?? {};
    const sessionEnvironment =
      runtimeProfile.home === undefined
        ? { ...runtimeEnvironment, ...runtimeProfile.settingsEnvironment }
        : {
            ...runtimeEnvironment,
            ...runtimeProfile.settingsEnvironment,
            HOME: runtimeProfile.home,
            USERPROFILE: runtimeProfile.home,
            FACTORY_HOME_OVERRIDE: runtimeProfile.home,
          };

    let session: DroidSdkSession | undefined;
    try {
      const environment = leaseEnvironment(sessionEnvironment, apiKeyEnv, apiKey, this.parentEnvironment);
      try {
        const sdk = await this.loader();
        session = await sdk.createSession({
          cwd: workspace,
          modelId: selected.model,
          apiKey,
          autonomyLevel: "high",
          ...(mcpServers.length === 0 ? {} : { mcpServers }),
          env: sessionEnvironment,
          ...(instruction === undefined || instruction === null
            ? {}
            : {
                systemPrompt:
                  instruction.mode === "append"
                    ? { type: "preset", preset: "droid", append: instruction.content }
                    : instruction.content,
              }),
        });
      } finally {
        environment.release();
      }
      await applyToolPolicy(session, input.config, mcpServers);
      await validateMcpTools(session, mcpServers, this.mcpStartupTimeoutMs);
      await validateSkills(session, runtimeProfile.entrypoints);
      return new DroidSdkSessionHandle(
        session,
        selected.model,
        new DroidUsageConventions(runtimeProfile.settingsFiles),
        runtimeProfile.root,
      );
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

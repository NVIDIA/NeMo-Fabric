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

import type { ClinePromptOutcome, ClineSessionFactory, ClineSessionHandle } from "./runtime.js";

const CLINE_HARNESS_INSTALL_COMMAND = "npm install @cline/sdk@0.0.83";
const CLINE_PLUGIN_NAME = "nvidia.fabric.cline";
const AGENT_PLUGIN_MANIFEST_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json";
const AGENT_PLUGIN_MCP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json";

interface ClineUsage {
  inputTokens?: number;
  outputTokens?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  totalCost?: number;
}

interface ClineResult {
  text?: unknown;
  finishReason?: unknown;
  iterations?: unknown;
  model?: unknown;
  usage?: ClineUsage;
}

interface ClineStartResult {
  sessionId: string;
  result?: ClineResult;
}

interface ClineCoreInstance {
  start(input: Record<string, unknown>): Promise<ClineStartResult>;
  send(input: { sessionId: string; prompt: string }): Promise<ClineResult | undefined>;
  stop(sessionId: string): Promise<void>;
  dispose(reason?: string): Promise<void>;
}

interface ClineSdkModule {
  ALL_DEFAULT_TOOL_NAMES: readonly string[];
  ClineCore: {
    create(options: Record<string, unknown>): Promise<ClineCoreInstance>;
  };
  getClineDefaultSystemPrompt(options: {
    workspaceRoot: string;
    overridePrompt?: string;
    providerId: string;
  }): string;
  loadAgentPluginPackages(options: {
    pluginPaths: readonly string[];
    cwd: string;
    searchPaths: readonly string[];
    pluginDataRoot: string;
  }): Promise<{
    skills: readonly unknown[];
    mcpServers: readonly unknown[];
    diagnostics: readonly { level: "warning" | "error"; message: string }[];
  }>;
}

export type ClineSdkLoader = () => Promise<ClineSdkModule>;

let clineDataDirTail = Promise.resolve();

async function withClineDataDir<T>(dataDir: string, operation: () => Promise<T>): Promise<T> {
  const predecessor = clineDataDirTail;
  let release: () => void = () => undefined;
  clineDataDirTail = new Promise<void>((resolveLease) => {
    release = resolveLease;
  });
  await predecessor;

  const previousDataDir = process.env.CLINE_DATA_DIR;
  process.env.CLINE_DATA_DIR = dataDir;
  try {
    return await operation();
  } finally {
    if (previousDataDir === undefined) {
      delete process.env.CLINE_DATA_DIR;
    } else {
      process.env.CLINE_DATA_DIR = previousDataDir;
    }
    release();
  }
}

function isMissingModuleError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error.code === "ERR_MODULE_NOT_FOUND" || error.code === "MODULE_NOT_FOUND")
  );
}

async function loadClineSdk(): Promise<ClineSdkModule> {
  let sdk: ClineSdkModule;
  try {
    const moduleName: string = "@cline/sdk";
    sdk = (await import(moduleName)) as ClineSdkModule;
  } catch (error) {
    if (isMissingModuleError(error)) {
      throw new LifecycleError(
        "cline_harness_unavailable",
        `The Cline SDK harness is not installed. Install the supported harness with: ${CLINE_HARNESS_INSTALL_COMMAND}`,
      );
    }
    throw new LifecycleError("cline_harness_load_failed", "The installed Cline SDK harness could not be loaded");
  }
  if (
    typeof sdk.ClineCore?.create !== "function" ||
    typeof sdk.getClineDefaultSystemPrompt !== "function" ||
    typeof sdk.loadAgentPluginPackages !== "function" ||
    !Array.isArray(sdk.ALL_DEFAULT_TOOL_NAMES)
  ) {
    throw new LifecycleError(
      "cline_harness_incompatible",
      "The installed Cline SDK harness does not expose the APIs required by this adapter",
    );
  }
  return sdk;
}

function selectModel(config: AgentConfig): AgentModelConfig {
  const entries = Object.entries(config.models ?? {});
  if (entries.length === 0) {
    throw new LifecycleError("cline_model_required", "The Cline adapter requires one configured model");
  }
  const selected = config.models?.default ?? (entries.length === 1 ? entries[0]?.[1] : undefined);
  if (selected === undefined) {
    throw new LifecycleError(
      "cline_model_ambiguous",
      "Configure a default model role when the Cline adapter receives multiple models",
    );
  }
  return selected;
}

function credentialValue(input: AdapterStartInput, name: string): string | undefined {
  return input.runtimeContext.environment.env?.[name] ?? process.env[name];
}

async function resolveWorkspace(input: AdapterStartInput): Promise<string> {
  try {
    const workspace = await realpath(resolve(input.runtimeContext.environment.workspace ?? input.baseDir));
    if (!(await stat(workspace)).isDirectory()) {
      throw new Error("not a directory");
    }
    return workspace;
  } catch {
    throw new LifecycleError("cline_workspace_invalid", "The NeMo Fabric runtime workspace must be a directory");
  }
}

function toolPolicies(
  config: AgentConfig,
  availableNames: readonly string[],
): Record<string, { enabled?: boolean; autoApprove: true }> {
  const available = new Set(availableNames);
  const enabled = config.tools?.enabled;
  const blocked = config.tools?.blocked ?? [];
  const requested = [...(enabled ?? []), ...blocked];
  const unknown = requested.filter((name) => !available.has(name));
  if (unknown.length > 0) {
    throw new LifecycleError("cline_tool_unknown", "One or more configured tools are not Cline built-in tools", {
      metadata: { tools: [...new Set(unknown)].sort() },
    });
  }

  const policies: Record<string, { enabled?: boolean; autoApprove: true }> = Object.fromEntries(
    availableNames.map((name) => [name, { autoApprove: true }]),
  );
  if (enabled !== undefined && enabled !== null) {
    const selected = new Set(enabled);
    for (const name of available) {
      policies[name] = { enabled: selected.has(name), autoApprove: true };
    }
  }
  for (const name of blocked) {
    policies[name] = { enabled: false, autoApprove: true };
  }
  return policies;
}

function normalizedMcpServer(name: string, server: AgentMcpServerConfig): JsonObject {
  if (server.authentication !== undefined && server.authentication !== null) {
    throw new LifecycleError(
      "cline_mcp_authentication_unsupported",
      "Cline MCP authentication is not supported by this adapter",
      { metadata: { server: name } },
    );
  }
  if (server.allowed_tools !== undefined && server.allowed_tools !== null) {
    throw new LifecycleError(
      "cline_mcp_allowlist_unsupported",
      "Cline MCP allowed_tools filtering is not supported by this adapter",
      { metadata: { server: name } },
    );
  }
  if (server.blocked_tools !== undefined && server.blocked_tools.length > 0) {
    throw new LifecycleError(
      "cline_mcp_blocklist_unsupported",
      "Cline MCP blocked_tools filtering is not supported by this adapter",
      { metadata: { server: name } },
    );
  }
  if (server.transport === "stdio") {
    if (server.custom_headers !== undefined && Object.keys(server.custom_headers).length > 0) {
      throw new LifecycleError(
        "cline_mcp_headers_unsupported",
        "Cline stdio MCP servers do not support custom_headers",
        { metadata: { server: name } },
      );
    }
    return {
      type: "stdio",
      command: server.url,
      ...(server.args === undefined || server.args.length === 0 ? {} : { args: server.args }),
      ...(server.env === undefined || Object.keys(server.env).length === 0 ? {} : { env: server.env }),
    };
  }
  if (server.transport === "sse" || server.transport === "streamable-http") {
    if ((server.args?.length ?? 0) > 0 || Object.keys(server.env ?? {}).length > 0) {
      throw new LifecycleError(
        "cline_mcp_process_fields_unsupported",
        "Cline network MCP servers do not support args or env",
        { metadata: { server: name } },
      );
    }
    return {
      type: server.transport === "sse" ? "sse" : "streamable-http",
      url: server.url,
      ...(server.custom_headers === undefined || Object.keys(server.custom_headers).length === 0
        ? {}
        : { headers: server.custom_headers }),
    };
  }
  throw new LifecycleError("cline_mcp_transport_unsupported", "Cline does not support the configured MCP transport", {
    metadata: { server: name, transport: server.transport },
  });
}

interface RuntimeFiles {
  root: string;
  dataDir: string;
  pluginRoot?: string;
  skillCount: number;
  mcpServerCount: number;
}

async function prepareRuntimeFiles(input: AdapterStartInput): Promise<RuntimeFiles> {
  const root = await mkdtemp(join(tmpdir(), "nemo-fabric-cline-"));
  const dataDir = join(root, "data");
  const skillPaths = input.config.skills?.paths ?? [];
  const mcpServers = input.config.mcp?.servers ?? {};
  if (skillPaths.length === 0 && Object.keys(mcpServers).length === 0) {
    return { root, dataDir, skillCount: 0, mcpServerCount: 0 };
  }

  const pluginRoot = join(root, "plugin");
  try {
    await mkdir(pluginRoot, { recursive: true });
    await writeFile(
      join(pluginRoot, "plugin.json"),
      `${JSON.stringify({
        $schema: AGENT_PLUGIN_MANIFEST_SCHEMA,
        name: CLINE_PLUGIN_NAME,
        version: "1.0.0",
        description: "Runtime-scoped NeMo Fabric skills and MCP servers.",
      }, null, 2)}\n`,
      "utf8",
    );

    if (skillPaths.length > 0) {
      const skillsRoot = join(pluginRoot, "skills");
      await mkdir(skillsRoot);
      const names = new Set<string>();
      for (const configured of skillPaths) {
        let source: string;
        try {
          source = await realpath(resolve(input.baseDir, configured));
          if (!(await stat(source)).isDirectory() || !(await stat(join(source, "SKILL.md"))).isFile()) {
            throw new Error("invalid skill directory");
          }
        } catch {
          throw new LifecycleError(
            "cline_skill_invalid",
            "Cline skill paths must be directories containing a SKILL.md file",
            { metadata: { path: configured } },
          );
        }
        const name = basename(source);
        if (names.has(name)) {
          throw new LifecycleError("cline_skill_collision", "Two configured Cline skills use the same directory name", {
            metadata: { skill: name },
          });
        }
        names.add(name);
        await cp(source, join(skillsRoot, name), { recursive: true, errorOnExist: true });
      }
    }

    if (Object.keys(mcpServers).length > 0) {
      const servers = Object.fromEntries(
        Object.entries(mcpServers).map(([name, server]) => [name, normalizedMcpServer(name, server)]),
      );
      await writeFile(
        join(pluginRoot, "mcp.json"),
        `${JSON.stringify({ $schema: AGENT_PLUGIN_MCP_SCHEMA, mcpServers: servers }, null, 2)}\n`,
        "utf8",
      );
    }
    return {
      root,
      dataDir,
      pluginRoot,
      skillCount: skillPaths.length,
      mcpServerCount: Object.keys(mcpServers).length,
    };
  } catch (error) {
    await rm(root, { recursive: true, force: true });
    throw error;
  }
}

async function validateRuntimePlugin(
  sdk: ClineSdkModule,
  files: RuntimeFiles,
  workspace: string,
): Promise<void> {
  if (files.pluginRoot === undefined) {
    return;
  }
  const report = await sdk.loadAgentPluginPackages({
    pluginPaths: [files.pluginRoot],
    cwd: workspace,
    searchPaths: [],
    pluginDataRoot: join(files.root, "plugin-data"),
  });
  const errors = report.diagnostics.filter((diagnostic) => diagnostic.level === "error");
  if (
    errors.length > 0 ||
    report.skills.length !== files.skillCount ||
    report.mcpServers.length !== files.mcpServerCount
  ) {
    throw new LifecycleError("cline_plugin_invalid", "Cline rejected configured skills or MCP servers", {
      metadata: {
        errors: errors.map((diagnostic) => diagnostic.message),
        expected_skills: files.skillCount,
        loaded_skills: report.skills.length,
        expected_mcp_servers: files.mcpServerCount,
        loaded_mcp_servers: report.mcpServers.length,
      },
    });
  }
}

function normalizeUsage(usage: ClineUsage | undefined): AgentUsage | undefined {
  if (usage === undefined) {
    return undefined;
  }
  if (typeof usage !== "object" || usage === null) {
    throw new LifecycleError("cline_malformed_result", "Cline returned malformed usage metadata");
  }
  for (const value of [usage.inputTokens, usage.outputTokens, usage.cacheReadTokens, usage.cacheWriteTokens]) {
    if (value !== undefined && (!Number.isSafeInteger(value) || value < 0)) {
      throw new LifecycleError("cline_malformed_result", "Cline returned malformed usage metadata");
    }
  }
  if (usage.totalCost !== undefined && (!Number.isFinite(usage.totalCost) || usage.totalCost < 0)) {
    throw new LifecycleError("cline_malformed_result", "Cline returned malformed usage metadata");
  }
  const input = usage.inputTokens;
  const output = usage.outputTokens;
  if (input !== undefined && output !== undefined && !Number.isSafeInteger(input + output)) {
    throw new LifecycleError("cline_malformed_result", "Cline returned malformed usage metadata");
  }
  const extensions: JsonObject = {
    ...(usage.cacheReadTokens === undefined ? {} : { cache_read_tokens: usage.cacheReadTokens }),
    ...(usage.cacheWriteTokens === undefined ? {} : { cache_write_tokens: usage.cacheWriteTokens }),
  };
  return {
    ...(input === undefined ? {} : { input_tokens: input }),
    ...(output === undefined ? {} : { output_tokens: output }),
    ...(input === undefined || output === undefined ? {} : { total_tokens: input + output }),
    ...(usage.totalCost === undefined ? {} : { cost_usd: usage.totalCost }),
    ...(Object.keys(extensions).length === 0 ? {} : { extensions }),
  };
}

function sdkErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message;
  }
  if (typeof error === "string" && error.trim().length > 0) {
    return error;
  }
  return "Cline session communication failed";
}

function normalizeResult(result: ClineResult | undefined, sessionId: string): ClinePromptOutcome {
  if (
    result === undefined ||
    !["completed", "aborted", "error", "mistake_limit", "max_iterations"].includes(String(result.finishReason))
  ) {
    throw new LifecycleError("cline_malformed_result", "Cline returned a malformed result");
  }
  if (typeof result.text !== "string") {
    throw new LifecycleError("cline_malformed_result", "Cline returned a malformed assistant response");
  }
  if (result.iterations !== undefined && (!Number.isInteger(result.iterations) || Number(result.iterations) < 0)) {
    throw new LifecycleError("cline_malformed_result", "Cline returned malformed iteration metadata");
  }
  let model: JsonObject | undefined;
  if (result.model !== undefined) {
    if (
      typeof result.model !== "object" ||
      result.model === null ||
      !("id" in result.model) ||
      !("provider" in result.model) ||
      typeof result.model.id !== "string" ||
      typeof result.model.provider !== "string"
    ) {
      throw new LifecycleError("cline_malformed_result", "Cline returned malformed model metadata");
    }
    model = { id: result.model.id, provider: result.model.provider };
  }
  const usage = normalizeUsage(result.usage);
  const finishReason = String(result.finishReason);
  const status = finishReason === "completed" ? "completed" : finishReason === "aborted" ? "aborted" : "failed";
  return {
    status,
    text: result.text,
    ...(status === "failed" && result.text.length > 0 ? { errorMessage: result.text } : {}),
    ...(usage === undefined ? {} : { usage }),
    extensions: {
      session_id: sessionId,
      finish_reason: finishReason,
      ...(result.iterations === undefined ? {} : { iterations: Number(result.iterations) }),
      ...(model === undefined ? {} : { model }),
    },
  };
}

class ClineSdkSessionHandle implements ClineSessionHandle {
  private readonly core: ClineCoreInstance;
  private readonly startInput: Record<string, unknown>;
  private sessionId?: string;
  private stopped = false;
  private readonly cleanup: () => Promise<void>;

  constructor(core: ClineCoreInstance, startInput: Record<string, unknown>, cleanup: () => Promise<void>) {
    this.core = core;
    this.startInput = startInput;
    this.cleanup = cleanup;
  }

  async prompt(text: string): Promise<ClinePromptOutcome> {
    if (this.stopped) {
      throw new LifecycleError("cline_session_stopped", "The Cline session is already stopped");
    }
    try {
      if (this.sessionId === undefined) {
        const started = await this.core.start({ ...this.startInput, prompt: text });
        if (typeof started.sessionId !== "string" || started.sessionId.length === 0) {
          throw new LifecycleError("cline_malformed_result", "Cline did not return a session identifier");
        }
        this.sessionId = started.sessionId;
        return normalizeResult(started.result, this.sessionId);
      }
      return normalizeResult(await this.core.send({ sessionId: this.sessionId, prompt: text }), this.sessionId);
    } catch (error) {
      if (error instanceof LifecycleError) {
        throw error;
      }
      throw new LifecycleError("cline_session_failed", sdkErrorMessage(error), {
        retryable: true,
      });
    }
  }

  async stop(): Promise<void> {
    if (this.stopped) {
      return;
    }
    this.stopped = true;
    let failure: unknown;
    if (this.sessionId !== undefined) {
      try {
        await this.core.stop(this.sessionId);
      } catch (error) {
        failure = error;
      }
    }
    try {
      await this.core.dispose("NeMo Fabric runtime stopped");
    } catch (error) {
      failure ??= error;
    }
    try {
      await this.cleanup();
    } catch (error) {
      failure ??= error;
    }
    if (failure !== undefined) {
      throw failure;
    }
  }
}

export class ClineSdkSessionFactory implements ClineSessionFactory {
  private readonly loader: ClineSdkLoader;

  constructor(loader: ClineSdkLoader = loadClineSdk) {
    this.loader = loader;
  }

  async create(input: AdapterStartInput): Promise<ClineSessionHandle> {
    const systemInstruction = input.config.instructions?.system;
    if (systemInstruction?.mode === "append") {
      throw new LifecycleError(
        "unsupported_system_instruction_mode",
        "Cline does not support instructions.system.mode='append'; supported modes: replace",
        {
          metadata: {
            field: "instructions.system.mode",
            mode: systemInstruction.mode,
            supported_modes: ["replace"],
          },
        },
      );
    }
    const selected = selectModel(input.config);
    const apiKeyEnv = selected.api_key_env;
    if (apiKeyEnv === undefined || apiKeyEnv === null || apiKeyEnv.length === 0) {
      throw new LifecycleError("cline_api_key_env_required", "The selected Cline model requires api_key_env");
    }
    const apiKey = credentialValue(input, apiKeyEnv);
    if (apiKey === undefined || apiKey.length === 0) {
      throw new LifecycleError("cline_credential_missing", `Credential environment variable ${apiKeyEnv} is not set`);
    }
    const workspace = await resolveWorkspace(input);
    const files = await prepareRuntimeFiles(input);
    let core: ClineCoreInstance | undefined;
    try {
      const sdk = await this.loader();
      await validateRuntimePlugin(sdk, files, workspace);
      const policies = {
        ...toolPolicies(input.config, sdk.ALL_DEFAULT_TOOL_NAMES),
      };
      core = await withClineDataDir(files.dataDir, () => sdk.ClineCore.create({
        clientName: "nemo-fabric",
        backendMode: "local",
        toolPolicies: policies,
      }));
      const systemPrompt =
        systemInstruction === undefined || systemInstruction === null
          ? sdk.getClineDefaultSystemPrompt({
              workspaceRoot: workspace,
              providerId: selected.provider,
            })
          : systemInstruction.content;
      return new ClineSdkSessionHandle(core, {
        config: {
          providerId: selected.provider,
          modelId: selected.model,
          apiKey,
          ...(selected.base_url === undefined || selected.base_url === null
            ? {}
            : { baseUrl: selected.base_url }),
          cwd: workspace,
          workspaceRoot: workspace,
          mode: "act",
          enableTools: input.config.tools?.enabled?.length !== 0,
          enableSpawnAgent: false,
          enableAgentTeams: false,
          systemPrompt,
          ...(files.pluginRoot === undefined ? {} : { agentPluginPaths: [files.pluginRoot] }),
        },
        // Cline removes non-interactive sessions after their initial turn. NeMo
        // Fabric owns a persistent runtime, so keep this SDK session interactive
        // and drive every turn through the lifecycle protocol instead.
        interactive: true,
        source: "sdk",
        toolPolicies: policies,
        localRuntime: { configExtensions: [] },
      }, async () => {
        await rm(files.root, { recursive: true, force: true });
      });
    } catch (error) {
      await core?.dispose("NeMo Fabric startup failed").catch(() => undefined);
      await rm(files.root, { recursive: true, force: true });
      if (error instanceof LifecycleError) {
        throw error;
      }
      throw new LifecycleError("cline_start_failed", "Cline SDK initialization failed");
    }
  }
}

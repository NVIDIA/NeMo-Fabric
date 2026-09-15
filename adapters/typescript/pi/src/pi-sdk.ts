// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pi SDK integration boundary. It translates normalized adapter configuration
// into a controlled Pi session, including model credentials, optional durable
// session state, skills, extensions, custom tools, and workspace containment.

import { lstat, mkdir, open, realpath, stat, unlink } from "node:fs/promises";
import { extname, isAbsolute, join, relative, resolve, sep } from "node:path";

import type {
  AgentSession,
  DefaultResourceLoader,
  ExtensionCommandContextActions,
  ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { createJiti } from "jiti/static";
import type { AgentConfig, AgentModelConfig, AgentToolDefinition, JsonObject } from "nemo-fabric-adapter-contract";
import { LifecycleError, type AdapterStartInput } from "nemo-fabric-adapters-common";

import type { PiPromptOutcome, PiSessionFactory, PiSessionHandle } from "./runtime.js";

interface PiHarnessSettings {
  extensions: string[];
  session?: PiSessionSettings;
}

interface PiSessionSettings {
  mode: "create" | "resume";
  id: string;
  directory: string;
}

interface PiToolFactoryContext {
  name: string;
  settings: JsonObject;
  workspace: string;
}

type PiToolFactory = (context: PiToolFactoryContext) => ToolDefinition | Promise<ToolDefinition>;

const PI_BUILTIN_TOOL_NAMES = new Set(["read", "bash", "edit", "write", "grep", "find", "ls"]);
const TOOL_MODULE_EXTENSIONS = new Set([".js", ".mjs", ".cjs", ".ts", ".mts", ".cts"]);
const PI_HARNESS_INSTALL_COMMAND =
  "npm install @earendil-works/pi-ai@^0.84.2 @earendil-works/pi-coding-agent@^0.84.2";
const DEFAULT_SESSION_DIRECTORY = ".fabric-pi/sessions";
const SESSION_ID_PATTERN = /^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$/;

interface PiSdkModules {
  InMemoryCredentialStore: typeof import("@earendil-works/pi-ai").InMemoryCredentialStore;
  createAgentSession: typeof import("@earendil-works/pi-coding-agent").createAgentSession;
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
  try {
    [ai, codingAgent] = await Promise.all([
      import("@earendil-works/pi-ai"),
      import("@earendil-works/pi-coding-agent"),
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
    typeof codingAgent.createAgentSession !== "function" ||
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
    createAgentSession: codingAgent.createAgentSession,
    DefaultResourceLoader: codingAgent.DefaultResourceLoader,
    ModelRuntime: codingAgent.ModelRuntime,
    SessionManager: codingAgent.SessionManager,
    SettingsManager: codingAgent.SettingsManager,
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

function harnessSettings(config: AgentConfig): PiHarnessSettings {
  const raw = config.harness?.settings;
  const extensions = raw?.extensions;
  if (extensions !== undefined && !Array.isArray(extensions)) {
    throw new LifecycleError("pi_invalid_settings", "Pi extension settings do not match the adapter schema");
  }
  const values: string[] = [];
  for (const entry of extensions ?? []) {
    if (typeof entry !== "string") {
      throw new LifecycleError("pi_invalid_settings", "Pi extension settings do not match the adapter schema");
    }
    values.push(entry);
  }
  const session = raw?.session;
  if (session === undefined) {
    return { extensions: values };
  }
  if (
    !isRecord(session) ||
    (session.mode !== "create" && session.mode !== "resume") ||
    typeof session.id !== "string" ||
    session.id.length > 128 ||
    !SESSION_ID_PATTERN.test(session.id) ||
    (session.directory !== undefined &&
      (typeof session.directory !== "string" || session.directory.length === 0))
  ) {
    throw new LifecycleError("pi_invalid_settings", "Pi session settings do not match the adapter schema");
  }
  return {
    extensions: values,
    session: {
      mode: session.mode,
      id: session.id,
      directory: session.directory ?? DEFAULT_SESSION_DIRECTORY,
    },
  };
}

function containedBy(root: string, candidate: string): boolean {
  const path = relative(root, candidate);
  return path === "" || (!path.startsWith(`..${sep}`) && path !== ".." && !isAbsolute(path));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorCode(error: unknown): string | undefined {
  if (isRecord(error) && typeof error.code === "string") {
    return error.code;
  }
  return undefined;
}

async function createSessionManager(
  pi: PiSdkModules,
  workspace: string,
  settings: PiSessionSettings | undefined,
): Promise<{
  sessionManager: ReturnType<PiSdkModules["SessionManager"]["inMemory"]>;
  cleanupOnFailure: () => Promise<void>;
}> {
  if (settings === undefined) {
    return {
      sessionManager: pi.SessionManager.inMemory(workspace),
      cleanupOnFailure: async () => undefined,
    };
  }
  if (isAbsolute(settings.directory)) {
    throw new LifecycleError(
      "pi_session_directory_outside_workspace",
      "The Pi session directory must be workspace-relative",
    );
  }
  const configuredDirectory = resolve(workspace, settings.directory);
  if (!containedBy(workspace, configuredDirectory)) {
    throw new LifecycleError(
      "pi_session_directory_outside_workspace",
      "The Pi session directory must be contained by the workspace",
    );
  }
  let sessionDirectory = workspace;
  for (const component of relative(workspace, configuredDirectory).split(sep).filter(Boolean)) {
    const candidate = join(sessionDirectory, component);
    try {
      await mkdir(candidate);
    } catch (error) {
      if (errorCode(error) !== "EEXIST") {
        throw new LifecycleError("pi_session_storage_failed", "The Pi session directory could not be prepared");
      }
    }
    try {
      sessionDirectory = await realpath(candidate);
    } catch {
      throw new LifecycleError("pi_session_storage_failed", "The Pi session directory could not be prepared");
    }
    if (!containedBy(workspace, sessionDirectory)) {
      throw new LifecycleError(
        "pi_session_directory_outside_workspace",
        "The Pi session directory resolves outside the workspace",
      );
    }
  }
  try {
    if (!(await stat(sessionDirectory)).isDirectory()) {
      throw new Error("not a directory");
    }
  } catch {
    throw new LifecycleError("pi_session_storage_failed", "The Pi session directory is not a directory");
  }

  const sessionFile = join(sessionDirectory, `${settings.id}.jsonl`);
  let cleanupOnFailure = async (): Promise<void> => undefined;
  if (settings.mode === "create") {
    try {
      const placeholder = await open(sessionFile, "wx");
      try {
        const reserved = await placeholder.stat();
        cleanupOnFailure = async () => {
          try {
            const current = await lstat(sessionFile);
            if (current.dev === reserved.dev && current.ino === reserved.ino) {
              await unlink(sessionFile);
            }
          } catch (error) {
            if (errorCode(error) !== "ENOENT") {
              throw error;
            }
          }
        };
      } finally {
        await placeholder.close();
      }
    } catch (error) {
      if (errorCode(error) === "EEXIST") {
        throw new LifecycleError("pi_session_exists", "A Pi session with the configured id already exists");
      }
      try {
        await cleanupOnFailure();
      } catch {
        // The original storage failure is more useful than cleanup diagnostics.
      }
      throw new LifecycleError("pi_session_storage_failed", "The Pi session file could not be created");
    }
  } else {
    try {
      const info = await lstat(sessionFile);
      if (info.isSymbolicLink()) {
        throw new LifecycleError(
          "pi_session_directory_outside_workspace",
          "The Pi session file must not be a symbolic link",
        );
      }
      if (!info.isFile() || info.size === 0) {
        throw new LifecycleError("pi_session_invalid", "The configured Pi session file is invalid");
      }
    } catch (error) {
      if (error instanceof LifecycleError) {
        throw error;
      }
      if (errorCode(error) === "ENOENT") {
        throw new LifecycleError("pi_session_not_found", "The configured Pi session does not exist");
      }
      throw new LifecycleError("pi_session_storage_failed", "The Pi session file could not be read");
    }
  }

  try {
    return {
      sessionManager: pi.SessionManager.open(sessionFile),
      cleanupOnFailure,
    };
  } catch {
    if (settings.mode === "create") {
      try {
        await cleanupOnFailure();
      } catch {
        // The original storage failure is more useful than cleanup diagnostics.
      }
      throw new LifecycleError("pi_session_storage_failed", "The Pi session file could not be initialized");
    }
    throw new LifecycleError("pi_session_invalid", "The configured Pi session file is invalid");
  }
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

function extensionToolNames(resourceLoader: DefaultResourceLoader): string[] {
  return resourceLoader
    .getExtensions()
    .extensions.flatMap((extension) => Array.from(extension.tools.keys()));
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
  private readonly session: AgentSession;
  private readonly state: { shutdownRequested: boolean };
  private stopped = false;

  constructor(session: AgentSession, state: { shutdownRequested: boolean }) {
    this.session = session;
    this.state = state;
  }

  async prompt(text: string): Promise<PiPromptOutcome> {
    let accepted = false;
    let finalAssistant:
      | { role: "assistant"; content: unknown; stopReason: string; errorMessage?: string }
      | undefined;
    const unsubscribe = this.session.subscribe((event) => {
      if (event.type === "message_end" && event.message.role === "assistant") {
        finalAssistant = event.message;
      }
    });
    try {
      await this.session.prompt(text, {
        expandPromptTemplates: true,
        source: "interactive",
        preflightResult: (result) => {
          accepted = result;
        },
      });
    } finally {
      unsubscribe();
    }
    return {
      accepted,
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
      this.session.dispose();
    } catch (error) {
      failure ??= error;
    }
    if (failure !== undefined) {
      throw failure;
    }
  }
}

export class PiSdkSessionFactory implements PiSessionFactory {
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
    const settings = harnessSettings(input.config);
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

    const piSettings = pi.SettingsManager.inMemory({}, { projectTrusted: false });
    const extensionPaths = await resolveExtensionPaths(workspace, settings.extensions);
    const skillPaths = await resolveSkillPaths(input.baseDir, input.config.skills?.paths ?? []);
    const customTools = await resolveCustomTools(workspace, input.config.tools?.definitions ?? {});
    const agentDir = join(workspace, ".fabric-pi");
    const resourceLoader = new pi.DefaultResourceLoader({
      cwd: workspace,
      agentDir,
      settingsManager: piSettings,
      additionalExtensionPaths: extensionPaths,
      additionalSkillPaths: skillPaths,
      noExtensions: true,
      noSkills: true,
      noPromptTemplates: true,
      noThemes: true,
      noContextFiles: true,
      systemPrompt: systemInstruction?.content,
    });
    await resourceLoader.reload();
    const extensionErrors = resourceLoader.getExtensions().errors;
    if (extensionErrors.length > 0) {
      throw new LifecycleError("pi_extension_load_failed", "One or more configured Pi extensions failed to load", {
        metadata: { count: extensionErrors.length },
      });
    }
    rejectToolCollisions(customTools, extensionToolNames(resourceLoader));
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

    const credentials = new pi.InMemoryCredentialStore();
    const modelRuntime = await pi.ModelRuntime.create({
      credentials,
      modelsPath: null,
      allowModelNetwork: false,
      refreshOnCreate: false,
    });
    await modelRuntime.setRuntimeApiKey(selected.provider, apiKey);
    const catalogModel = modelRuntime.getModel(selected.provider, selected.model);
    if (catalogModel === undefined) {
      throw new LifecycleError("pi_model_unknown", "The selected provider and model are not present in Pi's catalog");
    }
    const model = selected.base_url ? { ...catalogModel, baseUrl: selected.base_url } : catalogModel;
    const enabled = input.config.tools?.enabled;
    const blocked = input.config.tools?.blocked ?? [];
    const state = { shutdownRequested: false };
    const preparedSession = await createSessionManager(pi, workspace, settings.session);
    let handle: PiSdkSessionHandle | undefined;
    try {
      const { session } = await pi.createAgentSession({
        cwd: workspace,
        agentDir,
        model,
        modelRuntime,
        resourceLoader,
        sessionManager: preparedSession.sessionManager,
        settingsManager: piSettings,
        customTools,
        tools: enabled === null ? undefined : enabled,
        excludeTools: blocked,
      });
      handle = new PiSdkSessionHandle(session, state);
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
      return handle;
    } catch (error) {
      if (handle !== undefined) {
        try {
          await handle.stop();
        } catch {
          process.stderr.write("Pi session cleanup failed after adapter startup error\n");
        }
      }
      try {
        await preparedSession.cleanupOnFailure();
      } catch {
        process.stderr.write("Pi session file cleanup failed after adapter startup error\n");
      }
      throw error;
    }
  }
}

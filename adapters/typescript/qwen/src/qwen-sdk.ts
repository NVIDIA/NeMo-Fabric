// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The SDK owns one Qwen CLI child process. Keep its input stream open across
// Fabric invocations, and consume exactly one terminal SDK result per prompt.

import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";

import type { AdapterCheckResult, CheckLevel, AgentUsage } from "nemo-fabric-adapter-contract";
import { LifecycleError, type AdapterStartInput } from "nemo-fabric-adapters-common";
import type { Query, SDKResultMessage, SDKUserMessage } from "@qwen-code/sdk";

import { rejectUnclaimedConfig, selectBlockedTools, selectMcpServers, selectModel, selectPermissionMode, selectSkillPaths, selectSystemPrompt } from "./configuration.js";
import type { QwenSession, QwenSessionFactory, QwenTurn } from "./runtime.js";

const STREAM_CLOSE_TIMEOUT_MS = 10;
const MCP_FAILURE_MARKER = "Warning: MCP server(s) failed to start:";

type UsageSnapshot = { input: number | null; output: number | null; total: number | null; cached: number | null };

function tokenCount(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function tokenDelta(current: number | null, previous: number | null): number | null {
  return current !== null && previous !== null && current >= previous ? current - previous : null;
}

class QwenDiagnostics {
  private markerPrefix = "";
  private mcpUnavailable = false;

  consume(message: string): void {
    const combined = this.markerPrefix + message;
    if (combined.includes(MCP_FAILURE_MARKER)) this.mcpUnavailable = true;
    this.markerPrefix = "";
    for (let length = Math.min(MCP_FAILURE_MARKER.length - 1, combined.length); length > 0; length--) {
      const suffix = combined.slice(-length);
      if (MCP_FAILURE_MARKER.startsWith(suffix)) {
        this.markerPrefix = suffix;
        break;
      }
    }
  }

  hasMcpFailure(): boolean {
    return this.mcpUnavailable;
  }
}

class PromptQueue implements AsyncIterable<SDKUserMessage> {
  private readonly pending: SDKUserMessage[] = [];
  private readonly consumed: Promise<void>;
  private resolveConsumed!: () => void;
  private wake?: () => void;
  private ended = false;

  constructor() {
    this.consumed = new Promise<void>((resolveConsumed) => { this.resolveConsumed = resolveConsumed; });
  }

  push(message: SDKUserMessage): void {
    if (this.ended) throw new LifecycleError("qwen_session_closed", "Qwen session input is closed");
    this.pending.push(message);
    this.wake?.();
    this.wake = undefined;
  }

  close(): void {
    this.ended = true;
    this.wake?.();
    this.wake = undefined;
  }

  waitUntilConsumed(): Promise<void> {
    return this.consumed;
  }

  async *[Symbol.asyncIterator](): AsyncIterator<SDKUserMessage> {
    try {
      while (true) {
        const next = this.pending.shift();
        if (next !== undefined) {
          yield next;
        } else if (this.ended) {
          return;
        } else {
          await new Promise<void>((resolveWake) => { this.wake = resolveWake; });
        }
      }
    } finally {
      this.resolveConsumed();
    }
  }
}

export class QwenSdkSession implements QwenSession {
  private closed = false;
  private healthObservation?: Promise<boolean>;
  private previousUsage: UsageSnapshot = { input: 0, output: 0, total: 0, cached: 0 };

  constructor(
    private readonly query: Query,
    private readonly prompts: PromptQueue,
    private readonly directory: string,
    private readonly diagnostics: QwenDiagnostics,
  ) {}

  private turnUsage(result: SDKResultMessage): AgentUsage {
    // SDK result usage is cumulative for the live query. Fabric usage belongs
    // to one invocation, so report the delta since the previous result.
    const current: UsageSnapshot = {
      input: tokenCount(result.usage?.input_tokens),
      output: tokenCount(result.usage?.output_tokens),
      total: tokenCount(result.usage?.total_tokens),
      cached: tokenCount(result.usage?.cache_read_input_tokens),
    };
    const previous = this.previousUsage;
    // The CLI synthesizes zero metrics when the provider omits usage. Its
    // public result cannot distinguish that case from real zero consumption.
    if ((current.input ?? 0) === 0 && (current.output ?? 0) === 0 && (current.total ?? 0) === 0) {
      current.input = current.output = current.total = current.cached = null;
    }
    this.previousUsage = current;
    const reset = (["input", "output", "total"] as const).some(key => {
      const now = current[key];
      const before = previous[key];
      return now !== null && before !== null && now < before;
    });
    const input = reset ? null : tokenDelta(current.input, previous.input);
    const output = reset ? null : tokenDelta(current.output, previous.output);
    const total = reset || input === null || output === null ? null : tokenDelta(current.total, previous.total);
    // Missing usage after an earlier turn also leaves cumulative metrics
    // unchanged. Do not misreport that invocation as free model work.
    const unchanged = input === 0 && output === 0 && (total ?? 0) === 0;
    return {
      input_tokens: unchanged ? null : input,
      output_tokens: unchanged ? null : output,
      total_tokens: unchanged ? null : total,
      cached_input_tokens: reset || unchanged ? null : tokenDelta(current.cached, previous.cached),
      // This adapter fixes authType to OpenAI. The pinned CLI's OpenAI
      // converters retain prompt_tokens inclusive of cache and completion_tokens
      // inclusive of reasoning; adding either breakdown would count it twice.
      input_tokens_include_cache: true,
    };
  }

  async check(level: CheckLevel): Promise<AdapterCheckResult> {
    const source = level === "live" ? "fabric.adapter.liveness" : level === "active" ? "fabric.adapter.interface" : "fabric.adapter.readiness";
    if (this.closed || this.query.isClosed()) return { outcome: "failed", reason_code: "session_closed", source, observed_at: new Date().toISOString(), activity: "stopped" };
    if (this.healthObservation !== undefined) return { outcome: "unknown", reason_code: "probe_timeout", source, observed_at: null, activity: "unknown" };
    const sessionId = this.query.getSessionId();
    try {
      const observation: Promise<boolean> = this.query.supportedCommands().then(value => value !== null && Array.isArray(value.commands) && value.commands.every(command => typeof command === "string"));
      this.healthObservation = observation;
      void observation.finally(() => { if (this.healthObservation === observation) this.healthObservation = undefined; }).catch(() => { });
      const valid = await observation;
      if (this.closed || this.query.isClosed() || this.query.getSessionId() !== sessionId) return { outcome: "unknown", reason_code: "observation_changed", source, observed_at: null, activity: "unknown" };
      return valid
        ? { outcome: "passed", reason_code: null, source, observed_at: new Date().toISOString(), activity: "idle" }
        : { outcome: "unknown", reason_code: "probe_error", source, observed_at: new Date().toISOString(), activity: "unknown" };
    } catch {
      return { outcome: "unknown", reason_code: this.closed || this.query.isClosed() || this.query.getSessionId() !== sessionId ? "observation_changed" : "probe_error", source, observed_at: null, activity: "unknown" };
    }
  }

  async prompt(input: string): Promise<QwenTurn> {
    if (this.closed || this.query.isClosed()) throw new LifecycleError("qwen_session_closed", "Qwen session is closed");
    this.prompts.push({
      type: "user",
      session_id: this.query.getSessionId(),
      message: { role: "user", content: input },
      parent_tool_use_id: null,
    });
    while (true) {
      const next = await this.query.next();
      if (next.done) throw new LifecycleError("qwen_session_ended", "Qwen session ended before a result");
      if (next.value.type !== "result") continue;
      const result = next.value;
      const turnUsage = this.turnUsage(result);
      if (this.diagnostics.hasMcpFailure()) {
        return { error: "error_mcp_unavailable", usage: turnUsage };
      }
      return result.subtype === "success"
        ? { text: result.result, usage: turnUsage }
        : { error: result.subtype, usage: turnUsage };
    }
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.prompts.close();
    // Let the SDK's streamInput task observe EOF and end CLI stdin before
    // Query.close marks the query closed; otherwise normal stop logs an error.
    await this.prompts.waitUntilConsumed();
    await new Promise<void>((resolveClose) => setTimeout(resolveClose, STREAM_CLOSE_TIMEOUT_MS * 2));
    try {
      await this.query.close();
    } finally {
      await rm(this.directory, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
    }
  }
}

function settings(model: ReturnType<typeof selectModel>, blockedTools: string[], skillPaths: string[], mcpServerNames: string[]): object {
  const samplingParams = {
    ...(model.temperature === undefined ? {} : { temperature: model.temperature }),
    ...(model.topP === undefined ? {} : { top_p: model.topP }),
  };
  return {
    general: { enableAutoUpdate: false },
    privacy: { usageStatisticsEnabled: false },
    security: { auth: { selectedType: "openai", enforcedType: "openai" } },
    model: { name: model.id },
    modelProviders: {
      openai: [{
        id: model.id,
        envKey: model.apiKeyEnv,
        ...(model.baseUrl === undefined ? {} : { baseUrl: model.baseUrl }),
        ...(Object.keys(samplingParams).length === 0 ? {} : { generationConfig: { samplingParams } }),
      }],
    },
    // Explicit skills are loaded from isolated QWEN_HOME user scope; ambient
    // project, extension, and bundled skills remain hidden.
    skills: {
      disabledLevels: ["project", "extension", "bundled", ...(skillPaths.length ? [] : ["user"])],
      ...(skillPaths.length ? { directories: skillPaths } : {}),
    },
    mcp: { allowed: mcpServerNames },
    permissions: { deny: blockedTools },
  };
}

function workingDirectory(input: AdapterStartInput): string {
  const path = input.runtimeContext.environment.workspace ?? input.baseDir;
  if (!isAbsolute(path)) throw new LifecycleError("qwen_invalid_workspace", "Qwen requires an absolute workspace path");
  return resolve(path);
}

export class QwenSdkSessionFactory implements QwenSessionFactory {
  async create(input: AdapterStartInput): Promise<QwenSession> {
    rejectUnclaimedConfig(input.config);
    const model = selectModel(input.config);
    const systemPrompt = selectSystemPrompt(input.config);
    const permissionMode = selectPermissionMode(input.config);
    const blockedTools = selectBlockedTools(input.config);
    const skillPaths = await selectSkillPaths(input.config, input.baseDir);
    const configuredEnvironment = input.runtimeContext.environment.env ?? {};
    const mcpServers = selectMcpServers(input.config, configuredEnvironment, process.env);
    const mcpServerNames = Object.keys(mcpServers);
    const credential = input.runtimeContext.environment.env?.[model.apiKeyEnv] ?? process.env[model.apiKeyEnv];
    if (credential === undefined || credential.length === 0) {
      throw new LifecycleError("qwen_credential_missing", "The configured Qwen model credential variable is not set");
    }
    const directory = await mkdtemp(join(tmpdir(), "nemo-fabric-qwen-"));
    let queryHandle: Query | undefined;
    try {
      const home = join(directory, "home");
      const runtime = join(directory, "runtime");
      const systemSettings = join(directory, "system-settings.json");
      await mkdir(home);
      await mkdir(runtime);
      await writeFile(systemSettings, `${JSON.stringify(settings(model, blockedTools, skillPaths, mcpServerNames))}\n`, { mode: 0o600 });
      const prompts = new PromptQueue();
      const diagnostics = new QwenDiagnostics();
      let sdk: typeof import("@qwen-code/sdk");
      try {
        sdk = await import("@qwen-code/sdk");
      } catch {
        throw new LifecycleError("qwen_sdk_missing", "Install the supported @qwen-code/sdk package to run Qwen");
      }
      queryHandle = sdk.query({
        prompt: prompts,
        options: {
          cwd: workingDirectory(input),
          model: model.id,
          authType: "openai",
          permissionMode,
          // Qwen currently exposes external MCP discovery failures only on
          // diagnostic stderr. Capture, classify, and discard that stream so
          // configured server failures become a safe normalized result.
          stderr: (message) => diagnostics.consume(message),
          logLevel: "debug",
          excludeTools: blockedTools,
          extensions: ["none"],
          ...(systemPrompt === undefined ? {} : { systemPrompt }),
          env: {
            ...configuredEnvironment,
            [model.apiKeyEnv]: credential,
            QWEN_HOME: home,
            QWEN_RUNTIME_DIR: runtime,
            QWEN_CODE_SYSTEM_SETTINGS_PATH: systemSettings,
            QWEN_USAGE_STATISTICS_ENABLED: "0",
            QWEN_TELEMETRY_ENABLED: "0",
            // The CLI otherwise relaunches with a heap sized from host RAM,
            // overriding Node's container-aware limit and caller NODE_OPTIONS.
            QWEN_CODE_NO_RELAUNCH: "true",
          },
          mcpServers,
          // Only explicitly normalized servers may load; this excludes ambient
          // project MCP configuration without weakening Qwen's trust policy.
          allowedMcpServerNames: mcpServerNames.length === 0
            ? [`nemo-fabric-no-mcp-${randomUUID()}`]
            : mcpServerNames,
          // The adapter closes the input stream only while stopping. Avoid the
          // SDK's default 60-second wait for a first result when no prompt was
          // ever sent (for example, a start/stop health check).
          timeout: { streamClose: STREAM_CLOSE_TIMEOUT_MS },
        },
      });
      await queryHandle.initialized;
      return new QwenSdkSession(queryHandle, prompts, directory, diagnostics);
    } catch (error) {
      try { await queryHandle?.close(); } catch { /* Preserve startup failure. */ }
      await rm(directory, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
      if (error instanceof LifecycleError) throw error;
      throw new LifecycleError("qwen_start_failed", "Qwen session could not start");
    }
  }
}

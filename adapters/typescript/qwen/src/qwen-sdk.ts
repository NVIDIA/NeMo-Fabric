// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The SDK owns one Qwen CLI child process. Keep its input stream open across
// Fabric invocations, and consume exactly one terminal SDK result per prompt.

import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";

import type { AgentUsage } from "nemo-fabric-adapter-contract";
import { LifecycleError, type AdapterStartInput } from "nemo-fabric-adapters-common";
import type { Query, SDKResultMessage, SDKUserMessage } from "@qwen-code/sdk";

import { rejectUnclaimedConfig, selectBlockedTools, selectMcpServers, selectModel, selectPermissionMode, selectSkillPaths, selectSystemPrompt } from "./configuration.js";
import type { QwenSession, QwenSessionFactory, QwenTurn } from "./runtime.js";

const STREAM_CLOSE_TIMEOUT_MS = 10;
const MCP_FAILURE_MARKER = "Warning: MCP server(s) failed to start:";

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

class QwenSdkSession implements QwenSession {
  private closed = false;
  private previousUsage = { input: 0, output: 0, total: 0 };

  constructor(
    private readonly query: Query,
    private readonly prompts: PromptQueue,
    private readonly directory: string,
    private readonly diagnostics: QwenDiagnostics,
  ) {}

  private turnUsage(result: SDKResultMessage): AgentUsage {
    // SDK result usage is cumulative for the live query. Fabric usage belongs
    // to one invocation, so report the delta since the previous result.
    const current = {
      input: result.usage.input_tokens,
      output: result.usage.output_tokens,
      total: result.usage.total_tokens ?? result.usage.input_tokens + result.usage.output_tokens,
    };
    const previous = this.previousUsage;
    this.previousUsage = current;
    return {
      input_tokens: Math.max(0, current.input - previous.input),
      output_tokens: Math.max(0, current.output - previous.output),
      total_tokens: Math.max(0, current.total - previous.total),
    };
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

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import type { AdapterCheckResult, CheckLevel } from "nemo-fabric-adapter-contract";
import {
  AcpRunnerError,
  LifecycleError,
  startAcpSession,
  unobservedHealth,
  type AcpPromptResult,
  type AcpSession,
  type AdapterStartInput,
} from "nemo-fabric-adapters-common";

import {
  QUIET_ENVIRONMENT,
  configToml,
  runtimeDirectory,
  selectModel,
  selectSettings,
  workspacePath,
  type DeepAgentsCodeModel,
} from "./configuration.js";

export interface DeepAgentsCodePromptOutcome {
  text: string;
  stopReason: AcpPromptResult["stopReason"];
}

export interface DeepAgentsCodeSessionHandle {
  readonly id: string;
  check(level: CheckLevel): Promise<AdapterCheckResult>;
  prompt(text: string, timeoutMs?: number): Promise<DeepAgentsCodePromptOutcome>;
  stop(): Promise<void>;
}

export interface DeepAgentsCodeSessionFactory {
  create(input: AdapterStartInput): Promise<DeepAgentsCodeSessionHandle>;
}

// Fabric gives the adapter host 90 s to start; leave time for cleanup.
const STARTUP_TIMEOUT_MS = 60_000;

const INHERITED_ENVIRONMENT_NAMES = new Set([
  "HOME", "HTTP_PROXY", "HTTPS_PROXY", "LANG", "LC_ALL", "LC_CTYPE", "NO_COLOR", "NO_PROXY", "PATH", "SHELL",
  "SSL_CERT_DIR", "SSL_CERT_FILE", "TEMP", "TERM", "TMP", "TMPDIR", "USER", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
  "XDG_DATA_HOME", "XDG_RUNTIME_DIR", "http_proxy", "https_proxy", "no_proxy",
]);

const HEALTH_SOURCES: Partial<Record<CheckLevel, AdapterCheckResult["source"]>> = {
  live: "fabric.adapter.liveness",
  active: "fabric.adapter.interface",
  ready: "fabric.adapter.readiness",
};

function childEnvironment(input: AdapterStartInput, home: string, model: DeepAgentsCodeModel): NodeJS.ProcessEnv {
  const environment: NodeJS.ProcessEnv = {};
  for (const name of [...INHERITED_ENVIRONMENT_NAMES, model.apiKeyEnv]) {
    const value = process.env[name];
    if (value !== undefined) environment[name] = value;
  }
  Object.assign(environment, input.runtimeContext.environment.env ?? {}, QUIET_ENVIRONMENT);
  environment.DEEPAGENTS_HOME = home;
  return environment;
}

function translateRunnerError(error: AcpRunnerError, stage: "start" | "prompt"): LifecycleError {
  if (error.code === "acp_process_unavailable") {
    return new LifecycleError("deepagents_code_harness_unavailable", "Deep Agents Code could not be started");
  }
  if (error.code === "acp_protocol_unsupported") {
    return new LifecycleError(
      "deepagents_code_acp_protocol_unsupported",
      "The installed Deep Agents Code selected an unsupported ACP version",
    );
  }
  if (error.code === "acp_prompt_timeout") {
    return new LifecycleError("deepagents_code_prompt_timeout", "Deep Agents Code did not finish the invocation before the deadline");
  }
  return new LifecycleError(
    stage === "start" ? "deepagents_code_start_failed" : "deepagents_code_session_failed",
    stage === "start" ? "Deep Agents Code ACP initialization failed" : "Deep Agents Code ACP session communication failed",
    { retryable: stage === "start" && error.retryable },
  );
}

class DeepAgentsCodeAcpSession implements DeepAgentsCodeSessionHandle {
  private stopped = false;
  readonly id: string;
  private readonly session: AcpSession;

  constructor(session: AcpSession) {
    this.session = session;
    this.id = session.sessionId;
  }

  async check(level: CheckLevel): Promise<AdapterCheckResult> {
    if (this.stopped) return { ...unobservedHealth("session_closed"), outcome: "failed", activity: "stopped" };
    const source = HEALTH_SOURCES[level];
    if (source === undefined) return { ...unobservedHealth("unsupported_level"), outcome: "unsupported" };
    // The ACP process runs the agent in-process and keeps one open session;
    // every level observes that the process still serves it.
    const { alive } = this.session.observeProcess();
    const observed_at = new Date().toISOString();
    return alive
      ? { outcome: "passed", reason_code: null, activity: "unknown", source, observed_at }
      : { outcome: "failed", reason_code: "agent_exited", activity: "stopped", source, observed_at };
  }

  async prompt(text: string, timeoutMs?: number): Promise<DeepAgentsCodePromptOutcome> {
    try {
      const result = await this.session.prompt(text, timeoutMs);
      return { text: result.text, stopReason: result.stopReason };
    } catch (error) {
      if (error instanceof AcpRunnerError) throw translateRunnerError(error, "prompt");
      throw new LifecycleError("deepagents_code_session_failed", "Deep Agents Code ACP session communication failed");
    }
  }

  async stop(): Promise<void> {
    this.stopped = true;
    await this.session.stop();
  }
}

export class DeepAgentsCodeAcpSessionFactory implements DeepAgentsCodeSessionFactory {
  async create(input: AdapterStartInput): Promise<DeepAgentsCodeSessionHandle> {
    const settings = selectSettings(input.config);
    const model = selectModel(input.config);
    const workspace = await workspacePath(input);
    const directory = runtimeDirectory(input);
    const home = join(directory, "deepagents");
    await mkdir(home, { recursive: true, mode: 0o700 });
    await writeFile(join(home, "config.toml"), configToml(model), { mode: 0o600 });
    const environment = childEnvironment(input, home, model);
    let session: AcpSession;
    try {
      session = await startAcpSession({
        command: "dcode",
        args: ["--acp", "--no-tracing", ...(settings.approvalMode === "auto" ? ["--auto-approve"] : [])],
        cwd: workspace,
        env: environment,
        // No person can answer an approval through invoke: yolo allows each
        // gated action, and auto denies what its classifier escalates.
        permissionResolver: () => (settings.approvalMode === "yolo" ? "allow_once" : "deny"),
        startupTimeoutMs: STARTUP_TIMEOUT_MS,
        clientName: "nemo-fabric-deepagents-code",
        clientVersion: "0.7.0",
      });
    } catch (error) {
      if (error instanceof AcpRunnerError) throw translateRunnerError(error, "start");
      if (error instanceof LifecycleError) throw error;
      throw new LifecycleError("deepagents_code_start_failed", "Deep Agents Code ACP initialization failed", { retryable: true });
    }
    return new DeepAgentsCodeAcpSession(session);
  }
}

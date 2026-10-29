// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AdapterCheckResult, CheckLevel, AgentRunRequest, AgentRunResult, AgentUsage, RuntimeContext } from "nemo-fabric-adapter-contract";
import { unobservedHealth, LifecycleError, type AdapterRuntime, type AdapterStartInput } from "nemo-fabric-adapters-common";

export interface QwenTurn {
  text?: string;
  error?: "error_max_turns" | "error_during_execution" | "error_mcp_unavailable";
  usage?: AgentUsage;
}

export interface QwenSession {
  check(level: CheckLevel): Promise<AdapterCheckResult>;
  prompt(input: string): Promise<QwenTurn>;
  close(): Promise<void>;
}

export interface QwenSessionFactory {
  create(input: AdapterStartInput): Promise<QwenSession>;
}

function failed(code: string, message: string, usage?: AgentUsage): AgentRunResult {
  return { status: "failed", output: null, error: { code, message, retryable: false }, ...(usage ? { usage } : {}) };
}

export class QwenAdapterRuntime implements AdapterRuntime {
  private generation = 0;
  private activeInvocations = 0;
  private session?: QwenSession;

  constructor(private readonly factory: QwenSessionFactory) {}

  async start(input: AdapterStartInput): Promise<void> {
    if (this.session !== undefined) throw new LifecycleError("qwen_already_started", "Qwen runtime is already started");
    this.session = await this.factory.create(input);
    this.generation++;
  }

  async check(level: CheckLevel): Promise<AdapterCheckResult> {
    const session = this.session;
    const generation = this.generation;
    if (session === undefined) return unobservedHealth("runtime_unavailable");
    let result: AdapterCheckResult;
    try {
      result = await session.check(level);
    } catch {
      result = unobservedHealth("probe_error");
    }
    if (this.session !== session || this.generation !== generation) return unobservedHealth("observation_changed");
    return result.outcome === "passed" ? { ...result, activity: this.activeInvocations > 0 ? "busy" : "idle" } : result;
  }

  async invoke(request: AgentRunRequest, _context: RuntimeContext): Promise<AgentRunResult> {
    const session = this.session;
    if (session === undefined) throw new LifecycleError("qwen_not_started", "Qwen runtime is not started");
    if (typeof request.input !== "string") return failed("qwen_unsupported_input", "Qwen accepts only plain-text input");
    let turn: QwenTurn;
    this.activeInvocations++;
    try {
      turn = await session.prompt(request.input);
    } catch {
      this.session = undefined;
      try { await session.close(); } catch { /* Preserve the original failure. */ }
      throw new LifecycleError("qwen_session_failed", "Qwen session communication failed", { retryable: true });
    } finally {
      this.activeInvocations--;
    }
    if (turn.error !== undefined) {
      if (turn.error === "error_mcp_unavailable") {
        return failed("qwen_mcp_unavailable", "A configured Qwen MCP server was unavailable", turn.usage);
      }
      return failed(
        turn.error === "error_max_turns" ? "qwen_turn_limit" : "qwen_execution_failed",
        "Qwen could not complete the invocation",
        turn.usage,
      );
    }
    if (turn.text === undefined || turn.text.length === 0) {
      return failed("qwen_no_assistant_response", "Qwen completed without a final assistant text response", turn.usage);
    }
    return { status: "succeeded", output: { response: turn.text }, ...(turn.usage ? { usage: turn.usage } : {}) };
  }

  async stop(): Promise<void> {
    this.generation++;
    const session = this.session;
    this.session = undefined;
    if (session !== undefined) await session.close();
  }
}

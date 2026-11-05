// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AdapterCheckResult,
  AgentRunRequest,
  AgentRunResult,
  CheckLevel,
  JsonObject,
  RuntimeContext,
} from "nemo-fabric-adapter-contract";
import { LifecycleError, unobservedHealth, type AdapterRuntime, type AdapterStartInput } from "nemo-fabric-adapters-common";

import type {
  DeepAgentsCodePromptOutcome,
  DeepAgentsCodeSessionFactory,
  DeepAgentsCodeSessionHandle,
} from "./deepagents-code-acp.js";

function failed(code: string, message: string, extensions?: JsonObject): AgentRunResult {
  return {
    status: "failed",
    output: null,
    error: { code, message, retryable: false },
    ...(extensions === undefined ? {} : { extensions }),
  };
}

export class DeepAgentsCodeAdapterRuntime implements AdapterRuntime {
  private readonly factory: DeepAgentsCodeSessionFactory;
  private invoking = false;
  private stopping = false;
  private generation = 0;
  private session?: DeepAgentsCodeSessionHandle;

  constructor(factory: DeepAgentsCodeSessionFactory) {
    this.factory = factory;
  }

  async start(input: AdapterStartInput): Promise<void> {
    if (this.session !== undefined) {
      throw new LifecycleError("deepagents_code_already_started", "Deep Agents Code adapter runtime is already started");
    }
    this.session = await this.factory.create(input);
    this.generation += 1;
  }

  private async invalidate(session: DeepAgentsCodeSessionHandle): Promise<void> {
    if (this.session !== session) return;
    try {
      await this.stop();
    } catch {
      // Preserve the invocation failure that made the session unusable.
    }
  }

  async invoke(request: AgentRunRequest, context: RuntimeContext): Promise<AgentRunResult> {
    if (this.session === undefined || this.stopping) {
      throw new LifecycleError("deepagents_code_not_started", "Deep Agents Code adapter runtime is not started");
    }
    if (typeof request.input !== "string") {
      return failed("deepagents_code_unsupported_input", "The Deep Agents Code adapter accepts only plain-text input");
    }
    const deadline = context.deadline_millis;
    if (deadline !== undefined && deadline !== null && deadline <= Date.now()) {
      return failed("deepagents_code_deadline_exceeded", "The Deep Agents Code invocation deadline has already elapsed");
    }
    const session = this.session;
    let outcome: DeepAgentsCodePromptOutcome;
    this.invoking = true;
    try {
      outcome = await session.prompt(
        request.input,
        deadline === undefined || deadline === null ? undefined : deadline - Date.now(),
      );
    } catch (error) {
      await this.invalidate(session);
      if (error instanceof LifecycleError) throw error;
      throw new LifecycleError("deepagents_code_session_failed", "Deep Agents Code ACP session communication failed");
    } finally {
      this.invoking = false;
    }
    const extensions = { session_id: session.id, stop_reason: outcome.stopReason };
    if (outcome.stopReason === "cancelled") {
      return {
        status: "cancelled",
        output: null,
        error: { code: "deepagents_code_cancelled", message: "Deep Agents Code cancelled the invocation", retryable: false },
        extensions,
      };
    }
    if (outcome.stopReason !== "end_turn") {
      return failed(
        `deepagents_code_${outcome.stopReason}`,
        `Deep Agents Code stopped the invocation with ${outcome.stopReason}`,
        extensions,
      );
    }
    if (outcome.text.length === 0) {
      return failed(
        "deepagents_code_no_assistant_response",
        "Deep Agents Code completed without a final assistant text response",
        extensions,
      );
    }
    return { status: "succeeded", output: { response: outcome.text }, extensions };
  }

  async check(level: CheckLevel): Promise<AdapterCheckResult> {
    const session = this.session;
    const generation = this.generation;
    if (session === undefined || this.stopping) return { ...unobservedHealth("session_closed"), outcome: "failed", activity: "stopped" };
    let result: AdapterCheckResult;
    try {
      result = await session.check(level);
    } catch {
      result = unobservedHealth("probe_error");
    }
    if (this.session !== session || this.generation !== generation) return unobservedHealth("observation_changed");
    return result.outcome === "passed" ? { ...result, activity: this.invoking ? "busy" : "idle" } : result;
  }

  async stop(): Promise<void> {
    this.generation += 1;
    const session = this.session;
    if (session === undefined) return;
    this.stopping = true;
    // Keep the handle until the ACP stop settles so a failed stop can be retried.
    await session.stop();
    if (this.session === session) {
      this.session = undefined;
      this.stopping = false;
    }
  }
}

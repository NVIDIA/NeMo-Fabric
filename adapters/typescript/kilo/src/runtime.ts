// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AdapterCheckResult,
  CheckLevel,
  AgentRunRequest,
  AgentRunResult,
  AgentUsage,
  RuntimeContext,
} from "nemo-fabric-adapter-contract";
import {
  unobservedHealth,
  LifecycleError,
  type AdapterRuntime,
  type AdapterStartInput,
} from "nemo-fabric-adapters-common";

/** Failed communication retains its usage snapshot while the session is retired. */
export class KiloPromptError extends LifecycleError {
  constructor(error: LifecycleError, readonly usage: AgentUsage) {
    super(error.code, error.message, { retryable: error.retryable });
  }
}

export interface KiloPromptOutcome { text?: string; error?: boolean; usage?: AgentUsage; messageId?: string }
export interface KiloSessionHandle { id: string; check(level: CheckLevel): Promise<AdapterCheckResult>; prompt(text: string, deadlineMillis?: number): Promise<KiloPromptOutcome>; stop(): Promise<void> }
export interface KiloSessionFactory { create(input: AdapterStartInput): Promise<KiloSessionHandle> }

function failed(code: string, message: string, usage?: AgentUsage): AgentRunResult {
  return {status: "failed", output: null, error: {code, message, retryable: false}, ...(usage === undefined ? {} : {usage})};
}

export class KiloAdapterRuntime implements AdapterRuntime {
  private readonly factory: KiloSessionFactory;
  private invoking = false;
  private generation = 0;
  private session?: KiloSessionHandle;

  constructor(factory: KiloSessionFactory) { this.factory = factory; }

  async start(input: AdapterStartInput): Promise<void> {
    if (this.session !== undefined) throw new LifecycleError("kilo_already_started", "Kilo Code adapter runtime is already started");
    this.session = await this.factory.create(input);
    this.generation += 1;
  }

  async invoke(request: AgentRunRequest, context: RuntimeContext): Promise<AgentRunResult> {
    if (this.session === undefined) throw new LifecycleError("kilo_not_started", "Kilo Code adapter runtime is not started");
    if (typeof request.input !== "string") return failed("kilo_unsupported_input", "The Kilo Code adapter accepts only plain-text input");
    const session = this.session;
    let outcome: KiloPromptOutcome;
    this.invoking = true;
    try {
      outcome = await session.prompt(request.input, context.deadline_millis ?? undefined);
    } catch (error) {
      this.generation += 1;
      this.session = undefined;
      try { await session.stop(); } catch { /* Preserve the prompt failure. */ }
      if (error instanceof KiloPromptError) return failed(error.code, error.message, error.usage);
      if (error instanceof LifecycleError) throw error;
      throw new LifecycleError("kilo_session_failed", "Kilo Code session communication failed", {retryable: true});
    } finally {
      this.invoking = false;
    }
    if (outcome.error === true) return failed("kilo_model_error", "Kilo Code model invocation failed", outcome.usage);
    if (outcome.text === undefined || outcome.text.length === 0) {
      return failed("kilo_no_assistant_response", "Kilo Code completed without a final assistant text response", outcome.usage);
    }
    return {status: "succeeded", output: {response: outcome.text}, ...(outcome.usage === undefined ? {} : {usage: outcome.usage})};
  }

  async check(level: CheckLevel): Promise<AdapterCheckResult> {
    const session = this.session;
    const generation = this.generation;
    if (session === undefined)
      return {
        ...unobservedHealth("session_closed"),
        outcome: "failed",
        activity: "stopped",
      };
    let result: AdapterCheckResult;
    try {
      result = await session.check(level);
    } catch {
      result = unobservedHealth("probe_error");
    }
    if (this.session !== session || this.generation !== generation)
      return unobservedHealth("observation_changed");
    return result.outcome === "passed"
      ? { ...result, activity: this.invoking ? "busy" : "idle" }
      : result;
  }

  async stop(): Promise<void> {
    this.generation += 1;
    const session = this.session;
    this.session = undefined;
    if (session !== undefined) await session.stop();
  }
}

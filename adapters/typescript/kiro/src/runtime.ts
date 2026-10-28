// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AdapterCheckResult,
  CheckLevel,
  AgentRunRequest,
  AgentRunResult,
  AgentUsage,
  JsonObject,
  RuntimeContext,
} from "nemo-fabric-adapter-contract";
import {
  unobservedHealth,
  LifecycleError,
  type AdapterRuntime,
  type AdapterStartInput,
} from "nemo-fabric-adapters-common";

import type { KiroPromptOutcome, KiroSessionFactory, KiroSessionHandle } from "./kiro-acp.js";

function resultExtensions(session: KiroSessionHandle, outcome: KiroPromptOutcome): JsonObject {
  return { session_id: session.id, stop_reason: outcome.stopReason };
}

function failed(
  code: string,
  message: string,
  extensions: JsonObject,
  usage?: AgentUsage,
  retryable = false,
): AgentRunResult {
  return {
    status: "failed",
    output: null,
    error: { code, message, retryable },
    extensions,
    ...(usage === undefined ? {} : { usage }),
  };
}

export class KiroAdapterRuntime implements AdapterRuntime {
  private readonly factory: KiroSessionFactory;
  private invoking = false;
  private generation = 0;
  private session?: KiroSessionHandle;

  constructor(factory: KiroSessionFactory) {
    this.factory = factory;
  }

  async start(input: AdapterStartInput): Promise<void> {
    if (this.session !== undefined) {
      throw new LifecycleError("kiro_already_started", "Kiro adapter runtime is already started");
    }
    this.session = await this.factory.create(input);
    this.generation += 1;
  }

  private async invalidate(session: KiroSessionHandle): Promise<void> {
    if (this.session !== session) return;
    this.generation += 1;
    this.session = undefined;
    try {
      await session.stop();
    } catch {
      // Preserve the invocation failure that made the session unusable.
    }
  }

  async invoke(request: AgentRunRequest, context: RuntimeContext): Promise<AgentRunResult> {
    if (this.session === undefined) {
      throw new LifecycleError("kiro_not_started", "Kiro adapter runtime is not started");
    }
    if (typeof request.input !== "string") {
      return {
        status: "failed",
        output: null,
        error: { code: "kiro_unsupported_input", message: "The Kiro adapter accepts only plain-text input", retryable: false },
      };
    }
    const session = this.session;
    if (
      context.deadline_millis !== undefined &&
      context.deadline_millis !== null &&
      context.deadline_millis <= Date.now()
    ) {
      return {
        status: "failed",
        output: null,
        error: {
          code: "kiro_deadline_exceeded",
          message: "The Kiro invocation deadline has already elapsed",
          retryable: false,
        },
      };
    }
    const timeoutMs =
      context.deadline_millis === undefined || context.deadline_millis === null
        ? undefined
        : context.deadline_millis - Date.now();
    let outcome: KiroPromptOutcome;
    this.invoking = true;
    try {
      outcome = await session.prompt(request.input, timeoutMs);
    } catch (error) {
      await this.invalidate(session);
      if (error instanceof LifecycleError) throw error;
      throw new LifecycleError("kiro_session_failed", "Kiro ACP session communication failed");
    } finally {
      this.invoking = false;
    }
    const extensions = resultExtensions(session, outcome);
    if (outcome.stopReason === "cancelled") {
      return {
        status: "cancelled",
        output: null,
        error: { code: "kiro_cancelled", message: "Kiro cancelled the invocation", retryable: false },
        extensions,
        ...(outcome.usage === undefined ? {} : { usage: outcome.usage }),
      };
    }
    if (outcome.stopReason !== "end_turn") {
      return failed(
        `kiro_${outcome.stopReason}`,
        `Kiro stopped the invocation with ${outcome.stopReason}`,
        extensions,
        outcome.usage,
      );
    }
    if (outcome.text.length === 0) {
      return failed(
        "kiro_no_assistant_response",
        "Kiro completed without a final assistant text response",
        extensions,
        outcome.usage,
      );
    }
    return {
      status: "succeeded",
      output: { response: outcome.text },
      extensions,
      ...(outcome.usage === undefined ? {} : { usage: outcome.usage }),
    };
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

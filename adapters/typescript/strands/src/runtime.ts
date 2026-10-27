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

import { selectSettings } from "./configuration.js";

import type {
  StrandsPromptOutcome,
  StrandsSessionFactory,
  StrandsSessionHandle,
} from "./strands-acp.js";

function resultExtensions(
  session: StrandsSessionHandle,
  outcome: StrandsPromptOutcome,
): JsonObject {
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

export class StrandsAdapterRuntime implements AdapterRuntime {
  private readonly factory: StrandsSessionFactory;
  private generation = 0;
  private activeInvocations = 0;
  private session?: StrandsSessionHandle;
  private invalidated = false;
  private invocations = 0;
  private maxInvocations = 100;

  constructor(factory: StrandsSessionFactory) {
    this.factory = factory;
  }

  async start(input: AdapterStartInput): Promise<void> {
    if (this.session !== undefined) {
      throw new LifecycleError(
        "strands_already_started",
        "Strands adapter runtime is already started",
      );
    }
    this.maxInvocations = selectSettings(input.config).maxInvocations;
    this.session = await this.factory.create(input);
    this.generation++;
    this.invalidated = false;
    this.invocations = 0;
  }

  private async invalidate(session: StrandsSessionHandle): Promise<void> {
    if (this.session !== session) return;
    try {
      await this.stop();
    } catch {
      // Preserve the invocation failure that made the session unusable.
    }
  }

  async check(level: CheckLevel): Promise<AdapterCheckResult> {
    const session = this.session;
    const generation = this.generation;
    if (session === undefined) return unobservedHealth("runtime_unavailable");
    if (this.invalidated) return unobservedHealth("observation_changed");
    let result: AdapterCheckResult;
    try {
      result = await session.check(level);
    } catch {
      result = unobservedHealth("probe_error");
    }
    if (this.session !== session || this.generation !== generation) return unobservedHealth("observation_changed");
    return result.outcome === "passed" ? { ...result, activity: this.activeInvocations > 0 ? "busy" : "idle" } : result;
  }

  async invoke(
    request: AgentRunRequest,
    context: RuntimeContext,
  ): Promise<AgentRunResult> {
    if (this.session === undefined || this.invalidated) {
      throw new LifecycleError(
        "strands_not_started",
        "Strands adapter runtime is not started",
      );
    }
    if (typeof request.input !== "string") {
      return {
        status: "failed",
        output: null,
        error: {
          code: "strands_unsupported_input",
          message: "The Strands adapter accepts only plain-text input",
          retryable: false,
        },
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
          code: "strands_deadline_exceeded",
          message: "The Strands invocation deadline has already elapsed",
          retryable: false,
        },
      };
    }
    const timeoutMs =
      context.deadline_millis === undefined || context.deadline_millis === null
        ? undefined
        : context.deadline_millis - Date.now();
    if (this.invocations >= this.maxInvocations) {
      return failed(
        "strands_session_limit",
        "Start a new runtime after the Strands invocation limit",
        {},
      );
    }
    this.invocations++;
    let outcome: StrandsPromptOutcome;
    this.activeInvocations++;
    try {
      outcome = await session.prompt(request.input, timeoutMs);
    } catch (error) {
      await this.invalidate(session);
      if (error instanceof LifecycleError) throw error;
      throw new LifecycleError(
        "strands_session_failed",
        "Strands ACP session communication failed",
      );
    } finally {
      this.activeInvocations--;
    }
    const extensions = resultExtensions(session, outcome);
    if (outcome.stopReason === "cancelled") {
      return {
        status: "cancelled",
        output: null,
        error: {
          code: "strands_cancelled",
          message: "Strands cancelled the invocation",
          retryable: false,
        },
        extensions,
        ...(outcome.usage === undefined ? {} : { usage: outcome.usage }),
      };
    }
    if (outcome.stopReason !== "end_turn") {
      return failed(
        `strands_${outcome.stopReason}`,
        `Strands stopped the invocation with ${outcome.stopReason}`,
        extensions,
        outcome.usage,
      );
    }
    if (outcome.text.length === 0) {
      return failed(
        "strands_no_assistant_response",
        "Strands completed without a final assistant text response",
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

  async stop(): Promise<void> {
    this.generation++;
    const session = this.session;
    if (session !== undefined) {
      this.invalidated = true;
      await session.stop();
      this.session = undefined;
    }
  }
}

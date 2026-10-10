// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentRunRequest, AgentRunResult, AgentUsage, JsonObject, RuntimeContext } from "nemo-fabric-adapter-contract";
import { LifecycleError, type AdapterRuntime, type AdapterStartInput } from "nemo-fabric-adapters-common";

export interface DroidPromptOutcome {
  status: "completed" | "interrupted" | "failed";
  text?: string;
  errorMessage?: string;
  usage?: AgentUsage;
  extensions?: JsonObject;
}

export interface DroidSessionHandle {
  prompt(text: string): Promise<DroidPromptOutcome>;
  stop(): Promise<void>;
}

export interface DroidSessionFactory {
  create(input: AdapterStartInput): Promise<DroidSessionHandle>;
}

function failed(code: string, message: string, usage?: AgentUsage, extensions?: JsonObject): AgentRunResult {
  return {
    status: "failed",
    output: null,
    error: { code, message, retryable: false },
    ...(usage === undefined ? {} : { usage }),
    ...(extensions === undefined ? {} : { extensions }),
  };
}

export class DroidAdapterRuntime implements AdapterRuntime {
  private readonly factory: DroidSessionFactory;
  private session?: DroidSessionHandle;

  constructor(factory: DroidSessionFactory) {
    this.factory = factory;
  }

  async start(input: AdapterStartInput): Promise<void> {
    if (this.session !== undefined) {
      throw new LifecycleError("droid_already_started", "Droid adapter runtime is already started");
    }
    this.session = await this.factory.create(input);
  }

  private async invalidate(session: DroidSessionHandle): Promise<void> {
    if (this.session !== session) return;
    this.session = undefined;
    try {
      await session.stop();
    } catch {
      // Preserve the invocation failure after invalidating the unusable session.
    }
  }

  async invoke(request: AgentRunRequest, _context: RuntimeContext): Promise<AgentRunResult> {
    if (this.session === undefined) {
      throw new LifecycleError("droid_not_started", "Droid adapter runtime is not started");
    }
    if (typeof request.input !== "string") {
      return failed("droid_unsupported_input", "The Droid adapter accepts only plain-text input");
    }

    const session = this.session;
    let outcome: DroidPromptOutcome;
    try {
      outcome = await session.prompt(request.input);
    } catch (error) {
      await this.invalidate(session);
      if (error instanceof LifecycleError) throw error;
      throw new LifecycleError("droid_session_failed", "Droid session communication failed", { retryable: true });
    }

    if (outcome.status === "interrupted") {
      return {
        status: "cancelled",
        output: null,
        error: {
          code: "droid_interrupted",
          message: outcome.errorMessage ?? "The Droid invocation was interrupted",
          retryable: false,
        },
        ...(outcome.usage === undefined ? {} : { usage: outcome.usage }),
        ...(outcome.extensions === undefined ? {} : { extensions: outcome.extensions }),
      };
    }
    if (outcome.status === "failed") {
      return failed(
        "droid_invocation_failed",
        outcome.errorMessage ?? "The Droid invocation failed",
        outcome.usage,
        outcome.extensions,
      );
    }
    if (outcome.text === undefined || outcome.text.length === 0) {
      return failed(
        "droid_no_assistant_response",
        "Droid completed without a final assistant text response",
        outcome.usage,
        outcome.extensions,
      );
    }
    return {
      status: "succeeded",
      output: { response: outcome.text },
      ...(outcome.usage === undefined ? {} : { usage: outcome.usage }),
      ...(outcome.extensions === undefined ? {} : { extensions: outcome.extensions }),
    };
  }

  async stop(): Promise<void> {
    const session = this.session;
    this.session = undefined;
    if (session !== undefined) await session.stop();
  }
}

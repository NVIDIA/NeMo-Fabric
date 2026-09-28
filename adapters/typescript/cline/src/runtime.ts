// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentRunRequest, AgentRunResult, AgentUsage, JsonObject, RuntimeContext } from "nemo-fabric-adapter-contract";
import { LifecycleError, type AdapterRuntime, type AdapterStartInput } from "nemo-fabric-adapters-common";

export interface ClinePromptOutcome {
  status: "completed" | "aborted" | "failed";
  text?: string;
  errorMessage?: string;
  usage?: AgentUsage;
  extensions?: JsonObject;
}

export interface ClineSessionHandle {
  prompt(text: string): Promise<ClinePromptOutcome>;
  stop(): Promise<void>;
}

export interface ClineSessionFactory {
  create(input: AdapterStartInput): Promise<ClineSessionHandle>;
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

export class ClineAdapterRuntime implements AdapterRuntime {
  private readonly factory: ClineSessionFactory;
  private session?: ClineSessionHandle;

  constructor(factory: ClineSessionFactory) {
    this.factory = factory;
  }

  async start(input: AdapterStartInput): Promise<void> {
    if (this.session !== undefined) {
      throw new LifecycleError("cline_already_started", "Cline adapter runtime is already started");
    }
    this.session = await this.factory.create(input);
  }

  private async invalidate(session: ClineSessionHandle): Promise<void> {
    if (this.session !== session) {
      return;
    }
    this.session = undefined;
    try {
      await session.stop();
    } catch {
      // The failed session is unusable; preserve the invocation failure.
    }
  }

  async invoke(request: AgentRunRequest, _context: RuntimeContext): Promise<AgentRunResult> {
    if (this.session === undefined) {
      throw new LifecycleError("cline_not_started", "Cline adapter runtime is not started");
    }
    if (typeof request.input !== "string") {
      return failed("cline_unsupported_input", "The Cline adapter accepts only plain-text input");
    }

    const session = this.session;
    let outcome: ClinePromptOutcome;
    try {
      outcome = await session.prompt(request.input);
    } catch (error) {
      await this.invalidate(session);
      if (error instanceof LifecycleError) {
        throw error;
      }
      throw new LifecycleError("cline_session_failed", "Cline session communication failed", {
        retryable: true,
      });
    }

    if (outcome.status === "aborted") {
      return {
        status: "cancelled",
        output: null,
        error: {
          code: "cline_aborted",
          message: outcome.errorMessage ?? "The Cline invocation was aborted",
          retryable: false,
        },
        ...(outcome.usage === undefined ? {} : { usage: outcome.usage }),
        ...(outcome.extensions === undefined ? {} : { extensions: outcome.extensions }),
      };
    }
    if (outcome.status === "failed") {
      return failed(
        "cline_model_error",
        outcome.errorMessage ?? "The Cline invocation failed",
        outcome.usage,
        outcome.extensions,
      );
    }
    if (outcome.text === undefined || outcome.text.length === 0) {
      return failed(
        "cline_no_assistant_response",
        "Cline completed without a final assistant text response",
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
    if (session !== undefined) {
      await session.stop();
    }
  }
}

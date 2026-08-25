// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pi lifecycle state machine. It maps NeMo Fabric start, invoke, and stop calls
// to a PiSessionHandle and normalized results while keeping SDK construction
// behind a factory for focused lifecycle testing.

import type { AgentRunRequest, AgentRunResult, RuntimeContext } from "nemo-fabric-adapter-contract";
import { LifecycleError, type AdapterRuntime, type AdapterStartInput } from "nemo-fabric-adapters-common";

export type PiStopReason = "stop" | "length" | "toolUse" | "error" | "aborted" | string;

export interface PiPromptOutcome {
  accepted: boolean;
  text?: string;
  stopReason?: PiStopReason;
  errorMessage?: string;
  shutdownRequested?: boolean;
}

export interface PiSessionHandle {
  prompt(text: string): Promise<PiPromptOutcome>;
  stop(): Promise<void>;
}

export interface PiSessionFactory {
  create(input: AdapterStartInput): Promise<PiSessionHandle>;
}

function failed(code: string, message: string): AgentRunResult {
  return {
    status: "failed",
    output: null,
    error: { code, message, retryable: false },
  };
}

export class PiAdapterRuntime implements AdapterRuntime {
  private readonly factory: PiSessionFactory;
  private session?: PiSessionHandle;
  private unusable = false;

  constructor(factory: PiSessionFactory) {
    this.factory = factory;
  }

  async start(input: AdapterStartInput): Promise<void> {
    if (this.session !== undefined) {
      throw new LifecycleError("pi_already_started", "Pi adapter runtime is already started");
    }
    this.session = await this.factory.create(input);
    this.unusable = false;
  }

  async invoke(request: AgentRunRequest, _context: RuntimeContext): Promise<AgentRunResult> {
    if (this.session === undefined) {
      throw new LifecycleError("pi_not_started", "Pi adapter runtime is not started");
    }
    if (this.unusable) {
      throw new LifecycleError("pi_runtime_unusable", "Pi adapter runtime cannot accept another invocation");
    }
    if (typeof request.input !== "string") {
      return failed("pi_unsupported_input", "The Pi adapter accepts only plain-text input");
    }

    const outcome = await this.session.prompt(request.input);
    if (!outcome.accepted) {
      return failed("pi_prompt_rejected", "Pi rejected the prompt before starting an agent run");
    }
    if (outcome.shutdownRequested || outcome.stopReason === "aborted") {
      if (outcome.shutdownRequested) {
        this.unusable = true;
      }
      return {
        status: "cancelled",
        output: null,
        error: {
          code: outcome.shutdownRequested ? "pi_extension_shutdown" : "pi_aborted",
          message: outcome.shutdownRequested
            ? "A Pi extension requested adapter shutdown"
            : "The Pi invocation was aborted",
          retryable: false,
        },
      };
    }
    if (outcome.stopReason === "error") {
      return failed("pi_model_error", outcome.errorMessage || "The Pi model invocation failed");
    }
    if (outcome.text === undefined || outcome.text.length === 0) {
      return failed("pi_no_assistant_response", "Pi completed without a final assistant text response");
    }
    return { status: "succeeded", output: { response: outcome.text } };
  }

  async stop(): Promise<void> {
    const session = this.session;
    this.session = undefined;
    this.unusable = false;
    if (session !== undefined) {
      await session.stop();
    }
  }
}

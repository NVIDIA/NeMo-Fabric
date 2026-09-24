// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pi lifecycle state machine. It maps NeMo Fabric start, invoke, and stop calls
// to a PiSessionHandle and normalized results while keeping SDK construction
// behind a factory for focused lifecycle testing.

import type { AgentRunRequest, AgentRunResult, JsonObject, RuntimeContext } from "nemo-fabric-adapter-contract";
import { LifecycleError, type AdapterRuntime, type AdapterStartInput } from "nemo-fabric-adapters-common";

import { collectRelayArtifacts, type RelayArtifact } from "./relay-config.js";
import type { PiRelayRuntime } from "./relay.js";

export type PiStopReason = "stop" | "length" | "toolUse" | "error" | "aborted" | string;

export interface PiPromptOutcome {
  accepted: boolean;
  turnStarted: boolean;
  turnCount: number;
  text?: string;
  stopReason?: PiStopReason;
  errorMessage?: string;
  shutdownRequested?: boolean;
}

export interface PiSessionHandle {
  readonly relay?: PiRelayRuntime;
  readonly turnCount?: number;
  prompt(text: string): Promise<PiPromptOutcome>;
  /** Make a declared model role active for this and later invocations. */
  selectModel?(role: string): Promise<void>;
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

async function withRelayOutput(
  result: AgentRunResult,
  relay: PiRelayRuntime | undefined,
  turnStarted: boolean,
  turnCount: number,
): Promise<AgentRunResult> {
  const annotated = {
    ...result,
    extensions: { ...result.extensions, pi_turn_count: turnCount, pi_turn_started: turnStarted },
  };
  if (relay === undefined) {
    return annotated;
  }
  const current: JsonObject =
    typeof result.output === "object" && result.output !== null && !Array.isArray(result.output)
      ? (result.output as JsonObject)
      : {};
  return {
    ...annotated,
    output: { ...current, ...(await relay.output(await collectNonAtifArtifacts(relay))) },
  };
}

async function collectNonAtifArtifacts(relay: PiRelayRuntime): Promise<RelayArtifact[]> {
  return collectRelayArtifacts(relay.pluginConfig, []);
}

export class PiAdapterRuntime implements AdapterRuntime {
  private readonly factory: PiSessionFactory;
  private session?: PiSessionHandle;
  private turnCount = 0;
  private unusable = false;

  constructor(factory: PiSessionFactory) {
    this.factory = factory;
  }

  async start(input: AdapterStartInput): Promise<void> {
    if (this.session !== undefined) {
      throw new LifecycleError("pi_already_started", "Pi adapter runtime is already started");
    }
    this.session = await this.factory.create(input);
    this.turnCount = 0;
    this.unusable = false;
  }

  async invoke(request: AgentRunRequest, _context: RuntimeContext): Promise<AgentRunResult> {
    if (this.session === undefined) {
      throw new LifecycleError("pi_not_started", "Pi adapter runtime is not started");
    }
    if (this.unusable) {
      throw new LifecycleError("pi_runtime_unusable", "Pi adapter runtime cannot accept another invocation");
    }
    const input = promptInput(request.input);
    if (input?.model !== undefined) {
      try {
        if (this.session.selectModel === undefined) {
          throw new Error("model selection is unavailable");
        }
        await this.session.selectModel(input.model);
      } catch {
        return withRelayOutput(
          failed("pi_model_selection_failed", "The requested Pi model role could not be selected"),
          this.session.relay,
          false,
          this.turnCount,
        );
      }
    }
    if (input === undefined) {
      const observedTurnCount = this.session.turnCount;
      if (observedTurnCount !== undefined) {
        if (!Number.isInteger(observedTurnCount) || observedTurnCount < this.turnCount) {
          throw new LifecycleError("pi_invalid_turn_count", "Pi returned an invalid cumulative turn count");
        }
        this.turnCount = observedTurnCount;
      }
      return withRelayOutput(
        failed("pi_unsupported_input", "The Pi adapter accepts text or an object with prompt and model"),
        this.session.relay,
        false,
        this.turnCount,
      );
    }

    const relay = this.session.relay;
    const outcome = await this.session.prompt(input.prompt);
    if (
      !Number.isInteger(outcome.turnCount) ||
      outcome.turnCount < this.turnCount ||
      (outcome.turnStarted ? outcome.turnCount === this.turnCount : outcome.turnCount !== this.turnCount)
    ) {
      throw new LifecycleError("pi_invalid_turn_count", "Pi returned an invalid cumulative turn count");
    }
    this.turnCount = outcome.turnCount;
    if (!outcome.accepted) {
      return withRelayOutput(
        failed("pi_prompt_rejected", "Pi rejected the prompt before starting an agent run"),
        relay,
        outcome.turnStarted,
        outcome.turnCount,
      );
    }
    if (outcome.shutdownRequested || outcome.stopReason === "aborted") {
      if (outcome.shutdownRequested) {
        this.unusable = true;
      }
      return withRelayOutput(
        {
          status: "cancelled",
          output: null,
          error: {
            code: outcome.shutdownRequested ? "pi_extension_shutdown" : "pi_aborted",
            message: outcome.shutdownRequested
              ? "A Pi extension requested adapter shutdown"
              : "The Pi invocation was aborted",
            retryable: false,
          },
        },
        relay,
        outcome.turnStarted,
        outcome.turnCount,
      );
    }
    if (outcome.stopReason === "error") {
      return withRelayOutput(
        failed("pi_model_error", outcome.errorMessage || "The Pi model invocation failed"),
        relay,
        outcome.turnStarted,
        outcome.turnCount,
      );
    }
    if (outcome.text === undefined || outcome.text.length === 0) {
      return withRelayOutput(
        failed("pi_no_assistant_response", "Pi completed without a final assistant text response"),
        relay,
        outcome.turnStarted,
        outcome.turnCount,
      );
    }
    return withRelayOutput(
      { status: "succeeded", output: { response: outcome.text } },
      relay,
      outcome.turnStarted,
      outcome.turnCount,
    );
  }

  async stop(): Promise<void> {
    const session = this.session;
    if (session !== undefined) {
      let failure: unknown;
      try {
        await session.stop();
      } catch (error) {
        failure = error;
      }
      try {
        await session.relay?.stop();
      } catch (error) {
        if (failure === undefined) {
          failure = error;
        } else if (error instanceof LifecycleError) {
          failure = new LifecycleError(error.code, "Pi session and NeMo Relay cleanup failed", {
            retryable: error.retryable,
            metadata: {
              ...error.metadata,
              session_error: failure instanceof Error ? failure.message : String(failure),
            },
          });
        } else {
          failure = new AggregateError([failure, error], "Pi session and NeMo Relay cleanup failed");
        }
      }
      if (failure !== undefined) {
        this.unusable = true;
        // Retain the handle so a host retry can reach Relay cleanup that failed,
        // but prevent another invocation from reaching a disposed Pi session.
        throw failure;
      }
      this.session = undefined;
      this.unusable = false;
    }
  }
}

/** Plain text, or `{ prompt, model }` selecting a declared model role first. */
function promptInput(input: unknown): { prompt: string; model?: string } | undefined {
  if (typeof input === "string") {
    return { prompt: input };
  }
  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    return undefined;
  }
  const { prompt, model, ...rest } = input as Record<string, unknown>;
  if (typeof prompt !== "string" || typeof model !== "string" || Object.keys(rest).length > 0) {
    return undefined;
  }
  return { prompt, model };
}

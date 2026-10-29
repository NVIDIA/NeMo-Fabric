// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pi lifecycle state machine. It maps NeMo Fabric start, invoke, and stop calls
// to a PiSessionHandle and normalized results while keeping SDK construction
// behind a factory for focused lifecycle testing.

import type { AdapterCheckResult, CheckLevel, AgentRunRequest, AgentRunResult, AgentUsage, JsonObject, RuntimeContext } from "nemo-fabric-adapter-contract";
import { LifecycleError, unobservedHealth, type AdapterRuntime, type AdapterStartInput } from "nemo-fabric-adapters-common";

import { collectRelayArtifacts, type RelayArtifact } from "./relay-config.js";
import type { PiRelayRuntime } from "./relay.js";

export type PiStopReason = "stop" | "length" | "toolUse" | "error" | "aborted" | string;

export interface PiPromptOutcome {
  usage?: AgentUsage;
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
  stop(): Promise<void>;
  check(level: CheckLevel): Promise<AdapterCheckResult>;
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
  private stopping = false;
  private generation = 0;

  constructor(factory: PiSessionFactory) {
    this.factory = factory;
  }

  async start(input: AdapterStartInput): Promise<void> {
    if (this.session !== undefined) {
      throw new LifecycleError("pi_already_started", "Pi adapter runtime is already started");
    }
    this.session = await this.factory.create(input);
    this.generation += 1;
    this.stopping = false;
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
    if (typeof request.input !== "string") {
      const observedTurnCount = this.session.turnCount;
      if (observedTurnCount !== undefined) {
        if (!Number.isInteger(observedTurnCount) || observedTurnCount < this.turnCount) {
          throw new LifecycleError("pi_invalid_turn_count", "Pi returned an invalid cumulative turn count");
        }
        this.turnCount = observedTurnCount;
      }
      return withRelayOutput(
        failed("pi_unsupported_input", "The Pi adapter accepts only plain-text input"),
        this.session.relay,
        false,
        this.turnCount,
      );
    }

    const relay = this.session.relay;
    const outcome = await this.session.prompt(request.input);
    const annotate = (result: AgentRunResult) => withRelayOutput(
      { ...result, ...(outcome.usage === undefined ? {} : { usage: outcome.usage }) },
      relay, outcome.turnStarted, outcome.turnCount,
    );
    if (
      !Number.isInteger(outcome.turnCount) ||
      outcome.turnCount < this.turnCount ||
      (outcome.turnStarted ? outcome.turnCount === this.turnCount : outcome.turnCount !== this.turnCount)
    ) {
      throw new LifecycleError("pi_invalid_turn_count", "Pi returned an invalid cumulative turn count");
    }
    this.turnCount = outcome.turnCount;
    if (!outcome.accepted) {
      return annotate(
        failed("pi_prompt_rejected", "Pi rejected the prompt before starting an agent run"),
      );
    }
    if (outcome.shutdownRequested || outcome.stopReason === "aborted") {
      if (outcome.shutdownRequested) {
        this.unusable = true;
      }
      return annotate(
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
      );
    }
    if (outcome.stopReason === "error") {
      return annotate(
        failed("pi_model_error", outcome.errorMessage || "The Pi model invocation failed"),
      );
    }
    if (outcome.text === undefined || outcome.text.length === 0) {
      return annotate(
        failed("pi_no_assistant_response", "Pi completed without a final assistant text response"),
      );
    }
    return annotate(
      { status: "succeeded", output: { response: outcome.text } },
    );
  }

  async check(level: CheckLevel): Promise<AdapterCheckResult> {
    const session = this.session;
    const generation = this.generation;
    if (session === undefined) return {
      ...unobservedHealth("session_closed"), outcome: "failed", activity: "stopped",
    };
    if (this.stopping) return unobservedHealth("observation_changed");
    if (this.unusable && level !== "live") return {
      ...unobservedHealth("endpoint_unavailable"), outcome: "failed",
    };
    const result = await session.check(level);
    return this.session === session && this.generation === generation && !this.stopping
      ? result : unobservedHealth("observation_changed");
  }

  async stop(): Promise<void> {
    const session = this.session;
    if (session !== undefined) {
      this.stopping = true;
      this.generation += 1;
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

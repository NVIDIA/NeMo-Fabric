// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pi lifecycle state machine. It maps NeMo Fabric start, invoke, and stop calls
// to a PiSessionHandle and normalized results while keeping SDK construction
// behind a factory for focused lifecycle testing.

import type { AgentRunRequest, AgentRunResult, JsonObject, RuntimeContext } from "nemo-fabric-adapter-contract";
import { LifecycleError, type AdapterRuntime, type AdapterStartInput } from "nemo-fabric-adapters-common";

import {
  ATIF_FINALIZATION_TIMEOUT_MS,
  type AtifSnapshot,
  expectsLocalAtif,
  snapshotAtifFiles,
  waitForFinalizedAtif,
} from "./relay-artifacts.js";
import { collectRelayArtifacts, type RelayArtifact } from "./relay-config.js";
import type { PiRelayRuntime } from "./relay.js";

export type PiStopReason = "stop" | "length" | "toolUse" | "error" | "aborted" | string;

export interface PiPromptOutcome {
  accepted: boolean;
  text?: string;
  stopReason?: PiStopReason;
  errorMessage?: string;
  shutdownRequested?: boolean;
}

export interface PiSessionHandle {
  readonly relay?: PiRelayRuntime;
  prompt(text: string): Promise<PiPromptOutcome>;
  stop(): Promise<void>;
}

export interface PiSessionFactory {
  create(input: AdapterStartInput): Promise<PiSessionHandle>;
}

export interface PiAdapterRuntimeOptions {
  atifFinalizationTimeoutMs?: number;
}

interface PiStopFailure {
  stage: "session" | "relay_output" | "relay";
  error: unknown;
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
): Promise<AgentRunResult> {
  if (relay === undefined) {
    return result;
  }
  let artifacts: RelayArtifact[] = [];
  try {
    artifacts = await collectNonAtifArtifacts(relay);
  } catch (error) {
    const detail = error instanceof Error ? `: ${error.message}` : "";
    process.stderr.write(`NeMo Relay artifact collection failed${detail}\n`);
  }
  const current: JsonObject =
    typeof result.output === "object" && result.output !== null && !Array.isArray(result.output)
      ? (result.output as JsonObject)
      : {};
  return {
    ...result,
    output: { ...current, ...(await relay.output(artifacts)) },
  };
}

async function collectNonAtifArtifacts(relay: PiRelayRuntime): Promise<RelayArtifact[]> {
  return (await collectRelayArtifacts(relay.pluginConfig, relay.atifMatchers)).filter(
    (artifact) => artifact.kind !== "atif",
  );
}

function stopError(failures: PiStopFailure[]): JsonObject {
  const lifecycleFailure = failures.find((failure) => failure.error instanceof LifecycleError)?.error;
  const primary = lifecycleFailure instanceof LifecycleError ? lifecycleFailure : undefined;
  const details = failures.map(({ stage, error }) => ({
    stage,
    message: error instanceof Error ? error.message : String(error),
  }));
  return {
    stage: "stop",
    code: primary?.code ?? "pi_runtime_stop_failed",
    message:
      failures.length > 1
        ? "Pi session and NeMo Relay cleanup failed"
        : (primary?.message ?? "Pi runtime cleanup failed"),
    retryable: primary?.retryable ?? false,
    metadata: {
      ...primary?.metadata,
      failures: details,
    },
  };
}

export class PiAdapterRuntime implements AdapterRuntime {
  private readonly factory: PiSessionFactory;
  private readonly atifFinalizationTimeoutMs: number;
  private session?: PiSessionHandle;
  private unusable = false;

  constructor(factory: PiSessionFactory, options: PiAdapterRuntimeOptions = {}) {
    this.factory = factory;
    const configuredTimeout = options.atifFinalizationTimeoutMs;
    this.atifFinalizationTimeoutMs =
      configuredTimeout !== undefined && Number.isFinite(configuredTimeout) && configuredTimeout >= 0
        ? Math.min(configuredTimeout, ATIF_FINALIZATION_TIMEOUT_MS)
        : ATIF_FINALIZATION_TIMEOUT_MS;
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
      return withRelayOutput(
        failed("pi_unsupported_input", "The Pi adapter accepts only plain-text input"),
        this.session.relay,
      );
    }

    const outcome = await this.session.prompt(request.input);
    const relay = this.session.relay;
    if (!outcome.accepted) {
      return withRelayOutput(
        failed("pi_prompt_rejected", "Pi rejected the prompt before starting an agent run"),
        relay,
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
      );
    }
    if (outcome.stopReason === "error") {
      return withRelayOutput(
        failed("pi_model_error", outcome.errorMessage || "The Pi model invocation failed"),
        relay,
      );
    }
    if (outcome.text === undefined || outcome.text.length === 0) {
      return withRelayOutput(
        failed("pi_no_assistant_response", "Pi completed without a final assistant text response"),
        relay,
      );
    }
    return withRelayOutput({ status: "succeeded", output: { response: outcome.text } }, relay);
  }

  async stop(): Promise<JsonObject | void> {
    const session = this.session;
    if (session !== undefined) {
      const relay = session.relay;
      let atifBefore: AtifSnapshot | undefined;
      let atifSnapshotFailed = false;
      if (relay !== undefined && expectsLocalAtif(relay.pluginConfig, relay.atifMatchers)) {
        try {
          atifBefore = await snapshotAtifFiles(relay.pluginConfig, relay.atifMatchers);
        } catch (error) {
          atifSnapshotFailed = true;
          const detail = error instanceof Error ? `: ${error.message}` : "";
          process.stderr.write(`NeMo Relay ATIF artifact snapshot failed${detail}\n`);
        }
      }
      const failures: PiStopFailure[] = [];
      try {
        await session.stop();
      } catch (error) {
        failures.push({ stage: "session", error });
      }
      let relayOutput: JsonObject | undefined;
      if (relay !== undefined) {
        let artifacts: RelayArtifact[] = [];
        try {
          let includeAtif = !atifSnapshotFailed;
          let finalizedAtifPath: string | undefined;
          if (atifBefore !== undefined) {
            finalizedAtifPath = await waitForFinalizedAtif(relay.pluginConfig, atifBefore, {
              matchers: relay.atifMatchers,
              timeoutMs: this.atifFinalizationTimeoutMs,
            });
            includeAtif = finalizedAtifPath !== undefined;
            if (!includeAtif) {
              process.stderr.write(
                `NeMo Relay did not finalize an ATIF artifact within ${this.atifFinalizationTimeoutMs} ms after session shutdown\n`,
              );
            }
          }
          if (finalizedAtifPath !== undefined) {
            const finalizedArtifact: RelayArtifact = { kind: "atif", path: finalizedAtifPath };
            artifacts = [...(await collectNonAtifArtifacts(relay)), finalizedArtifact].sort((left, right) =>
              left.path.localeCompare(right.path),
            );
          } else {
            artifacts = includeAtif
              ? await collectRelayArtifacts(relay.pluginConfig, relay.atifMatchers)
              : await collectNonAtifArtifacts(relay);
          }
        } catch (error) {
          const detail = error instanceof Error ? `: ${error.message}` : "";
          process.stderr.write(`NeMo Relay final artifact collection failed${detail}\n`);
          try {
            artifacts = await collectNonAtifArtifacts(relay);
          } catch {
            artifacts = [];
          }
        }
        try {
          relayOutput = await relay.output(artifacts);
        } catch (error) {
          failures.push({ stage: "relay_output", error });
          relayOutput = {
            relay_artifacts: artifacts.map(({ kind, path }) => ({ kind, path })),
          };
        }
      }
      try {
        await relay?.stop();
      } catch (error) {
        failures.push({ stage: "relay", error });
      }
      if (failures.length > 0) {
        relayOutput = {
          ...(relayOutput ?? {}),
          runtime_stop_error: stopError(failures),
        };
      }
      this.session = undefined;
      this.unusable = false;
      return relayOutput;
    }
  }
}

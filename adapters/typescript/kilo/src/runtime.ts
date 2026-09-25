// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentRunRequest, AgentRunResult, AgentUsage, RuntimeContext } from "nemo-fabric-adapter-contract";
import { LifecycleError, type AdapterRuntime, type AdapterStartInput } from "nemo-fabric-adapters-common";

export interface KiloPromptOutcome { text?: string; error?: boolean; usage?: AgentUsage }
export interface KiloSessionHandle { id: string; prompt(text: string): Promise<KiloPromptOutcome>; stop(): Promise<void> }
export interface KiloSessionFactory { create(input: AdapterStartInput): Promise<KiloSessionHandle> }

function failed(code: string, message: string, usage?: AgentUsage): AgentRunResult {
  return {status: "failed", output: null, error: {code, message, retryable: false}, ...(usage === undefined ? {} : {usage})};
}

export class KiloAdapterRuntime implements AdapterRuntime {
  private readonly factory: KiloSessionFactory;
  private session?: KiloSessionHandle;

  constructor(factory: KiloSessionFactory) { this.factory = factory; }

  async start(input: AdapterStartInput): Promise<void> {
    if (this.session !== undefined) throw new LifecycleError("kilo_already_started", "Kilo Code adapter runtime is already started");
    this.session = await this.factory.create(input);
  }

  async invoke(request: AgentRunRequest, _context: RuntimeContext): Promise<AgentRunResult> {
    if (this.session === undefined) throw new LifecycleError("kilo_not_started", "Kilo Code adapter runtime is not started");
    if (typeof request.input !== "string") return failed("kilo_unsupported_input", "The Kilo Code adapter accepts only plain-text input");
    const session = this.session;
    let outcome: KiloPromptOutcome;
    try {
      outcome = await session.prompt(request.input);
    } catch (error) {
      this.session = undefined;
      try { await session.stop(); } catch { /* Preserve the prompt failure. */ }
      if (error instanceof LifecycleError) throw error;
      throw new LifecycleError("kilo_session_failed", "Kilo Code session communication failed", {retryable: true});
    }
    if (outcome.error === true) return failed("kilo_model_error", "Kilo Code model invocation failed", outcome.usage);
    if (outcome.text === undefined || outcome.text.length === 0) {
      return failed("kilo_no_assistant_response", "Kilo Code completed without a final assistant text response", outcome.usage);
    }
    return {status: "succeeded", output: {response: outcome.text}, ...(outcome.usage === undefined ? {} : {usage: outcome.usage})};
  }

  async stop(): Promise<void> {
    const session = this.session;
    this.session = undefined;
    if (session !== undefined) await session.stop();
  }
}

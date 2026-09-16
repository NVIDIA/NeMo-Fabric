// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Fabric lifecycle state for one OpenCode session. The SDK boundary supplies
// the session implementation so lifecycle tests do not need an installed SDK.

import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import type { AgentArtifact, AgentRunRequest, AgentRunResult, AgentUsage, RuntimeContext } from "nemo-fabric-adapter-contract";
import { LifecycleError, type AdapterRuntime, type AdapterStartInput } from "nemo-fabric-adapters-common";

export interface OpenCodePromptOutcome {
  text?: string;
  errorMessage?: string;
  patch?: string;
  usage?: AgentUsage;
}

export interface OpenCodeSessionHandle {
  id: string;
  prompt(text: string): Promise<OpenCodePromptOutcome>;
  stop(): Promise<void>;
}

export interface OpenCodeSessionFactory {
  create(input: AdapterStartInput): Promise<OpenCodeSessionHandle>;
}

function failed(code: string, message: string): AgentRunResult {
  return {
    status: "failed",
    output: null,
    error: { code, message, retryable: false },
  };
}

function artifactSegment(identifier: string): string {
  return createHash("sha256").update(identifier).digest("hex").slice(0, 32);
}

export class OpenCodeAdapterRuntime implements AdapterRuntime {
  private readonly factory: OpenCodeSessionFactory;
  private session?: OpenCodeSessionHandle;
  private artifactCount = 0;

  constructor(factory: OpenCodeSessionFactory) {
    this.factory = factory;
  }

  async start(input: AdapterStartInput): Promise<void> {
    if (this.session !== undefined) {
      throw new LifecycleError("opencode_already_started", "OpenCode adapter runtime is already started");
    }
    this.session = await this.factory.create(input);
    this.artifactCount = 0;
  }

  private async writePatch(patch: string, context: RuntimeContext): Promise<AgentArtifact[] | undefined> {
    const root = context.artifacts.root ?? context.environment.artifacts;
    if (root === undefined || root === null || root.length === 0) {
      return undefined;
    }
    const turn = this.artifactCount + 1;
    const runtime = artifactSegment(context.runtime_id);
    const invocation = artifactSegment(context.invocation_id);
    const path = `opencode/${runtime}/${invocation}-turn-${turn}.patch`;
    try {
      await mkdir(join(root, "opencode", runtime), { recursive: true });
      await writeFile(join(root, path), patch, { encoding: "utf8", flag: "wx" });
    } catch {
      throw new LifecycleError("opencode_artifact_write_failed", "OpenCode could not write its session diff artifact");
    }
    this.artifactCount = turn;
    return [{ kind: "patch", media_type: "text/x-diff", name: `opencode-diff-${turn}`, path }];
  }

  private async invalidate(session: OpenCodeSessionHandle): Promise<void> {
    if (this.session !== session) {
      return;
    }
    this.session = undefined;
    try {
      await session.stop();
    } catch {
      // The runtime is already invalid and must not conceal the original failure.
    }
  }

  async invoke(request: AgentRunRequest, context: RuntimeContext): Promise<AgentRunResult> {
    if (this.session === undefined) {
      throw new LifecycleError("opencode_not_started", "OpenCode adapter runtime is not started");
    }
    if (typeof request.input !== "string") {
      return failed("opencode_unsupported_input", "The OpenCode adapter accepts only plain-text input");
    }

    const session = this.session;
    let outcome: OpenCodePromptOutcome;
    try {
      outcome = await session.prompt(request.input);
    } catch (error) {
      await this.invalidate(session);
      if (error instanceof LifecycleError) {
        throw error;
      }
      throw new LifecycleError("opencode_session_failed", "OpenCode session communication failed", { retryable: true });
    }
    if (outcome.errorMessage !== undefined) {
      return failed("opencode_model_error", outcome.errorMessage);
    }
    if (outcome.text === undefined || outcome.text.length === 0) {
      return failed("opencode_no_assistant_response", "OpenCode completed without a final assistant text response");
    }
    const artifacts = outcome.patch === undefined ? undefined : await this.writePatch(outcome.patch, context);
    return {
      status: "succeeded",
      output: { response: outcome.text },
      ...(outcome.usage === undefined ? {} : { usage: outcome.usage }),
      ...(artifacts === undefined ? {} : { artifacts }),
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

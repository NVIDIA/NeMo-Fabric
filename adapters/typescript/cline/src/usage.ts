// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AsyncLocalStorage } from "node:async_hooks";
import { usageFromObservations, type AgentUsage, type UsageObservation } from "nemo-fabric-adapter-contract";

type ObjectValue = Record<string, unknown>;
const object = (value: unknown): ObjectValue => typeof value === "object" && value !== null ? value as ObjectValue : {};
const text = (value: unknown): string | undefined => typeof value === "string" && value.trim() ? value : undefined;
const owner = new AsyncLocalStorage<ClineInvocationUsage>();

/** Public Cline hooks preserve call boundaries; no global provider interception. */
export const clineUsageHooks = {
  beforeModel(context?: { snapshot?: { usage?: unknown } }): void { owner.getStore()?.started(context?.snapshot?.usage); },
  afterModel(context: { assistantMessage: unknown }): void { owner.getStore()?.completed(context.assistantMessage); },
  onEvent(event: { type: string; metadata?: unknown; usage?: unknown }): void {
    if (event.type === "usage-updated") owner.getStore()?.updated(event.usage);
    const metadata = object(event.metadata);
    if (event.type === "status-notice" && metadata.phase === "started" &&
        ["auto_compaction", "manual_compaction", "overflow_recovery_compaction"].includes(String(metadata.kind))) {
      // Cline performs this auxiliary model request before beforeModel runs.
      owner.getStore()?.compaction();
    }
  },
};

function reportedMetrics(value: unknown): UsageObservation["reported"] {
  const metrics = object(value);
  const reported: UsageObservation["reported"] = {};
  for (const [source, destination] of [
    ["inputTokens", "input_tokens"], ["cacheReadTokens", "cached_input_tokens"],
    ["cacheWriteTokens", "cache_write_input_tokens"], ["outputTokens", "output_tokens"],
    ["reasoningTokenCount", "reasoning_tokens"],
  ] as const) {
    const count = metrics[source];
    // @cline/agents 0.0.83 zero-fills omissions before assistantMessage.metrics.
    // A positive per-call value is evidence; zero cannot establish presence.
    if (typeof count === "number" && Number.isSafeInteger(count) && count > 0) reported[destination] = count;
  }
  if (reported.input_tokens !== undefined) {
    reported.input_tokens_include_cache = true;
  }
  if (reported.output_tokens !== undefined) reported.output_tokens_include_reasoning = true;
  // metrics.cost can be a client-side estimate, so never call it provider cost.
  return reported;
}

export class ClineInvocationUsage {
  private readonly records = new Map<string, UsageObservation>();
  private readonly pending: { baseline: ObjectValue; latest?: ObjectValue }[] = [];
  private unverified = false;
  private frozen = false;

  constructor(private readonly providerKnown = true) {}

  run<T>(operation: () => Promise<T>): Promise<T> { return owner.run(this, operation); }

  started(value?: unknown): void {
    if (!this.frozen) this.pending.push({ baseline: { ...object(value) } });
  }

  updated(value: unknown): void {
    if (this.frozen) return;
    const call = this.pending.at(-1);
    if (call !== undefined) call.latest = { ...object(value) };
  }

  completed(value: unknown): void {
    if (this.frozen) return;
    const message = object(value);
    const id = text(message.id);
    const modelInfo = object(message.modelInfo);
    const nativeProvider = text(modelInfo.provider);
    const provider = nativeProvider === "openai-native" || nativeProvider === "openai-compatible" ? "openai" : nativeProvider;
    const evidence: UsageObservation = {
      reported: reportedMetrics(message.metrics),
      ...(text(modelInfo.id) === undefined ? {} : { model: text(modelInfo.id) }),
      ...(this.providerKnown && provider !== undefined ? { provider } : {}),
      tokens_aligned: true,
      // A logical hook is not an exhaustive list of physical pricing requests.
      is_model_call: false,
      // The pinned Core hooks surround one logical call. Its llms provider can
      // hide physical retries and paid empty-response retries inside that call.
      gap: "scope_unverified",
    };
    if (id === undefined) {
      if (this.pending.length > 0) this.pending.pop();
      this.unverified = true;
      return;
    }
    const previous = this.records.get(id);
    if (previous !== undefined) {
      if (JSON.stringify(previous) !== JSON.stringify(evidence)) this.unverified = true;
      return;
    }
    if (this.pending.length === 0) this.unverified = true;
    else this.pending.pop();
    this.records.set(id, evidence);
  }

  compaction(): void { if (!this.frozen) this.unverified = true; }

  finish(terminalUsage?: unknown): AgentUsage {
    this.frozen = true;
    const records = [...this.records.values()];
    if (!records.length && !this.pending.length) {
      // A terminal aggregate cannot prove that every call reported each field.
      records.push({ reported: reportedMetrics(terminalUsage), gap: "scope_unverified" });
    }
    for (const call of this.pending) {
      const delta: ObjectValue = {};
      // Cline's usage-updated event is cumulative within this run. Recover only
      // the unfinished call's increments only when a baseline proves the delta.
      for (const field of ["inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens", "reasoningTokenCount"]) {
        const before = call.baseline[field];
        const after = call.latest?.[field];
        if (typeof before === "number" && Number.isSafeInteger(before) && before >= 0 &&
            typeof after === "number" && Number.isSafeInteger(after) && after >= before) delta[field] = after - before;
      }
      records.push({ reported: reportedMetrics(delta), gap: "unfinished_work" });
    }
    if (this.unverified) records.push({ reported: {}, gap: "scope_unverified" });
    return usageFromObservations(records);
  }
}

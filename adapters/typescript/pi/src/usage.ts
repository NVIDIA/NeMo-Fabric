// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pi 1.0.3 supplies raw provider events before its usage parsers fill omissions.
// Keep this source adaptation local; the contract reducer and core own totals.
import { AsyncLocalStorage } from "node:async_hooks";
import type { AgentSession } from "@earendil-works/pi-coding-agent";
import { usageFromObservations, type AgentUsage, type UsageObservation } from "nemo-fabric-adapter-contract";

const ownerScope = new AsyncLocalStorage<InvocationUsage>();
const FETCH_PROTOCOLS = new Set(["anthropic-messages", "azure-openai-responses", "mistral-conversations", "openai-codex-responses", "openai-completions", "openai-responses", "pi-messages"]);

interface Call {
  api: string;
  maxRetries: number;
  model?: string;
  provider?: string;
  routeObserved?: boolean;
  routeConflict?: boolean;
  returnedModel?: string;
  reported: UsageObservation["reported"];
  attempts: (number | undefined)[];
  finished: boolean;
  warmup?: boolean;
  transportFailure?: boolean;
  retryError?: boolean;
  overlappingWarmup?: boolean;
  terminal?: boolean;
  failed?: boolean;
}
interface PiUsageOptions {
  customToolNames?: ReadonlySet<string>;
  sessionId?: string;
  warmingStatus?: () => unknown;
  providerMaxRetries?: () => number | undefined;
}
interface CustomToolCall {
  evidence: UsageObservation;
  covered: boolean;
}
interface AttemptScope { delegate: typeof globalThis.fetch; current: () => Call }
const attemptScope = new AsyncLocalStorage<AttemptScope>();
// One function identity preserves Pi's Anthropic federation-client cache.
const observedFetch: typeof globalThis.fetch = async (input, init) => {
  const scope = attemptScope.getStore();
  if (scope === undefined) return globalThis.fetch(input, init);
  const call = scope.current();
  const index = call.attempts.length;
  call.attempts.push(undefined);
  const url = input instanceof Request ? input.url : String(input);
  observeRoute(call, url);
  const response = await scope.delegate(input, init);
  if (response.url) observeRoute(call, response.url);
  call.attempts[index] = response.status;
  return response;
};
const text = (value: unknown): string | undefined => typeof value === "string" && value.trim() ? value : undefined;
const count = (value: unknown): number | undefined => typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
const object = (value: unknown): Record<string, unknown> => typeof value === "object" && value !== null ? value as Record<string, unknown> : {};

/** Accounting identity follows a qualified source route, never Pi's provider label.
 * These are the native endpoint/protocol pairs established by the pinned APIs.
 * Other routes retain their model/counts and can use an explicit pricing catalog.
 */
function accountingProvider(api: string, endpoint: string | undefined): string | undefined {
  if (endpoint === undefined) return undefined;
  try {
    const url = new URL(endpoint);
    if (url.protocol !== "https:" || (url.port !== "" && url.port !== "443")) return undefined;
    if (url.hostname === "api.openai.com" && (api === "openai-completions" || api === "openai-responses")) return "openai";
    if (url.hostname === "api.anthropic.com" && api === "anthropic-messages") return "anthropic";
    if (url.hostname === "generativelanguage.googleapis.com" && api === "google-generative-ai") return "google";
    if (url.hostname === "api.mistral.ai" && api === "mistral-conversations") return "mistral";
  } catch {
    // An invalid or opaque endpoint cannot authorize provider-specific pricing.
  }
  return undefined;
}

function observeRoute(call: Call, endpoint: string): void {
  const provider = accountingProvider(call.api, endpoint);
  if (call.routeObserved && call.provider !== provider) call.routeConflict = true;
  call.routeObserved = true;
  call.provider = call.routeConflict ? undefined : provider;
}

/** Read only literal numeric fields before Pi fills absent values with zero. */
function sourceUsage(api: string, value: unknown, previous: UsageObservation["reported"]): UsageObservation["reported"] {
  const event = object(value);
  const fields = (usage: unknown, names: Record<string, string | string[]>, flags: { input: boolean; output: boolean }): UsageObservation["reported"] => {
    const raw = object(usage);
    const result: Record<string, number | boolean> = {};
    for (const [destination, paths] of Object.entries(names)) {
      for (const path of Array.isArray(paths) ? paths : [paths]) {
        let field: unknown = raw;
        for (const part of path.split(".")) field = object(field)[part];
        const measured = count(field);
        if (measured !== undefined) { result[destination] = measured; break; }
      }
    }
    if (result.input_tokens !== undefined) result.input_tokens_include_cache = flags.input;
    if (result.output_tokens !== undefined) result.output_tokens_include_reasoning = flags.output;
    if (result.input_tokens !== undefined && flags.input) result.peak_request_input_tokens = result.input_tokens;
    return result;
  };
  const openai = { input_tokens: "prompt_tokens", cached_input_tokens: ["prompt_tokens_details.cached_tokens", "prompt_cache_hit_tokens", "cached_tokens"], cache_write_input_tokens: "prompt_tokens_details.cache_write_tokens", output_tokens: "completion_tokens", reasoning_tokens: "completion_tokens_details.reasoning_tokens", total_tokens: "total_tokens" };
  switch (api) {
    case "openai-completions": {
      const choice = Array.isArray(event.choices) ? object(event.choices[0]) : {};
      const usage = event.usage ?? choice.usage;
      return usage === undefined ? previous : fields(usage, openai, { input: true, output: true });
    }
    case "openai-responses":
    case "azure-openai-responses":
    case "openai-codex-responses": {
      const usage = object(event.response).usage;
      if (!["response.completed", "response.incomplete", "response.done", "response.failed"].includes(String(event.type)) || usage === undefined) return previous;
      return fields(usage, { input_tokens: "input_tokens", cached_input_tokens: "input_tokens_details.cached_tokens", cache_write_input_tokens: "input_tokens_details.cache_write_tokens", output_tokens: "output_tokens", reasoning_tokens: "output_tokens_details.reasoning_tokens", total_tokens: "total_tokens" }, { input: true, output: true });
    }
    case "anthropic-messages": {
      const names = { input_tokens: "input_tokens", cached_input_tokens: "cache_read_input_tokens", cache_write_input_tokens: "cache_creation_input_tokens" };
      if (event.type === "message_start") return fields(object(event.message).usage, names, { input: false, output: true });
      if (event.type === "message_delta" && event.usage !== undefined) return { ...previous, ...fields(event.usage, { ...names, output_tokens: "output_tokens" }, { input: false, output: true }) };
      return previous;
    }
    case "google-generative-ai":
    case "google-vertex":
      return event.usageMetadata === undefined ? previous : fields(event.usageMetadata, { input_tokens: "promptTokenCount", cached_input_tokens: "cachedContentTokenCount", output_tokens: "candidatesTokenCount", reasoning_tokens: "thoughtsTokenCount", total_tokens: "totalTokenCount" }, { input: true, output: false });
    case "bedrock-converse-stream": {
      const usage = object(event.metadata).usage;
      return usage === undefined ? previous : fields(usage, { input_tokens: "inputTokens", cached_input_tokens: "cacheReadInputTokens", cache_write_input_tokens: "cacheWriteInputTokens", output_tokens: "outputTokens", total_tokens: "totalTokens" }, { input: false, output: true });
    }
    case "mistral-conversations":
      return event.usage === undefined ? previous : fields(event.usage, { input_tokens: "prompt_tokens", cached_input_tokens: ["promptTokensDetails.cachedTokens", "prompt_tokens_details.cached_tokens", "promptTokenDetails.cachedTokens", "prompt_token_details.cached_tokens", "numCachedTokens", "num_cached_tokens"], output_tokens: "completion_tokens", total_tokens: "total_tokens" }, { input: true, output: true });
    case "pi-messages":
      return event.usage === undefined || (event.type !== "done" && event.type !== "error") ? previous : fields(event.usage, { input_tokens: "input", cached_input_tokens: "cacheRead", cache_write_input_tokens: "cacheWrite", output_tokens: "output", reasoning_tokens: "reasoning" }, { input: false, output: true });
    default: return previous;
  }
}

function terminalEvent(api: string, value: unknown): boolean {
  const event = object(value);
  if (api === "anthropic-messages") {
    return event.type === "message_stop" || object(event.delta).stop_reason != null;
  }
  if (api.includes("responses")) {
    return ["response.completed", "response.incomplete", "response.done", "response.failed"].includes(String(event.type));
  }
  if (api === "pi-messages") return event.type === "done" || event.type === "error";
  if (api === "bedrock-converse-stream") return object(event.messageStop).stopReason != null;
  if (api === "google-generative-ai" || api === "google-vertex") {
    return Array.isArray(event.candidates) && event.candidates.some(candidate => object(candidate).finishReason != null);
  }
  return Array.isArray(event.choices) && event.choices.some(choice => object(choice).finish_reason != null);
}

function literalUsage(value: unknown): UsageObservation["reported"] {
  const usage = object(value);
  const input = count(usage.input);
  const cacheRead = count(usage.cacheRead);
  const cacheWrite = count(usage.cacheWrite);
  const output = count(usage.output);
  return {
    ...(input === undefined ? {} : { input_tokens: input, input_tokens_include_cache: false }),
    ...(cacheRead === undefined ? {} : { cached_input_tokens: cacheRead }),
    ...(cacheWrite === undefined ? {} : { cache_write_input_tokens: cacheWrite }),
    ...(output === undefined ? {} : { output_tokens: output, output_tokens_include_reasoning: true }),
  };
}

function observation(call: Call): UsageObservation {
  const reported = { ...call.reported };
  // A prior attempt can have consumed tokens that the final usage does not cover.
  // No HTTP error status proves a zero-token call.
  // Pi does not expose the Bedrock SDK attempt count, including transport retries.
  // Google callbacks likewise hide retries when the provider retry loop is enabled.
  const hiddenRetries = call.api === "bedrock-converse-stream" ||
    ((call.api === "google-generative-ai" || call.api === "google-vertex") && call.maxRetries > 0);
  const retryGap = hiddenRetries || call.retryError || call.api === "pi-messages" || call.attempts.length > 1 || call.attempts.some(status => status === undefined) || call.transportFailure || call.overlappingWarmup || call.failed && !call.terminal;
  return { reported, ...(call.model === undefined ? {} : { model: call.model }), ...(call.provider === undefined ? {} : { provider: call.provider }),
    is_model_call: true, tokens_aligned: true,
    // The retained raw snapshot is internally aligned. A whole-call gap
    // preserves it for observed pricing while withholding complete totals.
    ...(!call.finished ? { gap: "unfinished_work" as const }
      : retryGap ? { gap: "scope_unverified" as const } : {}),
  };
}

class InvocationUsage {
  readonly calls: Call[] = [];
  readonly extras: UsageObservation[] = [];
  readonly toolReports = new WeakMap<object, WeakSet<object>>();
  readonly customTools: CustomToolCall[] = [];
  readonly activeCustomTools = new Map<string, CustomToolCall>();
  closed = false;
  compactionStart: number | undefined;
  snapshot(): AgentUsage {
    this.closed = true;
    return usageFromObservations([
      ...this.calls.map(observation),
      ...this.extras,
      ...this.customTools.filter(call => !call.covered).map(call => call.evidence),
    ]);
  }
}

/** Source collector only. Snapshots freeze when the owning prompt returns. */
export class PiUsageCollector {
  private readonly messages = new WeakMap<object, InvocationUsage>();
  private readonly usages = new WeakMap<object, InvocationUsage>();

  private readonly options: PiUsageOptions;
  private warmingOwner: InvocationUsage | undefined;

  constructor(agent: AgentSession["agent"], options: PiUsageOptions = {}) {
    this.options = options;
    const stream = agent.streamFunction;
    agent.streamFunction = async (model, context, options) => {
      const owner = ownerScope.getStore();
      if (owner === undefined) return stream(model, context, options);
      if (this.options.sessionId !== undefined && options?.sessionId === this.options.sessionId) {
        this.warmingOwner = owner;
      }
      // Pi applies persisted provider settings inside the wrapped stream function.
      const maxRetries = options?.maxRetries ?? this.options.providerMaxRetries?.() ?? 0;
      const initial: Call = { api: model.api, maxRetries, model: model.id, provider: accountingProvider(model.api, model.baseUrl), reported: {}, attempts: [], finished: false };
      if (!owner.closed) owner.calls.push(initial);
      let current = initial;
      let payloads = 0;
      const ownFetch = options?.fetch;
      const scope: AttemptScope = { delegate: ownFetch === undefined || ownFetch === observedFetch ? globalThis.fetch : ownFetch, current: () => current };
      try {
        const response = await attemptScope.run(scope, () => stream(model, context, {
          ...options,
          ...(FETCH_PROTOCOLS.has(model.api) ? { fetch: observedFetch } : {}),
          onPayload: async (payload, payloadModel) => {
            if (++payloads > 1) {
              // Pi replays these options for a cache warm-up outside streamFunction.
              // Refresh awaits one result before scheduling the next. After the
              // original finishes, this payload and a raw terminal event identify
              // a sequential warm-up without needing its later usage entry.
              const overlappingWarmup = !initial.finished;
              if (overlappingWarmup) initial.overlappingWarmup = true;
              current = { api: payloadModel?.api ?? model.api, maxRetries: 0, model: payloadModel?.id ?? model.id, provider: accountingProvider(payloadModel?.api ?? model.api, payloadModel === undefined ? model.baseUrl : payloadModel.baseUrl), reported: {}, attempts: [], finished: false, warmup: true, overlappingWarmup };
              if (!owner.closed) owner.calls.push(current);
            }
            if (payloads === 1 && payloadModel !== undefined && !current.routeObserved) {
              current.api = payloadModel.api;
              current.model = current.returnedModel ?? payloadModel.id;
              current.provider = accountingProvider(payloadModel.api, payloadModel.baseUrl);
            }
            return options?.onPayload?.(payload, payloadModel);
          },
          onProviderStreamEvent: async (data, eventModel) => {
            const api = eventModel?.api ?? model.api;
            if (current.routeObserved && api !== current.api) {
              current.provider = undefined;
              current.routeConflict = true;
            }
            current.api = api;
            // Codex can retry these raw WebSocket errors without a transport diagnostic.
            const event = object(data);
            const retryCode = object(object(event.response).error).code;
            if (api === "openai-codex-responses" && (event.type === "error" ||
                event.type === "response.failed" && (retryCode === "previous_response_not_found" || retryCode === "websocket_connection_limit_reached"))) {
              current.retryError = true;
            }
            current.model = current.returnedModel ?? text(eventModel?.id) ?? model.id;
            if (!current.routeObserved) {
              current.provider = accountingProvider(current.api, eventModel === undefined ? model.baseUrl : eventModel.baseUrl);
            }
            // Replayed options carry no source-call identity. During overlap,
            // raw frames from the original stream and warmer are ambiguous.
            if (payloads <= 1 || initial.finished) {
              current.reported = sourceUsage(current.api, data, current.reported);
              // Warm-ups have no final-message observer. Retain the literal
              // returned identity from the same raw source as their counts.
              const event = object(data);
              const returnedModel = current.api === "anthropic-messages"
                ? text(object(event.message).model)
                : current.api.includes("responses") ? text(object(event.response).model)
                : text(event.model);
              current.returnedModel = returnedModel ?? current.returnedModel;
              current.model = current.returnedModel ?? current.model;
              current.terminal ||= terminalEvent(current.api, data);
              if (current.warmup && current.terminal) current.finished = true;
            }
            await options?.onProviderStreamEvent?.(data, eventModel);
          },
        }));
        void response.result().then(message => {
          initial.finished = true;
          initial.failed = message.stopReason === "error" || message.stopReason === "aborted";
          initial.model = text(message.responseModel) ?? text(message.model) ?? initial.model;
          initial.transportFailure = Array.isArray(message.diagnostics) && message.diagnostics.some(diagnostic => object(diagnostic).type === "provider_transport_failure");
          this.messages.set(message, owner);
          if (typeof message.usage === "object" && message.usage !== null) this.usages.set(message.usage, owner);
        }, () => { initial.finished = initial.failed = true; });
        return response;
      } catch (error) {
        initial.finished = initial.failed = true;
        throw error;
      }
    };
  }

  async prompt<T>(run: () => Promise<T>, eventSource: Pick<AgentSession, "subscribe">): Promise<{ ok: true; result: T; usage: AgentUsage } | { ok: false; error: unknown; usage: AgentUsage }> {
    const owner = new InvocationUsage();
    const unsubscribe = eventSource.subscribe(event => this.event(owner, event));
    try {
      return { ok: true, result: await ownerScope.run(owner, run), usage: this.snapshot(owner) };
    } catch (error) {
      return { ok: false, error, usage: this.snapshot(owner) };
    } finally { unsubscribe(); }
  }

  private snapshot(owner: InvocationUsage): AgentUsage {
    // The default streaming warmer stops at agent_settled. Idle warming can
    // still schedule future owned calls or be awaiting a decision before payload.
    // Only the real session request that installed that warmer owns this gap.
    const state = object(this.options.warmingStatus?.()).state;
    if (this.warmingOwner === owner && (state === "scheduled" || state === "refreshing")) {
      owner.extras.push({ reported: {}, gap: "unfinished_work" });
    }
    return owner.snapshot();
  }

  private event(owner: InvocationUsage, value: unknown): void {
    const event = object(value);
    if (owner.closed || ownerScope.getStore() !== owner) return;
    if (event.type === "tool_execution_start" &&
        typeof event.toolName === "string" && this.options.customToolNames?.has(event.toolName)) {
      const call: CustomToolCall = { evidence: { reported: {}, gap: "scope_unverified" }, covered: false };
      owner.customTools.push(call);
      const id = text(event.toolCallId);
      if (id !== undefined) owner.activeCustomTools.set(id, call);
      return;
    }
    if (event.type === "compaction_start") {
      owner.compactionStart = owner.calls.length;
      return;
    }
    if (event.type === "compaction_end") {
      if (owner.compactionStart === undefined || owner.calls.length === owner.compactionStart) {
        owner.extras.push({ reported: literalUsage(object(event.result).usage), gap: "scope_unverified" });
      }
      owner.compactionStart = undefined;
      return;
    }
    if (event.type !== "message_end") return;
    const message = object(event.message);
    if (message.role === "assistant") {
      if (this.messages.get(message) === owner) return;
      // A custom stream/extension bypassed the observed boundary. Do not treat
      // Pi's zero-filled assistant counters as provider reports.
      owner.extras.push({ reported: {}, gap: "scope_unverified" });
    } else if (message.role === "toolResult") {
      const id = text(message.toolCallId);
      const custom = id === undefined ? undefined : owner.activeCustomTools.get(id);
      if (typeof message.usage !== "object" || message.usage === null) return;
      if (this.usages.get(message.usage) === owner) {
        if (custom !== undefined) custom.covered = true;
        return;
      }
      // Pi creates a distinct message per tool call but preserves result.usage
      // by reference. Sharing a usage object does not make two calls duplicates.
      const reports = owner.toolReports.get(message) ?? new WeakSet<object>();
      if (reports.has(message.usage)) return;
      reports.add(message.usage);
      owner.toolReports.set(message, reports);
      const evidence: UsageObservation = {
        reported: literalUsage(message.usage),
        ...(message.nestedCalls === undefined ? {} : { gap: "scope_unverified" }),
      };
      if (custom === undefined) owner.extras.push(evidence);
      else {
        custom.evidence = evidence;
        owner.activeCustomTools.delete(id!);
      }
    }
  }
}

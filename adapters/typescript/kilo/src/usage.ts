// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentUsage, JsonObject } from "nemo-fabric-adapter-contract";

const EVIDENCE_KEY = "nvidia.fabric.kilo.usage_evidence.v1";
const DETAIL_KEY = "nemo_fabric.detailed_token_usage.v1";
type Counters = Partial<Record<"input_tokens" | "output_tokens" | "total_tokens" | "cached_input_tokens" | "cache_write_input_tokens" | "reasoning_tokens", number>>;
type Gap = "missing_usage" | "scope_unverified" | "unfinished_work";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function count(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

function positive(value: unknown): number | undefined {
  const result = count(value);
  return result !== undefined && result > 0 ? result : undefined;
}

/** Kilo 7.7.12 clamps subtraction at zero; only a positive remainder is reversible. */
function inclusive(adjusted: unknown, components: unknown[]): number | undefined {
  const main = positive(adjusted);
  const values = components.map(count);
  if (main === undefined || values.some(value => value === undefined)) return undefined;
  return count(main + (values as number[]).reduce((sum, value) => sum + value, 0));
}

/** Recover inclusive SDK counters without treating Kilo's inserted zeros as reports.
 * Source: https://github.com/Kilo-Org/kilocode/blob/8e842f1c945355b38eb4ce2641645a7ea078c855/packages/opencode/src/session/session.ts#L420
 */
export function reverseStoredUsage(value: unknown): Counters {
  const tokens = isRecord(value) ? value : {};
  const cache = isRecord(tokens.cache) ? tokens.cache : {};
  const candidates: Counters = {
    input_tokens: inclusive(tokens.input, [cache.read, cache.write]),
    output_tokens: inclusive(tokens.output, [tokens.reasoning]),
    // Unlike the other stored counters, Kilo passes total through without zero-filling.
    total_tokens: count(tokens.total),
    cached_input_tokens: positive(cache.read),
    cache_write_input_tokens: positive(cache.write),
    reasoning_tokens: positive(tokens.reasoning),
  };
  return Object.fromEntries(Object.entries(candidates).filter(([, value]) => value !== undefined));
}

export function messageId(value: unknown): string | undefined {
  const info = isRecord(value) && isRecord(value.info) ? value.info : {};
  return typeof info.id === "string" ? info.id : undefined;
}

/** Kilo marks its background results and compaction continuations as synthetic. */
export function findPrompt(messages: unknown, lastId: string | undefined): string | undefined {
  if (!Array.isArray(messages)) return undefined;
  const start = lastId === undefined ? 0 : messages.findIndex(message => messageId(message) === lastId) + 1;
  if (lastId !== undefined && start === 0) return undefined;
  const prompts = messages.slice(start).filter(message => isRecord(message) && isRecord(message.info) && message.info.role === "user" &&
    Array.isArray(message.parts) && message.parts.some(part => isRecord(part) && part.type === "text" && part.synthetic !== true));
  return prompts.length === 1 ? messageId(prompts[0]) : undefined;
}

export function unknownUsage(reason: Gap): AgentUsage {
  return { extensions: { [EVIDENCE_KEY]: { observed: {}, gaps: [reason] } } };
}

interface UsageScope {
  sessionId: string;
  prompt: string;
  prompts: ReadonlySet<string>;
  retried: boolean;
  eventCoverage: boolean;
  unfinished: boolean;
}

/** Only direct prompt ownership is proven here. Child and internal turns leave a gap. */
export function extractInvocationUsage(messages: unknown[], scope: UsageScope): AgentUsage {
  const gaps = new Set<Gap>();
  let completeScope = true;
  const gap = (reason: Gap): void => { gaps.add(reason); completeScope = false; };
  if (!scope.eventCoverage) gap("scope_unverified");
  if (scope.unfinished) gap("unfinished_work");
  if (scope.retried) gap("missing_usage");
  const start = messages.findIndex(message => messageId(message) === scope.prompt);
  if (start < 0) return unknownUsage("scope_unverified");
  const steps = new Map<string, { signature: string; counters: Counters }>();
  const conflicted = new Set<string>();
  let assistants = 0;
  for (const message of messages.slice(start + 1)) {
    if (!isRecord(message) || !isRecord(message.info)) { gap("scope_unverified"); continue; }
    const info = message.info;
    if (info.role === "user") {
      if (!scope.prompts.has(messageId(message) ?? "")) gap("scope_unverified");
      continue;
    }
    if (info.role !== "assistant") continue;
    if (info.parentID !== scope.prompt) {
      if (typeof info.parentID !== "string" || !scope.prompts.has(info.parentID)) gap("scope_unverified");
      continue;
    }
    if (info.sessionID !== scope.sessionId) { gap("scope_unverified"); continue; }
    assistants++;
    if ((!isRecord(info.time) || info.time.completed === undefined) && info.error === undefined) gap("unfinished_work");
    const id = messageId(message);
    if (id === undefined) gap("missing_usage");
    const parts = Array.isArray(message.parts) ? message.parts.filter(isRecord) : [];
    if (parts.some(part => part.type === "tool" && typeof part.tool === "string" && ["task", "schedule_wakeup", "cron_create", "goal"].includes(part.tool))) gap("scope_unverified");
    const finished = parts.filter(part => part.type === "step-finish");
    if (finished.length === 0) gap("missing_usage");
    for (const part of finished) {
      if (typeof part.id !== "string") { gap("scope_unverified"); continue; }
      if (conflicted.has(part.id)) continue;
      const counters = reverseStoredUsage(part.tokens);
      const signature = JSON.stringify(counters);
      const previous = steps.get(part.id);
      if (previous !== undefined && previous.signature !== signature) {
        steps.delete(part.id);
        conflicted.add(part.id);
        gap("scope_unverified");
      } else steps.set(part.id, { signature, counters });
    }
  }
  if (assistants === 0 || steps.size === 0) gap("missing_usage");
  const observed: Counters = {};
  const complete: Counters = {};
  const dimensions = ["input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "cache_write_input_tokens", "reasoning_tokens"] as const;
  for (const dimension of dimensions) {
    const values = [...steps.values()].map(step => step.counters[dimension]);
    const reported = values.filter((value): value is number => value !== undefined);
    const sum = count(reported.reduce((total, value) => total + value, 0));
    if (reported.length > 0 && sum !== undefined) observed[dimension] = sum;
    if (completeScope && values.length > 0 && reported.length === values.length && sum !== undefined) complete[dimension] = sum;
    if ((dimension === "input_tokens" || dimension === "output_tokens") && (reported.length < values.length || sum === undefined)) gaps.add("missing_usage");
  }
  const details: JsonObject = {};
  if (complete.input_tokens !== undefined) Object.assign(details, { input_tokens: complete.input_tokens, input_tokens_include_cache: true });
  if (complete.output_tokens !== undefined) Object.assign(details, { output_tokens: complete.output_tokens, output_tokens_include_reasoning: true });
  for (const dimension of ["cached_input_tokens", "cache_write_input_tokens", "reasoning_tokens"] as const) {
    if (complete[dimension] !== undefined) details[dimension] = complete[dimension];
  }
  return {
    ...(complete.input_tokens === undefined ? {} : { input_tokens: complete.input_tokens }),
    ...(complete.output_tokens === undefined ? {} : { output_tokens: complete.output_tokens }),
    ...(complete.total_tokens === undefined ? {} : { total_tokens: complete.total_tokens }),
    extensions: { [EVIDENCE_KEY]: { observed, gaps: [...gaps] }, ...(Object.keys(details).length === 0 ? {} : { [DETAIL_KEY]: details }) },
  };
}

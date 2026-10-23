// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentUsage } from "./generated/agent-run-result.js";
import type { AccountedModel, ReportedUsage, UsageAccounting, UsageDimension, UsageGap } from "./generated/usage-accounting.js";
import type { JsonObject } from "./json.js";

export const USAGE_ACCOUNTING_KEY = "nemo_fabric.usage_accounting.v1";
const DETAIL_KEY = "nemo_fabric.detailed_token_usage.v1";
type Reason = UsageGap["reason"];
const dimensions: UsageDimension[] = ["input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "peak_request_input_tokens", "cost_usd"];

/** One deduplicated source snapshot. Core owns normalization and pricing. */
export interface UsageObservation {
  reported: ReportedUsage;
  model?: string;
  provider?: string;
  gap?: Reason;
  tokens_aligned?: boolean;
  is_model_call?: boolean;
  gaps?: UsageGap[];
}

function validate(record: UsageObservation): void {
  for (const key of dimensions) {
    const value = record.reported[key];
    if (value == null) continue;
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0 ||
        (key === "cost_usd" ? value * 1e9 >= 2 ** 64 : !Number.isSafeInteger(value))) {
      throw new RangeError(`invalid usage ${key}`);
    }
  }
  for (const key of ["input_tokens_include_cache", "output_tokens_include_reasoning"] as const) {
    if (record.reported[key] != null && typeof record.reported[key] !== "boolean") throw new TypeError(`invalid usage ${key}`);
  }
  for (const value of [record.model, record.provider]) {
    if (value !== undefined && (typeof value !== "string" || !value.trim())) throw new TypeError("invalid usage identity");
  }
  if (record.gap !== undefined && !["unfinished_work", "missing_usage", "scope_unverified"].includes(record.gap)) throw new TypeError("invalid usage gap");
  for (const gap of record.gaps ?? []) {
    if (!gap.dimensions.length || new Set(gap.dimensions).size !== gap.dimensions.length ||
        gap.dimensions.some(dimension => !dimensions.includes(dimension)) ||
        !["unfinished_work", "missing_usage", "scope_unverified"].includes(gap.reason)) throw new TypeError("invalid dimension gap");
  }
}

// Keep low-order cost contributions until the final round, matching math.fsum.
// Inputs have already passed the finite, nonnegative cost validation.
function sumCosts(values: readonly number[]): number {
  const partials: number[] = [];
  for (let high of values) {
    let retained = 0;
    for (let low of partials) {
      if (Math.abs(high) < Math.abs(low)) [high, low] = [low, high];
      const sum = high + low;
      const residual = low - (sum - high);
      if (residual !== 0) partials[retained++] = residual;
      high = sum;
    }
    partials.length = retained;
    partials.push(high);
  }
  let index = partials.length;
  let total = index === 0 ? 0 : partials[--index]!;
  let residual = 0;
  while (index > 0) {
    const high = total;
    const low = partials[--index]!;
    total = high + low;
    residual = low - (total - high);
    if (residual !== 0) break;
  }
  // Preserve round-to-even when the remaining low-order terms share a sign.
  if (index > 0 && ((residual < 0 && partials[index - 1]! < 0) ||
      (residual > 0 && partials[index - 1]! > 0))) {
    const correction = residual * 2;
    const rounded = total + correction;
    if (correction === rounded - total) total = rounded;
  }
  return total;
}

function reduce(records: readonly UsageObservation[]): { reported: ReportedUsage; observed: ReportedUsage; gaps: UsageGap[] } {
  const reported: ReportedUsage = {};
  const observed: ReportedUsage = {};
  const gaps = new Map<Reason, Set<UsageDimension>>();
  const gap = (reason: Reason, dimension: UsageDimension): void => {
    const fields = gaps.get(reason) ?? new Set<UsageDimension>();
    fields.add(dimension);
    gaps.set(reason, fields);
  };
  for (const key of dimensions) {
    const values = records.map(record => record.reported[key]).filter((value): value is number => value != null);
    let value: number | undefined;
    if (values.length) {
      if (key === "cost_usd") {
        const total = sumCosts(values);
        if (Number.isFinite(total) && total * 1e9 < 2 ** 64) value = total;
      } else if (key === "peak_request_input_tokens") {
        value = values.reduce((maximum, item) => Math.max(maximum, item));
      } else {
        const total = values.reduce((sum, item) => sum + BigInt(item), 0n);
        if (total <= BigInt(Number.MAX_SAFE_INTEGER)) value = Number(total);
      }
    }
    if (value !== undefined) observed[key] = value;
    const causes = new Set(records.flatMap(record => record.gap === undefined ? [] : [record.gap]));
    for (const record of records) for (const item of record.gaps ?? []) {
      if (item.dimensions.includes(key)) causes.add(item.reason);
    }
    if (values.length !== records.length) causes.add("missing_usage");
    if (values.length && value === undefined) causes.add("scope_unverified");
    if (value !== undefined && !causes.size) reported[key] = value;
    else if (!causes.size) causes.add("missing_usage");
    for (const reason of causes) gap(reason, key);
  }
  for (const [flag, counter] of [["input_tokens_include_cache", "input_tokens"], ["output_tokens_include_reasoning", "output_tokens"]] as const) {
    const flags = new Set(records.filter(record => record.reported[counter] != null).map(record => record.reported[flag]));
    const value = flags.values().next().value;
    if (flags.size === 1 && value != null) {
      observed[flag] = value;
      if (reported[counter] !== undefined) reported[flag] = value;
    }
    else if (records.filter(record => record.reported[counter] != null).length === 1 && flags.size === 1 && value == null) {
      // Keep a single literal source even when its convention is unknown.
      // Core decides whether a normalized value can be established.
    } else if (observed[counter] !== undefined) {
      delete reported[counter];
      delete observed[counter];
      if (!records.every(record => record.is_model_call === true && record.model !== undefined)) gap("scope_unverified", counter);
    }
  }
  return { reported, observed, gaps: [...gaps].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([reason, fields]) => ({ reason, dimensions: [...fields] as UsageGap["dimensions"] })) };
}

function aligned(records: readonly UsageObservation[]): boolean {
  const fields: UsageDimension[] = ["input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens", "peak_request_input_tokens"];
  if (records.some(record => record.reported.output_tokens_include_reasoning === false)) fields.push("reasoning_tokens");
  const relevant = records.filter(record => fields.some(field => record.reported[field] != null));
  return relevant.length > 0 && relevant.every(record => record.tokens_aligned === true && !record.gaps?.length) &&
    fields.every(field => new Set(relevant.map(record => record.reported[field] != null)).size === 1);
}

function requestUsage(record: UsageObservation): ReportedUsage {
  // Retain uncertain literal counts in model observations, not complete calls.
  const fields: ReportedUsage = {};
  const keys = [...dimensions, "input_tokens_include_cache", "output_tokens_include_reasoning"] as const;
  for (const key of keys) {
    const value = record.reported[key];
    if (value !== undefined) (fields as Record<string, unknown>)[key] = value;
  }
  for (const gap of record.gaps ?? []) for (const dimension of gap.dimensions) delete fields[dimension];
  return fields;
}

function textOrder(a: string, b: string): number {
  const left = Array.from(a, value => value.codePointAt(0)!);
  const right = Array.from(b, value => value.codePointAt(0)!);
  for (let i = 0; i < Math.min(left.length, right.length); i++) {
    if (left[i] !== right[i]) return left[i]! - right[i]!;
  }
  return left.length - right.length;
}

/** Preserve independent raw dimensions; never infer zero or change scheduling.
 * Collectors deduplicate source identities before calling this reducer.
 */
export function usageFromObservations(records: readonly UsageObservation[]): AgentUsage {
  records.forEach(validate);
  const aggregate = reduce(records);
  const groups = new Map<string, UsageObservation[]>();
  for (const record of records) {
    if (record.model === undefined) continue;
    const key = JSON.stringify([record.provider ?? null, record.model]);
    const group = groups.get(key) ?? [];
    group.push(record);
    groups.set(key, group);
  }
  const models: AccountedModel[] = [...groups.values()].sort((a, b) =>
    textOrder(a[0]!.provider ?? "", b[0]!.provider ?? "") || textOrder(a[0]!.model!, b[0]!.model!))
    .map(group => ({ model: group[0]!.model!, ...(group[0]!.provider === undefined ? {} : { provider: group[0]!.provider! }),
      ...reduce(group), observed_tokens_aligned: aligned(group),
      requests_exhaustive: group.every(record => record.is_model_call === true),
      ...(group.every(record => record.is_model_call === true) ? { requests: group.map(record => ({ usage: requestUsage(record),
        tokens_aligned: record.tokens_aligned === true, ...(record.gap !== undefined ? { gap: record.gap } : {}) })) } : {}) }));
  const accounting: UsageAccounting = { models_exhaustive: records.length > 0 && records.every(record => record.model !== undefined && record.gap === undefined),
    observed: aggregate.observed, gaps: aggregate.gaps, ...(models.length ? { models } : {}) };
  const base = ["input_tokens", "cached_input_tokens", "input_tokens_include_cache", "output_tokens", "total_tokens", "cost_usd"];
  const detail = ["input_tokens", "cached_input_tokens", "cache_write_input_tokens", "input_tokens_include_cache", "output_tokens", "reasoning_tokens", "output_tokens_include_reasoning", "peak_request_input_tokens"];
  const select = (keys: string[]): JsonObject => Object.fromEntries(Object.entries(aggregate.reported).filter(([key]) => keys.includes(key))) as JsonObject;
  return { ...select(base), extensions: { [USAGE_ACCOUNTING_KEY]: accounting as unknown as JsonObject, [DETAIL_KEY]: select(detail) } };
}

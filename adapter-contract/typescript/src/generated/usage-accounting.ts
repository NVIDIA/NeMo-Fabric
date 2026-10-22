// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical adapter-contract JSON Schemas.
// Do not edit it directly; run `npm run generate` instead.

/**
 * A dimension in raw adapter evidence.
 */
export type UsageDimension =
  | "input_tokens"
  | "cached_input_tokens"
  | "cache_write_input_tokens"
  | "output_tokens"
  | "reasoning_tokens"
  | "total_tokens"
  | "peak_request_input_tokens"
  | "cost_usd";
/**
 * Categories that affect whether a consumer can treat a value as complete.
 */
export type GapReason = "unfinished_work" | "missing_usage" | "scope_unverified";

/**
 * The declared adapter transport for invocation accounting.
 */
export interface UsageAccounting {
  /**
   * Invocation-wide gaps.
   */
  gaps: UsageGap[];
  /**
   * One entry per literal provider/model pair.
   */
  models?: AccountedModel[];
  /**
   * Every model identity used by owned work is represented; this alone does
   * not certify that any dimension is complete.
   */
  models_exhaustive: boolean;
  /**
   * Valid deduplicated observations, not a promised mathematical lower bound.
   */
  observed?: ReportedUsage | null;
}
/**
 * Missing evidence for one or more raw dimensions.
 */
export interface UsageGap {
  /**
   * Nonempty, unique set of affected dimensions.
   *
   * @minItems 1
   */
  dimensions: [UsageDimension, ...UsageDimension[]];
  /**
   * Why those dimensions are incomplete.
   */
  reason: "unfinished_work" | "missing_usage" | "scope_unverified";
}
/**
 * Evidence for one literal provider/model identity.
 */
export interface AccountedModel {
  /**
   * Gaps affecting this model.
   */
  gaps: UsageGap[];
  /**
   * Model identity as reported by the source.
   */
  model: string;
  /**
   * Valid deduplicated observations, potentially incomplete.
   */
  observed?: ReportedUsage | null;
  /**
   * Every present observed token dimension describes the same source-call subset.
   * False unless the producer establishes this, including peak and normalization components.
   */
  observed_tokens_aligned?: boolean;
  /**
   * Literal provider identity, when available.
   */
  provider?: string | null;
  reported?: ReportedUsage1;
  /**
   * Optional per-call evidence for exact request-tier pricing.
   */
  requests?: UsageRequest[];
  /**
   * The request list covers every source unit contributing to this model.
   */
  requests_exhaustive?: boolean;
}
/**
 * Raw counts in provider/harness conventions. Missing values are never zero.
 */
export interface ReportedUsage {
  /**
   * Cache writes, independently reported.
   */
  cache_write_input_tokens?: number | null;
  /**
   * Cache reads, independently reported.
   */
  cached_input_tokens?: number | null;
  /**
   * Provider-reported USD, never a client-side price estimate.
   */
  cost_usd?: number | null;
  /**
   * Raw input count.
   */
  input_tokens?: number | null;
  /**
   * Whether input includes both cache categories.
   */
  input_tokens_include_cache?: boolean | null;
  /**
   * Raw output count.
   */
  output_tokens?: number | null;
  /**
   * Whether output includes reasoning.
   */
  output_tokens_include_reasoning?: boolean | null;
  /**
   * Largest request's cache-inclusive input.
   */
  peak_request_input_tokens?: number | null;
  /**
   * Reasoning count, independently reported.
   */
  reasoning_tokens?: number | null;
  /**
   * Provider-reported total, independent of normalized input plus output.
   */
  total_tokens?: number | null;
}
/**
 * Raw counts in provider/harness conventions. Missing values are never zero.
 */
export interface ReportedUsage1 {
  /**
   * Cache writes, independently reported.
   */
  cache_write_input_tokens?: number | null;
  /**
   * Cache reads, independently reported.
   */
  cached_input_tokens?: number | null;
  /**
   * Provider-reported USD, never a client-side price estimate.
   */
  cost_usd?: number | null;
  /**
   * Raw input count.
   */
  input_tokens?: number | null;
  /**
   * Whether input includes both cache categories.
   */
  input_tokens_include_cache?: boolean | null;
  /**
   * Raw output count.
   */
  output_tokens?: number | null;
  /**
   * Whether output includes reasoning.
   */
  output_tokens_include_reasoning?: boolean | null;
  /**
   * Largest request's cache-inclusive input.
   */
  peak_request_input_tokens?: number | null;
  /**
   * Reasoning count, independently reported.
   */
  reasoning_tokens?: number | null;
  /**
   * Provider-reported total, independent of normalized input plus output.
   */
  total_tokens?: number | null;
}
/**
 * One model call retained for request-tier pricing and evidence reconciliation.
 */
export interface UsageRequest {
  /**
   * Known incompleteness of this call's source coverage.
   */
  gap?: GapReason | null;
  /**
   * Present observed fields describe the same request/subset.
   */
  tokens_aligned?: boolean;
  usage: ReportedUsage2;
}
/**
 * Raw counts in provider/harness conventions. Missing values are never zero.
 */
export interface ReportedUsage2 {
  /**
   * Cache writes, independently reported.
   */
  cache_write_input_tokens?: number | null;
  /**
   * Cache reads, independently reported.
   */
  cached_input_tokens?: number | null;
  /**
   * Provider-reported USD, never a client-side price estimate.
   */
  cost_usd?: number | null;
  /**
   * Raw input count.
   */
  input_tokens?: number | null;
  /**
   * Whether input includes both cache categories.
   */
  input_tokens_include_cache?: boolean | null;
  /**
   * Raw output count.
   */
  output_tokens?: number | null;
  /**
   * Whether output includes reasoning.
   */
  output_tokens_include_reasoning?: boolean | null;
  /**
   * Largest request's cache-inclusive input.
   */
  peak_request_input_tokens?: number | null;
  /**
   * Reasoning count, independently reported.
   */
  reasoning_tokens?: number | null;
  /**
   * Provider-reported total, independent of normalized input plus output.
   */
  total_tokens?: number | null;
}

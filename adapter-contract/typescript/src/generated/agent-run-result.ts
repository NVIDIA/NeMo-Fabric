// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical adapter-contract JSON Schemas.
// Do not edit it directly; run `npm run generate` instead.

import type { JsonObject, JsonValue } from "../json.js";

/** Preview southbound terminal result. */
export type AgentRunResult =
  | AgentRunSucceeded
  | AgentRunFailed
  | AgentRunCancelled;

/** Successful terminal result. Successful results can carry only a null error. */
export interface AgentRunSucceeded extends AgentRunResultCommon {
  status: "succeeded";
  error?: null;
}

/** Failed terminal result. Failed results must carry a non-null error. */
export interface AgentRunFailed extends AgentRunResultCommon {
  status: "failed";
  error: AgentRunError;
}

/** Cancelled terminal result. Cancellation details are optional. */
export interface AgentRunCancelled extends AgentRunResultCommon {
  status: "cancelled";
  error?: AgentRunError | null;
}

/**
 * Fields shared by every preview terminal result variant.
 */
export interface AgentRunResultCommon {
  /**
   * Artifacts produced by the adapter target.
   */
  artifacts?: AgentArtifact[];
  /**
   * Adapter-owned result fields.
   */
  extensions?: JsonObject;
  /**
   * Primary adapter-target output.
   */
  output: JsonValue;
  /**
   * Normalized model usage when reported by the adapter target.
   */
  usage?: AgentUsage | null;
}
/**
 * One artifact produced by an adapter target.
 *
 * This interface was referenced by `AgentRunResultCommon`'s JSON-Schema
 * via the `definition` "AgentArtifact".
 */
export interface AgentArtifact {
  /**
   * Adapter-owned artifact fields.
   */
  extensions?: JsonObject;
  /**
   * Artifact kind.
   */
  kind: string;
  /**
   * Optional media type.
   */
  media_type?: string | null;
  /**
   * Logical artifact name.
   */
  name: string;
  /**
   * Path relative to the artifact root supplied in `RuntimeContext`.
   */
  path: string;
}
/**
 * Normalized model usage reported by an adapter target.
 *
 * This interface was referenced by `AgentRunResultCommon`'s JSON-Schema
 * via the `definition` "AgentUsage".
 */
export interface AgentUsage {
  /**
   * Input tokens written to the provider prompt cache, when reported.
   */
  cache_write_input_tokens?: number | null;
  /**
   * Cached input tokens consumed by the invocation, when reported.
   */
  cached_input_tokens?: number | null;
  /**
   * Invocation cost in US dollars when reported by the provider.
   */
  cost_usd?: number | null;
  /**
   * Adapter-owned usage fields.
   */
  extensions?: JsonObject;
  /**
   * Input tokens consumed by the invocation.
   */
  input_tokens?: number | null;
  /**
   * Whether input_tokens already includes cached_input_tokens and
   * cache_write_input_tokens; absent means unknown.
   */
  input_tokens_include_cache?: boolean | null;
  /**
   * Usage broken out by the model that served the requests; empty when not
   * reported. Each provider and model pair appears at most once, compared
   * case-insensitively.
   */
  models?: AgentModelUsage[];
  /**
   * Output tokens produced by the invocation.
   */
  output_tokens?: number | null;
  /**
   * Whether output_tokens already includes reasoning_tokens; absent means unknown.
   */
  output_tokens_include_reasoning?: boolean | null;
  /**
   * Input tokens of the largest single model request in the invocation,
   * including cached and cache-write input, when reported.
   */
  peak_request_input_tokens?: number | null;
  /**
   * Reasoning tokens produced by the invocation, when reported.
   */
  reasoning_tokens?: number | null;
  /**
   * Total tokens reported by the provider.
   */
  total_tokens?: number | null;
}
/**
 * Usage of one model within one invocation.
 *
 * Token fields have the same meaning as the matching `AgentUsage` fields,
 * restricted to requests served by this model.
 *
 * This interface was referenced by `AgentRunResultCommon`'s JSON-Schema
 * via the `definition` "AgentModelUsage".
 */
export interface AgentModelUsage {
  /**
   * Input tokens this model wrote to the provider prompt cache, when reported.
   */
  cache_write_input_tokens?: number | null;
  /**
   * Cached input tokens consumed by this model, when reported.
   */
  cached_input_tokens?: number | null;
  /**
   * Cost in US dollars for this model when reported by the provider.
   */
  cost_usd?: number | null;
  /**
   * Input tokens consumed by this model.
   */
  input_tokens?: number | null;
  /**
   * Whether input_tokens already includes cached_input_tokens and
   * cache_write_input_tokens; absent means unknown.
   */
  input_tokens_include_cache?: boolean | null;
  /**
   * Model identifier as reported by the harness or provider response.
   */
  model: string;
  /**
   * Output tokens produced by this model.
   */
  output_tokens?: number | null;
  /**
   * Whether output_tokens already includes reasoning_tokens; absent means unknown.
   */
  output_tokens_include_reasoning?: boolean | null;
  /**
   * Input tokens of the largest single request served by this model,
   * including cached and cache-write input, when reported.
   */
  peak_request_input_tokens?: number | null;
  /**
   * Provider or route that served the model, when known.
   */
  provider?: string | null;
  /**
   * Reasoning tokens produced by this model, when reported.
   */
  reasoning_tokens?: number | null;
  /**
   * Total tokens reported by the provider for this model.
   */
  total_tokens?: number | null;
}
/**
 * Error reported by an adapter target.
 *
 * This interface was referenced by `AgentRunResultCommon`'s JSON-Schema
 * via the `definition` "AgentRunError".
 */
export interface AgentRunError {
  /**
   * Stable adapter error code.
   */
  code: string;
  /**
   * Adapter-owned error fields.
   */
  extensions?: JsonObject;
  /**
   * Human-readable error message.
   */
  message: string;
  /**
   * Whether the adapter considers the failure safe for a consumer-level retry.
   */
  retryable?: boolean;
}

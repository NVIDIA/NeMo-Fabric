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

/** Successful terminal result. Successful results cannot carry an error. */
export interface AgentRunSucceeded extends AgentRunResultCommon {
  status: "succeeded";
  error?: never;
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
   * Output tokens produced by the invocation.
   */
  output_tokens?: number | null;
  /**
   * Total tokens reported by the provider.
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

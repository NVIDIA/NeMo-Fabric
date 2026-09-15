// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical adapter-contract JSON Schemas.
// Do not edit it directly; run `npm run generate` instead.

import type { JsonObject } from "../json.js";

/**
 * Optional adapter-specific contribution to a runtime health report.
 */
export interface AdapterHealthResult {
  /**
   * Adapter or dependency checks.
   */
  checks?: HealthCheck[];
  /**
   * Adapter-owned readiness override when one is known.
   */
  readiness?: AdapterReadiness | null;
}
/**
 * One timestamped runtime or adapter health observation.
 */
export interface HealthCheck {
  /**
   * Age of the evidence when this report was assembled.
   */
  age_millis: number;
  /**
   * Optional human-readable diagnostic detail.
   */
  message?: string | null;
  /**
   * Additional non-sensitive check metadata.
   */
  metadata?: JsonObject;
  /**
   * Stable, namespaced check name.
   */
  name: string;
  /**
   * Unix timestamp in milliseconds when the evidence was observed.
   */
  observed_at_millis: number;
  /**
   * Stable machine-readable reason.
   */
  reason_code: string;
  /**
   * Structured check outcome.
   */
  status: "ok" | "failed" | "unknown" | "unsupported";
}
/**
 * Adapter-owned readiness observation.
 */
export interface AdapterReadiness {
  /**
   * Stable machine-readable reason.
   */
  reason_code: string;
  /**
   * Adapter readiness state.
   */
  state: "ready" | "not_ready" | "unknown";
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical adapter-contract JSON Schemas.
// Do not edit it directly; run `npm run generate` instead.

/**
 * Diagnostic codes for health observations.
 */
export type CheckReason =
  | "server_stopping"
  | "capabilities_unverified"
  | "runtime_unavailable"
  | "agent_exited"
  | "session_closed"
  | "endpoint_unavailable"
  | "health_failed"
  | "probe_timeout"
  | "probe_error"
  | "observation_changed"
  | "unsupported_adapter"
  | "unsupported_level"
  | "unsupported_scope";

/**
 * Result of one adapter health probe.
 */
export interface AdapterCheckResult {
  /**
   * Current agent activity.
   */
  activity: "idle" | "busy" | "stopping" | "stopped" | "unknown";
  /**
   * Native observation time in UTC, or null if no observation was made.
   */
  observed_at: string | null;
  /**
   * Probe outcome.
   */
  outcome: "passed" | "failed" | "unknown" | "unsupported";
  /**
   * Fixed diagnostic or null on pass.
   */
  reason_code: CheckReason | null;
  /**
   * Fixed evidence source.
   */
  source:
    | "fabric.runtime_server"
    | "fabric.runtime_server.transport"
    | "fabric.adapter"
    | "fabric.adapter.pi.session"
    | "fabric.adapter.pi.interface"
    | "fabric.adapter.pi.setup";
}

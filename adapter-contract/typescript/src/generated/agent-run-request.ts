// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical adapter-contract JSON Schemas.
// Do not edit it directly; run `npm run generate` instead.

import type { JsonObject, JsonValue } from "../json.js";

/**
 * Preview southbound invocation request.
 *
 * The current local-host transport does not enforce this type. It will join
 * the negotiated adapter contract when typed invoke transport is implemented.
 */
export interface AgentRunRequest {
  /**
   * Caller-provided task, rollout, workflow, or application context.
   */
  context?: JsonObject;
  /**
   * Adapter-owned request fields.
   */
  extensions?: JsonObject;
  /**
   * Request payload for the adapter target.
   */
  input: JsonValue;
}

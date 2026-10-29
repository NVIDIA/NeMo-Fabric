// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical NeMo Fabric JSON Schemas.
// Do not edit it directly; run `npm run generate` instead.

/**
 * One adapter health probe using the remaining shared deadline budget.
 */
export interface AdapterCheckRequest {
  /**
   * One native level to observe.
   */
  level: "live" | "active" | "ready";
  /**
   * Exact adapter host selected by the core, including service host IDs.
   */
  runtime_id: string;
  /**
   * Remaining shared deadline budget.
   */
  timeout_seconds: number;
}

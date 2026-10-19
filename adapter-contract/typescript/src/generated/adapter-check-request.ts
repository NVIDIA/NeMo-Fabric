// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical adapter-contract JSON Schemas.
// Do not edit it directly; run `npm run generate` instead.

/**
 * One adapter health probe using the remaining shared deadline budget.
 */
export interface AdapterCheckRequest {
  /**
   * One native level to observe.
   */
  level: "live" | "active" | "ready" | "operational";
  /**
   * Exact runtime selected by the core.
   */
  runtime_id: string;
  /**
   * Remaining shared deadline budget.
   */
  timeout_seconds: number;
}

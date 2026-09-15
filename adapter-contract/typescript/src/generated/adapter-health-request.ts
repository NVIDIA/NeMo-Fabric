// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical adapter-contract JSON Schemas.
// Do not edit it directly; run `npm run generate` instead.

/**
 * Request passed to an optional adapter health hook.
 */
export interface AdapterHealthRequest {
  /**
   * Runtime being inspected.
   */
  runtime_id: string;
  /**
   * Remaining health budget in milliseconds.
   */
  timeout_millis: number;
}

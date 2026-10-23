// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { UsageAccounting } from "../src/index.js";
import { accountingVectors } from "./generated/usage-accounting-vectors.js";

for (const vector of accountingVectors) {
  const accounting: UsageAccounting = vector.accounting;
  void accounting;
}

// @ts-expect-error a producer must state whether model identities are exhaustive
const missingCoverage: UsageAccounting = { gaps: [] };
// @ts-expect-error unknown reasons cannot silently enter the supported contract
const invalidReason: UsageAccounting = { models_exhaustive: false, gaps: [{ dimensions: ["input_tokens"], reason: "probably_complete" }] };
void missingCoverage;
void invalidReason;

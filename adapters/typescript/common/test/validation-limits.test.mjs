// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { z } from "zod";

test("rejects oversized constrained arrays before accumulating element errors", () => {
  const result = z.array(z.string()).max(1).safeParse([1, 2, 3, 4]);
  assert.equal(result.success, false);
  assert.equal(result.error.issues.length, 1);
  assert.equal(result.error.issues[0].code, "too_big");
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("published accounting schema is byte-identical to the canonical schema", async () => {
  const canonical = await readFile(new URL("../../../schemas/extensions/usage-accounting.schema.json", import.meta.url));
  const packaged = await readFile(new URL("../schemas/usage-accounting.schema.json", import.meta.url));
  assert.deepEqual(packaged, canonical);
});

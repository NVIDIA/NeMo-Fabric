// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import test from "node:test";
import { fileURLToPath } from "node:url";

test("native OpenCode session inspection stays available while a model response is blocked", { timeout: 45_000 }, async () => {
  const { stdout } = await promisify(execFile)(process.env.BUN_EXECUTABLE ?? "bun", [fileURLToPath(new URL("./fixtures/health-native.mjs", import.meta.url))], { timeout: 40_000 });
  assert.match(stdout, /native OpenCode health passed/);
});

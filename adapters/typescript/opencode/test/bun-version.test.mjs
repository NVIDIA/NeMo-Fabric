// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { assertSupportedBunVersion } from "../dist/bun-version.js";

test("accepts Bun versions that satisfy the OpenCode package requirement", () => {
  assert.doesNotThrow(() => assertSupportedBunVersion("1.4.2", ">=1.4.2"));
  assert.doesNotThrow(() => assertSupportedBunVersion("1.5.0", ">=1.4.2"));
});

test("reports an unsupported Bun version with a stable lifecycle error", () => {
  for (const version of ["1.4.1", "1.4.2-rc.1"]) {
    assert.throws(
      () => assertSupportedBunVersion(version, ">=1.4.2"),
      (error) =>
        error.code === "opencode_bun_version_unsupported" &&
        error.message === `The OpenCode adapter requires Bun >=1.4.2; current runtime is ${version}`,
    );
  }
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  assertAdapterSchemaInventory,
  assertRunResultConditionals,
} from "../scripts/projection-guards.mjs";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const runResultSchema = JSON.parse(
  await readFile(
    resolve(
      packageRoot,
      "../..",
      "schemas/adapter-contract/agent-run-result.schema.json",
    ),
    "utf8",
  ),
);
const currentConditionals = runResultSchema.allOf;

test("result projection rejects an unhandled conditional constraint", () => {
  const changed = structuredClone(currentConditionals);
  changed[0].then.required.push("usage");

  assert.throws(
    () => assertRunResultConditionals(changed),
    /AgentRunResult conditionals changed/,
  );
});

test("result projection accepts the exact supported conditional shape", () => {
  assert.doesNotThrow(() => assertRunResultConditionals(currentConditionals));
});

test("schema inventory rejects an unhandled canonical schema", () => {
  assert.throws(
    () =>
      assertAdapterSchemaInventory(
        ["adapter-descriptor.schema.json", "new-contract.schema.json"],
        ["adapter-descriptor.schema.json"],
      ),
    /schema inventory changed/,
  );
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  assertAdapterSchemaInventory,
  assertEvidenceConditionals,
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

const evidenceSchema = JSON.parse(await readFile(
  resolve(packageRoot, "../..", "schemas/adapter-contract/execution-evidence.schema.json"), "utf8",
));

test("software projection accepts the canonical observation and coverage conditions", () => {
  assert.doesNotThrow(() => assertEvidenceConditionals(evidenceSchema.$defs));
});

test("software projection rejects unhandled observation or coverage constraints", () => {
  for (const name of ["VersionObservation", "SoftwareEnvironment"]) {
    const changed = structuredClone(evidenceSchema.$defs);
    changed[name].allOf.push({ required: ["new_evidence"] });
    assert.throws(() => assertEvidenceConditionals(changed), /ExecutionEvidence conditionals changed/);
  }
});

test("result projection rejects an unhandled conditional constraint", () => {
  const changed = structuredClone(currentConditionals);
  const failedConditional = changed.find(
    (entry) => entry.if?.properties?.status?.const === "failed",
  );
  assert.ok(failedConditional, "expected a failed-status conditional");
  failedConditional.then.required.push("usage");

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

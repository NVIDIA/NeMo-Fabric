// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isDeepStrictEqual } from "node:util";

const supportedRunResultConditionals = [
  {
    if: {
      properties: { status: { const: "failed" } },
      required: ["status"],
    },
    then: {
      properties: { error: { $ref: "#/$defs/AgentRunError" } },
      required: ["error"],
    },
  },
  {
    if: {
      properties: { status: { const: "succeeded" } },
      required: ["status"],
    },
    then: { properties: { error: { type: "null" } } },
  },
];

export function assertRunResultConditionals(actual) {
  if (!isDeepStrictEqual(actual, supportedRunResultConditionals)) {
    throw new Error(
      "AgentRunResult conditionals changed; update the discriminated union projection",
    );
  }
}

export function assertEvidenceConditionals(definitions) {
  const nonempty = { type: "string", minLength: 1, pattern: "\\S" };
  const available = { properties: { status: { const: "available" } }, required: ["status"] };
  const expectedVersion = [{
    if: available,
    then: { properties: { value: nonempty, reason: { type: "null" } }, required: ["value"] },
    else: { properties: { value: { type: "null" }, reason: nonempty }, required: ["reason"] },
  }];
  const expectedInventory = [{
    if: available,
    else: { properties: { complete: { const: false }, entries: { maxItems: 0 } } },
  }, {
    if: { anyOf: [
      { properties: { status: { not: { const: "available" } } }, required: ["status"] },
      { properties: { complete: { const: false } }, required: ["complete"] },
    ] },
    then: { properties: { reason: nonempty }, required: ["reason"] },
  }];
  if (!isDeepStrictEqual(definitions?.VersionObservation?.allOf, expectedVersion)
      || !isDeepStrictEqual(definitions?.SoftwareEnvironment?.allOf, expectedInventory)) {
    throw new Error("ExecutionEvidence conditionals changed; update the discriminated union projection");
  }
}

export function assertAdapterSchemaInventory(actual, expected) {
  const actualSorted = [...actual].sort();
  const expectedSorted = [...expected].sort();
  if (!isDeepStrictEqual(actualSorted, expectedSorted)) {
    throw new Error(
      `Adapter-contract schema inventory changed:\n  expected: ${expectedSorted.join(", ")}\n  actual: ${actualSorted.join(", ")}\nUpdate the TypeScript schema projection explicitly.`,
    );
  }
}

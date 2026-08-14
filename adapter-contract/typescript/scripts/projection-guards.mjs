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
    then: { not: { required: ["error"] } },
  },
];

export function assertRunResultConditionals(actual) {
  if (!isDeepStrictEqual(actual, supportedRunResultConditionals)) {
    throw new Error(
      "AgentRunResult conditionals changed; update the discriminated union projection",
    );
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

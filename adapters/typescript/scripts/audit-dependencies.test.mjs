// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { checkAudit } from "./audit-dependencies.mjs";

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const lockfile = JSON.parse(
  readFileSync(join(packageRoot, "package-lock.json"), "utf8"),
);
const piPath =
  "node_modules/@earendil-works/pi-coding-agent/node_modules/brace-expansion";
const report = {
  metadata: {
    vulnerabilities: { low: 10, moderate: 18, high: 1, critical: 0 },
  },
  vulnerabilities: {
    "brace-expansion": {
      name: "brace-expansion",
      severity: "high",
      isDirect: false,
      nodes: [piPath],
      via: [
        {
          severity: "moderate",
          url: "https://github.com/advisories/GHSA-q2hr-2g5m-vwhr",
        },
        {
          severity: "high",
          url: "https://github.com/advisories/GHSA-qhr7-859c-m2p7",
        },
        {
          severity: "high",
          url: "https://github.com/advisories/GHSA-6j4f-fj2g-mc7p",
        },
      ],
    },
  },
};

test("allows only the known Pi shrinkwrap finding", () => {
  assert.equal(checkAudit(report, lockfile).high, 1);
});

test("rejects another high-severity package", () => {
  const changed = structuredClone(report);
  changed.metadata.vulnerabilities.high = 2;
  changed.vulnerabilities.other = { name: "other", severity: "high" };
  assert.throws(() => checkAudit(changed, lockfile));
});

test("rejects a new high-severity advisory on brace-expansion", () => {
  const changed = structuredClone(report);
  changed.vulnerabilities["brace-expansion"].via.push({
    severity: "high",
    url: "https://github.com/advisories/new",
  });
  assert.throws(() => checkAudit(changed, lockfile));
});

test("rejects a new vulnerable path", () => {
  const changed = structuredClone(report);
  changed.vulnerabilities["brace-expansion"].nodes.push(
    "node_modules/brace-expansion",
  );
  assert.throws(() => checkAudit(changed, lockfile));
});

test("requires review when Pi updates its shrinkwrap", () => {
  const changed = structuredClone(lockfile);
  changed.packages[piPath].version = "5.0.12";
  assert.throws(() => checkAudit(report, changed));
});

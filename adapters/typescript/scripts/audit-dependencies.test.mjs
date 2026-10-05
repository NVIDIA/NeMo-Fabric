// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";

import { checkAudit } from "./audit-dependencies.mjs";

const report = {
  metadata: {
    vulnerabilities: { low: 10, moderate: 0, high: 0, critical: 0 },
  },
  vulnerabilities: {},
};

test("allows an audit report without high- or critical-severity findings", () => {
  assert.equal(checkAudit(report).high, 0);
});

test("rejects a high-severity package", () => {
  const changed = structuredClone(report);
  changed.metadata.vulnerabilities.high = 1;
  changed.vulnerabilities.other = { name: "other", severity: "high" };
  assert.throws(() => checkAudit(changed));
});

test("rejects a critical-severity package", () => {
  const changed = structuredClone(report);
  changed.metadata.vulnerabilities.critical = 1;
  changed.vulnerabilities.other = { name: "other", severity: "critical" };
  assert.throws(() => checkAudit(changed));
});

test("rejects a severe finding omitted from the summary counts", () => {
  const changed = structuredClone(report);
  changed.vulnerabilities.other = { name: "other", severity: "high" };
  assert.throws(() => checkAudit(changed));
});

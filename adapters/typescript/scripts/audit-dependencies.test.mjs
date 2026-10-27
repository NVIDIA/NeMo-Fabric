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

const advisory = {
  name: "zod",
  severity: "high",
  url: "https://github.com/advisories/GHSA-mrhx-c7xc-m3w5",
  range: "<4.6.6",
};
const zodPath = "node_modules/@factory/droid-sdk/node_modules/zod";
const exceptionLock = {
  packages: {
    [zodPath]: { version: "3.25.76" },
    "node_modules/zod": { version: "4.1.8" },
    "node_modules/@factory/droid-sdk": { version: "0.9.1" },
    "node_modules/@opencode/core": { version: "2.0.23" },
    "node_modules/@opencode/plugin": { version: "2.0.23" },
  },
};
function exceptionReport() {
  const findings = {
    zod: { name: "zod", severity: "high", nodes: [zodPath], via: [advisory] },
    "@factory/droid-sdk": {
      name: "@factory/droid-sdk", severity: "high",
      nodes: ["node_modules/@factory/droid-sdk"], via: ["zod", "uuid"],
    },
    uuid: { name: "uuid", severity: "moderate", via: [] },
    "@opencode/core": {
      name: "@opencode/core", severity: "high",
      nodes: ["node_modules/@opencode/core"], via: ["@opencode/plugin", "zod"],
    },
    "@opencode/plugin": {
      name: "@opencode/plugin", severity: "high",
      nodes: ["node_modules/@opencode/plugin"], via: ["@opencode/core"],
    },
  };
  return structuredClone({
    vulnerabilities: findings,
    metadata: { vulnerabilities: { high: 4, critical: 0, moderate: 1, low: 0 } },
  });
}

test("allows only the known Zod advisory and propagated SDK findings, including cycles", () => {
  assert.equal(checkAudit(exceptionReport(), exceptionLock).high, 4);
});

test("allows removal of a patched dependency from the audit report", () => {
  const changed = exceptionReport();
  delete changed.vulnerabilities["@factory/droid-sdk"];
  changed.metadata.vulnerabilities.high--;
  assert.equal(checkAudit(changed, exceptionLock).high, 3);
});

for (const [description, mutate] of [
  ["a second high advisory on Zod", (r) => r.vulnerabilities.zod.via.push({ ...advisory, url: "https://github.com/advisories/OTHER" })],
  ["a new high advisory on an approved parent", (r) => r.vulnerabilities["@opencode/core"].via.push({ ...advisory })],
  ["a changed advisory range", (r) => { r.vulnerabilities.zod.via[0].range = "*"; }],
  ["an additional Zod install path", (r) => r.vulnerabilities.zod.nodes.push("node_modules/other/node_modules/zod")],
  ["the patched OpenCode Zod install path", (r) => r.vulnerabilities.zod.nodes.push("node_modules/zod")],
  ["an additional SDK install path", (r) => r.vulnerabilities["@factory/droid-sdk"].nodes.push("other")],
  ["a critical finding", (r) => { r.vulnerabilities.zod.severity = "critical"; r.metadata.vulnerabilities.high--; r.metadata.vulnerabilities.critical++; }],
  ["an unrelated high package", (r) => { r.vulnerabilities.other = { name: "other", severity: "high", nodes: ["node_modules/other"], via: ["zod"] }; r.metadata.vulnerabilities.high++; }],
  ["an unresolved dependency reference", (r) => r.vulnerabilities["@factory/droid-sdk"].via.push("missing")],
  ["a cycle without an approved cause", (r) => { r.vulnerabilities["@opencode/core"].via = ["@opencode/plugin"]; }],
  ["a missing Zod advisory", (r) => { r.vulnerabilities.zod.via = []; }],
  ["an audit service error", (r) => { r.error = { code: "E503" }; }],
  ["an inconsistent high count", (r) => { r.metadata.vulnerabilities.high--; }],
]) {
  test(`rejects ${description}`, () => {
    const changed = exceptionReport();
    mutate(changed);
    assert.throws(() => checkAudit(changed, exceptionLock));
  });
}

for (const path of [zodPath, "node_modules/@factory/droid-sdk"]) {
  test(`rejects a changed or missing locked version at ${path}`, () => {
    const changed = structuredClone(exceptionLock);
    changed.packages[path].version = "99.0.0";
    assert.throws(() => checkAudit(exceptionReport(), changed));
    delete changed.packages[path];
    assert.throws(() => checkAudit(exceptionReport(), changed));
  });
}

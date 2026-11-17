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

const picomatchAdvisory = {
  name: "picomatch",
  severity: "high",
  url: "https://github.com/advisories/GHSA-fc5q-39xc-vjpf",
  range: "<=4.0.7",
};

function picomatchFixture(scope) {
  const packages =
    scope === "contract"
      ? {
          picomatch: ["node_modules/picomatch", "4.0.5", [picomatchAdvisory]],
          fdir: ["node_modules/fdir", "6.5.0", ["picomatch"]],
          tinyglobby: [
            "node_modules/tinyglobby",
            "0.2.17",
            ["fdir", "picomatch"],
          ],
          "json-schema-to-typescript": [
            "node_modules/json-schema-to-typescript",
            "15.0.4",
            ["tinyglobby"],
          ],
        }
      : {
          picomatch: ["node_modules/picomatch", "4.0.7", [picomatchAdvisory]],
          fdir: [
            "node_modules/tinyglobby/node_modules/fdir",
            "6.5.0",
            ["picomatch"],
          ],
          tinyglobby: [
            "node_modules/tinyglobby",
            "0.2.17",
            ["fdir", "picomatch"],
          ],
          "@parcel/watcher": [
            "node_modules/@parcel/watcher",
            "2.6.0",
            ["picomatch"],
          ],
          "node-gyp": ["node_modules/node-gyp", "12.4.0", ["tinyglobby"]],
          "@npmcli/run-script": [
            "node_modules/@npmcli/run-script",
            "10.0.4",
            ["node-gyp"],
          ],
        };
  const audit = structuredClone(report);
  const lock = { packages: {} };
  for (const [name, [path, version, via]] of Object.entries(packages)) {
    audit.vulnerabilities[name] = structuredClone({
      name,
      severity: "high",
      nodes: [path],
      via,
    });
    lock.packages[path] = { version };
    audit.metadata.vulnerabilities.high++;
  }
  return { audit, lock };
}

for (const scope of ["contract", "adapters"]) {
  test(`${scope}: allows the exact picomatch advisory and known parent findings`, () => {
    const { audit, lock } = picomatchFixture(scope);
    assert.equal(
      checkAudit(audit, lock, scope).high,
      scope === "contract" ? 4 : 6,
    );
  });

  for (const [description, mutate] of [
    [
      "a second high advisory",
      (r) =>
        r.vulnerabilities.picomatch.via.push({
          ...picomatchAdvisory,
          url: "https://github.com/advisories/OTHER",
        }),
    ],
    [
      "a changed advisory range",
      (r) => {
        r.vulnerabilities.picomatch.via[0].range = "*";
      },
    ],
    [
      "a changed advisory package",
      (r) => {
        r.vulnerabilities.picomatch.via[0].name = "other";
      },
    ],
    [
      "a new high advisory on a parent",
      (r) => r.vulnerabilities.tinyglobby.via.push({ ...picomatchAdvisory }),
    ],
    [
      "an additional picomatch path",
      (r) =>
        r.vulnerabilities.picomatch.nodes.push(
          "node_modules/other/node_modules/picomatch",
        ),
    ],
    [
      "an additional parent path",
      (r) =>
        r.vulnerabilities.tinyglobby.nodes.push(
          "node_modules/other/node_modules/tinyglobby",
        ),
    ],
    [
      "a critical finding",
      (r) => {
        r.vulnerabilities.picomatch.severity = "critical";
        r.metadata.vulnerabilities.high--;
        r.metadata.vulnerabilities.critical++;
      },
    ],
    [
      "an unrelated high package",
      (r) => {
        r.vulnerabilities.other = {
          name: "other",
          severity: "high",
          nodes: ["node_modules/other"],
          via: ["picomatch"],
        };
        r.metadata.vulnerabilities.high++;
      },
    ],
    [
      "a parent cycle without the advisory",
      (r) => {
        r.vulnerabilities.fdir.via = ["tinyglobby"];
        r.vulnerabilities.tinyglobby.via = ["fdir"];
      },
    ],
    [
      "a missing root finding",
      (r) => {
        delete r.vulnerabilities.picomatch;
        r.metadata.vulnerabilities.high--;
      },
    ],
    [
      "an audit service error",
      (r) => {
        r.error = { code: "E503" };
      },
    ],
    [
      "an inconsistent high count",
      (r) => {
        r.metadata.vulnerabilities.high--;
      },
    ],
  ]) {
    test(`${scope}: rejects ${description}`, () => {
      const { audit, lock } = picomatchFixture(scope);
      mutate(audit);
      assert.throws(() => checkAudit(audit, lock, scope));
    });
  }

  test(`${scope}: rejects changed or missing locked versions for every excepted package`, () => {
    const { audit, lock } = picomatchFixture(scope);
    for (const path of Object.keys(lock.packages)) {
      const changed = structuredClone(lock);
      changed.packages[path].version = "99.0.0";
      assert.throws(() => checkAudit(audit, changed, scope), path);
      delete changed.packages[path];
      assert.throws(() => checkAudit(audit, changed, scope), path);
    }
  });
}

test("allows both existing Zod and picomatch exceptions in the adapter scope", () => {
  const { audit, lock } = picomatchFixture("adapters");
  const zod = exceptionReport();
  Object.assign(audit.vulnerabilities, zod.vulnerabilities);
  Object.assign(lock.packages, exceptionLock.packages);
  audit.metadata.vulnerabilities.high += zod.metadata.vulnerabilities.high;
  assert.equal(checkAudit(audit, lock).high, 10);
});

test("keeps the Zod exception out of the contract scope", () => {
  assert.throws(() => checkAudit(exceptionReport(), exceptionLock, "contract"));
});

test("rejects a lockfile from the other TypeScript scope", () => {
  const contract = picomatchFixture("contract");
  const adapters = picomatchFixture("adapters");
  assert.throws(() => checkAudit(contract.audit, adapters.lock, "contract"));
  assert.throws(() => checkAudit(adapters.audit, contract.lock, "adapters"));
});

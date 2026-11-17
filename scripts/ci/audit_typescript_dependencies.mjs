// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// Temporary CI exception; removal and exposure assessment: NVIDIA/NeMo-Fabric#485.
// Only this advisory and these installed versions are excepted. Never force Zod 4
// into SDKs that require Zod 3. Compatible copies are patched separately.
const zodAdvisory = "https://github.com/advisories/GHSA-mrhx-c7xc-m3w5";
const zodPackages = {
  "@factory/droid-sdk": { "node_modules/@factory/droid-sdk": "0.9.1" },
  "@qwen-code/sdk": { "node_modules/@qwen-code/sdk": "0.1.16" },
  "gitlab-ai-provider": { "node_modules/gitlab-ai-provider": "6.18.0" },
  zod: {
    "node_modules/@factory/droid-sdk/node_modules/zod": "3.25.76",
    "node_modules/@qwen-code/sdk/node_modules/zod": "3.25.76",
    "node_modules/gitlab-ai-provider/node_modules/zod": "3.25.76",
  },
};
for (const name of [
  "core",
  "plugin",
  "plugin-browser",
  "sdk",
  "server",
  "simulation",
]) {
  zodPackages[`@opencode/${name}`] = {
    [`node_modules/@opencode/${name}`]: "2.0.23",
  };
}

// Temporary repository CI exception for GHSA-fc5q-39xc-vjpf (no patch as of
// 2026-11-17). Maintainers: review by 2026-11-24 and remove after a compatible
// upstream fix passes clean installs, adapter tests and both audit gates.
// This does not establish that affected code is unreachable or approve consumers.
const picomatchAdvisory = "https://github.com/advisories/GHSA-fc5q-39xc-vjpf";
const picomatchPackages = {
  contract: {
    picomatch: { "node_modules/picomatch": "4.0.5" },
    fdir: { "node_modules/fdir": "6.5.0" },
    tinyglobby: { "node_modules/tinyglobby": "0.2.17" },
    "json-schema-to-typescript": {
      "node_modules/json-schema-to-typescript": "15.0.4",
    },
  },
  adapters: {
    picomatch: { "node_modules/picomatch": "4.0.7" },
    fdir: { "node_modules/tinyglobby/node_modules/fdir": "6.5.0" },
    tinyglobby: { "node_modules/tinyglobby": "0.2.17" },
    "@parcel/watcher": { "node_modules/@parcel/watcher": "2.6.0" },
    "node-gyp": { "node_modules/node-gyp": "12.4.0" },
    "@npmcli/run-script": { "node_modules/@npmcli/run-script": "10.0.4" },
  },
};

export function checkAudit(report, lockfile = {}, scope = "adapters") {
  if (!Object.hasOwn(picomatchPackages, scope)) {
    throw new Error(`Unknown audit scope: ${scope}`);
  }
  const exceptions = [
    {
      root: "picomatch",
      advisory: picomatchAdvisory,
      range: "<=4.0.7",
      packages: picomatchPackages[scope],
    },
  ];
  if (scope === "adapters") {
    exceptions.push({
      root: "zod",
      advisory: zodAdvisory,
      range: "<4.6.6",
      packages: zodPackages,
    });
  }
  if (
    report.error ||
    !report.vulnerabilities ||
    !report.metadata?.vulnerabilities
  ) {
    throw new Error("npm audit did not return a complete vulnerability report");
  }
  const severe = Object.entries(report.vulnerabilities).filter(
    ([, finding]) =>
      finding.severity === "high" || finding.severity === "critical",
  );
  if (
    report.metadata.vulnerabilities.critical !== 0 ||
    report.metadata.vulnerabilities.high !== severe.length ||
    severe.some(([, finding]) => finding.severity === "critical")
  ) {
    throw new Error(
      "Unexpected high/critical audit counts or critical finding",
    );
  }
  const severeNames = new Set(severe.map(([name]) => name));
  // npm reports parent-package metavulnerabilities, including cyclic OpenCode
  // dependencies. Each severe package must trace to its approved advisory cause.
  function reachesRoot(name, root, visiting = new Set()) {
    if (name === root) return true;
    if (visiting.has(name)) return false;
    const next = new Set(visiting).add(name);
    return report.vulnerabilities[name].via.some(
      (cause) =>
        typeof cause === "string" &&
        severeNames.has(cause) &&
        reachesRoot(cause, root, next),
    );
  }
  for (const [name, finding] of severe) {
    const exception = exceptions.find((entry) =>
      Object.hasOwn(entry.packages, name),
    );
    const allowedNodes = exception?.packages[name];
    if (
      finding.name !== name ||
      !allowedNodes ||
      !Array.isArray(finding.nodes) ||
      finding.nodes.length === 0 ||
      !Array.isArray(finding.via) ||
      finding.via.length === 0 ||
      finding.nodes.some(
        (path) =>
          !Object.hasOwn(allowedNodes, path) ||
          lockfile.packages?.[path]?.version !== allowedNodes[path],
      )
    ) {
      throw new Error(`Unapproved audit package, path or version: ${name}`);
    }
    for (const cause of finding.via) {
      if (typeof cause === "string") {
        if (!report.vulnerabilities[cause] || name === exception.root) {
          throw new Error(`Unrecognized audit dependency: ${name} -> ${cause}`);
        }
      } else if (
        cause.severity === "critical" ||
        (cause.severity === "high" &&
          (name !== exception.root ||
            cause.name !== exception.root ||
            cause.url !== exception.advisory ||
            cause.range !== exception.range))
      ) {
        throw new Error(
          `Advisory outside the temporary audit exception: ${name}`,
        );
      }
    }
    if (
      !reachesRoot(name, exception.root) ||
      (name === exception.root &&
        !finding.via.some(
          (cause) =>
            cause.severity === "high" && cause.url === exception.advisory,
        ))
    ) {
      throw new Error(
        `High-severity finding has no approved advisory cause: ${name}`,
      );
    }
  }
  return report.metadata.vulnerabilities;
}

export function auditDependencies(packageRoot, scope) {
  const npmCli = process.env.npm_execpath;
  const audit = spawnSync(
    npmCli ? process.execPath : "npm",
    npmCli ? [npmCli, "audit", "--json"] : ["audit", "--json"],
    { cwd: packageRoot, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
  );
  if (audit.error) throw audit.error;
  if ((audit.status !== 0 && audit.status !== 1) || audit.signal !== null) {
    throw new Error(
      `npm audit exited unexpectedly: status ${audit.status}, signal ${audit.signal}`,
    );
  }
  let report;
  try {
    report = JSON.parse(audit.stdout);
  } catch {
    throw new Error("npm audit failed without valid JSON output");
  }
  const lockfile = JSON.parse(
    readFileSync(join(packageRoot, "package-lock.json"), "utf8"),
  );
  const counts = checkAudit(report, lockfile, scope);
  const exceptionNames =
    scope === "adapters" ? "Zod (#485) and picomatch" : "picomatch";
  console.log(
    `npm audit: ${counts.high} high covered by temporary ${exceptionNames} CI exceptions (picomatch review 2026-11-24); ` +
      `${counts.critical} critical; ` +
      `${counts.moderate} moderate and ${counts.low} low findings remain.`,
  );
}

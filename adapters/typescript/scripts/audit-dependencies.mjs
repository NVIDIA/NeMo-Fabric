// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));

// Temporary CI exception; removal and exposure assessment: NVIDIA/NeMo-Fabric#485.
// Only this advisory and these installed versions are excepted. Never force Zod 4
// into SDKs that require Zod 3. Compatible copies are patched separately.
const zodAdvisory = "https://github.com/advisories/GHSA-mrhx-c7xc-m3w5";
const allowedPackages = {
  "@factory/droid-sdk": { "node_modules/@factory/droid-sdk": "0.9.1" },
  "@qwen-code/sdk": { "node_modules/@qwen-code/sdk": "0.1.16" },
  "gitlab-ai-provider": { "node_modules/gitlab-ai-provider": "6.18.0" },
  zod: {
    "node_modules/@factory/droid-sdk/node_modules/zod": "3.25.76",
    "node_modules/@qwen-code/sdk/node_modules/zod": "3.25.76",
    "node_modules/gitlab-ai-provider/node_modules/zod": "3.25.76",
    "node_modules/zod": "4.1.8",
  },
};
for (const name of ["core", "plugin", "plugin-browser", "sdk", "server", "simulation"]) {
  allowedPackages[`@opencode/${name}`] = { [`node_modules/@opencode/${name}`]: "2.0.23" };
}

export function checkAudit(report, lockfile = {}) {
  if (
    report.error ||
    !report.vulnerabilities ||
    !report.metadata?.vulnerabilities
  ) {
    throw new Error("npm audit did not return a complete vulnerability report");
  }
  const severe = Object.entries(report.vulnerabilities).filter(
    ([, finding]) => finding.severity === "high" || finding.severity === "critical",
  );
  if (
    report.metadata.vulnerabilities.critical !== 0 ||
    report.metadata.vulnerabilities.high !== severe.length ||
    severe.some(([, finding]) => finding.severity === "critical")
  ) {
    throw new Error("Unexpected high/critical audit counts or critical finding");
  }
  const severeNames = new Set(severe.map(([name]) => name));
  // npm reports parent-package metavulnerabilities, including cyclic OpenCode
  // dependencies. Each severe package must trace to the approved Zod cause.
  function reachesZod(name, visiting = new Set()) {
    if (name === "zod") return true;
    if (visiting.has(name)) return false;
    const next = new Set(visiting).add(name);
    return report.vulnerabilities[name].via.some(
      (cause) => typeof cause === "string" && severeNames.has(cause) && reachesZod(cause, next),
    );
  }
  for (const [name, finding] of severe) {
    const allowedNodes = allowedPackages[name];
    if (
      finding.name !== name ||
      !allowedNodes ||
      !Array.isArray(finding.nodes) ||
      finding.nodes.length === 0 ||
      !Array.isArray(finding.via) ||
      finding.via.length === 0 ||
      finding.nodes.some((path) => !Object.hasOwn(allowedNodes, path) ||
        lockfile.packages?.[path]?.version !== allowedNodes[path])
    ) {
      throw new Error(`Unapproved audit package, path or version: ${name}`);
    }
    for (const cause of finding.via) {
      if (typeof cause === "string") {
        if (!report.vulnerabilities[cause] || (name === "zod")) {
          throw new Error(`Unrecognized audit dependency: ${name} -> ${cause}`);
        }
      } else if (
        cause.severity === "critical" ||
        (cause.severity === "high" && (name !== "zod" || cause.name !== "zod" ||
          cause.url !== zodAdvisory || cause.range !== "<4.6.6"))
      ) {
        throw new Error(`Advisory outside the temporary Zod exception: ${name}`);
      }
    }
    if (!reachesZod(name) || (name === "zod" && !finding.via.some(
      (cause) => cause.severity === "high" && cause.url === zodAdvisory,
    ))) {
      throw new Error(`High-severity finding has no approved Zod cause: ${name}`);
    }
  }
  return report.metadata.vulnerabilities;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
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
    throw new Error(`npm audit failed without JSON output: ${audit.stderr}`);
  }
  const lockfile = JSON.parse(readFileSync(join(packageRoot, "package-lock.json"), "utf8"));
  const counts = checkAudit(report, lockfile);
  console.log(
    `npm audit: ${counts.high} high covered by temporary Zod CI exception (#485); ` +
    `${counts.critical} critical; ` +
    `${counts.moderate} moderate and ${counts.low} low findings remain.`,
  );
}

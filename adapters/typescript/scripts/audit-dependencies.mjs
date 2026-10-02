// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Remove this exception when @earendil-works/pi-coding-agent publishes a
// shrinkwrap with brace-expansion >=5.0.12. npm overrides and the parent
// package-lock cannot replace its shrinkwrapped 5.0.9 copy during npm ci.
// Tracked for removal in NVIDIA/NeMo-Fabric#354.

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const piPath =
  "node_modules/@earendil-works/pi-coding-agent/node_modules/brace-expansion";
const allowedHighAdvisories = [
  "https://github.com/advisories/GHSA-6j4f-fj2g-mc7p",
  "https://github.com/advisories/GHSA-qhr7-859c-m2p7",
];

export function checkAudit(report, lockfile) {
  if (
    report.error ||
    !report.vulnerabilities ||
    !report.metadata?.vulnerabilities
  ) {
    throw new Error("npm audit did not return a complete vulnerability report");
  }
  const severe = Object.values(report.vulnerabilities).filter(
    (finding) => finding.severity === "high" || finding.severity === "critical",
  );
  if (
    severe.length !== 1 ||
    report.metadata.vulnerabilities.high !== 1 ||
    report.metadata.vulnerabilities.critical !== 0
  ) {
    throw new Error(
      "Expected only the known Pi shrinkwrap high-severity finding",
    );
  }
  const finding = severe[0];
  const actualAdvisories = finding.via
    .filter((advisory) => advisory.severity === "high")
    .map((advisory) => advisory.url)
    .sort();
  if (
    finding.name !== "brace-expansion" ||
    finding.isDirect !== false ||
    JSON.stringify(finding.nodes) !== JSON.stringify([piPath]) ||
    JSON.stringify(actualAdvisories) !== JSON.stringify(allowedHighAdvisories)
  ) {
    throw new Error(
      "High-severity finding is outside the approved Pi shrinkwrap exception",
    );
  }
  if (
    lockfile.packages?.[piPath]?.version !== "5.0.9" ||
    lockfile.packages?.["node_modules/brace-expansion"]?.version !== "5.0.12" ||
    lockfile.packages?.["node_modules/@earendil-works/pi-coding-agent"]
      ?.hasShrinkwrap !== true
  ) {
    throw new Error(
      "Pi dependency versions changed; remove or review the audit exception",
    );
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
  if (audit.status !== 1 || audit.signal !== null) {
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
  const lockfile = JSON.parse(
    readFileSync(join(packageRoot, "package-lock.json"), "utf8"),
  );
  const counts = checkAudit(report, lockfile);
  console.log(
    `Allowed Pi shrinkwrap brace-expansion exception: ${counts.high} high; ` +
      `${counts.moderate} moderate and ${counts.low} low findings remain.`,
  );
}

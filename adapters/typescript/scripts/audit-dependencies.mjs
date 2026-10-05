// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawnSync } from "node:child_process";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = dirname(dirname(fileURLToPath(import.meta.url)));

export function checkAudit(report) {
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
    severe.length !== 0 ||
    report.metadata.vulnerabilities.high !== 0 ||
    report.metadata.vulnerabilities.critical !== 0
  ) {
    throw new Error("npm audit reported a high- or critical-severity finding");
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
  const counts = checkAudit(report);
  console.log(
    `npm audit: ${counts.high} high and ${counts.critical} critical; ` +
      `${counts.moderate} moderate and ${counts.low} low findings remain.`,
  );
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifest = JSON.parse(
  await readFile(resolve(packageRoot, "package.json"), "utf8"),
);
const lockfile = JSON.parse(
  await readFile(resolve(packageRoot, "package-lock.json"), "utf8"),
);
const reviewedPermissiveLicenses = new Set(["Apache-2.0", "MIT", "Python-2.0"]);

for (const field of [
  "dependencies",
  "optionalDependencies",
  "peerDependencies",
  "bundledDependencies",
  "bundleDependencies",
]) {
  if (manifest[field] !== undefined) {
    throw new Error(`Package must not declare ${field}`);
  }
}

const dependencies = Object.entries(lockfile.packages)
  .filter(([path]) => path.length > 0)
  .map(([path, dependency]) => ({ path, ...dependency }));
const productionDependencies = dependencies.filter(
  (dependency) => dependency.dev !== true,
);
if (productionDependencies.length > 0) {
  throw new Error(
    `Package lock contains production dependencies: ${productionDependencies
      .map((dependency) => dependency.path)
      .join(", ")}`,
  );
}

const unreviewedLicenses = dependencies.filter(
  (dependency) => !reviewedPermissiveLicenses.has(dependency.license),
);
if (unreviewedLicenses.length > 0) {
  const details = unreviewedLicenses
    .map(
      (dependency) =>
        `${dependency.path} (${JSON.stringify(dependency.license) ?? "missing"})`,
    )
    .join(", ");
  throw new Error(
    `Dependency licenses require explicit dependency-approver review: ${details}. ` +
      "Add only reviewed permissive SPDX identifiers to the allowlist.",
  );
}

const licenseInventory = [
  ...new Set(dependencies.map((dependency) => dependency.license)),
].sort();
console.log(`Development dependency licenses: ${licenseInventory.join(", ")}`);

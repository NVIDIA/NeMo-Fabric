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

const missingLicenses = dependencies.filter(
  (dependency) =>
    typeof dependency.license !== "string" || dependency.license.length === 0,
);
if (missingLicenses.length > 0) {
  throw new Error(
    `Package lock contains dependencies without license metadata: ${missingLicenses
      .map((dependency) => dependency.path)
      .join(", ")}`,
  );
}

const licenseInventory = [
  ...new Set(dependencies.map((dependency) => dependency.license)),
].sort();
console.log(`Development dependency licenses: ${licenseInventory.join(", ")}`);

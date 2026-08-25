// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Validate one TypeScript adapter package before publication. This check
// enforces manifest policy and exact npm tarball contents; it does not install
// or execute the packed package. check-install.mjs owns consumer-install and
// runtime verification across the related packages.

import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const packageRoot = process.cwd();
const manifest = JSON.parse(await readFile(join(packageRoot, "package.json"), "utf8"));
const npmCli = process.env.npm_execpath;
if (npmCli === undefined) {
  throw new Error("npm_execpath is required; run this check through npm");
}

const expectedByPackage = {
  "nemo-fabric-adapters-common": [
    "LICENSE",
    "README.md",
    "dist/index.d.ts",
    "dist/index.js",
    "dist/lifecycle.d.ts",
    "dist/lifecycle.js",
    "package.json",
  ],
  "nemo-fabric-adapters-pi": [
    "LICENSE",
    "README.md",
    "dist/cli.d.ts",
    "dist/cli.js",
    "dist/node-version.d.ts",
    "dist/node-version.js",
    "dist/pi-sdk.d.ts",
    "dist/pi-sdk.js",
    "dist/runtime.d.ts",
    "dist/runtime.js",
    "package.json",
    "pi.fabric-adapter.json",
  ],
};
const expectedFiles = expectedByPackage[manifest.name];
if (expectedFiles === undefined) {
  throw new Error(`Unsupported TypeScript adapter package: ${manifest.name}`);
}
if (manifest.private === true) {
  throw new Error("Published adapter packages must not be private");
}
if (
  manifest.name === "nemo-fabric-adapters-common" &&
  manifest.exports?.["."]?.import !== "./dist/index.js"
) {
  throw new Error("The common package must export its lifecycle host entry point");
}
if (manifest.name === "nemo-fabric-adapters-pi") {
  if (manifest.exports?.["./descriptor"] !== "./pi.fabric-adapter.json") {
    throw new Error("The Pi package must export its adapter descriptor");
  }
  const descriptor = JSON.parse(
    await readFile(join(packageRoot, "pi.fabric-adapter.json"), "utf8"),
  );
  if (descriptor.runner?.command !== "node" || descriptor.runner?.script !== "dist/cli.js") {
    throw new Error("The Pi descriptor runner must resolve inside the npm package");
  }
  for (const name of ["@earendil-works/pi-ai", "@earendil-works/pi-coding-agent"]) {
    if (manifest.dependencies?.[name] !== undefined) {
      throw new Error(`The Pi harness package ${name} must not be a production dependency`);
    }
    if (manifest.peerDependencies?.[name] !== "^0.84.2") {
      throw new Error(`The Pi harness package ${name} must declare the supported peer range`);
    }
    if (manifest.peerDependenciesMeta?.[name]?.optional !== true) {
      throw new Error(`The Pi harness package ${name} must be an optional peer`);
    }
    if (manifest.devDependencies?.[name] !== "0.84.2") {
      throw new Error(`The Pi harness package ${name} must be exact-pinned for development`);
    }
  }
}
for (const [name, specifier] of Object.entries(manifest.dependencies ?? {})) {
  if (specifier.startsWith("file:") || specifier.startsWith("workspace:")) {
    throw new Error(`Published dependency ${name} must use a registry version`);
  }
}
for (const hook of ["preinstall", "install", "postinstall"]) {
  if (manifest.scripts?.[hook] !== undefined) {
    throw new Error(`Published package must not define an ${hook} hook`);
  }
}

const temporaryRoot = await mkdtemp(join(tmpdir(), "nemo-fabric-ts-adapter-"));
try {
  const packOutput = execFileSync(
    process.execPath,
    [npmCli, "pack", "--json", "--ignore-scripts", "--pack-destination", temporaryRoot],
    { cwd: packageRoot, encoding: "utf8" },
  );
  const [packResult] = JSON.parse(packOutput);
  const packedFiles = new Set(packResult.files.map((file) => file.path));
  const missingFiles = expectedFiles.filter((file) => !packedFiles.has(file));
  if (missingFiles.length > 0) {
    throw new Error(`Packed artifact is missing: ${missingFiles.join(", ")}`);
  }
  const unexpectedFiles = [...packedFiles].filter((file) => !expectedFiles.includes(file));
  if (unexpectedFiles.length > 0) {
    throw new Error(`Packed artifact contains unexpected files: ${unexpectedFiles.join(", ")}`);
  }
} finally {
  await rm(temporaryRoot, { recursive: true, force: true });
}

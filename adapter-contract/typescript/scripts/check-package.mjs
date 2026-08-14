// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const npmCli = process.env.npm_execpath;
if (npmCli === undefined) {
  throw new Error("npm_execpath is required; run this check through npm");
}

const temporaryRoot = await mkdtemp(join(tmpdir(), "nemo-fabric-ts-contract-"));
try {
  const packOutput = execFileSync(
    process.execPath,
    [
      npmCli,
      "pack",
      "--json",
      "--ignore-scripts",
      "--pack-destination",
      temporaryRoot,
    ],
    { cwd: packageRoot, encoding: "utf8" },
  );
  const [packResult] = JSON.parse(packOutput);
  const packedFiles = new Set(packResult.files.map((file) => file.path));
  const expectedFiles = new Set([
    "LICENSE",
    "README.md",
    "dist/generated/adapter-descriptor.d.ts",
    "dist/generated/adapter-descriptor.js",
    "dist/generated/agent-config.d.ts",
    "dist/generated/agent-config.js",
    "dist/generated/agent-run-request.d.ts",
    "dist/generated/agent-run-request.js",
    "dist/generated/agent-run-result.d.ts",
    "dist/generated/agent-run-result.js",
    "dist/generated/runtime-context.d.ts",
    "dist/generated/runtime-context.js",
    "dist/index.d.ts",
    "dist/index.js",
    "dist/json.d.ts",
    "dist/json.js",
    "dist/version.d.ts",
    "dist/version.js",
    "package.json",
    "schemas/adapter-descriptor.schema.json",
    "schemas/agent-config.schema.json",
    "schemas/agent-run-request.schema.json",
    "schemas/agent-run-result.schema.json",
    "schemas/runtime-context.schema.json",
  ]);
  const missingFiles = [...expectedFiles].filter(
    (file) => !packedFiles.has(file),
  );
  if (missingFiles.length > 0) {
    throw new Error(`Packed artifact is missing: ${missingFiles.join(", ")}`);
  }
  const unexpectedFiles = [...packedFiles].filter(
    (file) => !expectedFiles.has(file),
  );
  if (unexpectedFiles.length > 0) {
    throw new Error(
      `Packed artifact contains unexpected files: ${unexpectedFiles.join(", ")}`,
    );
  }

  const tarball = join(temporaryRoot, packResult.filename);
  const consumerRoot = join(temporaryRoot, "consumer");
  await writeFile(
    join(temporaryRoot, "package.json"),
    `${JSON.stringify({ private: true, type: "module" }, null, 2)}\n`,
  );
  execFileSync(
    process.execPath,
    [
      npmCli,
      "install",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      "--package-lock=false",
      tarball,
    ],
    { cwd: temporaryRoot, stdio: "inherit" },
  );

  await mkdir(consumerRoot, { recursive: true });
  await writeFile(
    join(consumerRoot, "package.json"),
    `${JSON.stringify({ private: true, type: "module" }, null, 2)}\n`,
  );
  await writeFile(
    join(consumerRoot, "tsconfig.json"),
    `${JSON.stringify(
      {
        compilerOptions: {
          exactOptionalPropertyTypes: true,
          module: "NodeNext",
          moduleResolution: "NodeNext",
          noUncheckedIndexedAccess: true,
          outDir: "dist",
          resolveJsonModule: true,
          strict: true,
          target: "ES2022",
        },
        include: ["index.ts"],
      },
      null,
      2,
    )}\n`,
  );
  await writeFile(
    join(consumerRoot, "index.ts"),
    `import { ADAPTER_CONTRACT_VERSION } from "nemo-fabric-adapter-contract";
import agentConfigSchema from "nemo-fabric-adapter-contract/schemas/agent-config" with { type: "json" };
import type { AdapterDescriptor } from "nemo-fabric-adapter-contract";
import type { AgentRunResult } from "nemo-fabric-adapter-contract";

const descriptor: AdapterDescriptor = {
  adapter_id: "example",
  adapter_kind: "process",
  contract_version: ADAPTER_CONTRACT_VERSION,
  harness: "example",
};
const result: AgentRunResult = {
  output: ["ok", null],
  status: "succeeded",
};
if (
  descriptor.contract_version !== ADAPTER_CONTRACT_VERSION ||
  result.status !== "succeeded" ||
  agentConfigSchema.title !== "AgentConfig"
) {
  throw new Error("Unexpected adapter contract values");
}
`,
  );

  const tsc = resolve(packageRoot, "node_modules/typescript/bin/tsc");
  execFileSync(
    process.execPath,
    [tsc, "-p", join(consumerRoot, "tsconfig.json")],
    {
      cwd: temporaryRoot,
      stdio: "inherit",
    },
  );
  execFileSync(process.execPath, [join(consumerRoot, "dist/index.js")], {
    cwd: temporaryRoot,
    stdio: "inherit",
  });

  const installedManifest = JSON.parse(
    await readFile(
      join(
        temporaryRoot,
        "node_modules/nemo-fabric-adapter-contract/package.json",
      ),
      "utf8",
    ),
  );
  if (installedManifest.main !== undefined) {
    throw new Error("The ESM-only package must not advertise a CommonJS main");
  }
  if (installedManifest.exports?.["."]?.import !== "./dist/index.js") {
    throw new Error("The package root must expose the ESM entry point");
  }
  for (const field of [
    "dependencies",
    "optionalDependencies",
    "peerDependencies",
    "bundledDependencies",
    "bundleDependencies",
  ]) {
    if (installedManifest[field] !== undefined) {
      throw new Error(`Published package must not declare ${field}`);
    }
  }
  for (const hook of ["preinstall", "install", "postinstall"]) {
    if (installedManifest.scripts?.[hook] !== undefined) {
      throw new Error(`Published package must not define an ${hook} hook`);
    }
  }
} finally {
  await rm(temporaryRoot, { recursive: true, force: true });
}

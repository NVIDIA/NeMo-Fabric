// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Validate the TypeScript packages from a consumer's perspective. This check
// packs and installs the contract, common host, and Pi adapter in a temporary
// project, verifies adapter-only behavior, then installs the consumer-managed
// Pi harness and exercises the packaged lifecycle. check-package.mjs owns each
// package's manifest policy and exact tarball contents.

import { execFileSync, spawnSync } from "node:child_process";
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const npmCli = process.env.npm_execpath;
if (npmCli === undefined) {
  throw new Error("npm_execpath is required; run this check through npm");
}

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(scriptDirectory, "../../..");
const packageRoots = [
  join(repositoryRoot, "adapter-contract/typescript"),
  join(repositoryRoot, "adapters/typescript/common"),
  join(repositoryRoot, "adapters/typescript/pi"),
];

function npm(args, cwd) {
  return execFileSync(process.execPath, [npmCli, ...args], {
    cwd,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
  });
}

async function pathExists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function runPiCli(piRoot, consumerRoot, requests) {
  const invocation = spawnSync(process.execPath, [join(piRoot, "dist/cli.js")], {
    cwd: consumerRoot,
    encoding: "utf8",
    input: `${requests.map((request) => JSON.stringify(request)).join("\n")}\n`,
    timeout: 60_000,
  });
  if (invocation.error) {
    throw invocation.error;
  }
  if (invocation.status !== 0) {
    throw new Error(
      `Installed Pi CLI failed (status ${invocation.status}, signal ${invocation.signal}): ${invocation.stderr}`,
    );
  }
  return invocation.stdout.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
}

function startRequest(consumerRoot) {
  return {
    operation: "start",
    payload: {
      agent_name: "pi-install-check",
      base_dir: consumerRoot,
      config: {
        models: {
          default: {
            api_key_env: "TEST_API_KEY",
            model: "gpt-4.1-mini",
            provider: "openai",
          },
        },
        tools: { enabled: [] },
      },
      runtime_context: {
        artifacts: {},
        environment: {
          control_location: "external_control",
          env: { TEST_API_KEY: "not-a-real-key" },
          environment_id: "environment-install-check",
          ownership: "caller_owned",
          provider: "local",
          workspace: consumerRoot,
        },
        invocation_id: "invocation-install-check",
        request_id: "request-install-check",
        runtime_id: "runtime-install-check",
      },
    },
  };
}

const temporaryRoot = await mkdtemp(join(tmpdir(), "nemo-fabric-ts-install-"));
try {
  const tarballs = [];
  for (const packageRoot of packageRoots) {
    npm(["run", "build"], packageRoot);
    const result = JSON.parse(
      npm(
        ["pack", "--json", "--ignore-scripts", "--pack-destination", temporaryRoot],
        packageRoot,
      ),
    );
    tarballs.push(join(temporaryRoot, result[0].filename));
  }

  const consumerRoot = join(temporaryRoot, "consumer");
  await mkdir(consumerRoot);
  await writeFile(
    join(consumerRoot, "package.json"),
    `${JSON.stringify({ name: "nemo-fabric-adapter-install-check", private: true }, null, 2)}\n`,
    "utf8",
  );
  npm(
    [
      "install",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      "--package-lock=false",
      ...tarballs,
    ],
    consumerRoot,
  );

  const piRoot = join(consumerRoot, "node_modules/nemo-fabric-adapters-pi");
  const descriptor = JSON.parse(await readFile(join(piRoot, "pi.fabric-adapter.json"), "utf8"));
  if (descriptor.runner?.command !== "node" || descriptor.runner?.script !== "dist/cli.js") {
    throw new Error("Installed Pi descriptor does not reference its packaged CLI");
  }

  for (const name of ["@earendil-works/pi-ai", "@earendil-works/pi-coding-agent"]) {
    if (await pathExists(join(consumerRoot, "node_modules", name, "package.json"))) {
      throw new Error(`Adapter-only install unexpectedly included ${name}`);
    }
  }

  const [invalidResponse] = runPiCli(piRoot, consumerRoot, [{}]);
  if (invalidResponse.outcome?.error?.code !== "lifecycle_invalid_operation") {
    throw new Error(`Installed Pi CLI returned an unexpected response: ${JSON.stringify(invalidResponse)}`);
  }

  const [missingHarnessResponse] = runPiCli(piRoot, consumerRoot, [startRequest(consumerRoot)]);
  if (missingHarnessResponse.outcome?.error?.code !== "pi_harness_unavailable") {
    throw new Error(
      `Adapter-only install did not report the missing Pi harness: ${JSON.stringify(missingHarnessResponse)}`,
    );
  }

  npm(
    [
      "install",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      "--package-lock=false",
      "@earendil-works/pi-ai@0.84.2",
      "@earendil-works/pi-coding-agent@0.84.2",
    ],
    consumerRoot,
  );
  const responses = runPiCli(piRoot, consumerRoot, [
    startRequest(consumerRoot),
    { operation: "stop", payload: { runtime_id: "runtime-install-check" } },
  ]);
  if (responses.length !== 2 || responses[0].outcome?.status !== "succeeded") {
    throw new Error(`Consumer-managed Pi harness failed to start: ${JSON.stringify(responses)}`);
  }
} finally {
  await rm(temporaryRoot, { recursive: true, force: true });
}

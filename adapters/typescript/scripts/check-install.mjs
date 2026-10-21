// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Validate the TypeScript packages from a consumer's perspective. This check
// packs and installs the contract, common host, and harness adapters in a
// temporary project, verifies adapter-only behavior, then installs each
// consumer-managed harness and exercises the packaged lifecycle.

import { execFileSync, spawnSync } from "node:child_process";
import { access, chmod, copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
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
  join(repositoryRoot, "adapters/typescript/cline"),
  join(repositoryRoot, "adapters/typescript/droid"),
  join(repositoryRoot, "adapters/typescript/pi"),
  join(repositoryRoot, "adapters/typescript/opencode"),
  join(repositoryRoot, "adapters/typescript/qwen"),
  join(repositoryRoot, "adapters/typescript/kilo"),
  join(repositoryRoot, "adapters/typescript/kiro"),
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

function runClineCli(clineRoot, consumerRoot, requests) {
  const invocation = spawnSync(process.execPath, [join(clineRoot, "dist/cli.js")], {
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
      `Installed Cline CLI failed (status ${invocation.status}, signal ${invocation.signal}): ${invocation.stderr}`,
    );
  }
  return invocation.stdout.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
}

function runDroidCli(droidRoot, consumerRoot, requests) {
  const invocation = spawnSync(process.execPath, [join(droidRoot, "dist/cli.js")], {
    cwd: consumerRoot,
    encoding: "utf8",
    input: `${requests.map((request) => JSON.stringify(request)).join("\n")}\n`,
    timeout: 60_000,
  });
  if (invocation.error) throw invocation.error;
  if (invocation.status !== 0) {
    throw new Error(`Installed Droid CLI failed (status ${invocation.status}): ${invocation.stderr}`);
  }
  return invocation.stdout.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
}

function runOpenCodeCli(opencodeRoot, consumerRoot, requests) {
  const invocation = spawnSync(process.env.BUN_EXECUTABLE ?? "bun", [join(opencodeRoot, "dist/cli.js")], {
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
      `Installed OpenCode CLI failed (status ${invocation.status}, signal ${invocation.signal}): ${invocation.stderr}`,
    );
  }
  return invocation.stdout.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
}

function runQwenCli(qwenRoot, consumerRoot, requests) {
  const invocation = spawnSync(process.execPath, [join(qwenRoot, "dist/cli.js")], {
    cwd: consumerRoot,
    encoding: "utf8",
    input: `${requests.map((request) => JSON.stringify(request)).join("\n")}\n`,
    timeout: 60_000,
  });
  if (invocation.error) throw invocation.error;
  if (invocation.status !== 0) {
    throw new Error(`Installed Qwen CLI failed (status ${invocation.status}): ${invocation.stderr}`);
  }
  return invocation.stdout.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
}

function runKiloCli(kiloRoot, consumerRoot, requests) {
  const invocation = spawnSync(process.execPath, [join(kiloRoot, "dist/cli.js")], {
    cwd: consumerRoot,
    encoding: "utf8",
    input: `${requests.map((request) => JSON.stringify(request)).join("\n")}\n`,
    timeout: 60_000,
  });
  if (invocation.error) throw invocation.error;
  if (invocation.status !== 0) throw new Error(`Installed Kilo Code CLI failed: ${invocation.stderr}`);
  return invocation.stdout.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
}

function runKiroCli(kiroRoot, consumerRoot, requests, env = process.env) {
  const invocation = spawnSync(process.execPath, [join(kiroRoot, "dist/cli.js")], {
    cwd: consumerRoot,
    encoding: "utf8",
    env,
    input: `${requests.map((request) => JSON.stringify(request)).join("\n")}\n`,
    timeout: 60_000,
  });
  if (invocation.error) throw invocation.error;
  if (invocation.status !== 0) throw new Error(`Installed Kiro CLI adapter failed: ${invocation.stderr}`);
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

function clineStartRequest(consumerRoot) {
  return {
    ...startRequest(consumerRoot),
    payload: {
      ...startRequest(consumerRoot).payload,
      agent_name: "cline-install-check",
    },
  };
}

function droidStartRequest(consumerRoot) {
  const request = startRequest(consumerRoot);
  return {
    ...request,
    payload: {
      ...request.payload,
      agent_name: "droid-install-check",
      config: {
        models: {
          default: {
            api_key_env: "TEST_API_KEY",
            model: "auto",
            provider: "factory",
          },
        },
      },
    },
  };
}

function openCodeStartRequest(consumerRoot) {
  return {
    operation: "start",
    payload: {
      agent_name: "opencode-install-check",
      base_dir: consumerRoot,
      config: {
        models: {
          default: {
            api_key_env: "TEST_API_KEY",
            model: "gpt-4.1-mini",
            provider: "openai",
          },
        },
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

function qwenStartRequest(consumerRoot) {
  return {
    operation: "start",
    payload: {
      ...startRequest(consumerRoot).payload,
      agent_name: "qwen-install-check",
      config: { models: { default: { api_key_env: "TEST_API_KEY", model: "test-model", provider: "openai" } } },
    },
  };
}

function kiloStartRequest(consumerRoot) {
  const request = openCodeStartRequest(consumerRoot);
  return {
    ...request,
    payload: {...request.payload, agent_name: "kilo-install-check"},
  };
}

function kiroStartRequest(consumerRoot) {
  const request = openCodeStartRequest(consumerRoot);
  return {
    ...request,
    payload: {
      ...request.payload,
      agent_name: "kiro-install-check",
      config: {},
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

  const clineRoot = join(consumerRoot, "node_modules/nemo-fabric-adapters-cline");
  const clineDescriptor = JSON.parse(
    await readFile(join(clineRoot, "cline.fabric-adapter.json"), "utf8"),
  );
  if (clineDescriptor.runner?.command !== "node" || clineDescriptor.runner?.script !== "dist/cli.js") {
    throw new Error("Installed Cline descriptor does not reference its packaged CLI");
  }
  if (await pathExists(join(consumerRoot, "node_modules/@cline/sdk/package.json"))) {
    throw new Error("Adapter-only install unexpectedly included @cline/sdk");
  }
  const [invalidClineResponse] = runClineCli(clineRoot, consumerRoot, [{}]);
  if (invalidClineResponse.outcome?.error?.code !== "lifecycle_invalid_operation") {
    throw new Error(
      `Installed Cline CLI returned an unexpected response: ${JSON.stringify(invalidClineResponse)}`,
    );
  }
  const [missingClineHarnessResponse] = runClineCli(clineRoot, consumerRoot, [clineStartRequest(consumerRoot)]);
  if (missingClineHarnessResponse.outcome?.error?.code !== "cline_harness_unavailable") {
    throw new Error(
      `Adapter-only install did not report the missing Cline harness: ${JSON.stringify(missingClineHarnessResponse)}`,
    );
  }
  npm(
    [
      "install",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      "--package-lock=false",
      "@cline/sdk@0.0.83",
    ],
    consumerRoot,
  );
  const clineResponses = runClineCli(clineRoot, consumerRoot, [
    clineStartRequest(consumerRoot),
    { operation: "stop", payload: { runtime_id: "runtime-install-check" } },
  ]);
  if (clineResponses.length !== 2 || clineResponses[0].outcome?.status !== "succeeded") {
    throw new Error(`Consumer-managed Cline harness failed to start: ${JSON.stringify(clineResponses)}`);
  }

  const droidRoot = join(consumerRoot, "node_modules/nemo-fabric-adapters-droid");
  const droidDescriptor = JSON.parse(await readFile(join(droidRoot, "droid.fabric-adapter.json"), "utf8"));
  if (droidDescriptor.runner?.command !== "node" || droidDescriptor.runner?.script !== "dist/cli.js") {
    throw new Error("Installed Droid descriptor does not reference its packaged CLI");
  }
  const [invalidDroidResponse] = runDroidCli(droidRoot, consumerRoot, [{}]);
  if (invalidDroidResponse.outcome?.error?.code !== "lifecycle_invalid_operation") {
    throw new Error(`Installed Droid CLI returned an unexpected response: ${JSON.stringify(invalidDroidResponse)}`);
  }
  if (!(await pathExists(join(consumerRoot, "node_modules/@factory/droid-sdk/package.json")))) {
    const [missingDroidHarnessResponse] = runDroidCli(droidRoot, consumerRoot, [droidStartRequest(consumerRoot)]);
    if (missingDroidHarnessResponse.outcome?.error?.code !== "droid_harness_unavailable") {
      throw new Error(
        `Adapter-only install did not report the missing Droid SDK: ${JSON.stringify(missingDroidHarnessResponse)}`,
      );
    }
    npm(
      [
        "install",
        "--ignore-scripts",
        "--no-audit",
        "--no-fund",
        "--package-lock=false",
        "@factory/droid-sdk@0.9.1",
      ],
      consumerRoot,
    );
  }
  const [missingDroidCliResponse] = runDroidCli(droidRoot, consumerRoot, [droidStartRequest(consumerRoot)]);
  if (missingDroidCliResponse.outcome?.error?.code !== "droid_cli_unavailable") {
    throw new Error(
      `Consumer-managed Droid SDK did not report the missing CLI: ${JSON.stringify(missingDroidCliResponse)}`,
    );
  }

  const piRoot = join(consumerRoot, "node_modules/nemo-fabric-adapters-pi");
  const descriptor = JSON.parse(await readFile(join(piRoot, "pi.fabric-adapter.json"), "utf8"));
  if (descriptor.runner?.command !== "node" || descriptor.runner?.script !== "dist/cli.js") {
    throw new Error("Installed Pi descriptor does not reference its packaged CLI");
  }

  for (const name of ["@earendil-works/pi-ai", "@earendil-works/pi-coding-agent", "@earendil-works/pi-mcp"]) {
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
      "@earendil-works/pi-ai@1.2.0",
      "@earendil-works/pi-coding-agent@1.2.0",
      "@earendil-works/pi-mcp@1.2.0",
      "@earendil-works/pi-codemode@1.2.0",
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

  const opencodeRoot = join(consumerRoot, "node_modules/nemo-fabric-adapters-opencode");
  const opencodeDescriptor = JSON.parse(await readFile(join(opencodeRoot, "opencode.fabric-adapter.json"), "utf8"));
  if (opencodeDescriptor.runner?.command !== "bun" || opencodeDescriptor.runner?.script !== "dist/cli.js") {
    throw new Error("Installed OpenCode descriptor does not reference its packaged Bun CLI");
  }
  if (await pathExists(join(consumerRoot, "node_modules/@opencode/sdk/package.json"))) {
    throw new Error("Adapter-only install unexpectedly included @opencode/sdk");
  }
  if (await pathExists(join(consumerRoot, "node_modules/@opencode/core/package.json"))) {
    throw new Error("Adapter-only install unexpectedly included @opencode/core");
  }

  const [invalidOpenCodeResponse] = runOpenCodeCli(opencodeRoot, consumerRoot, [{}]);
  if (invalidOpenCodeResponse.outcome?.error?.code !== "lifecycle_invalid_operation") {
    throw new Error(`Installed OpenCode CLI returned an unexpected response: ${JSON.stringify(invalidOpenCodeResponse)}`);
  }

  const [missingOpenCodeHarnessResponse] = runOpenCodeCli(opencodeRoot, consumerRoot, [openCodeStartRequest(consumerRoot)]);
  if (missingOpenCodeHarnessResponse.outcome?.error?.code !== "opencode_harness_unavailable") {
    throw new Error(
      `Adapter-only install did not report the missing OpenCode harness: ${JSON.stringify(missingOpenCodeHarnessResponse)}`,
    );
  }

  npm(
    [
      "install",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      "--package-lock=false",
      "@opencode/core@2.0.23",
      "@opencode/sdk@2.0.23",
    ],
    consumerRoot,
  );
  if (!(await pathExists(join(consumerRoot, "node_modules/@opencode/core/package.json")))) {
    throw new Error("Consumer-managed OpenCode core harness was not installed at the adapter resolution level");
  }
  const opencodeResponses = runOpenCodeCli(opencodeRoot, consumerRoot, [
    openCodeStartRequest(consumerRoot),
    { operation: "stop", payload: { runtime_id: "runtime-install-check" } },
  ]);
  if (opencodeResponses.length !== 2 || opencodeResponses[0].outcome?.status !== "succeeded") {
    throw new Error(`Consumer-managed OpenCode harness failed to start: ${JSON.stringify(opencodeResponses)}`);
  }

  const qwenRoot = join(consumerRoot, "node_modules/nemo-fabric-adapters-qwen");
  const qwenDescriptor = JSON.parse(await readFile(join(qwenRoot, "qwen.fabric-adapter.json"), "utf8"));
  if (qwenDescriptor.runner?.command !== "node" || qwenDescriptor.runner?.script !== "dist/cli.js") {
    throw new Error("Installed Qwen descriptor does not reference its packaged Node CLI");
  }
  if (await pathExists(join(consumerRoot, "node_modules/@qwen-code/sdk/package.json"))) {
    throw new Error("Adapter-only install unexpectedly included @qwen-code/sdk");
  }
  const [invalidQwenResponse] = runQwenCli(qwenRoot, consumerRoot, [{}]);
  if (invalidQwenResponse.outcome?.error?.code !== "lifecycle_invalid_operation") {
    throw new Error(`Installed Qwen CLI returned an unexpected response: ${JSON.stringify(invalidQwenResponse)}`);
  }
  const [missingQwenSdkResponse] = runQwenCli(qwenRoot, consumerRoot, [qwenStartRequest(consumerRoot)]);
  if (missingQwenSdkResponse.outcome?.error?.code !== "qwen_sdk_missing") {
    throw new Error(`Adapter-only install did not report the missing Qwen SDK: ${JSON.stringify(missingQwenSdkResponse)}`);
  }
  npm(["install", "--ignore-scripts", "--no-audit", "--no-fund", "--package-lock=false", "@qwen-code/sdk@0.1.16"], consumerRoot);
  const qwenResponses = runQwenCli(qwenRoot, consumerRoot, [
    qwenStartRequest(consumerRoot),
    { operation: "stop", payload: { runtime_id: "runtime-install-check" } },
  ]);
  if (qwenResponses.length !== 2 ||
      qwenResponses[0].outcome?.status !== "succeeded" ||
      qwenResponses[1].outcome?.status !== "succeeded") {
    throw new Error(`Consumer-managed Qwen SDK failed to start: ${JSON.stringify(qwenResponses)}`);
  }

  const kiloRoot = join(consumerRoot, "node_modules/nemo-fabric-adapters-kilo");
  const kiloDescriptor = JSON.parse(await readFile(join(kiloRoot, "kilo.fabric-adapter.json"), "utf8"));
  if (kiloDescriptor.runner?.command !== "node" || kiloDescriptor.runner?.script !== "dist/cli.js") {
    throw new Error("Installed Kilo Code descriptor does not reference its packaged Node.js CLI");
  }
  for (const name of ["@kilocode/cli", "@kilocode/sdk"]) {
    if (await pathExists(join(consumerRoot, "node_modules", name, "package.json"))) {
      throw new Error(`Adapter-only install unexpectedly included ${name}`);
    }
  }
  const [invalidKiloResponse] = runKiloCli(kiloRoot, consumerRoot, [{}]);
  if (invalidKiloResponse.outcome?.error?.code !== "lifecycle_invalid_operation") {
    throw new Error(`Installed Kilo Code CLI returned an unexpected response: ${JSON.stringify(invalidKiloResponse)}`);
  }
  const [missingKiloResponse] = runKiloCli(kiloRoot, consumerRoot, [kiloStartRequest(consumerRoot)]);
  if (missingKiloResponse.outcome?.error?.code !== "kilo_harness_unavailable") {
    throw new Error(`Adapter-only install did not report the missing Kilo Code harness: ${JSON.stringify(missingKiloResponse)}`);
  }

  const kiroRoot = join(consumerRoot, "node_modules/nemo-fabric-adapters-kiro");
  const kiroDescriptor = JSON.parse(await readFile(join(kiroRoot, "kiro.fabric-adapter.json"), "utf8"));
  if (kiroDescriptor.runner?.command !== "node" || kiroDescriptor.runner?.script !== "dist/cli.js") {
    throw new Error("Installed Kiro descriptor does not reference its packaged Node.js CLI");
  }
  const [invalidKiroResponse] = runKiroCli(kiroRoot, consumerRoot, [{}]);
  if (invalidKiroResponse.outcome?.error?.code !== "lifecycle_invalid_operation") {
    throw new Error(`Installed Kiro CLI returned an unexpected response: ${JSON.stringify(invalidKiroResponse)}`);
  }
  const missingKiroBin = join(temporaryRoot, "missing-kiro-bin");
  await mkdir(missingKiroBin);
  const [missingKiroResponse] = runKiroCli(
    kiroRoot,
    consumerRoot,
    [kiroStartRequest(consumerRoot)],
    { ...process.env, PATH: missingKiroBin },
  );
  if (missingKiroResponse.outcome?.error?.code !== "kiro_harness_unavailable") {
    throw new Error(`Adapter-only install did not report the missing Kiro CLI: ${JSON.stringify(missingKiroResponse)}`);
  }

  const kiroBin = join(temporaryRoot, "kiro-bin");
  await mkdir(kiroBin);
  const kiroExecutable = join(kiroBin, process.platform === "win32" ? "kiro-cli.exe" : "kiro-cli");
  if (process.platform === "win32") {
    // Direct spawn requires an executable, not a .cmd wrapper. Node receives
    // "acp" as its script argument and runs this workspace-local fixture.
    await copyFile(process.execPath, kiroExecutable);
    await writeFile(
      join(consumerRoot, "acp"),
      'import(require("node:url").pathToFileURL(process.env.TEST_ACP_FIXTURE).href);\n',
      "utf8",
    );
  } else {
    await writeFile(
      kiroExecutable,
      "#!/bin/sh\nexec \"$TEST_NODE_EXECUTABLE\" \"$TEST_ACP_FIXTURE\" \"$@\"\n",
      "utf8",
    );
    await chmod(kiroExecutable, 0o755);
  }
  const kiroEnvironment = {
    ...process.env,
    PATH: `${kiroBin}${process.platform === "win32" ? ";" : ":"}${process.env.PATH ?? ""}`,
    TEST_ACP_FIXTURE: join(repositoryRoot, "adapters/typescript/common/test/fixtures/acp-agent.mjs"),
    TEST_NODE_EXECUTABLE: process.execPath,
  };
  const kiroFixtureStart = kiroStartRequest(consumerRoot);
  kiroFixtureStart.payload.runtime_context.environment.env = {
    ...kiroFixtureStart.payload.runtime_context.environment.env,
    TEST_ACP_FIXTURE: kiroEnvironment.TEST_ACP_FIXTURE,
    TEST_NODE_EXECUTABLE: kiroEnvironment.TEST_NODE_EXECUTABLE,
  };
  const kiroResponses = runKiroCli(
    kiroRoot,
    consumerRoot,
    [
      kiroFixtureStart,
      {
        operation: "invoke",
        payload: {
          request: { input: "one" },
          runtime_context: {
            ...kiroStartRequest(consumerRoot).payload.runtime_context,
            invocation_id: "invocation-kiro-one",
          },
        },
      },
      {
        operation: "invoke",
        payload: {
          request: { input: "two" },
          runtime_context: {
            ...kiroStartRequest(consumerRoot).payload.runtime_context,
            invocation_id: "invocation-kiro-two",
          },
        },
      },
      { operation: "stop", payload: { runtime_id: "runtime-install-check" } },
    ],
    kiroEnvironment,
  );
  if (
    kiroResponses.length !== 4 ||
    kiroResponses[0].outcome?.status !== "succeeded" ||
    kiroResponses[1].outcome?.output?.output?.response !==
      "one:1:build:model-a:medium:mcp=0:permission=none:ready=false" ||
    kiroResponses[2].outcome?.output?.output?.response !==
      "two:2:build:model-a:medium:mcp=0:permission=none:ready=false" ||
    kiroResponses[3].outcome?.status !== "succeeded"
  ) {
    throw new Error(`Packaged Kiro lifecycle failed: ${JSON.stringify(kiroResponses)}`);
  }
} finally {
  await rm(temporaryRoot, { recursive: true, force: true });
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { chmod, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { KiroAcpSessionFactory } from "../dist/kiro-acp.js";

test("runs a persistent Kiro-compatible ACP session and reports per-turn usage", async () => {
  const bin = await mkdtemp(join(tmpdir(), "fabric-kiro-bin-"));
  const executable = join(bin, "kiro-cli");
  await writeFile(
    executable,
    "#!/bin/sh\nexec \"$TEST_NODE_EXECUTABLE\" \"$TEST_ACP_FIXTURE\"\n",
    "utf8",
  );
  await chmod(executable, 0o755);
  const fixture = resolve(
    fileURLToPath(new URL("../../common/test/fixtures/acp-agent.mjs", import.meta.url)),
  );
  const environment = {
    PATH: `${bin}${delimiter}${process.env.PATH ?? ""}`,
    TEST_NODE_EXECUTABLE: process.execPath,
    TEST_ACP_FIXTURE: fixture,
  };
  const session = await new KiroAcpSessionFactory().create({
    agentName: "kiro-test",
    baseDir: process.cwd(),
    config: {
      models: { default: { provider: "kiro", model: "model-b" } },
      harness: { settings: { mode_id: "plan", effort: "high", permission_mode: "allow_once" } },
    },
    runtimeContext: {
      runtime_id: "runtime",
      invocation_id: "start",
      request_id: "request",
      artifacts: {},
      environment: {
        environment_id: "environment",
        provider: "local",
        ownership: "caller_owned",
        control_location: "external_control",
        workspace: process.cwd(),
        env: environment,
      },
    },
  });
  try {
    const first = await session.prompt("one");
    const second = await session.prompt("permission");
    assert.equal(first.text, "one:1:plan:model-b:high:mcp=0:permission=none:ready=false");
    assert.equal(second.text, "permission:2:plan:model-b:high:mcp=0:permission=allow:ready=false");
    assert.deepEqual(first.usage, { input_tokens: 2, output_tokens: 3, total_tokens: 5 });
    assert.deepEqual(second.usage, { input_tokens: 2, output_tokens: 3, total_tokens: 5 });

    assert.equal((await session.prompt("usage-missing")).usage, undefined);
    assert.equal((await session.prompt("after-gap")).usage, undefined);
    assert.deepEqual((await session.prompt("after-baseline")).usage, {
      input_tokens: 2,
      output_tokens: 3,
      total_tokens: 5,
    });
    assert.deepEqual((await session.prompt("cache-first")).usage, {
      input_tokens: 2,
      output_tokens: 3,
      total_tokens: 5,
    });
    assert.deepEqual((await session.prompt("cache-second")).usage, {
      input_tokens: 2,
      output_tokens: 3,
      total_tokens: 5,
      cached_input_tokens: 7,
    });
  } finally {
    await session.stop();
  }
});

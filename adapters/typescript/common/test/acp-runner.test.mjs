// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { access, mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { AcpRunnerError, startAcpSession } from "../dist/index.js";

const fixture = fileURLToPath(new URL("./fixtures/acp-agent.mjs", import.meta.url));

test("runs two turns through one ACP session with negotiated settings", async () => {
  const session = await startAcpSession({
    command: process.execPath,
    args: [fixture],
    cwd: process.cwd(),
    env: process.env,
    modeId: "plan",
    configOptions: [
      { id: "model", value: "model-b" },
      { id: "effortLevel", value: "high" },
    ],
    mcpServers: [{ name: "fixture", command: process.execPath, args: [], env: [] }],
    permissionResolver: () => "allow_once",
  });
  try {
    const first = await session.prompt("one");
    const second = await session.prompt("permission");
    assert.equal(first.text, "one:1:plan:model-b:high:mcp=1:permission=none:ready=false");
    assert.equal(second.text, "permission:2:plan:model-b:high:mcp=1:permission=allow:ready=false");
    assert.deepEqual(second.usage, { inputTokens: 4, outputTokens: 6, totalTokens: 10 });
  } finally {
    await session.stop();
  }
});

test("waits for an agent readiness notification before returning the session", async () => {
  const session = await startAcpSession({
    command: process.execPath,
    args: [fixture],
    cwd: process.cwd(),
    env: { ...process.env, TEST_READINESS_NOTIFICATION: "1" },
    readinessNotification: {
      method: "_test/mcp/status",
      isReady: (params, sessionId) =>
        params?.sessionId === sessionId && params.servers?.[0]?.status === "connected",
    },
  });
  try {
    const result = await session.prompt("first");
    assert.equal(result.text, "first:1:build:model-a:medium:mcp=0:permission=none:ready=true");
  } finally {
    await session.stop();
  }
});

test("rejects values the ACP agent did not advertise and cleans up", async () => {
  await assert.rejects(
    startAcpSession({
      command: process.execPath,
      args: [fixture],
      cwd: process.cwd(),
      env: process.env,
      configOptions: [{ id: "model", value: "unknown" }],
    }),
    (error) => error instanceof AcpRunnerError && error.code === "acp_config_option_unsupported",
  );
});

test("cancels a prompt deadline and waits for child shutdown", async () => {
  const directory = await mkdtemp(join(tmpdir(), "fabric-acp-runner-"));
  const marker = join(directory, "stopped");
  const session = await startAcpSession({
    command: process.execPath,
    args: [fixture],
    cwd: process.cwd(),
    env: { ...process.env, TEST_EXIT_MARKER: marker },
    promptTimeoutMs: 25,
    shutdownTimeoutMs: 1_000,
  });
  await assert.rejects(
    session.prompt("hang"),
    (error) => error instanceof AcpRunnerError && error.code === "acp_prompt_timeout" && error.retryable === false,
  );
  await session.stop();
  await access(marker);
  assert.equal(await readFile(marker, "utf8"), "stopped\n");
});

test("requests session close only when the capability is advertised", async () => {
  const directory = await mkdtemp(join(tmpdir(), "fabric-acp-runner-"));
  const supportedMarker = join(directory, "supported-close");
  const supported = await startAcpSession({
    command: process.execPath,
    args: [fixture],
    cwd: process.cwd(),
    env: { ...process.env, TEST_CLOSE_MARKER: supportedMarker },
  });
  await supported.stop();
  assert.equal(await readFile(supportedMarker, "utf8"), "closed\n");

  const unsupportedMarker = join(directory, "unsupported-close");
  const unsupported = await startAcpSession({
    command: process.execPath,
    args: [fixture],
    cwd: process.cwd(),
    env: { ...process.env, TEST_CLOSE_CAPABILITY_NULL: "1", TEST_CLOSE_MARKER: unsupportedMarker },
  });
  await unsupported.stop();
  await assert.rejects(access(unsupportedMarker), (error) => error.code === "ENOENT");
});

test("returns a stable missing-process error", async () => {
  await assert.rejects(
    startAcpSession({
      command: join(tmpdir(), "missing-fabric-acp-agent"),
      cwd: process.cwd(),
      env: process.env,
      shutdownTimeoutMs: 50,
    }),
    (error) => error instanceof AcpRunnerError && error.code === "acp_process_unavailable",
  );
});

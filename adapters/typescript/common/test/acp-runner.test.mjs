// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { access, mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { setTimeout as delay } from "node:timers/promises";
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

test("handles readiness failure while session configuration is pending", async () => {
  await assert.rejects(
    startAcpSession({
      command: process.execPath,
      args: [fixture],
      cwd: process.cwd(),
      env: { ...process.env, TEST_READINESS_NOTIFICATION: "1", TEST_DELAY_MODE: "1" },
      modeId: "plan",
      readinessNotification: {
        method: "_test/mcp/status",
        isReady: () => { throw new Error("readiness callback failed"); },
      },
    }),
    (error) => error instanceof AcpRunnerError && error.code === "acp_start_failed",
  );
});

async function waitForProcessToStop(pid) {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    try {
      process.kill(pid, 0);
      if (process.platform === "linux") {
        // Orphaned children can remain as zombies until the system reaps them.
        const status = await readFile(`/proc/${pid}/stat`, "utf8");
        if (status.slice(status.lastIndexOf(")") + 2).startsWith("Z")) return;
      }
    } catch (error) {
      if (error.code === "ESRCH" || error.code === "ENOENT") return;
      throw error;
    }
    await delay(10);
  }
  assert.fail(`Process ${pid} is still running after cleanup`);
}

for (const scenario of ["parent exits", "forced shutdown", "partial startup"]) {
  test(`cleans up descendants after ${scenario}`, async () => {
    const directory = await mkdtemp(join(tmpdir(), "fabric-acp-tree-"));
    const marker = join(directory, "pids.json");
    const options = {
      command: process.execPath,
      args: [fixture],
      cwd: process.cwd(),
      env: {
        ...process.env,
        TEST_DESCENDANT_MARKER: marker,
        TEST_IGNORE_SIGTERM: scenario === "parent exits" ? "0" : "1",
        TEST_PROTOCOL_MISMATCH: scenario === "partial startup" ? "1" : "0",
      },
      shutdownTimeoutMs: 100,
    };
    let session;
    try {
      if (scenario === "partial startup") {
        await assert.rejects(startAcpSession(options), (error) => error.code === "acp_protocol_unsupported");
      } else {
        session = await startAcpSession(options);
        await assert.rejects(session.prompt("hang", 25), (error) => error.code === "acp_prompt_timeout");
        await session.stop();
      }
      const pids = JSON.parse(await readFile(marker, "utf8"));
      await waitForProcessToStop(pids.parent);
      await waitForProcessToStop(pids.child);
    } finally {
      await session?.stop();
      const pids = JSON.parse(await readFile(marker, "utf8"));
      for (const pid of [pids.parent, pids.child]) {
        try { process.kill(pid, "SIGKILL"); } catch (error) {
          if (error.code !== "ESRCH") throw error;
        }
      }
    }
  });
}

test("retries process-group termination after a failed stop", { skip: process.platform === "win32" }, async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "fabric-acp-retry-"));
  const marker = join(directory, "pids.json");
  const session = await startAcpSession({
    command: process.execPath,
    args: [fixture],
    cwd: process.cwd(),
    env: { ...process.env, TEST_DESCENDANT_MARKER: marker, TEST_IGNORE_SIGTERM: "1" },
    shutdownTimeoutMs: 100,
  });
  const pids = JSON.parse(await readFile(marker, "utf8"));
  const originalKill = process.kill.bind(process);
  let failTermination = true;
  const kill = t.mock.method(process, "kill", (pid, signal) => {
    if (pid === -pids.parent && signal === "SIGTERM" && failTermination) {
      failTermination = false;
      throw Object.assign(new Error("termination denied"), { code: "EPERM" });
    }
    return originalKill(pid, signal);
  });
  let cleanupError;
  try {
    await assert.rejects(session.stop(), { code: "EPERM" });
    originalKill(pids.child, 0);
    await assert.rejects(session.prompt("after failed stop"), { code: "acp_session_stopped" });
    await Promise.all([session.stop(), session.stop()]);
    await waitForProcessToStop(pids.parent);
    await waitForProcessToStop(pids.child);
    assert.equal(kill.mock.calls.filter(({ arguments: [pid, signal] }) => pid === -pids.parent && signal === "SIGTERM").length, 2);
    const signalCount = kill.mock.callCount();
    await session.stop();
    assert.equal(kill.mock.callCount(), signalCount);
  } finally {
    try {
      kill.mock.restore();
    } catch (error) {
      cleanupError ??= error;
    }
    for (const pid of [pids.parent, pids.child]) {
      try { originalKill(pid, "SIGKILL"); } catch (error) {
        if (error.code !== "ESRCH") cleanupError ??= error;
      }
    }
    try {
      await session.stop();
    } catch (error) {
      cleanupError ??= error;
    }
  }
  if (cleanupError !== undefined) throw cleanupError;
});

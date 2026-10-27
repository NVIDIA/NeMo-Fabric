// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { chmod, copyFile, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { KiroAcpSessionFactory } from "../dist/kiro-acp.js";

for (const fixtureType of process.platform === "win32" ? ["executable"] : ["shell", "executable"]) {
  test(`runs a persistent Kiro-compatible ACP session with a ${fixtureType} fixture`, async () => {
    const bin = await mkdtemp(join(tmpdir(), "fabric-kiro-bin-"));
    const executable = join(bin, process.platform === "win32" ? "kiro-cli.exe" : "kiro-cli");
    if (fixtureType === "executable") {
      await copyFile(process.execPath, executable);
      await writeFile(join(bin, "acp"), 'import(require("node:url").pathToFileURL(process.env.TEST_ACP_FIXTURE).href);\n');
    } else {
      await writeFile(
        executable,
        "#!/bin/sh\nexec \"$TEST_NODE_EXECUTABLE\" \"$TEST_ACP_FIXTURE\"\n",
        "utf8",
      );
      await chmod(executable, 0o755);
    }
    const fixture = resolve(
      fileURLToPath(new URL("../../common/test/fixtures/acp-agent.mjs", import.meta.url)),
    );
    const environment = {
      PATH: `${bin}${delimiter}${process.env.PATH ?? ""}`,
      TEST_NODE_EXECUTABLE: process.execPath,
      TEST_ACP_FIXTURE: fixture,
      TEST_PROMPT_DELAY_MS: "100",
      TEST_PROMPT_STARTED: join(bin, "prompt-started"),
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
          workspace: bin,
          env: environment,
        },
      },
    });
    try {
      assert.equal((await session.check("live")).outcome, "passed");
      assert.equal((await session.check("active")).outcome, "unsupported");
      assert.equal((await session.check("ready")).outcome, "unsupported");
      const firstPending = session.prompt("one");
      const deadline = Date.now() + 2000;
      while (true) {
        try { await readFile(environment.TEST_PROMPT_STARTED); break; } catch (error) { if (error.code !== "ENOENT") throw error; }
        assert.ok(Date.now() < deadline);
        await new Promise((resolveDelay) => setTimeout(resolveDelay, 5));
      }
      assert.equal((await session.check("live")).outcome, "passed");
      const first = await firstPending;
      assert.equal((await session.check("live")).outcome, "passed");
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

      for (const input of ["usage-reset", "usage-malformed"]) {
        const result = await session.prompt(input);
        assert.equal(result.stopReason, "end_turn");
        assert.ok(result.text.startsWith(`${input}:`));
        assert.equal(result.usage, undefined);
        assert.equal((await session.prompt("after-invalid-usage")).usage, undefined);
        assert.deepEqual((await session.prompt("after-new-baseline")).usage, {
          input_tokens: 2,
          output_tokens: 3,
          total_tokens: 5,
        });
      }
    } finally {
      await session.stop();
      assert.equal((await session.check("live")).reason_code, "session_closed");
    }
  });
}

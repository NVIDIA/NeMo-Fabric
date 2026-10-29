// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { chmod, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import test from "node:test";

import { KiroAcpSessionFactory } from "../dist/kiro-acp.js";

test("unsupported BYOM CLI versions close the owned listener and remove the isolated home", {
  skip: process.platform === "win32",
}, async t => {
  const workspace = await mkdtemp(join(tmpdir(), "kiro-version-test-"));
  t.after(() => rm(workspace, { recursive: true, force: true }));
  const command = join(workspace, "kiro-cli");
  await writeFile(command, '#!/bin/sh\nprintf "%s\\n%s\\n" "$HOME" "$KIRO_KAS_ENDPOINT" > "$CAPTURE"\nprintf "kiro-cli 2.34.1\\n"\n');
  await chmod(command, 0o700);
  const capture = join(workspace, "capture");
  await assert.rejects(new KiroAcpSessionFactory().create({
    agentName: "version-test", baseDir: workspace,
    config: {
      harness: { settings: { experimental_byom: true } },
      models: { default: { provider: "openai", model: "test", base_url: "http://127.0.0.1:1/v1" } },
    },
    runtimeContext: { runtime_id: "test", request_id: "test", invocation_id: "test", artifacts: {},
      environment: { environment_id: "test", provider: "local", ownership: "caller_owned", control_location: "external_control",
        workspace, env: { PATH: `${workspace}${delimiter}${process.env.PATH}`, CAPTURE: capture } } },
  }), error => error.code === "kiro_byom_version_unsupported");
  const [home, endpoint] = (await readFile(capture, "utf8")).trim().split("\n");
  assert.notEqual(home, process.env.HOME);
  await assert.rejects(stat(home), { code: "ENOENT" });
  await assert.rejects(fetch(endpoint));
});

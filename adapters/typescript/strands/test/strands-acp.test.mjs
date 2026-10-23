// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import {
  chmod,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
  access,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import { StrandsAdapterRuntime } from "../dist/runtime.js";
import {
  normalizeUsage,
  StrandsAcpSessionFactory,
} from "../dist/strands-acp.js";

async function fixture(t) {
  const bin = await mkdtemp(join(tmpdir(), "fabric-strands-test-"));
  t.after(() => rm(bin, { recursive: true, force: true }));
  const source = fileURLToPath(
    new URL("./fixtures/acp-agent.mjs", import.meta.url),
  );
  const script = `import(${JSON.stringify(new URL("./fixtures/acp-agent.mjs", import.meta.url).href)});\n`;
  if (process.platform === "win32") {
    const root = join(bin, "node_modules/@strands-agents/cli/bin");
    await mkdir(root, { recursive: true });
    await writeFile(join(root, "strands.js"), script);
  } else {
    await writeFile(
      join(bin, "strands"),
      '#!/bin/sh\nexec "$TEST_NODE_EXECUTABLE" "$TEST_STRANDS_FIXTURE" "$@"\n',
    );
    await chmod(join(bin, "strands"), 0o755);
  }
  return {
    agentName: "test",
    baseDir: bin,
    config: {
      models: {
        default: {
          provider: "openai",
          model: "local",
          base_url: "http://localhost:8000/v1",
        },
      },
      harness: { settings: { builtin_tools: [] } },
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
        env: {
          PATH: `${bin}${delimiter}${process.env.PATH ?? ""}`,
          TEST_NODE_EXECUTABLE: process.execPath,
          TEST_STRANDS_FIXTURE: source,
          TEST_STRANDS_STATE: join(bin, "state.json"),
        },
      },
    },
  };
}

test("preserves real ACP session history, isolates runtime state, and uses per-invocation usage", async (t) => {
  const input = await fixture(t);
  input.config.instructions = {
    system: { mode: "append", content: "--instruction-text" },
  };
  const previousSecret = process.env.TEST_UNRELATED_SECRET;
  process.env.TEST_UNRELATED_SECRET = "must-not-leak";
  t.after(() => {
    if (previousSecret === undefined) delete process.env.TEST_UNRELATED_SECRET;
    else process.env.TEST_UNRELATED_SECRET = previousSecret;
  });
  const factory = new StrandsAcpSessionFactory();
  const first = await factory.create(input);
  t.after(() => first.stop());
  const state = JSON.parse(
    await readFile(
      input.runtimeContext.environment.env.TEST_STRANDS_STATE,
      "utf8",
    ),
  );
  assert.notEqual(state.home, process.env.HOME);
  assert.equal(state.endpoint, "http://localhost:8000/v1");
  assert.equal(state.hasKey, true);
  assert.equal(state.unrelatedSecret, undefined);
  assert.ok(state.args.includes("litellm/local"));
  assert.ok(state.args.includes("--instructions=--instruction-text"));
  assert.ok(state.args.includes("agentConfig.backgroundTasks=false"));
  const one = await first.prompt("secret-token");
  const two = await first.prompt("recall");
  assert.equal(two.text, "secret-token");
  assert.equal(one.usage.input_tokens, 11);
  assert.equal(two.usage.input_tokens, 12);
  assert.equal(
    two.usage.extensions["nemo_fabric.detailed_token_usage.v1"]
      .cache_write_input_tokens,
    1,
  );
  const second = await factory.create(input);
  t.after(() => second.stop());
  assert.notEqual(second.id, first.id);
  assert.equal((await second.prompt("recall")).text, "EMPTY");
  await first.stop();
  await first.stop();
  await assert.rejects(access(state.home), { code: "ENOENT" });
});

test("passes MCP servers into the first ACP session and cleans up on timeout", async (t) => {
  const input = await fixture(t);
  input.config.mcp = {
    servers: {
      probe: { transport: "stdio", url: process.execPath, args: ["probe.mjs"] },
    },
  };
  const session = await new StrandsAcpSessionFactory().create(input);
  t.after(() => session.stop());
  const result = await session.prompt("mcp");
  assert.equal(JSON.parse(result.text)[0].name, "probe");
  await assert.rejects(session.prompt("hang", 25), {
    code: "strands_prompt_timeout",
  });
  await session.stop();
});

test("keeps missing and invalid usage unknown without rejecting the completed result", () => {
  for (const usage of [
    undefined,
    { inputTokens: 1, outputTokens: 1, totalTokens: 3 },
    { inputTokens: 1.5, outputTokens: 2, totalTokens: 3.5 },
    { inputTokens: 1, outputTokens: 2, totalTokens: 3, cachedReadTokens: 5 },
  ])
    assert.equal(normalizeUsage(usage), undefined);
  assert.equal(
    normalizeUsage({ inputTokens: 2, outputTokens: 1, totalTokens: 3 })
      .cached_input_tokens,
    undefined,
  );
});

test("fails before launch when an explicitly named model credential is missing", async (t) => {
  const input = await fixture(t);
  input.config.models.default.api_key_env = "FABRIC_TEST_STRANDS_MISSING_KEY";
  await assert.rejects(new StrandsAcpSessionFactory().create(input), {
    code: "strands_credential_missing",
  });
});

test("removes temporary CLI state after partial startup and keeps diagnostics private", async (t) => {
  const input = await fixture(t);
  input.runtimeContext.environment.env.TEST_STRANDS_FAIL_START = "1";
  await assert.rejects(
    new StrandsAcpSessionFactory().create(input),
    (error) =>
      error.code === "strands_start_failed" &&
      !error.message.includes("PRIVATE_STARTUP_DETAIL"),
  );
  const state = JSON.parse(
    await readFile(
      input.runtimeContext.environment.env.TEST_STRANDS_STATE,
      "utf8",
    ),
  );
  await assert.rejects(access(state.home), { code: "ENOENT" });
});

test("omits ambiguous zero cache counts while preserving positive counts and token totals", () => {
  for (const [read, write] of [[0, 0], [2, 0], [0, 3], [2, 3]]) {
    const usage = normalizeUsage({
      inputTokens: 10, outputTokens: 5, totalTokens: 15,
      cachedReadTokens: read, cachedWriteTokens: write,
    });
    const detail = usage.extensions["nemo_fabric.detailed_token_usage.v1"];
    assert.equal(usage.input_tokens, 10);
    assert.equal(usage.output_tokens, 5);
    assert.equal(usage.total_tokens, 15);
    assert.equal(usage.input_tokens_include_cache, true);
    assert.equal(Object.hasOwn(usage, "cached_input_tokens"), read > 0);
    assert.equal(Object.hasOwn(detail, "cached_input_tokens"), read > 0);
    assert.equal(Object.hasOwn(detail, "cache_write_input_tokens"), write > 0);
    if (read > 0) {
      assert.equal(usage.cached_input_tokens, read);
      assert.equal(detail.cached_input_tokens, read);
    }
    if (write > 0) assert.equal(detail.cache_write_input_tokens, write);
  }
});

test("retains a timed-out runtime session until process cleanup succeeds", { skip: process.platform === "win32" }, async (t) => {
  const input = await fixture(t);
  const descendantState = join(input.baseDir, "descendant.json");
  input.runtimeContext.environment.env.TEST_STRANDS_DESCENDANT_STATE = descendantState;
  const runtime = new StrandsAdapterRuntime(new StrandsAcpSessionFactory());
  await runtime.start(input);
  const { pid: parent } = JSON.parse(await readFile(input.runtimeContext.environment.env.TEST_STRANDS_STATE, "utf8"));
  const { pid: child } = JSON.parse(await readFile(descendantState, "utf8"));
  const originalKill = process.kill.bind(process);
  let failTermination = true;
  const kill = t.mock.method(process, "kill", (pid, signal) => {
    if (pid === -parent && signal === "SIGTERM" && failTermination) {
      failTermination = false;
      throw Object.assign(new Error("termination denied"), { code: "EPERM" });
    }
    return originalKill(pid, signal);
  });
  t.after(async () => {
    const cleanup = [
      () => kill.mock.restore(),
      ...[parent, child].map((pid) => () => {
        try { originalKill(pid, "SIGKILL"); } catch (error) {
          if (error.code !== "ESRCH") throw error;
        }
      }),
      () => runtime.stop(),
    ];
    const results = await Promise.allSettled(cleanup.map(async (step) => step()));
    const failures = results.filter((result) => result.status === "rejected");
    if (failures.length > 0) {
      throw new AggregateError(failures.map((result) => result.reason), "Test process cleanup failed");
    }
  });
  await assert.rejects(
    runtime.invoke({ input: "hang" }, { ...input.runtimeContext, deadline_millis: Date.now() + 50 }),
    { code: "strands_prompt_timeout" },
  );
  originalKill(child, 0);
  await assert.rejects(runtime.invoke({ input: "next" }, input.runtimeContext), { code: "strands_not_started" });
  await assert.rejects(runtime.start(input), { code: "strands_already_started" });
  await runtime.stop();
  for (const pid of [parent, child]) {
    let running = true;
    for (let attempt = 0; attempt < 200; attempt++) {
      try {
        originalKill(pid, 0);
        if (process.platform === "linux") {
          const status = await readFile(`/proc/${pid}/stat`, "utf8");
          if (status.slice(status.lastIndexOf(")") + 2).startsWith("Z")) { running = false; break; }
        }
      } catch (error) {
        if (error.code !== "ESRCH" && error.code !== "ENOENT") throw error;
        running = false;
        break;
      }
      await delay(10);
    }
    assert.equal(running, false, `Process ${pid} survived runtime.stop()`);
  }
  const attempts = kill.mock.callCount();
  await runtime.stop();
  assert.equal(kill.mock.callCount(), attempts);
});

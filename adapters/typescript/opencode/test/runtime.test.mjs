// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { OpenCodeAdapterRuntime } from "../dist/runtime.js";
import { LifecycleError } from "nemo-fabric-adapters-common";

function startInput(runtimeId = "runtime-1") {
  return {
    agentName: "opencode-test",
    baseDir: "/tmp",
    config: {},
    runtimeContext: {
      artifacts: {},
      environment: {
        control_location: "external_control",
        environment_id: "environment-1",
        ownership: "caller_owned",
        provider: "local",
        workspace: "/tmp",
      },
      invocation_id: "start",
      request_id: "request-start",
      runtime_id: runtimeId,
    },
  };
}

const context = startInput().runtimeContext;

test("enforces the runtime start and invoke lifecycle guards", async () => {
  const runtime = new OpenCodeAdapterRuntime({
    async create() {
      return { id: "session-1", async prompt() { return { text: "ok" }; }, async stop() {} };
    },
  });

  await assert.rejects(
    runtime.invoke({ input: "before start" }, context),
    (error) => error.code === "opencode_not_started",
  );
  await runtime.start(startInput());
  await assert.rejects(runtime.start(startInput()), (error) => error.code === "opencode_already_started");
  await runtime.stop();
});

test("creates one session per runtime and stops it exactly once", async () => {
  const created = [];
  let stops = 0;
  const factory = {
    async create(input) {
      created.push(input.runtimeContext.runtime_id);
      return {
        id: `session-${input.runtimeContext.runtime_id}`,
        async prompt() {
          return { text: "ok" };
        },
        async stop() {
          stops += 1;
        },
      };
    },
  };
  const first = new OpenCodeAdapterRuntime(factory);
  const second = new OpenCodeAdapterRuntime(factory);

  await first.start(startInput("runtime-1"));
  await second.start(startInput("runtime-2"));
  await first.stop();
  await first.stop();
  await second.stop();

  assert.deepEqual(created, ["runtime-1", "runtime-2"]);
  assert.equal(stops, 2);
});

test("normalizes successful plain-text prompts and reuses its OpenCode session", async () => {
  const prompts = [];
  const runtime = new OpenCodeAdapterRuntime({
    async create() {
      return {
        id: "session-1",
        async prompt(text) {
          prompts.push(text);
          return {
            text: `reply:${text}`,
            usage: { input_tokens: 3, output_tokens: 5, total_tokens: 8, cost_usd: 0.01 },
          };
        },
        async stop() {},
      };
    },
  });

  await runtime.start(startInput());
  const first = await runtime.invoke({ input: "one" }, context);
  const second = await runtime.invoke({ input: "two" }, context);

  assert.deepEqual(prompts, ["one", "two"]);
  assert.deepEqual(first, {
    status: "succeeded",
    output: { response: "reply:one" },
    usage: { input_tokens: 3, output_tokens: 5, total_tokens: 8, cost_usd: 0.01 },
  });
  assert.deepEqual(second.output, { response: "reply:two" });
});

test("writes an OpenCode session diff as a per-invocation Fabric artifact", async () => {
  const artifacts = await mkdtemp(join(tmpdir(), "fabric-opencode-artifacts-"));
  try {
    const runtime = new OpenCodeAdapterRuntime({
      async create() {
        return {
          id: "session-1",
          async prompt() {
            return { text: "done", patch: "diff --git a/example.txt b/example.txt\n+added\n" };
          },
          async stop() {},
        };
      },
    });
    await runtime.start(startInput());
    const result = await runtime.invoke(
      { input: "edit the file" },
      { ...context, artifacts: { root: artifacts }, invocation_id: "invocation-1" },
    );

    const artifact = result.artifacts?.[0];
    assert.equal(artifact?.kind, "patch");
    assert.equal(artifact?.media_type, "text/x-diff");
    assert.equal(artifact?.name, "opencode-diff-1");
    assert.match(artifact?.path ?? "", /^opencode\/[a-f0-9]{32}\/[a-f0-9]{32}-turn-1\.patch$/);
    assert.equal(await readFile(join(artifacts, artifact?.path), "utf8"), "diff --git a/example.txt b/example.txt\n+added\n");
  } finally {
    await rm(artifacts, { recursive: true, force: true });
  }
});

test("returns usage and a patch artifact for a terminal OpenCode model error", async () => {
  const artifacts = await mkdtemp(join(tmpdir(), "fabric-opencode-failed-artifacts-"));
  try {
    const runtime = new OpenCodeAdapterRuntime({
      async create() {
        return {
          id: "session-1",
          async prompt() {
            return {
              errorMessage: "OpenCode model invocation failed",
              patch: "failed turn diff",
              usage: { input_tokens: 3, output_tokens: 5, total_tokens: 8, cost_usd: 0.01 },
            };
          },
          async stop() {},
        };
      },
    });
    await runtime.start(startInput());

    const result = await runtime.invoke(
      { input: "edit" },
      { ...context, artifacts: { root: artifacts }, invocation_id: "failed-invocation" },
    );

    assert.equal(result.status, "failed");
    assert.equal(result.error?.code, "opencode_model_error");
    assert.deepEqual(result.usage, { input_tokens: 3, output_tokens: 5, total_tokens: 8, cost_usd: 0.01 });
    assert.match(result.artifacts?.[0]?.path ?? "", /^opencode\/[a-f0-9]{32}\/[a-f0-9]{32}-turn-1\.patch$/);
    assert.equal(await readFile(join(artifacts, result.artifacts?.[0]?.path), "utf8"), "failed turn diff");
  } finally {
    await rm(artifacts, { recursive: true, force: true });
  }
});

test("returns a successful result without an artifact when no artifact root is configured", async () => {
  const runtime = new OpenCodeAdapterRuntime({
    async create() {
      return { id: "session-1", async prompt() { return { text: "done", patch: "diff" }; }, async stop() {} };
    },
  });

  await runtime.start(startInput());
  const result = await runtime.invoke({ input: "edit" }, context);

  assert.equal(result.status, "succeeded");
  assert.equal(result.artifacts, undefined);
});

test("reports a stable error when it cannot write an artifact", async () => {
  const root = await mkdtemp(join(tmpdir(), "fabric-opencode-artifact-file-"));
  const artifactFile = join(root, "not-a-directory");
  await writeFile(artifactFile, "file", "utf8");
  try {
    const runtime = new OpenCodeAdapterRuntime({
      async create() {
        return { id: "session-1", async prompt() { return { text: "done", patch: "diff" }; }, async stop() {} };
      },
    });
    await runtime.start(startInput());

    await assert.rejects(
      runtime.invoke({ input: "edit" }, { ...context, artifacts: { root: artifactFile } }),
      (error) => error.code === "opencode_artifact_write_failed",
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("keeps patch artifacts from distinct runtimes under a shared root", async () => {
  const artifacts = await mkdtemp(join(tmpdir(), "fabric-opencode-shared-artifacts-"));
  try {
    const createRuntime = (patch) => new OpenCodeAdapterRuntime({
      async create() {
        return { id: "session", async prompt() { return { text: "done", patch }; }, async stop() {} };
      },
    });
    const first = createRuntime("first patch");
    const second = createRuntime("second patch");
    await first.start(startInput("runtime-one"));
    await second.start(startInput("runtime-two"));

    const firstResult = await first.invoke(
      { input: "first" },
      { ...context, artifacts: { root: artifacts }, runtime_id: "runtime-one", invocation_id: "invocation-one" },
    );
    const secondResult = await second.invoke(
      { input: "second" },
      { ...context, artifacts: { root: artifacts }, runtime_id: "runtime-two", invocation_id: "invocation-two" },
    );

    const firstPath = firstResult.artifacts?.[0]?.path;
    const secondPath = secondResult.artifacts?.[0]?.path;
    assert.notEqual(firstPath, secondPath);
    assert.equal(await readFile(join(artifacts, firstPath), "utf8"), "first patch");
    assert.equal(await readFile(join(artifacts, secondPath), "utf8"), "second patch");
  } finally {
    await rm(artifacts, { recursive: true, force: true });
  }
});

test("rejects non-text input and normalizes terminal OpenCode failures", async () => {
  const outcomes = [{ errorMessage: "provider rejected" }, {}, { text: "" }];
  let prompts = 0;
  const runtime = new OpenCodeAdapterRuntime({
    async create() {
      return {
        id: "session-1",
        async prompt() {
          prompts += 1;
          return outcomes.shift();
        },
        async stop() {},
      };
    },
  });

  await runtime.start(startInput());
  const unsupported = await runtime.invoke({ input: { task: "not text" } }, context);
  const failed = await runtime.invoke({ input: "one" }, context);
  const missing = await runtime.invoke({ input: "two" }, context);
  const empty = await runtime.invoke({ input: "three" }, context);

  assert.equal(unsupported.status, "failed");
  assert.equal(unsupported.error.code, "opencode_unsupported_input");
  assert.equal(prompts, 3);
  assert.equal(failed.error.code, "opencode_model_error");
  assert.equal(failed.error.message, "provider rejected");
  assert.equal(missing.error.code, "opencode_no_assistant_response");
  assert.equal(empty.error.code, "opencode_no_assistant_response");
});

test("invalidates a runtime after an OpenCode session transport failure", async () => {
  let stops = 0;
  const runtime = new OpenCodeAdapterRuntime({
    async create() {
      return {
        id: "session-transport-failure",
        async prompt() {
          throw new LifecycleError("opencode_session_failed", "OpenCode session communication failed");
        },
        async stop() {
          stops += 1;
        },
      };
    },
  });

  await runtime.start(startInput());
  await assert.rejects(
    runtime.invoke({ input: "private prompt text" }, context),
    (error) => error.code === "opencode_session_failed" && error.message === "OpenCode session communication failed",
  );
  await assert.rejects(
    runtime.invoke({ input: "another prompt" }, context),
    (error) => error.code === "opencode_not_started",
  );
  assert.equal(stops, 1);
});

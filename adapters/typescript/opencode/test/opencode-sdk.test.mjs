// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { extractOpenCodePromptOutcome, loadOpenCodeSdk, OpenCodeSdkSessionFactory } from "../dist/opencode-sdk.js";

function startInput() {
  return {
    agentName: "opencode-sdk-test",
    baseDir: "/fallback",
    config: {
      models: {
        default: { provider: "openai", model: "gpt-4.1-mini", api_key_env: "OPENCODE_TEST_KEY" },
      },
    },
    runtimeContext: {
      artifacts: {},
      environment: {
        control_location: "external_control",
        env: { OPENCODE_TEST_KEY: "test-secret" },
        environment_id: "environment-1",
        ownership: "caller_owned",
        provider: "local",
        workspace: "/workspace",
      },
      invocation_id: "start",
      request_id: "request-start",
      runtime_id: "runtime-1",
    },
  };
}

test("reports an unresolvable OpenCode peer as unavailable before attempting to load it", async () => {
  const resolved = [];
  await assert.rejects(
    loadOpenCodeSdk(
      (specifier) => {
        resolved.push(specifier);
        throw new Error("package is not installed");
      },
      async () => {
        throw new Error("should not load an unresolved package");
      },
    ),
    (error) => error.code === "opencode_harness_unavailable",
  );
  assert.deepEqual(resolved, ["@opencode/core/config"]);
});

test("reports an installed OpenCode peer with a missing transitive module as a load failure", async () => {
  const loaded = [];
  await assert.rejects(
    loadOpenCodeSdk(
      (specifier) => `resolved:${specifier}`,
      async (specifier) => {
        loaded.push(specifier);
        if (specifier === "resolved:@opencode/sdk") {
          const error = new Error("Cannot find @effect/platform-node");
          Object.assign(error, { code: "ERR_MODULE_NOT_FOUND" });
          throw error;
        }
        return {};
      },
    ),
    (error) => error.code === "opencode_harness_load_failed",
  );
  assert.deepEqual(loaded, ["resolved:@opencode/core/config", "resolved:@opencode/sdk"]);
});

test("extracts the final assistant text and usage from OpenCode session history", () => {
  const outcome = extractOpenCodePromptOutcome([
    { type: "user", text: "first" },
    {
      id: "assistant-first",
      type: "assistant",
      content: [{ type: "text", text: "first answer" }],
      finish: "stop",
      tokens: { input: 2, output: 3, reasoning: 0, cache: { read: 0, write: 0 } },
      cost: 0.012,
    },
    { type: "user", text: "second" },
    {
      id: "assistant-second",
      type: "assistant",
      content: [{ type: "reasoning", text: "hidden" }, { type: "text", text: "second answer" }],
      finish: "stop",
      tokens: { input: 5, output: 7, reasoning: 2, cache: { read: 1, write: 0 } },
      cost: 0.023,
    },
  ], new Set(["assistant-first"]));

  assert.deepEqual(outcome, {
    text: "second answer",
    usage: { input_tokens: 5, output_tokens: 7, total_tokens: 12, cost_usd: 0.023 },
  });
});

test("sums usage across every new assistant message in a tool-using prompt", () => {
  const outcome = extractOpenCodePromptOutcome(
    [
      {
        id: "assistant-tool-call",
        type: "assistant",
        content: [{ type: "tool", name: "read", input: { path: "example.txt" } }],
        finish: "tool-calls",
        tokens: { input: 11, output: 13 },
        cost: 0.01,
      },
      {
        id: "assistant-terminal",
        type: "assistant",
        content: [{ type: "text", text: "done" }],
        finish: "stop",
        tokens: { input: 17, output: 19 },
        cost: 0.02,
      },
    ],
    new Set(),
  );

  assert.deepEqual(outcome, {
    text: "done",
    usage: { input_tokens: 28, output_tokens: 32, total_tokens: 60, cost_usd: 0.03 },
  });
});

test("preserves a redacted OpenCode terminal error and handles malformed assistant output", () => {
  assert.deepEqual(
    extractOpenCodePromptOutcome([
      {
        type: "assistant",
        content: [{ type: "text", text: "partial" }],
        finish: "error",
        error: { message: "provider returned HTTP 401: Authorization: Bearer test-secret" },
      },
    ]),
    {
      errorMessage: "OpenCode model invocation failed",
    },
  );
  assert.deepEqual(extractOpenCodePromptOutcome([{ type: "assistant", content: "invalid", finish: "stop" }]), {});
  assert.deepEqual(extractOpenCodePromptOutcome([]), {});
});

test("does not return a prior assistant response for a new prompt", async () => {
  let contextCalls = 0;
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: {
      async create() {
        return {
          sessions: {
            async create() {
              return { id: "session-stale-response" };
            },
            async prompt() {},
            async wait() {},
            async context() {
              contextCalls += 1;
              const priorTurn = [
                { id: "user-first", type: "user", text: "first" },
                {
                  id: "assistant-first",
                  type: "assistant",
                  content: [{ type: "text", text: "first answer" }],
                  finish: "stop",
                },
              ];
              return contextCalls === 1
                ? priorTurn
                : [...priorTurn, { id: "user-second", type: "user", text: "second" }];
            },
            async diff() {
              throw new Error("no diff available");
            },
            async remove() {},
          },
          async close() {},
        };
      },
    },
  }));

  const handle = await factory.create(startInput());
  assert.deepEqual(await handle.prompt("second"), {});
  assert.equal(contextCalls, 2);
  await handle.stop();
});

test("finds a new assistant response after OpenCode compacts session history", async () => {
  let contextCalls = 0;
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: {
      async create() {
        return {
          sessions: {
            async create() {
              return { id: "session-compaction" };
            },
            async prompt() {},
            async wait() {},
            async context() {
              contextCalls += 1;
              if (contextCalls === 1) {
                return Array.from({ length: 8 }, (_, index) => ({
                  id: `prior-${index}`,
                  type: index % 2 === 0 ? "user" : "assistant",
                  content: [{ type: "text", text: `prior ${index}` }],
                  finish: "stop",
                }));
              }
              return [
                { id: "compaction-marker", type: "compaction" },
                {
                  id: "assistant-after-compaction",
                  type: "assistant",
                  content: [{ type: "text", text: "compacted answer" }],
                  finish: "stop",
                },
              ];
            },
            async diff() {
              throw new Error("no diff available");
            },
            async remove() {},
          },
          async close() {},
        };
      },
    },
  }));

  const handle = await factory.create(startInput());
  assert.deepEqual(await handle.prompt("continue"), { text: "compacted answer" });
  await handle.stop();
});

test("creates, invokes, and cleans up one embedded OpenCode session", async () => {
  const calls = [];
  let contextCalls = 0;
  let closed = false;
  const previous = process.env.OPENCODE_TEST_KEY;
  delete process.env.OPENCODE_TEST_KEY;
  try {
    const factory = new OpenCodeSdkSessionFactory(async () => ({
      OpenCode: {
        async create(options) {
          calls.push(["create", options, process.env.OPENCODE_TEST_KEY]);
          return {
            sessions: {
              async create(input) {
                calls.push(["session.create", input]);
                return { id: "session-1" };
              },
              async prompt(input) {
                calls.push(["session.prompt", input]);
              },
              async wait(input) {
                calls.push(["session.wait", input]);
              },
              async context(input) {
                calls.push(["session.context", input]);
                contextCalls += 1;
                return contextCalls === 1
                  ? []
                  : [{ id: "assistant-done", type: "assistant", content: [{ type: "text", text: "done" }], finish: "stop" }];
              },
              async diff(input) {
                calls.push(["session.diff", input]);
                return [{ file: "example.txt", patch: "diff --git a/example.txt b/example.txt\n+added\n" }];
              },
              async remove(input) {
                calls.push(["session.remove", input]);
              },
            },
            async close() {
              closed = true;
            },
          };
        },
      },
    }));

    const handle = await factory.create(startInput());
    assert.equal(handle.id, "session-1");
    assert.deepEqual(await handle.prompt("hello"), {
      text: "done",
      patch: "diff --git a/example.txt b/example.txt\n+added\n",
    });
    await handle.stop();

    assert.equal(closed, true);
    assert.equal(process.env.OPENCODE_TEST_KEY, undefined);
    assert.deepEqual(calls, [
      [
        "create",
        {
          config: {
            directory: "/workspace",
            project: false,
            content: JSON.stringify({
              providers: { openai: { settings: { apiKey: "{env:OPENCODE_TEST_KEY}" } } },
            }),
          },
          fs: { filewatcher: false },
          models: { fetch: false },
        },
        "test-secret",
      ],
      ["session.create", { location: { directory: "/workspace" }, model: { providerID: "openai", id: "gpt-4.1-mini" } }],
      ["session.context", { sessionID: "session-1" }],
      ["session.prompt", { sessionID: "session-1", text: "hello" }],
      ["session.wait", { sessionID: "session-1" }],
      ["session.context", { sessionID: "session-1" }],
      ["session.diff", { sessionID: "session-1" }],
      ["session.remove", { sessionID: "session-1" }],
    ]);
  } finally {
    if (previous === undefined) {
      delete process.env.OPENCODE_TEST_KEY;
    } else {
      process.env.OPENCODE_TEST_KEY = previous;
    }
  }
});

test("uses a configured credential name for a native OpenCode provider", async () => {
  let createOptions;
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: {
      async create(options) {
        createOptions = options;
        return {
          sessions: {
            async create() {
              return { id: "session-native-provider" };
            },
            async remove() {},
          },
          async close() {},
        };
      },
    },
  }));

  const handle = await factory.create(startInput());
  await handle.stop();

  assert.deepEqual(JSON.parse(createOptions.config.content), {
    providers: { openai: { settings: { apiKey: "{env:OPENCODE_TEST_KEY}" } } },
  });
});

test("turns SDK transport failures into a stable lifecycle error without request details", async () => {
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: {
      async create() {
        return {
          sessions: {
            async create() {
              return { id: "session-transport-failure" };
            },
            async context() {
              throw new Error("request https://example.test/?token=supersecret failed for private prompt text");
            },
            async remove() {},
          },
          async close() {},
        };
      },
    },
  }));

  const handle = await factory.create(startInput());
  await assert.rejects(
    handle.prompt("private prompt text"),
    (error) =>
      error.code === "opencode_session_failed" &&
      error.message === "OpenCode session communication failed" &&
      !error.message.includes("supersecret") &&
      !error.message.includes("private prompt"),
  );
  await handle.stop();
});

test("does not mark a failure after prompt submission as retryable", async () => {
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: {
      async create() {
        return {
          sessions: {
            async create() {
              return { id: "session-wait-failure" };
            },
            async context() {
              return [];
            },
            async prompt() {},
            async wait() {
              throw new Error("connection lost after the prompt was submitted");
            },
            async remove() {},
          },
          async close() {},
        };
      },
    },
  }));

  const handle = await factory.create(startInput());
  await assert.rejects(
    handle.prompt("edit a file"),
    (error) =>
      error.code === "opencode_session_failed" &&
      error.message === "OpenCode session communication failed" &&
      error.retryable === false,
  );
  await handle.stop();
});

test("configures an OpenAI-compatible OpenCode provider for an explicit endpoint", async () => {
  let createOptions;
  const factory = new OpenCodeSdkSessionFactory(async () => ({
    OpenCode: {
      async create(options) {
        createOptions = options;
        return {
          sessions: {
            async create() {
              return { id: "session-endpoint" };
            },
            async remove() {},
          },
          async close() {},
        };
      },
    },
  }));

  const input = startInput();
  input.config.models.default = {
    provider: "local-test",
    model: "test-model",
    api_key_env: "OPENCODE_TEST_KEY",
    base_url: "http://127.0.0.1:8080/v1",
  };
  const handle = await factory.create(input);
  await handle.stop();

  const provider = JSON.parse(createOptions.config.content).providers["local-test"];
  assert.equal(provider.package, "aisdk:@ai-sdk/openai-compatible");
  assert.equal(provider.settings.apiKey, "{env:OPENCODE_TEST_KEY}");
  assert.equal(new URL(provider.settings.baseURL).hostname, "127.0.0.1");
  assert.deepEqual(provider.models, { "test-model": {} });
});

test("restores the credential lease when startup cleanup also fails", async () => {
  const previous = process.env.OPENCODE_TEST_KEY;
  delete process.env.OPENCODE_TEST_KEY;
  try {
    const factory = new OpenCodeSdkSessionFactory(async () => ({
      OpenCode: {
        async create() {
          return {
            sessions: {
              async create() {
                throw new Error("session creation failed");
              },
            },
            async close() {
              throw new Error("close failed");
            },
          };
        },
      },
    }));

    await assert.rejects(factory.create(startInput()), (error) => error.code === "opencode_start_failed");
    assert.equal(process.env.OPENCODE_TEST_KEY, undefined);
  } finally {
    if (previous === undefined) {
      delete process.env.OPENCODE_TEST_KEY;
    } else {
      process.env.OPENCODE_TEST_KEY = previous;
    }
  }
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtemp, realpath, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { PiSdkSessionFactory } from "../dist/pi-sdk.js";
import { invocationUsage } from "../dist/usage.js";

const [major, minor] = process.versions.node.split(".").map(Number);
const skip =
  major > 22 || (major === 22 && minor >= 19)
    ? false
    : "Pi requires Node >=22.19";

test(
  "real SDK raw ledger retains other branches and compaction usage",
  { skip },
  async () => {
    const { SessionManager } = await import("@earendil-works/pi-coding-agent");
    const manager = SessionManager.inMemory("/tmp");
    const usage = {
      input: 10,
      output: 3,
      cacheRead: 2,
      cacheWrite: 1,
      totalTokens: 16,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    };
    const message = {
      role: "assistant",
      content: [{ type: "text", text: "answer" }],
      api: "openai-completions",
      provider: "test",
      model: "test",
      stopReason: "stop",
      timestamp: Date.now(),
      usage,
    };
    const old = manager.appendMessage(message);
    const before = manager.getEntries().map((entry) => entry.id);
    manager.appendMessage({ ...message, stopReason: "error" });
    manager.branch(old);
    const current = manager.appendMessage(message);
    manager.appendCompaction("summary", current, 50, undefined, false, usage);
    assert.ok(manager.getEntries().length > manager.getBranch().length);
    const result = invocationUsage(before, manager.getEntries());
    assert.equal(result.input_tokens, 39);
    assert.equal(result.output_tokens, 9);
    assert.equal(result.total_tokens, 48);
  },
);

test(
  "real SDK transports inclusive cache/reasoning usage without leaking previous invocations",
  { skip },
  async () => {
    const workspace = await realpath(
      await mkdtemp(join(tmpdir(), "fabric-pi-usage-")),
    );
    const requests = [];
    const server = createServer(async (request, response) => {
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      requests.push(JSON.parse(Buffer.concat(chunks).toString("utf8")));
      const common = {
        id: `chatcmpl-${requests.length}`,
        object: "chat.completion.chunk",
        created: 0,
        model: "openai/gpt-oss-20b",
      };
      const frame = (value) =>
        `data: ${JSON.stringify({ ...common, ...value })}\n\n`;
      response.writeHead(200, { "content-type": "text/event-stream" });
      response.end(
        frame({
          choices: [
            {
              index: 0,
              delta: {
                role: "assistant",
                content: `answer-${requests.length}`,
              },
              finish_reason: "stop",
            },
          ],
        }) +
          (requests.length === 3
            ? ""
            : frame({
                choices: [],
                usage: {
                  prompt_tokens: requests.length * 100,
                  completion_tokens: 20,
                  total_tokens: requests.length * 100 + 20,
                  prompt_tokens_details: {
                    cached_tokens: 12,
                    cache_write_tokens: 8,
                  },
                  completion_tokens_details: { reasoning_tokens: 7 },
                },
              })) +
          "data: [DONE]\n\n",
      );
    });
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    let session;
    try {
      session = await new PiSdkSessionFactory().create({
        agentName: "usage-test",
        baseDir: workspace,
        config: {
          models: {
            default: {
              provider: "nvidia",
              model: "openai/gpt-oss-20b",
              base_url: `http://127.0.0.1:${server.address().port}/v1`,
              api_key_env: "TEST_API_KEY",
            },
          },
          tools: { enabled: [] },
        },
        runtimeContext: {
          artifacts: {},
          environment: {
            control_location: "external_control",
            environment_id: "usage-test",
            ownership: "caller_owned",
            provider: "local",
            workspace,
            env: { TEST_API_KEY: "unused" },
          },
          invocation_id: "start",
          request_id: "start",
          runtime_id: "usage",
        },
      });
      // Observe the actual SDK settings, not just the adapter constructor arguments.
      assert.equal(
        session.session.settingsManager.getCacheWarmingMode(),
        "off",
      );
      const first = await session.prompt("Remember the nonce alpha.");
      assert.equal(first.usage.input_tokens, 100);
      assert.equal(first.usage.output_tokens, 20);
      assert.equal(first.usage.total_tokens, 120);
      assert.equal(first.usage.cost_usd, undefined);
      assert.equal(
        first.usage.extensions["nemo_fabric.detailed_token_usage.v1"]
          .cache_write_input_tokens,
        8,
      );
      assert.equal(
        first.usage.extensions["nemo_fabric.detailed_token_usage.v1"]
          .reasoning_tokens,
        7,
      );
      const second = await session.prompt("Continue from the previous answer.");
      assert.equal(second.usage.input_tokens, 200);
      assert.equal(second.usage.output_tokens, 20);
      assert.match(JSON.stringify(requests[1]), /answer-1/);
      const missing = await session.prompt("Provider omits usage this time.");
      assert.equal(missing.usage, undefined);
      assert.equal(missing.text, "answer-3");
    } finally {
      await session?.stop();
      server.closeAllConnections();
      await new Promise((resolve) => server.close(resolve));
      await rm(workspace, { recursive: true, force: true });
    }
  },
);

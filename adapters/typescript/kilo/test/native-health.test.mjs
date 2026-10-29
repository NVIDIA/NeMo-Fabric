// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { KiloSdkSessionFactory, startKiloServer } from "../dist/kilo-sdk.js";

const executable = process.env.FABRIC_KILO_EXECUTABLE;

test(
  "pinned native Kilo health stays responsive before, during, and after a blocked model invocation",
  {
    skip:
      executable === undefined
        ? "Set FABRIC_KILO_EXECUTABLE to the pinned Kilo 7.7.12 binary"
        : false,
    timeout: 60000,
  },
  async (t) => {
    const workspace = await mkdtemp(
      join(tmpdir(), "fabric-kilo-native-health-"),
    );
    t.after(() => rm(workspace, { recursive: true, force: true }));
    const entered = Promise.withResolvers();
    const release = Promise.withResolvers();
    let requests = 0;
    const model = createServer(async (request, response) => {
      if (request.url !== "/v1/chat/completions") {
        response.writeHead(404).end();
        return;
      }
      for await (const _chunk of request) {
        /* Consume the body without retaining prompts. */
      }
      requests += 1;
      entered.resolve();
      await release.promise;
      response.writeHead(200, { "Content-Type": "text/event-stream" });
      for (const [delta, finish_reason] of [
        [{ role: "assistant", content: "finished" }, null],
        [{}, "stop"],
      ]) {
        response.write(
          `data: ${JSON.stringify({ id: "native-health", object: "chat.completion.chunk", created: 0, model: "test-model", choices: [{ index: 0, delta, finish_reason }] })}\n\n`,
        );
      }
      response.end("data: [DONE]\n\n");
    });
    await new Promise((resolve) => model.listen(0, "127.0.0.1", resolve));
    t.after(() => {
      release.resolve();
      model.closeAllConnections();
      return new Promise((resolve) => model.close(resolve));
    });
    const factory = new KiloSdkSessionFactory(
      undefined,
      (directory, environment) =>
        startKiloServer(directory, environment, executable),
    );
    const handle = await factory.create({
      agentName: "native-health",
      baseDir: workspace,
      config: {
        models: {
          default: {
            provider: "native-health",
            model: "test-model",
            api_key_env: "TEST_NATIVE_KEY",
            base_url: `http://127.0.0.1:${model.address().port}/v1`,
          },
        },
        tools: { enabled: [] },
      },
      runtimeContext: {
        runtime_id: "runtime",
        invocation_id: "start",
        request_id: "start",
        artifacts: {},
        environment: {
          environment_id: "environment",
          provider: "local",
          ownership: "caller_owned",
          control_location: "external_control",
          workspace,
          env: { TEST_NATIVE_KEY: "PRIVATE-TEST-KEY" },
        },
      },
    });
    t.after(() => handle.stop());
    for (const level of ["live", "active", "ready"])
      assert.equal((await handle.check(level)).outcome, "passed");
    assert.equal(requests, 0);
    const invocation = handle.prompt("PRIVATE-PROMPT");
    // Register a handler immediately so a native prompt failure cannot become unhandled.
    const completed = invocation.then(
      (value) => ({ value }),
      (error) => ({ error }),
    );
    try {
      await Promise.race([
        entered.promise,
        completed.then(() => {
          throw new Error(
            "Native invocation ended before reaching the local model",
          );
        }),
        new Promise((_, reject) =>
          setTimeout(
            () =>
              reject(
                new Error("Native invocation did not reach the local model"),
              ),
            15000,
          ).unref(),
        ),
      ]);
      for (const level of ["live", "active", "ready"]) {
        const result = await handle.check(level);
        assert.equal(result.outcome, "passed");
        assert.equal(JSON.stringify(result).includes("PRIVATE"), false);
      }
      assert.equal(requests, 1);
    } finally {
      release.resolve();
    }
    const result = await completed;
    assert.equal(result.error, undefined);
    assert.equal(result.value.text, "finished");
    for (const level of ["live", "active", "ready"])
      assert.equal((await handle.check(level)).outcome, "passed");
    assert.equal(requests, 1);
  },
);

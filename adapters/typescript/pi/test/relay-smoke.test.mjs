// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, mkdir, readFile, readdir, realpath, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";

import { PiSdkSessionFactory } from "../dist/pi-sdk.js";
import { PiAdapterRuntime } from "../dist/runtime.js";

const execFileAsync = promisify(execFile);
const relayCommand = process.env.FABRIC_TEST_NEMO_RELAY_COMMAND;

function openAiStream(content) {
  const chunk = (delta, finishReason = null) =>
    `data: ${JSON.stringify({
      id: "chatcmpl-fabric-relay-smoke",
      object: "chat.completion.chunk",
      created: 0,
      model: "openai/gpt-oss-20b",
      choices: [{ index: 0, delta, finish_reason: finishReason }],
    })}\n\n`;
  return `${chunk({ role: "assistant" })}${chunk({ content })}${chunk({}, "stop")}data: [DONE]\n\n`;
}

async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  assert.notEqual(typeof address, "string");
  return `http://127.0.0.1:${address.port}`;
}

test(
  "runs the released Relay CLI and shipped Pi extension end to end",
  { skip: relayCommand === undefined ? "set FABRIC_TEST_NEMO_RELAY_COMMAND to the released CLI" : false },
  async () => {
    const root = await realpath(await mkdtemp(join(tmpdir(), "fabric-pi-relay-smoke-")));
    const workspace = join(root, "workspace");
    const piAgentDir = join(root, "pi-agent");
    const runtimeConfigPath = join(root, "relay-runtime.json");
    const providerRequests = [];
    const upstream = createServer(async (request, response) => {
      if (request.method !== "POST" || request.url !== "/v1/chat/completions") {
        response.writeHead(404).end();
        return;
      }
      const chunks = [];
      for await (const chunk of request) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
      }
      const payload = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      providerRequests.push(payload);
      response.writeHead(200, { "content-type": "text/event-stream" });
      response.end(openAiStream("relay smoke ok"));
    });
    const upstreamUrl = await listen(upstream);
    const previousConfigPath = process.env.FABRIC_RELAY_CONFIG_PATH;
    let runtime;
    try {
      await mkdir(workspace);
      await execFileAsync(relayCommand, ["install", "pi", "--skip-doctor"], {
        env: { ...process.env, PI_CODING_AGENT_DIR: piAgentDir },
        timeout: 10_000,
      });
      const extensionPath = join(piAgentDir, "extensions", "nemo-relay");
      await writeFile(
        runtimeConfigPath,
        JSON.stringify({
          relay: {
            config: {
              version: 1,
              components: [
                {
                  kind: "observability",
                  enabled: true,
                  config: {
                    version: 3,
                    atof: {
                      enabled: true,
                      sinks: [
                        {
                          type: "file",
                          output_directory: join(root, "atof"),
                          filename: "events.atof.jsonl",
                        },
                      ],
                    },
                    atif: {
                      enabled: true,
                      output_directory: join(root, "atif"),
                    },
                  },
                },
              ],
            },
          },
        }),
        "utf8",
      );
      process.env.FABRIC_RELAY_CONFIG_PATH = runtimeConfigPath;
      const runtimeContext = {
        artifacts: {},
        environment: {
          control_location: "external_control",
          env: { TEST_API_KEY: "credential-free-smoke" },
          environment_id: "environment-1",
          ownership: "caller_owned",
          provider: "local",
          workspace,
        },
        invocation_id: "start",
        request_id: "request-start",
        runtime_id: "relay-smoke",
        telemetry: { relay_enabled: true },
      };
      runtime = new PiAdapterRuntime(new PiSdkSessionFactory());
      await runtime.start({
        agentName: "pi-relay-smoke",
        baseDir: root,
        config: {
          harness: { settings: { relay_extension_path: extensionPath } },
          models: {
            default: {
              api_key_env: "TEST_API_KEY",
              base_url: `${upstreamUrl}/v1`,
              model: "openai/gpt-oss-20b",
              provider: "nvidia",
            },
          },
          tools: { enabled: [] },
        },
        runtimeContext,
      });
      const result = await runtime.invoke({ input: "Reply with relay smoke ok." }, runtimeContext);

      assert.equal(result.status, "succeeded");
      assert.equal(result.output.response, "relay smoke ok");
      assert.equal(result.extensions.pi_turn_started, true);
      assert.equal(providerRequests.length, 1);
      assert.equal(providerRequests[0].model, "openai/gpt-oss-20b");
      assert.equal(result.output.relay_artifacts.some((artifact) => artifact.kind === "atif"), false);

      await runtime.stop();
      runtime = undefined;

      const atifDirectory = join(root, "atif", "relay-smoke");
      const atifFiles = await readdir(atifDirectory);
      assert.equal(atifFiles.length, 1);
      const atif = JSON.parse(await readFile(join(atifDirectory, atifFiles[0]), "utf8"));
      assert.equal(atif.agent.model_name, "openai/gpt-oss-20b");
      assert.equal(
        atif.extra.observed_events.some((event) => event.metadata?.hook_event_name === "turn_end"),
        true,
      );
      const atof = await readFile(join(root, "atof", "relay-smoke", "events.atof.jsonl"), "utf8");
      assert.match(atof, /session_start/u);
      assert.match(atof, /session_shutdown/u);
    } finally {
      await runtime?.stop().catch(() => undefined);
      if (previousConfigPath === undefined) {
        delete process.env.FABRIC_RELAY_CONFIG_PATH;
      } else {
        process.env.FABRIC_RELAY_CONFIG_PATH = previousConfigPath;
      }
      await new Promise((resolve) => upstream.close(resolve));
      await rm(root, { recursive: true, force: true });
    }
  },
);

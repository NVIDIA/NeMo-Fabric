// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { writeFileSync } from "node:fs";
import { Readable, Writable } from "node:stream";

import { PROTOCOL_VERSION, agent, methods, ndJsonStream } from "@agentclientprotocol/sdk";

let turn = 0;
let mode = "build";
let model = "model-a";
let effort = "medium";
let mcpCount = 0;
let ready = false;

function configOptions() {
  return [
    {
      id: "model",
      name: "Model",
      type: "select",
      currentValue: model,
      options: [
        { value: "model-a", name: "Model A" },
        { value: "model-b", name: "Model B" },
      ],
    },
    {
      id: "effortLevel",
      name: "Effort",
      type: "select",
      currentValue: effort,
      options: [
        { value: "low", name: "Low" },
        { value: "medium", name: "Medium" },
        { value: "high", name: "High" },
      ],
    },
  ];
}

const app = agent({ name: "fabric-acp-test-agent" });
app.onRequest(methods.agent.initialize, ({ params }) => ({
  protocolVersion: process.env.TEST_PROTOCOL_MISMATCH === "1" ? params.protocolVersion + 1 : PROTOCOL_VERSION,
  agentCapabilities: {
    mcpCapabilities: { http: true, sse: true },
    sessionCapabilities: { close: process.env.TEST_CLOSE_CAPABILITY_NULL === "1" ? null : {} },
  },
  agentInfo: { name: "fixture-agent", version: "1.0.0" },
}));
app.onRequest(methods.agent.session.new, ({ params, client }) => {
  mcpCount = params.mcpServers.length;
  if (process.env.TEST_READINESS_NOTIFICATION === "1") {
    setTimeout(() => {
      ready = true;
      void client.notify("_test/mcp/status", {
        sessionId: "fixture-session",
        servers: [{ name: "fixture", status: "connected" }],
      });
    }, 25);
  }
  return {
    sessionId: "fixture-session",
    modes: {
      currentModeId: mode,
      availableModes: [
        { id: "build", name: "Build" },
        { id: "plan", name: "Plan" },
      ],
    },
    configOptions: configOptions(),
  };
});
app.onRequest(methods.agent.session.setMode, ({ params }) => {
  mode = params.modeId;
  return {};
});
app.onRequest(methods.agent.session.setConfigOption, ({ params }) => {
  if (params.configId === "model") model = String(params.value);
  if (params.configId === "effortLevel") effort = String(params.value);
  return { configOptions: configOptions() };
});
app.onRequest(methods.agent.session.prompt, async ({ params, client }) => {
  const input = params.prompt.find((block) => block.type === "text")?.text ?? "";
  if (input === "hang") return await new Promise(() => {});
  let permission = "none";
  if (input === "permission") {
    const response = await client.request(methods.client.session.requestPermission, {
      sessionId: params.sessionId,
      toolCall: { toolCallId: "tool-1", name: "shell" },
      options: [
        { optionId: "allow", name: "Allow once", kind: "allow_once" },
        { optionId: "reject", name: "Reject once", kind: "reject_once" },
      ],
    });
    permission = response.outcome.outcome === "selected" ? response.outcome.optionId : "cancelled";
  }
  turn += 1;
  await client.notify(methods.client.session.update, {
    sessionId: params.sessionId,
    update: {
      sessionUpdate: "agent_message_chunk",
      content: {
        type: "text",
        text: `${input}:${turn}:${mode}:${model}:${effort}:mcp=${mcpCount}:permission=${permission}:ready=${ready}`,
      },
    },
  });
  const usage = {
    inputTokens: turn * 2,
    outputTokens: turn * 3,
    totalTokens: turn * 5,
    ...(input.startsWith("cache-") ? { cachedReadTokens: turn * 7 } : {}),
  };
  return { stopReason: "end_turn", ...(input === "usage-missing" ? {} : { usage }) };
});
app.onNotification(methods.agent.session.cancel, () => {});
app.onRequest(methods.agent.session.close, () => {
  if (process.env.TEST_CLOSE_MARKER !== undefined) writeFileSync(process.env.TEST_CLOSE_MARKER, "closed\n");
  return {};
});

process.on("SIGTERM", () => {
  if (process.env.TEST_EXIT_MARKER !== undefined) writeFileSync(process.env.TEST_EXIT_MARKER, "stopped\n");
  process.exit(0);
});

const stream = ndJsonStream(
  Writable.toWeb(process.stdout),
  Readable.toWeb(process.stdin),
);
const connection = app.connect(stream);
await connection.closed;

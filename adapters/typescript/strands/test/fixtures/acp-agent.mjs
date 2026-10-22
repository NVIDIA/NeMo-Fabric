// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { randomUUID } from "node:crypto";
import { writeFile } from "node:fs/promises";
import { Readable, Writable } from "node:stream";
import {
  agent,
  methods,
  ndJsonStream,
  PROTOCOL_VERSION,
} from "@agentclientprotocol/sdk";

let history = [];
let servers = [];
const id = randomUUID();
if (process.env.TEST_STRANDS_STATE) {
  await writeFile(
    process.env.TEST_STRANDS_STATE,
    JSON.stringify({
      pid: process.pid,
      home: process.env.HOME,
      args: process.argv.slice(2),
      endpoint: process.env.LITELLM_BASE_URL,
      hasKey: Boolean(process.env.LITELLM_API_KEY),
      unrelatedSecret: process.env.TEST_UNRELATED_SECRET,
    }),
  );
}
const app = agent({ name: "strands-fixture" });
app.onRequest(methods.agent.initialize, () => ({
  protocolVersion: PROTOCOL_VERSION,
  agentCapabilities: { mcpCapabilities: { http: true, sse: true } },
}));
app.onRequest(methods.agent.session.new, ({ params }) => {
  if (process.env.TEST_STRANDS_FAIL_START === "1")
    throw new Error("PRIVATE_STARTUP_DETAIL");
  servers = params.mcpServers;
  return { sessionId: id };
});
app.onRequest(methods.agent.session.prompt, async ({ params, client }) => {
  const text = params.prompt[0].text;
  if (text === "hang") return await new Promise(() => {});
  if (text === "crash") process.exit(2);
  const response =
    text === "recall"
      ? (history[0] ?? "EMPTY")
      : text === "mcp"
        ? JSON.stringify(servers)
        : text;
  history.push(text);
  await client.notify(methods.client.session.update, {
    sessionId: id,
    update: {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: response },
    },
  });
  // Deliberately unequal per-turn counters expose accidental cumulative differencing.
  return {
    stopReason: text === "refuse" ? "refusal" : "end_turn",
    usage: {
      inputTokens: 10 + history.length,
      outputTokens: 3,
      totalTokens: 13 + history.length,
      cachedReadTokens: 2,
      cachedWriteTokens: 1,
    },
  };
});
await app.connect(
  ndJsonStream(Writable.toWeb(process.stdout), Readable.toWeb(process.stdin)),
).closed;

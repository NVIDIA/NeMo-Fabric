// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

let buffered = "";

function send(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function result(id, value) {
  send({ jsonrpc: "2.0", id, result: value });
}

function handle(message) {
  if (message.method === "initialize") {
    result(message.id, {
      protocolVersion: message.params?.protocolVersion ?? "2024-11-05",
      capabilities: { tools: {} },
      serverInfo: { name: "fabric-qwen-test-mcp", version: "1.0.0" },
    });
    return;
  }
  if (message.method === "tools/list") {
    result(message.id, {
      tools: [{
        name: "echo",
        description: "Echo text for the Fabric Qwen MCP test",
        inputSchema: {
          type: "object",
          properties: { text: { type: "string" } },
          required: ["text"],
          additionalProperties: false,
        },
      }],
    });
    return;
  }
  if (message.method === "tools/call") {
    result(message.id, {
      content: [{ type: "text", text: `echo:${message.params?.arguments?.text ?? ""}` }],
    });
    return;
  }
  if (message.method === "ping") {
    result(message.id, {});
    return;
  }
  if (message.id !== undefined) {
    send({
      jsonrpc: "2.0",
      id: message.id,
      error: { code: -32601, message: `Method not found: ${message.method}` },
    });
  }
}

process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  buffered += chunk;
  while (true) {
    const newline = buffered.indexOf("\n");
    if (newline < 0) break;
    const line = buffered.slice(0, newline).trim();
    buffered = buffered.slice(newline + 1);
    if (line.length > 0) handle(JSON.parse(line));
  }
});

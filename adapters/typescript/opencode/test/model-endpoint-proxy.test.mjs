// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { createServer } from "node:http";
import test from "node:test";
import { gzipSync } from "node:zlib";

import { forwardedHeaders, ModelEndpointProxy } from "../dist/model-endpoint-proxy.js";

async function listen(server) {
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
  const address = server.address();
  assert.notEqual(address, null);
  assert.notEqual(typeof address, "string");
  return `http://127.0.0.1:${address.port}`;
}

async function close(server) {
  await new Promise((resolve, reject) => {
    server.close((error) => (error === undefined ? resolve() : reject(error)));
  });
}

test("removes hop-by-hop and framing request headers", () => {
  const headers = forwardedHeaders({
    authorization: "Bearer test-key",
    connection: "keep-alive, x-local-only",
    "content-length": "42",
    host: "proxy.example",
    "keep-alive": "timeout=5",
    "proxy-authenticate": "Basic realm=upstream",
    "proxy-authorization": "Basic credentials",
    te: "trailers",
    trailer: "x-checksum",
    "transfer-encoding": "chunked",
    upgrade: "websocket",
    "x-local-only": "do-not-forward",
  });

  assert.deepEqual(Object.fromEntries(headers), { authorization: "Bearer test-key" });
});

test("forwards decoded compressed responses without stale content headers", async () => {
  const responseBody = JSON.stringify({ response: "compressed provider response" });
  const upstream = createServer((request, response) => {
    assert.equal(request.url, "/v1/chat/completions");
    const compressed = gzipSync(responseBody);
    response.writeHead(200, {
      connection: "close",
      "content-encoding": "gzip",
      "content-length": String(compressed.length),
      "content-type": "application/json",
    });
    response.end(compressed);
  });
  const upstreamUrl = await listen(upstream);
  const proxy = await ModelEndpointProxy.create(`${upstreamUrl}/v1`);
  try {
    const response = await fetch(`${proxy.url}/chat/completions`, { method: "POST", body: "{}" });

    assert.equal(response.status, 200);
    assert.equal(response.headers.get("content-encoding"), null);
    assert.equal(response.headers.get("content-length"), null);
    assert.equal(await response.text(), responseBody);
  } finally {
    await proxy.close();
    await close(upstream);
  }
});

test("preserves configured endpoint query parameters when forwarding a request", async () => {
  const upstream = createServer((request, response) => {
    assert.equal(request.url, "/v1/chat/completions?api-version=fixed&trace=on");
    response.writeHead(200, { "content-type": "application/json" });
    response.end("{}");
  });
  const upstreamUrl = await listen(upstream);
  const proxy = await ModelEndpointProxy.create(`${upstreamUrl}/v1?api-version=fixed`);
  try {
    const response = await fetch(`${proxy.url}/chat/completions?trace=on`, { method: "POST", body: "{}" });

    assert.equal(response.status, 200);
  } finally {
    await proxy.close();
    await close(upstream);
  }
});

test("terminates a partial upstream response without hanging or crashing", async () => {
  const upstream = createServer((request, response) => {
    assert.equal(request.url, "/v1/chat/completions");
    response.writeHead(200, { "content-type": "text/event-stream" });
    response.write("data: {\\\"partial\\\":true}\\n\\n");
    setImmediate(() => response.socket?.destroy());
  });
  const upstreamUrl = await listen(upstream);
  const proxy = await ModelEndpointProxy.create(`${upstreamUrl}/v1`);
  try {
    const outcome = await Promise.race([
      fetch(`${proxy.url}/chat/completions`, { method: "POST", body: "{}" })
        .then((response) => response.text())
        .then(
          () => "completed",
          () => "failed",
        ),
      new Promise((resolve) => setTimeout(() => resolve("timed_out"), 2000)),
    ]);

    assert.equal(outcome, "failed");
  } finally {
    await proxy.close();
    await close(upstream);
  }
});

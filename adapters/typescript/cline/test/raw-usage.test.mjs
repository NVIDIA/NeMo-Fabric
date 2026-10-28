// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";

import { RawUsageCollector } from "../dist/raw-usage.js";

const endpoint = "https://example.test/v1/chat/completions";
const request = { method: "POST", body: JSON.stringify({ stream: true, stream_options: { include_usage: true } }) };
const finished = 'data: {"choices":[{"finish_reason":"stop"}]}\n\n';
const done = "data: [DONE]\n\n";

function sse(usage) {
  return `${finished}data: ${JSON.stringify({ choices: [], usage })}\n\n${done}`;
}

async function withResponse(response, run) {
  const original = globalThis.fetch;
  globalThis.fetch = async () => response;
  try { await run(); } finally { globalThis.fetch = original; }
}

test("raw NVIDIA usage preserves reported zero and forwards the stream unchanged", async () => {
  const raw = sse({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0,
    prompt_tokens_details: { cached_tokens: 0 }, completion_tokens_details: { reasoning_tokens: 0 } });
  await withResponse(new Response(raw, { headers: { "content-type": "text/event-stream" } }), async () => {
    const collector = new RawUsageCollector("https://example.test/v1");
    const turn = collector.begin();
    const response = await collector.fetch(endpoint, request);
    assert.equal(await response.text(), raw);
    assert.deepEqual(collector.end(turn), {
      input_tokens: 0, output_tokens: 0, total_tokens: 0,
      cached_input_tokens: 0, input_tokens_include_cache: true,
    });
  });
});

test("raw NVIDIA input stays inclusive when cache detail is absent", async () => {
  const raw = sse({ prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 });
  await withResponse(new Response(raw, { headers: { "content-type": "text/event-stream" } }), async () => {
    const collector = new RawUsageCollector("https://example.test/v1");
    const turn = collector.begin();
    await (await collector.fetch(endpoint, request)).text();
    assert.deepEqual(collector.end(turn), {
      input_tokens: 7, input_tokens_include_cache: true, output_tokens: 3, total_tokens: 10,
    });
  });
});

test("buffered small events do not invalidate complete usage", async () => {
  const raw = `${'data: {"choices":[]}\n\n'.repeat(4000)}${sse({
    prompt_tokens: 7, completion_tokens: 3, total_tokens: 10,
  })}`;
  await withResponse(new Response(new ReadableStream({ start(controller) {
    controller.enqueue(new TextEncoder().encode(raw));
    controller.close();
  } }), { headers: { "content-type": "text/event-stream" } }), async () => {
    const collector = new RawUsageCollector("https://example.test/v1");
    const turn = collector.begin();
    assert.equal(await (await collector.fetch(endpoint, request)).text(), raw);
    assert.deepEqual(collector.end(turn), {
      input_tokens: 7, input_tokens_include_cache: true, output_tokens: 3, total_tokens: 10,
    });
  });
});

test("truncated raw stream keeps usage unknown and forwards received bytes", async () => {
  const chunks = [finished, 'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}\n\n'];
  await withResponse(new Response(new ReadableStream({ start(controller) {
    for (const chunk of chunks) controller.enqueue(new TextEncoder().encode(chunk));
    controller.close();
  } }), { headers: { "content-type": "text/event-stream" } }), async () => {
    const collector = new RawUsageCollector("https://example.test/v1");
    const turn = collector.begin();
    assert.equal(await (await collector.fetch(endpoint, request)).text(), chunks.join(""));
    assert.equal(collector.end(turn), undefined);
  });
});

test("raw NVIDIA usage remains unknown for failed attempts, missing usage, and invalid accounting", async (t) => {
  const cases = [
    { name: "failed retry", status: 503, body: "provider unavailable", contentType: "text/plain" },
    { name: "missing usage", status: 200, body: `${finished}${done}`, contentType: "text/event-stream" },
    { name: "cache exceeds input", status: 200, body: sse({ prompt_tokens: 2, completion_tokens: 1, total_tokens: 3,
      prompt_tokens_details: { cached_tokens: 3 } }), contentType: "text/event-stream" },
    { name: "total disagrees", status: 200, body: sse({ prompt_tokens: 2, completion_tokens: 1, total_tokens: 4 }),
      contentType: "text/event-stream" },
  ];
  for (const sample of cases) {
    await t.test(sample.name, async () => {
      await withResponse(new Response(sample.body, {
        status: sample.status, headers: { "content-type": sample.contentType },
      }), async () => {
        const collector = new RawUsageCollector("https://example.test/v1");
        const turn = collector.begin();
        const response = await collector.fetch(endpoint, request);
        assert.equal(await response.text(), sample.body);
        assert.equal(collector.end(turn), undefined);
      });
    });
  }
});

test("raw NVIDIA usage remains unknown for overlapping turns or an uncaptured request", async () => {
  const raw = sse({ prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 });
  await withResponse(new Response(raw, { headers: { "content-type": "text/event-stream" } }), async () => {
    const collector = new RawUsageCollector("https://example.test/v1");
    const first = collector.begin();
    const second = collector.begin();
    const response = await collector.fetch(endpoint, request);
    await response.text();
    assert.equal(collector.end(first), undefined);
    assert.equal(collector.end(second), undefined);
  });
});

test("a third turn stays incomplete until all overlapping turns end", async () => {
  const raw = sse({ prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 });
  await withResponse(new Response(raw, { headers: { "content-type": "text/event-stream" } }), async () => {
    const collector = new RawUsageCollector("https://example.test/v1");
    const first = collector.begin();
    const second = collector.begin();
    assert.equal(collector.end(second), undefined);
    const third = collector.begin();
    await (await collector.fetch(endpoint, request)).text();
    assert.equal(collector.end(third), undefined);
    assert.equal(collector.end(first), undefined);
  });
});

test("a failed model retry keeps earlier valid counts incomplete", async () => {
  const original = globalThis.fetch;
  let requests = 0;
  globalThis.fetch = async () => ++requests === 1
    ? new Response(sse({ prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 }),
      { headers: { "content-type": "text/event-stream" } })
    : new Response("provider unavailable", { status: 503 });
  try {
    const collector = new RawUsageCollector("https://example.test/v1");
    const turn = collector.begin();
    await (await collector.fetch(endpoint, request)).text();
    await (await collector.fetch(endpoint, request)).text();
    assert.equal(collector.end(turn), undefined);
  } finally {
    globalThis.fetch = original;
  }
});

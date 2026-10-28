// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { OpenCodeUsageCollector } from "../dist/usage.js";

const KEY = "nvidia.fabric.opencode.usage_evidence.v1";
const request = () => new Request("https://model.example/v1/chat/completions", { method: "POST" });
const frame = (usage) => `data: ${JSON.stringify({ choices: [], usage })}\n\n`;
const stream = (data) => new Response(data, { headers: { "content-type": "text/event-stream" } });

async function fixture() {
  const collector = new OpenCodeUsageCollector();
  const hooks = new Map();
  const hook = async (name, callback) => { hooks.set(name, callback); return { async dispose() {} }; };
  await collector.plugin().setup({ session: {
    async get() { return { parentID: "root" }; },
    hook,
  }, tool: { hook } });
  const start = (req = request(), kind = "primary", sessionID = "root") => {
    hooks.get("http.request")({ sessionID, kind, request: req });
    return req;
  };
  const respond = async (req, body) => {
    const event = { sessionID: "root", request: req, response: body };
    await hooks.get("http.response")(event);
    return event.response;
  };
  return { collector, hooks, start, respond };
}

test("sums physical requests once and preserves raw zero and cache conventions", async () => {
  const { collector, start, respond } = await fixture();
  const invocation = collector.begin("root");
  const data = frame({ prompt_tokens: 100, completion_tokens: 20 }) + frame({
    prompt_tokens: 120, completion_tokens: 30, total_tokens: 150,
    prompt_tokens_details: { cached_tokens: 20 }, completion_tokens_details: { reasoning_tokens: 5 },
  }) + "data: [DONE]\n\n";
  assert.equal(await (await respond(start(), stream(data))).text(), data);
  await (await respond(start(), new Response(JSON.stringify({ usage: {
    prompt_tokens: 0, completion_tokens: 0, total_tokens: 0,
  } }), { headers: { "content-type": "application/json" } }))).text();
  const usage = collector.end(invocation);
  assert.deepEqual(usage, { extensions: { [KEY]: {
    observed: { input_tokens: 120, output_tokens: 30, total_tokens: 150, cached_input_tokens: 20, reasoning_tokens: 5 },
    gaps: ["scope_unverified"],
  } } });
});

test("does not replace omitted or invalid counters with zero", async () => {
  const { collector, start, respond } = await fixture();
  const invocation = collector.begin("root");
  await (await respond(start(), stream(frame({ prompt_tokens: 0, completion_tokens: -1, total_tokens: Number.MAX_SAFE_INTEGER + 1 }) + "data: [DONE]\n\n"))).text();
  assert.deepEqual(collector.end(invocation).extensions[KEY], {
    observed: { input_tokens: 0 }, gaps: ["scope_unverified", "missing_usage"],
  });
});

test("preserves usage and response bytes across valid SSE line endings", async () => {
  for (const newline of ["\r", "\r\n", "\n"]) {
    const { collector, start, respond } = await fixture();
    const invocation = collector.begin("root");
    const data = [
      'data: {"choices":[],"usage":',
      'data: {"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}',
      "",
      "data: [DONE]",
      "",
      "",
    ].join(newline);
    const bytes = new TextEncoder().encode(data);
    let offset = 0;
    const body = new ReadableStream({ pull(controller) {
      if (offset === bytes.length) controller.close();
      else controller.enqueue(bytes.subarray(offset, ++offset));
    } });
    const response = await respond(start(), new Response(body, { headers: { "content-type": "text/event-stream" } }));
    assert.equal(await response.text(), data);
    assert.deepEqual(collector.end(invocation).extensions[KEY], {
      observed: { input_tokens: 12, output_tokens: 3, total_tokens: 15 },
      gaps: ["scope_unverified"],
    });
  }
});

test("ignores the compatible proxy keepalive accepted by OpenCode", async () => {
  const { collector, start, respond } = await fixture();
  const invocation = collector.begin("root");
  const data = "data: : keepalive\n\n" + frame({ prompt_tokens: 12, completion_tokens: 3 }) + "data: [DONE]\n\n";
  assert.equal(await (await respond(start(), stream(data))).text(), data);
  assert.deepEqual(collector.end(invocation).extensions[KEY], {
    observed: { input_tokens: 12, output_tokens: 3 }, gaps: ["scope_unverified"],
  });
});

test("ignores an empty SSE flush without marking valid usage missing", async () => {
  const { collector, start, respond } = await fixture();
  const invocation = collector.begin("root");
  const data = "data:\n\n" + frame({ prompt_tokens: 12, completion_tokens: 3 }) + "data: [DONE]\n\n";
  await (await respond(start(), stream(data))).text();
  assert.deepEqual(collector.end(invocation).extensions[KEY], {
    observed: { input_tokens: 12, output_tokens: 3 }, gaps: ["scope_unverified"],
  });
});

test("recognizes a mixed-case SSE media type with parameters", async () => {
  const { collector, start, respond } = await fixture();
  const invocation = collector.begin("root");
  const data = frame({ prompt_tokens: 12, completion_tokens: 3 }) + "data: [DONE]\n\n";
  await (await respond(start(), new Response(data, { headers: { "content-type": "Text/Event-Stream; charset=utf-8" } }))).text();
  assert.deepEqual(collector.end(invocation).extensions[KEY], {
    observed: { input_tokens: 12, output_tokens: 3 }, gaps: ["scope_unverified"],
  });
});

test("bounds oversized SSE events and preserves later usage and response bytes", async () => {
  const usage = { prompt_tokens: 900, completion_tokens: 90 };
  const oversizedLine = `data: ${JSON.stringify({ padding: "x".repeat(1 << 20), usage })}\n\n`;
  const oversizedEvent = `data: ${" ".repeat(1 << 19)}\ndata: ${" ".repeat(1 << 19)}\n${frame(usage)}`;
  for (const oversized of [oversizedLine, oversizedEvent]) {
    for (const recover of [false, true]) {
      const { collector, start, respond } = await fixture();
      const invocation = collector.begin("root");
      const data = oversized + (recover ? frame({ prompt_tokens: 12, completion_tokens: 3 }) : "") + "data: [DONE]\n\n";
      const bytes = new TextEncoder().encode(data);
      let offset = 0;
      const body = new ReadableStream({ pull(controller) {
        if (offset === bytes.length) controller.close();
        else {
          const end = Math.min(offset + 65536, bytes.length);
          controller.enqueue(bytes.subarray(offset, end));
          offset = end;
        }
      } });
      const response = await respond(start(), new Response(body, { headers: { "content-type": "text/event-stream" } }));
      assert.deepEqual(new Uint8Array(await response.arrayBuffer()), bytes);
      assert.deepEqual(collector.end(invocation).extensions[KEY], {
        observed: recover ? { input_tokens: 12, output_tokens: 3 } : {},
        gaps: ["scope_unverified", "missing_usage"],
      });
    }
  }
});

test("omits usage from an oversized JSON body without changing its bytes", async () => {
  const { collector, start, respond } = await fixture();
  const invocation = collector.begin("root");
  const data = JSON.stringify({ padding: "x".repeat(1 << 20), usage: { prompt_tokens: 12, completion_tokens: 3 } });
  const response = await respond(start(), new Response(data, { headers: { "content-type": "application/json" } }));
  assert.equal(await response.text(), data);
  assert.deepEqual(collector.end(invocation).extensions[KEY], {
    observed: {}, gaps: ["scope_unverified", "missing_usage"],
  });
});

test("collects root-session compaction usage", async () => {
  const { collector, start, respond } = await fixture();
  const invocation = collector.begin("root");
  const data = frame({ prompt_tokens: 12, completion_tokens: 3 }) + "data: [DONE]\n\n";
  await (await respond(start(undefined, "compaction"), stream(data))).text();
  assert.deepEqual(collector.end(invocation).extensions[KEY], {
    observed: { input_tokens: 12, output_tokens: 3 }, gaps: ["scope_unverified"],
  });
});

test("retains observations when another physical attempt fails or remains unfinished", async () => {
  const { collector, start, respond } = await fixture();
  const invocation = collector.begin("root");
  await (await respond(start(), stream(frame({ prompt_tokens: 12, completion_tokens: 3 }) + "data: [DONE]\n\n"))).text();
  await respond(start(), new Response("failed", { status: 503 }));
  start();
  assert.deepEqual(collector.end(invocation).extensions[KEY], {
    observed: { input_tokens: 12, output_tokens: 3 },
    gaps: ["scope_unverified", "missing_usage", "unfinished_work"],
  });
});

test("freezes returned evidence and keeps late responses out of the next invocation", async () => {
  const { collector, start, respond } = await fixture();
  const first = collector.begin("root");
  const earlier = start();
  const snapshot = collector.end(first);
  const saved = structuredClone(snapshot);
  const second = collector.begin("root");
  await (await respond(earlier, stream(frame({ prompt_tokens: 900, completion_tokens: 90 }) + "data: [DONE]\n\n"))).text();
  await (await respond(start(), stream(frame({ prompt_tokens: 20, completion_tokens: 2 }) + "data: [DONE]\n\n"))).text();
  assert.deepEqual(snapshot, saved);
  assert.deepEqual(collector.end(second).extensions[KEY].observed, { input_tokens: 20, output_tokens: 2 });
});

test("excludes title and child requests and suppresses ambiguous later root continuations", async () => {
  const { collector, hooks, start, respond } = await fixture();
  const first = collector.begin("root");
  await (await respond(start(undefined, "title"), stream(frame({ prompt_tokens: 500, completion_tokens: 50 }) + "data: [DONE]\n\n"))).text();
  await hooks.get("prompt")({ sessionID: "child" });
  await (await respond(start(undefined, "primary", "child"), stream(frame({ prompt_tokens: 600, completion_tokens: 60 }) + "data: [DONE]\n\n"))).text();
  assert.deepEqual(collector.end(first).extensions[KEY].observed, {});
  const second = collector.begin("root");
  await (await respond(start(), stream(frame({ prompt_tokens: 700, completion_tokens: 70 }) + "data: [DONE]\n\n"))).text();
  assert.deepEqual(collector.end(second).extensions[KEY].observed, {});
});

test("marks truncated streams and preserves observations before the interruption", async () => {
  const { collector, start, respond } = await fixture();
  const invocation = collector.begin("root");
  await (await respond(start(), stream(frame({ prompt_tokens: 12, completion_tokens: 3 })))).text();
  assert.deepEqual(collector.end(invocation).extensions[KEY], {
    observed: { input_tokens: 12, output_tokens: 3 }, gaps: ["scope_unverified", "unfinished_work"],
  });
});

test("omits overflowing sums and leaves unsupported responses unknown", async () => {
  const { collector, start, respond } = await fixture();
  const invocation = collector.begin("root");
  for (let index = 0; index < 2; index += 1) {
    await (await respond(start(), stream(frame({ prompt_tokens: Number.MAX_SAFE_INTEGER, completion_tokens: 0 }) + "data: [DONE]\n\n"))).text();
  }
  await (await respond(start(new Request("https://model.example/v1/messages")), new Response("{}"))).text();
  assert.deepEqual(collector.end(invocation).extensions[KEY], {
    observed: { output_tokens: 0 }, gaps: ["scope_unverified", "missing_usage"],
  });
});

test("retains usage but marks an errored stream unfinished after a finish frame", async () => {
  const { collector, start, respond } = await fixture();
  const invocation = collector.begin("root");
  let sent = false;
  const body = new ReadableStream({ pull(controller) {
    if (sent) { controller.error(new Error("connection reset")); return; }
    sent = true;
    controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({
      choices: [{ finish_reason: "stop" }], usage: { prompt_tokens: 12, completion_tokens: 3 },
    })}\n\n`));
  } }, { highWaterMark: 0 });
  const response = await respond(start(), new Response(body, { headers: { "content-type": "text/event-stream" } }));
  await assert.rejects(response.text(), /connection reset/);
  assert.deepEqual(collector.end(invocation).extensions[KEY], {
    observed: { input_tokens: 12, output_tokens: 3 }, gaps: ["scope_unverified", "unfinished_work"],
  });
});

test("distinguishes early cancellation from discarding a terminal stream tail", async () => {
  for (const terminal of [false, true]) {
    const { collector, start, respond } = await fixture();
    const invocation = collector.begin("root");
    const data = `data: ${JSON.stringify({
      choices: terminal ? [{ finish_reason: "stop" }] : [],
      usage: { prompt_tokens: 12, completion_tokens: 3 },
    })}\n\n`;
    const reader = (await respond(start(), stream(data))).body.getReader();
    await reader.read();
    await reader.cancel();
    assert.deepEqual(collector.end(invocation).extensions[KEY].gaps,
      terminal ? ["scope_unverified"] : ["scope_unverified", "unfinished_work"]);
  }
});

test("excludes later root work after a background shell while preserving ordinary warm prompts", async () => {
  for (const [tool, status, suppressLater] of [["shell", "running", true], ["shell", "completed", false], ["read", "running", false]]) {
    const { collector, hooks, start, respond } = await fixture();
    const first = collector.begin("root");
    await (await respond(start(), stream(frame({ prompt_tokens: 12, completion_tokens: 3 }) + "data: [DONE]\n\n"))).text();
    await hooks.get("execute.after")?.({ sessionID: "root", tool, status: "completed", result: { output: { status } } });
    const snapshot = collector.end(first);
    const second = collector.begin("root");
    await (await respond(start(), stream(frame({ prompt_tokens: 999, completion_tokens: 99 }) + "data: [DONE]\n\n"))).text();
    assert.deepEqual(collector.end(second).extensions[KEY].observed,
      suppressLater ? {} : { input_tokens: 999, output_tokens: 99 });
    assert.deepEqual(snapshot.extensions[KEY].observed, { input_tokens: 12, output_tokens: 3 });
  }
});

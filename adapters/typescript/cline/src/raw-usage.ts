// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentUsage } from "nemo-fabric-adapter-contract";

const MAX_EVENT_CHARS = 64 * 1024;

interface Counts {
  input: number;
  output: number;
  cached?: number;
}

interface Attempt {
  counts?: Counts;
  complete: boolean;
}

export interface RawUsageTurn {
  attempts: Attempt[];
  invalid: boolean;
}

function count(value: unknown): value is number {
  return Number.isSafeInteger(value) && typeof value === "number" && value >= 0;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function parseCounts(value: unknown): Counts | undefined {
  const usage = record(value);
  if (usage === undefined || !count(usage.prompt_tokens) || !count(usage.completion_tokens)) return;
  const input = usage.prompt_tokens;
  const output = usage.completion_tokens;
  const total = input + output;
  if (!Number.isSafeInteger(total) || !count(usage.total_tokens) || usage.total_tokens !== total) return;
  const promptDetails = usage.prompt_tokens_details == null ? undefined : record(usage.prompt_tokens_details);
  const completionDetails = usage.completion_tokens_details == null ? undefined : record(usage.completion_tokens_details);
  if ((usage.prompt_tokens_details != null && promptDetails === undefined) ||
      (usage.completion_tokens_details != null && completionDetails === undefined)) return;
  const cached = promptDetails?.cached_tokens;
  const reasoning = completionDetails?.reasoning_tokens;
  if ((cached != null && (!count(cached) || cached > input)) ||
      (reasoning != null && (!count(reasoning) || reasoning > output))) return;
  return { input, output, ...(cached == null ? {} : { cached }) };
}

function observeStream(body: ReadableStream<Uint8Array>, attempt: Attempt): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  let pending = "";
  let data: string[] = [];
  let finished = false;
  let done = false;
  let invalid = false;
  let counts: Counts | undefined;

  function event(): void {
    if (data.length === 0) return;
    const value = data.join("\n");
    data = [];
    if (done) { invalid = true; return; }
    if (value === "[DONE]") { done = true; return; }
    let payload: Record<string, unknown> | undefined;
    try { payload = record(JSON.parse(value)); } catch { invalid = true; return; }
    if (payload === undefined || payload.error !== undefined) { invalid = true; return; }
    if (payload.usage !== undefined && payload.usage !== null) {
      if (counts !== undefined || !finished || !Array.isArray(payload.choices) || payload.choices.length !== 0) {
        invalid = true;
        return;
      }
      counts = parseCounts(payload.usage);
      if (counts === undefined) invalid = true;
    }
    if (Array.isArray(payload.choices)) {
      for (const choice of payload.choices) {
        const item = record(choice);
        if (item?.finish_reason != null) {
          if (counts !== undefined || typeof item.finish_reason !== "string" ||
              item.finish_reason.length === 0 || item.finish_reason === "error") invalid = true;
          else finished = true;
        }
      }
    }
  }

  function feed(text: string): void {
    pending += text;
    if (pending.length > MAX_EVENT_CHARS) { invalid = true; pending = ""; data = []; return; }
    let newline = pending.indexOf("\n");
    while (newline !== -1) {
      const line = pending.slice(0, newline).replace(/\r$/, "");
      pending = pending.slice(newline + 1);
      if (line === "") event();
      else if (line.startsWith("data:")) {
        data.push(line.slice(5).trimStart());
        if (data.join("\n").length > MAX_EVENT_CHARS) { invalid = true; data = []; }
      }
      newline = pending.indexOf("\n");
    }
  }

  return body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      if (!invalid) {
        if (chunk.byteLength > MAX_EVENT_CHARS) invalid = true;
        else feed(decoder.decode(chunk, { stream: true }));
      }
      controller.enqueue(chunk);
    },
    flush() {
      if (!invalid) feed(decoder.decode());
      if (pending.length > 0 || data.length > 0) invalid = true;
      if (!invalid && finished && done && counts !== undefined) {
        attempt.counts = counts;
        attempt.complete = true;
      }
    },
  }));
}

/** Captures only the local NVIDIA OpenAI-compatible route in Cline 0.0.83. */
export class RawUsageCollector {
  private active?: RawUsageTurn;
  private readonly inFlight = new Set<RawUsageTurn>();
  private readonly endpoint: string;

  constructor(baseUrl: string) {
    this.endpoint = `${baseUrl.replace(/\/$/, "")}/chat/completions`;
  }

  begin(): RawUsageTurn {
    const turn: RawUsageTurn = { attempts: [], invalid: false };
    if (this.inFlight.size > 0) {
      for (const pending of this.inFlight) pending.invalid = true;
      turn.invalid = true;
      this.active = undefined;
    } else {
      this.active = turn;
    }
    this.inFlight.add(turn);
    return turn;
  }

  end(turn: RawUsageTurn): AgentUsage | undefined {
    this.inFlight.delete(turn);
    if (this.active === turn) this.active = undefined;
    if (turn.invalid || turn.attempts.length === 0 || turn.attempts.some((attempt) => !attempt.complete)) return;
    let input = 0;
    let output = 0;
    let cached = 0;
    let allCached = true;
    for (const attempt of turn.attempts) {
      const counts = attempt.counts;
      if (counts === undefined) return;
      input += counts.input;
      output += counts.output;
      if (counts.cached === undefined) allCached = false;
      else cached += counts.cached;
    }
    const total = input + output;
    if (![input, output, total, cached].every(Number.isSafeInteger)) return;
    return {
      input_tokens: input,
      input_tokens_include_cache: true,
      output_tokens: output,
      total_tokens: total,
      ...(allCached ? { cached_input_tokens: cached } : {}),
    };
  }

  fetch: typeof fetch = async (resource, init) => {
    const turn = this.active;
    const attempt: Attempt = { complete: false };
    if (turn !== undefined) {
      turn.attempts.push(attempt);
      const url = resource instanceof Request ? resource.url : String(resource);
      const method = init?.method ?? (resource instanceof Request ? resource.method : "GET");
      let request: Record<string, unknown> | undefined;
      if (typeof init?.body === "string") {
        try { request = record(JSON.parse(init.body)); } catch { turn.invalid = true; }
      }
      if (url !== this.endpoint || method.toUpperCase() !== "POST" ||
          request?.stream !== true || record(request.stream_options)?.include_usage !== true) {
        turn.invalid = true;
      }
    }
    const response = await globalThis.fetch(resource, init);
    if (turn === undefined || turn.invalid || !response.ok || response.redirected || response.body === null ||
        !response.headers.get("content-type")?.toLowerCase().includes("text/event-stream")) return response;
    return new Response(observeStream(response.body, attempt), {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };
}

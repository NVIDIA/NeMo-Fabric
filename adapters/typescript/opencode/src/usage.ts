// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentUsage } from "nemo-fabric-adapter-contract";

type CreateOptions = NonNullable<Parameters<(typeof import("@opencode/sdk"))["OpenCode"]["create"]>[0]>;
type Plugin = NonNullable<CreateOptions["plugins"]>[number];
type Counter = "input_tokens" | "output_tokens" | "total_tokens" | "cached_input_tokens" | "reasoning_tokens";
type Counts = Partial<Record<Counter, number>>;
type Gap = "scope_unverified" | "missing_usage" | "unfinished_work";
interface HookEvents {
  prompt: { sessionID: string };
  "http.request": { sessionID: string; kind: string; request: Request };
  "http.response": { sessionID: string; request: Request; response: Response };
}
interface PluginContext {
  session: {
    get(input: { sessionID: string }): Promise<{ parentID?: string }>;
    hook<Name extends keyof HookEvents>(name: Name, callback: (event: HookEvents[Name]) => Promise<void> | void): Promise<{
      dispose(): Promise<void>;
    }>;
  };
  tool: {
    hook(name: "execute.after", callback: (event: {
      sessionID: string;
      tool: string;
      status: string;
      result?: { output?: unknown };
    }) => void): Promise<{ dispose(): Promise<void> }>;
  };
}
interface Call {
  counts: Counts;
  settled: boolean;
  missing: boolean;
}
const KEY = "nvidia.fabric.opencode.usage_evidence.v1";
const MAX_EVENT_CHARACTERS = 1 << 20;
const counters: Counter[] = ["input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens"];

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function count(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

/** Root HTTP observations cannot establish title and child-work completeness. */
export class OpenCodeInvocationUsage {
  readonly calls: Call[] = [];
  private frozen?: AgentUsage;

  finish(): AgentUsage {
    if (this.frozen !== undefined) return this.frozen;
    const observed: Counts = {};
    const gaps = new Set<Gap>(["scope_unverified"]);
    for (const field of counters) {
      const values = this.calls.flatMap((call) => call.counts[field] === undefined ? [] : [call.counts[field]!]);
      if (values.length === 0) continue;
      const sum = values.reduce((total, value) => total + value, 0);
      if (Number.isSafeInteger(sum)) observed[field] = sum;
    }
    if (this.calls.length === 0 || this.calls.some((call) => call.missing)) gaps.add("missing_usage");
    if (this.calls.some((call) => !call.settled)) gaps.add("unfinished_work");
    this.frozen = { extensions: { [KEY]: { observed, gaps: [...gaps] } } };
    return this.frozen;
  }
}

/** Observe only bytes consumed by OpenCode, without buffering the full stream. */
function observeResponse(call: Call, response: Response): Response {
  if (!response.ok || response.body === null) {
    call.settled = true;
    return response;
  }
  const streaming = response.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase() === "text/event-stream";
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "";
  let overflow = false;
  let parseLoss = false;
  let terminal = false;
  let discardLineFeed = false;
  const report = (data: string): void => {
    if (data === "\n" || data.trim() === ": keepalive") return;
    if (data.trim() === "[DONE]") { terminal = true; call.settled = true; return; }
    let value: unknown;
    try { value = JSON.parse(data); } catch { parseLoss = true; return; }
    if (!record(value)) return;
    if (Array.isArray(value.choices) && value.choices.some((choice) => record(choice) && typeof choice.finish_reason === "string")) {
      terminal = true;
      call.settled = true;
    }
    if (!record(value.usage)) return;
    const raw = value.usage;
    const input = record(raw.prompt_tokens_details) ? raw.prompt_tokens_details : {};
    const output = record(raw.completion_tokens_details) ? raw.completion_tokens_details : {};
    const values: Record<Counter, unknown> = {
      input_tokens: raw.prompt_tokens,
      output_tokens: raw.completion_tokens,
      total_tokens: raw.total_tokens,
      cached_input_tokens: input.cached_tokens,
      reasoning_tokens: output.reasoning_tokens,
    };
    // Repeated streaming usage frames are snapshots of one physical request.
    call.counts = Object.fromEntries(counters.flatMap((field) => {
      const value = count(values[field]);
      return value === undefined ? [] : [[field, value]];
    }));
    call.missing = count(raw.prompt_tokens) === undefined || count(raw.completion_tokens) === undefined;
  };
  const feed = (chunk: string): void => {
    if (streaming && discardLineFeed && chunk.length > 0) {
      discardLineFeed = false;
      if (chunk.startsWith("\n")) chunk = chunk.slice(1);
    }
    buffer += chunk;
    if (!streaming) {
      if (buffer.length > MAX_EVENT_CHARACTERS) { buffer = ""; overflow = true; parseLoss = true; }
      return;
    }
    let newline: number;
    while ((newline = buffer.search(/[\r\n]/u)) !== -1) {
      const line = buffer.slice(0, newline);
      const carriageReturn = buffer[newline] === "\r";
      buffer = buffer.slice(newline + 1);
      if (carriageReturn) {
        // CRLF is one separator even when its bytes arrive in separate chunks.
        if (buffer.length === 0) discardLineFeed = true;
        else if (buffer.startsWith("\n")) buffer = buffer.slice(1);
      }
      if (line === "") {
        if (event.length > 0 && !overflow) report(event);
        event = "";
        overflow = false;
      } else if (line.startsWith("data:")) {
        event += `${line.slice(5).replace(/^ /u, "")}\n`;
        if (event.length > MAX_EVENT_CHARACTERS) { event = ""; overflow = true; parseLoss = true; }
      }
    }
    if (buffer.length > MAX_EVENT_CHARACTERS) { buffer = ""; overflow = true; parseLoss = true; }
  };
  const reader = response.body.getReader();
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const next = await reader.read();
        if (next.done) {
          feed(decoder.decode());
          if (streaming) feed("\n\n");
          else if (!overflow) report(buffer);
          call.missing ||= parseLoss;
          call.settled = !streaming || terminal;
          controller.close();
        } else {
          feed(decoder.decode(next.value, { stream: true }));
          controller.enqueue(next.value);
        }
      } catch (error) {
        call.settled = false;
        call.missing ||= parseLoss;
        controller.error(error);
      }
    },
    async cancel(reason) {
      // The SDK cancels the unused stream tail after a terminal model frame.
      call.settled = terminal;
      call.missing ||= parseLoss;
      await reader.cancel(reason);
    },
  }, { highWaterMark: 0 });
  return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
}

export class OpenCodeUsageCollector {
  private root?: string;
  private active?: OpenCodeInvocationUsage;
  private continuationUncertain = false;
  private readonly requests = new WeakMap<Request, Call>();

  begin(root: string): OpenCodeInvocationUsage {
    this.root = root;
    const usage = new OpenCodeInvocationUsage();
    // Earlier child or background shell work can wake the root during a later prompt.
    // Without an invocation ID in that wakeup, its root requests are ambiguous.
    this.active = this.continuationUncertain ? undefined : usage;
    return usage;
  }

  end(usage: OpenCodeInvocationUsage): AgentUsage {
    if (this.active === usage) this.active = undefined;
    return usage.finish();
  }

  plugin(): Plugin {
    return {
      id: "nemo-fabric-usage",
      setup: async (context: PluginContext) => {
        const registrations = await Promise.all([
          context.session.hook("prompt", async (event) => {
            if (event.sessionID === this.root) return;
            try {
              const session = await context.session.get({ sessionID: event.sessionID });
              if (session.parentID !== undefined) this.continuationUncertain = true;
            } catch {
              // A failed ownership lookup must not make later root usage look attributable.
              this.continuationUncertain = true;
            }
          }),
          context.tool.hook("execute.after", (event) => {
            if (event.sessionID !== this.root || event.tool !== "shell" || event.status !== "completed") return;
            const output = event.result?.output;
            if (record(output) && output.status === "running") this.continuationUncertain = true;
          }),
          context.session.hook("http.request", (event) => {
            if (event.sessionID !== this.root || (event.kind !== "primary" && event.kind !== "compaction")) return;
            const owner = this.active;
            if (owner === undefined) return;
            const call: Call = { counts: {}, settled: false, missing: true };
            owner.calls.push(call);
            this.requests.set(event.request, call);
          }),
          context.session.hook("http.response", (event) => {
            const call = this.requests.get(event.request);
            if (call === undefined) return;
            if (!new URL(event.request.url).pathname.replace(/\/$/u, "").endsWith("/chat/completions")) {
              call.settled = true;
              return;
            }
            event.response = observeResponse(call, event.response);
          }),
        ]);
        return async () => { await Promise.all(registrations.map((registration) => registration.dispose())); };
      },
    };
  }
}

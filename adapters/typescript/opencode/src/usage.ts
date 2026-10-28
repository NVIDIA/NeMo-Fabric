// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentUsage } from "nemo-fabric-adapter-contract";

type CreateOptions = NonNullable<Parameters<(typeof import("@opencode/sdk"))["OpenCode"]["create"]>[0]>;
type Plugin = NonNullable<CreateOptions["plugins"]>[number];
type Counter = "input_tokens" | "output_tokens" | "total_tokens" | "cached_input_tokens" | "reasoning_tokens";
type Counts = Partial<Record<Counter, number>>;
type Gap = "scope_unverified" | "missing_usage" | "unfinished_work";
export type ScopeReason =
  | "observer_unavailable"
  | "observer_lost"
  | "configuration_changed"
  | "prepare_scope_read_failed"
  | "unsupported_configuration"
  | "root_execution_unconfirmed"
  | "title_completion_unconfirmed"
  | "final_scope_read_failed"
  | "child_ancestry_unverified"
  | "child_ownership_changed"
  | "manual_session_change"
  | "tool_result_unavailable"
  | "background_shell"
  | "background_subagent"
  | "unsupported_protocol"
  | "auxiliary_generation"
  | "unsupported_request_kind"
  | "prompt_failed"
  | "unfinished_work";
interface HookEvents {
  prompt: { sessionID: string };
  "http.request": { sessionID: string; kind: string; request: Request };
  "http.response": { sessionID: string; request: Request; response: Response };
  "experimental.ws.handshake": { sessionID: string };
}
interface ToolEvent {
  sessionID: string;
  tool: string;
  id?: string;
  status?: string;
  result?: { output?: unknown };
}
interface PluginContext {
  session: {
    get(input: { sessionID: string }): Promise<{ parentID?: string }>;
    hook<Name extends keyof HookEvents>(name: Name, callback: (event: HookEvents[Name]) => Promise<void> | void): Promise<{
      dispose(): Promise<void>;
    }>;
  };
  tool: {
    hook(name: "execute.before" | "execute.after", callback: (event: ToolEvent) => void): Promise<{ dispose(): Promise<void> }>;
  };
}
interface Call {
  counts: Counts;
  settled: boolean;
  missing: boolean;
  invalid: boolean;
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

/** Freeze request-owned evidence only after the caller proves its scope. */
export class OpenCodeInvocationUsage {
  readonly calls: Call[] = [];
  readonly children = new Set<string>();
  readonly tools = new Map<string, number>();
  readonly scopeReasons = new Set<ScopeReason>();
  uncertain = false;
  closed = false;
  private frozen?: AgentUsage;

  unfinished(): boolean {
    return this.children.size > 0 || this.tools.size > 0 || this.calls.some((call) => !call.settled);
  }

  finish(scopeProven = false): AgentUsage {
    if (this.frozen !== undefined) return this.frozen;
    this.closed = true;
    const observed: Counts = {};
    const complete: AgentUsage = {};
    const gaps = new Set<Gap>();
    if (!scopeProven || this.uncertain) gaps.add("scope_unverified");
    let overflow = false;
    for (const field of counters) {
      const values = this.calls.flatMap((call) => call.counts[field] === undefined ? [] : [call.counts[field]!]);
      if (values.length === 0) continue;
      const sum = values.reduce((total, value) => total + value, 0);
      if (!Number.isSafeInteger(sum)) { overflow = true; continue; }
      observed[field] = sum;
      if (field !== "reasoning_tokens" && scopeProven && !this.uncertain && !this.unfinished() &&
          this.calls.every((call) => !call.invalid && call.counts[field] !== undefined)) complete[field] = sum;
    }
    if (complete.input_tokens !== undefined) complete.input_tokens_include_cache = true;
    if (this.calls.length === 0 || overflow || this.calls.some((call) => call.missing)) gaps.add("missing_usage");
    if (this.unfinished()) gaps.add("unfinished_work");
    this.frozen = { ...complete, extensions: { [KEY]: { observed, gaps: [...gaps], ...(this.scopeReasons.size === 0 ? {} : { scope_reasons: [...this.scopeReasons] }) } } };
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
  let finished = false;
  let discardLineFeed = false;
  const report = (data: string): void => {
    if (terminal) return;
    if (data === "\n" || data.trim() === ": keepalive") return;
    if (data.trim() === "[DONE]") { terminal = true; call.settled = true; return; }
    let value: unknown;
    try { value = JSON.parse(data); } catch { parseLoss = true; return; }
    if (!record(value)) return;
    if (Array.isArray(value.choices) && value.choices.some((choice) => record(choice) && typeof choice.finish_reason === "string")) {
      finished = true;
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
    if (terminal) return;
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
    while (!terminal && (newline = buffer.search(/[\r\n]/u)) !== -1) {
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
    if (terminal) buffer = "";
    else if (buffer.length > MAX_EVENT_CHARACTERS) { buffer = ""; overflow = true; parseLoss = true; }
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
          call.invalid ||= parseLoss;
          call.settled = !streaming || terminal || finished;
          controller.close();
        } else {
          feed(decoder.decode(next.value, { stream: true }));
          controller.enqueue(next.value);
        }
      } catch (error) {
        call.settled = false;
        call.missing ||= parseLoss;
        call.invalid ||= parseLoss;
        controller.error(error);
      }
    },
    async cancel(reason) {
      // OpenCode discards the unused tail after [DONE], not after finish_reason.
      call.settled = terminal;
      call.missing ||= parseLoss;
      call.invalid ||= parseLoss;
      await reader.cancel(reason);
    },
  }, { highWaterMark: 0 });
  return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
}

export class OpenCodeUsageCollector {
  private root?: string;
  private active?: OpenCodeInvocationUsage;
  private titleOwner?: OpenCodeInvocationUsage;
  private continuationUncertain = false;
  private readonly continuationReasons = new Set<ScopeReason>();
  private readonly requests = new WeakMap<Request, Call>();
  private readonly sessions = new Map<string, OpenCodeInvocationUsage>();

  begin(root: string): OpenCodeInvocationUsage {
    this.root = root;
    const usage = new OpenCodeInvocationUsage();
    // Earlier child or background shell work can wake the root during a later prompt.
    // Without an invocation ID in that wakeup, its root requests are ambiguous.
    this.active = this.continuationUncertain ? undefined : usage;
    if (!this.continuationUncertain) this.titleOwner = usage;
    usage.uncertain = this.continuationUncertain;
    for (const reason of this.continuationReasons) usage.scopeReasons.add(reason);
    return usage;
  }

  markUncertain(reason?: ScopeReason): void {
    this.continuationUncertain = true;
    if (reason !== undefined) {
      this.continuationReasons.add(reason);
      this.active?.scopeReasons.add(reason);
    }
    if (this.active !== undefined) this.active.uncertain = true;
  }

  end(usage: OpenCodeInvocationUsage, scopeProven = false): AgentUsage {
    if (usage.closed) return usage.finish(scopeProven);
    if (usage.unfinished()) this.markUncertain("unfinished_work");
    if (this.active === usage) this.active = undefined;
    return usage.finish(scopeProven);
  }

  private owner(sessionID: string): OpenCodeInvocationUsage | undefined {
    return sessionID === this.root ? this.active : this.sessions.get(sessionID);
  }

  plugin(): Plugin {
    return {
      id: "nemo-fabric-usage",
      setup: async (context: PluginContext) => {
        const registrations = await Promise.all([
          context.session.hook("prompt", async (event) => {
            if (event.sessionID === this.root) return;
            try {
              const visited = new Set<string>();
              let sessionID: string | undefined = event.sessionID;
              while (sessionID !== undefined) {
                if (visited.has(sessionID)) { this.markUncertain("child_ancestry_unverified"); return; }
                visited.add(sessionID);
                const session = await context.session.get({ sessionID });
                sessionID = session.parentID;
                const owner = sessionID === undefined ? undefined : this.owner(sessionID);
                if (owner === undefined) continue;
                const previous = this.sessions.get(event.sessionID);
                if (previous !== undefined && previous !== owner && previous.unfinished()) {
                  owner.uncertain = true;
                  owner.scopeReasons.add("child_ownership_changed");
                  this.markUncertain("child_ownership_changed");
                  return;
                }
                this.sessions.set(event.sessionID, owner);
                owner.children.add(event.sessionID);
                return;
              }
            } catch {
              this.markUncertain("child_ancestry_unverified");
            }
          }),
          context.tool.hook("execute.before", (event) => {
            const owner = this.owner(event.sessionID);
            if (owner === undefined) return;
            const key = `${event.sessionID}:${event.tool}:${event.id ?? ""}`;
            owner.tools.set(key, (owner.tools.get(key) ?? 0) + 1);
            if (event.tool === "opencode_session_rename" || event.tool === "opencode_session_move") {
              owner.uncertain = true;
              owner.scopeReasons.add("manual_session_change");
              this.markUncertain("manual_session_change");
            }
          }),
          context.tool.hook("execute.after", (event) => {
            const owner = this.owner(event.sessionID);
            if (owner === undefined) return;
            const key = `${event.sessionID}:${event.tool}:${event.id ?? ""}`;
            const pending = owner.tools.get(key) ?? 0;
            if (pending <= 1) owner.tools.delete(key);
            else owner.tools.set(key, pending - 1);
            if (event.tool !== "shell" && event.tool !== "subagent") return;
            const output = event.result?.output;
            if (event.status !== "completed" || !record(output)) { owner.uncertain = true; owner.scopeReasons.add("tool_result_unavailable"); return; }
            if (output.status === "running") {
              owner.uncertain = true;
              const reason = event.tool === "shell" ? "background_shell" : "background_subagent";
              owner.scopeReasons.add(reason);
              this.markUncertain(reason);
            } else if (event.tool === "subagent" && output.status === "completed" && typeof output.sessionID === "string") {
              owner.children.delete(output.sessionID);
            }
          }),
          context.session.hook("experimental.ws.handshake", (event) => {
            const owner = this.owner(event.sessionID);
            if (owner !== undefined) { owner.uncertain = true; owner.scopeReasons.add("unsupported_protocol"); }
          }),
          context.session.hook("http.request", (event) => {
            const owner = event.sessionID === this.root && event.kind === "title" ? this.titleOwner : this.owner(event.sessionID);
            if (owner === undefined) return;
            if (event.kind === "generate") { owner.uncertain = true; owner.scopeReasons.add("auxiliary_generation"); this.markUncertain("auxiliary_generation"); return; }
            if (event.kind !== "primary" && event.kind !== "compaction" && event.kind !== "title") { owner.uncertain = true; owner.scopeReasons.add("unsupported_request_kind"); }
            const call: Call = { counts: {}, settled: false, missing: true, invalid: false };
            owner.calls.push(call);
            this.requests.set(event.request, call);
          }),
          context.session.hook("http.response", (event) => {
            const call = this.requests.get(event.request);
            if (call === undefined) return;
            if (!new URL(event.request.url).pathname.replace(/\/$/u, "").endsWith("/chat/completions")) {
              call.settled = true;
              call.invalid = true;
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

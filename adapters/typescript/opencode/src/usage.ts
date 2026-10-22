// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Public OpenCode 2.0.23 session hooks retain physical request evidence before
// the session projector replaces omitted counts with zero.

import type { AgentUsage, ReportedUsage, UsageObservation } from "nemo-fabric-adapter-contract";
import { usageFromObservations } from "nemo-fabric-adapter-contract";
type CreateOptions = NonNullable<Parameters<(typeof import("@opencode/sdk"))["OpenCode"]["create"]>[0]>;
type UsagePlugin = NonNullable<CreateOptions["plugins"]>[number];
interface SessionHook {
  sessionID: string;
}
interface HookEvents {
  prompt: SessionHook;
  "http.request": SessionHook & {
    request: Request;
  };
  "http.response": SessionHook & {
    request: Request;
    response: Response;
  };
  "experimental.ws.handshake": SessionHook;
}
interface PluginContext {
  session: {
    context(input: {sessionID: string}): Promise<unknown>;
    get(input: {
      sessionID: string;
    }): Promise<{
      parentID?: string;
    }>;
    hook<Name extends keyof HookEvents>(name: Name, callback: (event: HookEvents[Name]) => Promise<void>): Promise<{
      dispose(): Promise<void>;
    }>;
  };
}
type Gap = NonNullable<UsageObservation["gap"]>;
interface BeforeSession {
  running: boolean;
  newestMessageId?: string;
}
interface Call {
  readonly sessionID: string;
  model?: string;
  provider?: string;
  reported: ReportedUsage;
  gap?: Gap;
  settled: boolean;
  include: boolean;
}
const MAX_EVENT_BYTES = 1 << 20;
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function count(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}
function fields(value: unknown): Record<string, unknown> { return record(value) ? value : {}; }
function nonempty(value: unknown): string | undefined { return typeof value === "string" && value.trim() ? value : undefined; }
function providerFor(url: string): string | undefined {
  try {
    const route = new URL(url);
    if (route.protocol !== "https:" || (route.port && route.port !== "443"))
      return undefined;
    return ({ "api.openai.com": "openai", "api.anthropic.com": "anthropic" } as Record<string, string>)[route.hostname];
  }
  catch {
    return undefined;
  }
}
export class OpenCodeInvocationEvidence {
  readonly calls: Call[] = [];
  readonly observations: UsageObservation[] = [];
  readonly before: ReadonlyMap<string, BeforeSession>;
  private snapshot?: AgentUsage;
  suppressLater = false;
  constructor(readonly root: string, before: ReadonlyMap<string, BeforeSession>, readonly initialRootMessages: ReadonlySet<string>) { this.before = before; }
  get frozen(): boolean { return this.snapshot !== undefined; }
  start(sessionID: string, model?: string, provider?: string): Call {
    const call: Call = { sessionID, ...(model === undefined ? {} : { model }), ...(provider === undefined ? {} : { provider }), reported: {}, settled: false, include: !this.suppressLater };
    this.calls.push(call);
    return call;
  }
  hasCalls(sessionID: string): boolean { return this.calls.some(call => call.sessionID === sessionID); }
  callCount(sessionID: string): number { return this.calls.filter(call => call.sessionID === sessionID).length; }
  gap(reason: Gap): void { this.observations.push({ reported: {}, gap: reason }); }
  finish(): AgentUsage {
    this.snapshot ??= usageFromObservations([
      ...this.calls.filter(call => call.include).map(call => ({ reported: { ...call.reported }, ...(call.model === undefined ? {} : { model: call.model }), ...(call.provider === undefined ? {} : { provider: call.provider }),
        ...(call.settled ? (call.gap === undefined ? {} : { gap: call.gap }) : { gap: "unfinished_work" as const }), tokens_aligned: true, is_model_call: true })),
      ...this.observations,
    ]);
    return this.snapshot;
  }
}
/** A bounded passive parser; it never consumes bytes the harness did not request. */
class ResponseUsage {
  private buffer = "";
  private event: string[] = [];
  private eventSize = 0;
  private eventLost = false;
  private parseLoss = false;
  private terminal = false;
  private readonly decoder = new TextDecoder();
  private readonly streaming: boolean;
  private readonly messages: boolean;
  private anthropic: Record<string, unknown> = {};
  private anthropicOutputFinal = false;
  constructor(private readonly call: Call, request: Request, response: Response) {
    this.streaming = response.headers.get("content-type")?.includes("text/event-stream") === true;
    this.messages = new URL(request.url).pathname.replace(/\/$/u, "").endsWith("/messages");
  }
  feed(chunk: Uint8Array): void {
    this.buffer += this.decoder.decode(chunk, { stream: true });
    if (!this.streaming) {
      if (this.buffer.length > MAX_EVENT_BYTES) {
        this.buffer = "";
        this.parseLoss = true;
      }
      return;
    }
    let newline: number;
    while ((newline = this.buffer.indexOf("\n")) !== -1) {
      const line = this.buffer.slice(0, newline).replace(/\r$/u, "");
      this.buffer = this.buffer.slice(newline + 1);
      if (!line)
        this.flushEvent();
      else if (line.startsWith("data:")) {
        const data = line.slice(5).replace(/^ /u, "");
        this.eventSize += data.length;
        if (this.eventSize <= MAX_EVENT_BYTES)
          this.event.push(data);
        else
          this.eventLost = this.parseLoss = true;
      }
    }
    if (this.buffer.length > MAX_EVENT_BYTES) {
      this.buffer = "";
      this.eventLost = this.parseLoss = true;
    }
  }
  private flushEvent(): void {
    if (this.event.length && !this.eventLost) {
      const data = this.event.join("\n");
      if (data.trim() === "[DONE]")
        this.terminal = true;
      else
        this.json(data);
    }
    this.event = [];
    this.eventSize = 0;
    this.eventLost = false;
  }
  private json(data: string): void {
    try {
      this.response(JSON.parse(data));
    }
    catch {
      this.parseLoss = true;
    }
  }
  private response(value: unknown): void {
    if (!record(value)) {
      this.parseLoss = true;
      return;
    }
    if (["response.completed", "response.incomplete", "response.failed"].includes(String(value.type)))
      this.terminal = true;
    if (this.messages && this.streaming) {
      if (value.type === "message_start") {
        const message = fields(value.message);
        this.call.model = nonempty(message.model) ?? this.call.model;
        this.anthropic = { ...fields(message.usage) };
      }
      else if (value.type === "message_delta") {
        const delta = fields(value.usage);
        this.anthropic = { ...this.anthropic, ...delta };
        this.anthropicOutputFinal ||= Object.hasOwn(delta, "output_tokens");
      }
      else if (value.type === "message_stop")
        this.terminal = true;
      const raw = { ...this.anthropic };
      if (!this.anthropicOutputFinal)
        delete raw.output_tokens;
      this.report(raw, true);
      return;
    }
    const body = record(value.response) ? value.response : value;
    this.call.model = nonempty(body.model) ?? this.call.model;
    if (record(body.usage))
      this.report(body.usage, this.messages);
    else if (this.terminal)
      this.parseLoss = true;
  }
  private report(raw: Record<string, unknown>, messages: boolean): void {
    const input = fields(raw.input_tokens_details ?? raw.prompt_tokens_details);
    const output = fields(raw.output_tokens_details ?? raw.completion_tokens_details);
    const found: ReportedUsage = {
      ...(count(raw.input_tokens ?? raw.prompt_tokens) === undefined ? {} : { input_tokens: count(raw.input_tokens ?? raw.prompt_tokens)! }),
      input_tokens_include_cache: !messages,
      ...(count(raw.cache_read_input_tokens ?? input.cached_tokens) === undefined ? {} : { cached_input_tokens: count(raw.cache_read_input_tokens ?? input.cached_tokens)! }),
      ...(count(raw.cache_creation_input_tokens) === undefined ? {} : { cache_write_input_tokens: count(raw.cache_creation_input_tokens)! }),
      ...(count(raw.output_tokens ?? raw.completion_tokens) === undefined ? {} : { output_tokens: count(raw.output_tokens ?? raw.completion_tokens)! }),
      output_tokens_include_reasoning: true,
      ...(count(output.reasoning_tokens) === undefined ? {} : { reasoning_tokens: count(output.reasoning_tokens)! }),
      ...(count(raw.total_tokens) === undefined ? {} : { total_tokens: count(raw.total_tokens)! }),
    };
    this.call.reported = found;
  }
  finish(completed: boolean): void {
    if (completed) {
      this.buffer += this.decoder.decode();
      if (this.streaming) {
        this.feed(new TextEncoder().encode("\n\n"));
      }
      else if (!this.parseLoss)
        this.json(this.buffer);
    }
    this.call.settled = true;
    this.call.gap = this.parseLoss ? "scope_unverified" : this.streaming && !this.terminal ? "unfinished_work" : completed || this.terminal ? undefined : "missing_usage";
  }
}
function observeBody(call: Call, request: Request, response: Response): Response {
  if (!response.ok) {
    call.settled = true;
    call.gap = "missing_usage";
    return response;
  }
  if (response.body === null) {
    call.settled = true;
    return response;
  }
  const parser = new ResponseUsage(call, request, response);
  const reader = response.body.getReader();
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const next = await reader.read();
        if (next.done) {
          parser.finish(true);
          controller.close();
        }
        else {
          parser.feed(next.value);
          controller.enqueue(next.value);
        }
      }
      catch (error) {
        parser.finish(false);
        controller.error(error);
      }
    },
    async cancel(reason) { parser.finish(false); await reader.cancel(reason); },
  }, { highWaterMark: 0 });
  return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
}
/** One plugin lease per embedded host, with ownership captured before each request. */
export class OpenCodeUsageCollector {
  installed = false;
  private readonly owners = new Map<string, OpenCodeInvocationEvidence>();
  private readonly requests = new WeakMap<Request, {
    owner: OpenCodeInvocationEvidence;
    call: Call;
  }>();
  private context?: PluginContext;
  begin(root: string, before: ReadonlyMap<string, BeforeSession>, initialRootMessages: ReadonlySet<string> = new Set()): OpenCodeInvocationEvidence {
    const owner = new OpenCodeInvocationEvidence(root, before, initialRootMessages);
    if (before.get(root)?.running === false || !this.owners.has(root))
      this.owners.set(root, owner);
    return owner;
  }
  private async owner(sessionID: string, seen = new Set<string>()): Promise<OpenCodeInvocationEvidence | undefined> {
    const current = this.owners.get(sessionID);
    if (current !== undefined)
      return current;
    if (this.context === undefined || seen.has(sessionID))
      return undefined;
    seen.add(sessionID);
    const session = await this.context.session.get({ sessionID });
    if (session.parentID === undefined)
      return undefined;
    const parent = await this.owner(session.parentID, seen);
    if (parent !== undefined)
      this.owners.set(sessionID, parent);
    return parent;
  }
  /** Earlier background completions can wake the parent without the prompt hook.
   * Inspect only sessions with an earlier child, before dispatch; retain earlier
   * proven-owned requests and suppress ambiguous continuations and descendants.
   */
  private async verifyContinuation(sessionID: string, owner: OpenCodeInvocationEvidence): Promise<void> {
    if (owner.frozen || owner.suppressLater) return;
    const earlierChildren = new Set([...owner.before.keys()].filter(id =>
      id !== owner.root && id !== sessionID && this.owners.get(id) !== owner));
    if (!earlierChildren.size) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      const history = await Promise.race([
        this.context!.session.context({sessionID}),
        new Promise<never>((_, reject) => {timer = setTimeout(() => reject(new Error("usage context deadline")), 1000);}),
      ]);
      if (!Array.isArray(history)) throw new Error("usage context shape");
      const boundaryID = owner.before.get(sessionID)?.newestMessageId;
      const boundary = history.findIndex(message => record(message) && message.id === boundaryID);
      const recent = boundary >= 0 ? history.slice(boundary + 1) : history.filter(message =>
        sessionID !== owner.root || !record(message) || typeof message.id !== "string" || !owner.initialRootMessages.has(message.id));
      if (!recent.some(message => record(message) && message.type === "synthetic" && record(message.metadata) &&
        message.metadata.source === "subagent" && typeof message.metadata.childID === "string" && earlierChildren.has(message.metadata.childID))) return;
    } catch {
      // Unknown provenance must not put possibly earlier-owned usage on this turn.
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
    owner.suppressLater = true;
    owner.gap("scope_unverified");
  }
  plugin(): UsagePlugin {
    return {
      id: "nemo-fabric-usage",
      setup: async (context: PluginContext) => {
        this.context = context;
        const registrations: {
          dispose(): Promise<void>;
        }[] = [];
        const observe = <Name extends keyof HookEvents>(name: Name, callback: (event: HookEvents[Name]) => Promise<void>): Promise<{
          dispose(): Promise<void>;
        }> => context.session.hook(name, async (event) => {
          try {
            await callback(event);
          }
          catch {
            this.owners.get(event.sessionID)?.gap("scope_unverified");
          }
        });
        try {
          registrations.push(await observe("prompt", async (event) => {
            const session = await context.session.get({ sessionID: event.sessionID });
            if (session.parentID === undefined)
              return;
            const parent = await this.owner(session.parentID);
            // Re-prompting an already-running child cannot separate its old work.
            if (parent !== undefined && parent.before.get(event.sessionID)?.running !== true)
              this.owners.set(event.sessionID, parent);
          }));
          registrations.push(await observe("http.request", async (event) => {
            const owner = await this.owner(event.sessionID);
            if (owner === undefined)
              return;
            await this.verifyContinuation(event.sessionID, owner);
            let model: string | undefined;
            try {
              const body: unknown = await event.request.clone().json();
              model = record(body) ? nonempty(body.model) : undefined;
            }
            catch { /* Preserve execution; missing wire identity cannot enable pricing. */ }
            const call = owner.start(event.sessionID, model, providerFor(event.request.url));
            this.requests.set(event.request, { owner, call });
          }));
          registrations.push(await observe("http.response", async (event) => {
            let captured = this.requests.get(event.request);
            if (captured === undefined) {
              const owner = await this.owner(event.sessionID);
              if (owner === undefined)
                return;
              owner.gap("scope_unverified");
              captured = { owner, call: owner.start(event.sessionID) };
            }
            const requestedProvider = providerFor(event.request.url);
            // Fetch may follow redirects before this hook runs. Both visible
            // endpoints must qualify the same native billing identity.
            captured.call.provider = event.response.url && providerFor(event.response.url) !== requestedProvider
              ? undefined : requestedProvider;
            event.response = observeBody(captured.call, event.request, event.response);
          }));
          registrations.push(await observe("experimental.ws.handshake", async (event) => {
            const owner = await this.owner(event.sessionID);
            if (owner !== undefined)
              owner.gap("scope_unverified");
          }));
          this.installed = true;
        }
        catch {
          await Promise.allSettled(registrations.map(registration => registration.dispose()));
          this.context = undefined;
          // Unsupported public hooks leave execution available through reader fallback.
        }
        return async () => {
          this.installed = false;
          this.context = undefined;
          await Promise.allSettled(registrations.map(registration => registration.dispose()));
        };
      },
    };
  }
}

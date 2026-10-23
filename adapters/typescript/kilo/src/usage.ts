// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {usageFromObservations, type AgentUsage, type UsageObservation, type ReportedUsage} from "nemo-fabric-adapter-contract";
import type {KiloUsageEvents} from "./usage-events.js";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function count(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}
function positive(value: unknown): number | undefined {
  const result = count(value);
  return result !== undefined && result > 0 ? result : undefined;
}

/** Undo Session.getUsage's subtraction only when its clamp lost no information.
 * A positive adjusted count proves the original main counter exceeded the
 * subtracted components. Zero cannot distinguish a real 0 from source loss.
 */
function originalInclusive(adjusted: unknown, components: unknown[]): number | undefined {
  const main = positive(adjusted);
  const values = components.map(count);
  if (main === undefined || values.some(value => value === undefined)) return undefined;
  const total = main + (values as number[]).reduce((sum, value) => sum + value, 0);
  return Number.isSafeInteger(total) ? total : undefined;
}

export function reverseStoredUsage(tokens: unknown): ReportedUsage {
  const raw = isRecord(tokens) ? tokens : {};
  const cache = isRecord(raw.cache) ? raw.cache : {};
  const input = originalInclusive(raw.input, [cache.read, cache.write]);
  const output = originalInclusive(raw.output, [raw.reasoning]);
  const reported: ReportedUsage = {};
  if (input !== undefined) {
    reported.input_tokens = input;
    reported.input_tokens_include_cache = true;
  } else if (positive(raw.input) !== undefined) {
    // A malformed/missing operand prevents inversion, but the positive stored
    // exclusive component remains evidence with its actual convention.
    reported.input_tokens = positive(raw.input);
    reported.input_tokens_include_cache = false;
  }
  if (output !== undefined) {
    reported.output_tokens = output;
    reported.output_tokens_include_reasoning = true;
  } else if (positive(raw.output) !== undefined) {
    reported.output_tokens = positive(raw.output);
    reported.output_tokens_include_reasoning = false;
  }
  if (positive(cache.read) !== undefined) reported.cached_input_tokens = positive(cache.read);
  if (positive(cache.write) !== undefined) reported.cache_write_input_tokens = positive(cache.write);
  if (positive(raw.reasoning) !== undefined) reported.reasoning_tokens = positive(raw.reasoning);
  return reported;
}

export interface UsageSourceScope {
  events?: KiloUsageEvents;
  eventCoverage?: boolean;
  /** Custom configured provider overrides cannot establish native billing. */
  customProvider?: string;
  unfinished?: boolean;
}

type Message = Record<string, unknown>;

function messageInfo(message: Message): Record<string, unknown> {
  return isRecord(message.info) ? message.info : {};
}

export function messageId(message: unknown): string | undefined {
  const id = isRecord(message) ? messageInfo(message).id : undefined;
  return typeof id === "string" ? id : undefined;
}

function messageParts(message: Message): Message[] {
  return Array.isArray(message.parts) ? message.parts.filter(isRecord) : [];
}

/** Whether Kilo Code finished processing an assistant message, normally or with an error. */
function assistantDone(info: Record<string, unknown>): boolean {
  return (isRecord(info.time) && info.time.completed !== undefined) || info.error !== undefined;
}

/** A `task` tool call: its status, the child session it runs in, when it started, and whether it runs in the background. */
interface TaskCall { status?: string; sessionId?: string; start?: number; background: boolean }

function taskCalls(message: Message): TaskCall[] {
  return messageParts(message).filter((part) => part.type === "tool" && part.tool === "task").map((part) => {
    const state = isRecord(part.state) ? part.state : {};
    const metadata = isRecord(state.metadata) ? state.metadata : {};
    const input = isRecord(state.input) ? state.input : {};
    const time = isRecord(state.time) ? state.time : {};
    return {
      ...(typeof state.status === "string" ? {status: state.status} : {}),
      ...(typeof metadata.sessionId === "string" ? {sessionId: metadata.sessionId} : {}),
      ...(typeof time.start === "number" ? {start: time.start} : {}),
      background: metadata.background === true || input.background === true,
    };
  });
}

/**
 * The child session whose result a user message delivers, when Kilo Code's
 * task tool added it for a finished background subagent. Kilo Code marks its
 * text part as synthetic with `metadata.background` and renders the child
 * session ID as `<task id="...">`.
 */
function backgroundResultSession(message: Message): string | undefined {
  if (messageInfo(message).role !== "user") return undefined;
  for (const part of messageParts(message)) {
    if (part.type !== "text" || part.synthetic !== true || !isRecord(part.metadata) || part.metadata.background !== true) continue;
    const match = typeof part.text === "string" ? /<task id="([^"]+)"/u.exec(part.text) : null;
    return match?.[1] ?? "";
  }
  return undefined;
}

const AMBIGUOUS = Symbol("ambiguous");
type Owner = string | typeof AMBIGUOUS;

interface Turn { owner: Owner; adapterPrompt: boolean }

/**
 * Ownership of a session's messages: the adapter prompt whose turn produced
 * each one. This is the single place that decides which invocation a model
 * call belongs to.
 *
 * Kilo Code runs one turn at a time per session and sets each assistant
 * message's `parentID` to the user message of its turn. A user message is
 * owned by the adapter prompt it is (`prompts`), or, for a background
 * subagent's result, by the owner of the `task` call that started that
 * subagent. Kilo Code also adds user messages inside a running turn for
 * compaction and its follow-ups. Such a message belongs to the turn that was
 * running, which is the latest assistant message's turn or a turn whose
 * prompt or background result has no assistant message yet. When those
 * candidates share an owner, the message has that owner. When all of them
 * come from adapter prompts, it belongs to the newest prompt, because the
 * adapter submits a prompt only after its previous prompt returned. Any
 * other case, such as a background result waiting before a newer prompt, is
 * ambiguous.
 */
/** The turn running when Kilo Code added a user message of its own, from the latest assistant message's turn and the turns not started yet. */
function runningTurn(latestAssistant: Turn | undefined, openRoots: Turn[]): Turn {
  const candidates = [...(latestAssistant === undefined ? [] : [latestAssistant]), ...openRoots];
  const newest = candidates.at(-1);
  if (newest === undefined || newest.owner === AMBIGUOUS) return {owner: AMBIGUOUS, adapterPrompt: false};
  if (candidates.every((turn) => turn.owner === newest.owner) || candidates.every((turn) => turn.adapterPrompt)) return newest;
  return {owner: AMBIGUOUS, adapterPrompt: false};
}

function sessionOwners(messages: Message[], prompts: ReadonlySet<string>): Map<string, Owner> {
  const users = new Map<string, Turn>();
  const tasks = new Map<string, Owner>();
  let latestAssistant: Turn | undefined;
  // Prompts and background results whose turn has no assistant message yet.
  let openRoots: Turn[] = [];
  for (const message of messages) {
    const info = messageInfo(message);
    const id = messageId(message);
    if (info.role === "user" && id !== undefined) {
      let turn: Turn;
      const delivered = backgroundResultSession(message);
      if (prompts.has(id)) {
        turn = {owner: id, adapterPrompt: true};
        openRoots.push(turn);
      } else if (delivered !== undefined) {
        turn = {owner: tasks.get(delivered) ?? AMBIGUOUS, adapterPrompt: false};
        openRoots.push(turn);
      } else {
        turn = runningTurn(latestAssistant, openRoots);
      }
      users.set(id, turn);
    } else if (info.role === "assistant") {
      const turn = typeof info.parentID === "string" ? users.get(info.parentID) : undefined;
      latestAssistant = turn ?? {owner: AMBIGUOUS, adapterPrompt: false};
      openRoots = openRoots.filter((root) => root !== turn);
      for (const task of taskCalls(message)) {
        if (task.sessionId !== undefined) tasks.set(task.sessionId, latestAssistant.owner);
      }
    }
  }
  return new Map([...users].map(([id, turn]) => [id, turn.owner]));
}

/** Keep provably owned assistant messages and flag any ambiguous ownership. */
function ownedAssistants(messages: Message[], prompt: string, prompts: ReadonlySet<string>): {messages: Message[]; ambiguous: boolean} {
  const owners = sessionOwners(messages, prompts);
  const promptIndex = messages.findIndex((message) => messageId(message) === prompt);
  if (promptIndex < 0) return {messages: [], ambiguous: true};
  const owned: Message[] = [];
  let ambiguous = false;
  // The prompt's work starts at its message, so only later messages can be its own.
  for (const message of messages.slice(promptIndex + 1)) {
    const info = messageInfo(message);
    if (info.role === "user" && owners.get(messageId(message) ?? "") === AMBIGUOUS) ambiguous = true;
    if (info.role !== "assistant") continue;
    const owner = typeof info.parentID === "string" ? owners.get(info.parentID) : undefined;
    if (owner === undefined || owner === AMBIGUOUS) { ambiguous = true; continue; }
    if (owner === prompt) owned.push(message);
  }
  return {messages: owned, ambiguous};
}

/**
 * The adapter's prompt among the messages after `lastMessageIdBefore`, the
 * newest message before the prompt (undefined when the session was empty):
 * the first user message with a text part that Kilo Code did not add itself.
 * Kilo Code marks the text of background results, compaction replays and
 * compaction follow-ups as synthetic.
 */
export function findPrompt(messages: unknown, lastMessageIdBefore: string | undefined): string | undefined {
  if (!Array.isArray(messages)) return undefined;
  const start = lastMessageIdBefore === undefined ? 0 : messages.findIndex((message) => messageId(message) === lastMessageIdBefore) + 1;
  if (start === 0 && lastMessageIdBefore !== undefined) return undefined;
  const prompt = messages.slice(start).filter(isRecord).find((message) => messageInfo(message).role === "user" && messageParts(message).some((part) => part.type === "text" && part.synthetic !== true));
  return prompt === undefined ? undefined : messageId(prompt);
}

/** The foreground `task` calls of `prompt`'s turn whose child sessions hold model calls to add. */
export function ownedChildSessions(messages: unknown, prompt: string, prompts: ReadonlySet<string>): string[] {
  if (!Array.isArray(messages)) return [];
  const owned = ownedAssistants(messages.filter(isRecord), prompt, prompts).messages;
  return [...new Set(owned.flatMap(taskCalls).filter((task) => !task.background).flatMap((task) => task.sessionId === undefined ? [] : [task.sessionId]))];
}

/** The explicit false overflow marker distinguishes preflight from a provider
 * context-overflow error. Both otherwise leave the assistant error absent.
 */
function preflightNoRequest(session: Message[], message: Message): boolean {
  const index = session.indexOf(message);
  const next = session[index + 1];
  return index >= 0 && messageInfo(message).error === undefined && next !== undefined &&
    messageInfo(next).role === "user" && messageParts(next).some(part => part.type === "compaction" && part.overflow === false);
}

const gap = (reason: UsageObservation["gap"]): UsageObservation => ({reported: {}, gap: reason});

function stepObservations(assistants: Message[], session: Message[], source: UsageSourceScope): UsageObservation[] {
  const observations: UsageObservation[] = [];
  const seen = new Map<string, {tokens: string; index: number}>();
  for (const message of assistants) {
    const info = messageInfo(message);
    const id = messageId(message);
    const parts = messageParts(message).filter(part => part.type === "step-finish");
    for (const [index, part] of parts.entries()) {
      const identity = typeof part.id === "string" ? part.id : `${id}:${index}`;
      const encoded = JSON.stringify(part.tokens);
      const previous = seen.get(identity);
      if (previous !== undefined) {
        if (previous.tokens !== encoded) observations[previous.index] = gap("scope_unverified");
        continue;
      }
      seen.set(identity, {tokens: encoded, index: observations.length});
      const model = typeof info.modelID === "string" && info.modelID.trim() ? info.modelID : undefined;
      const provider = typeof info.providerID === "string" && info.providerID.trim() && info.providerID !== source.customProvider ? info.providerID : undefined;
      const reported = reverseStoredUsage(part.tokens);
      const gaps: NonNullable<UsageObservation["gaps"]> = [];
      if (reported.input_tokens_include_cache === false) gaps.push({dimensions: ["input_tokens"], reason: "scope_unverified"});
      if (reported.output_tokens_include_reasoning === false) gaps.push({dimensions: ["output_tokens"], reason: "scope_unverified"});
      observations.push({reported, ...(gaps.length ? {gaps} : {}), ...(model === undefined ? {} : {model}),
        ...(provider === undefined ? {} : {provider}), tokens_aligned: true, is_model_call: true});
    }
    if (!assistantDone(info)) observations.push(gap("unfinished_work"));
    else if (parts.length === 0 && !preflightNoRequest(session, message)) observations.push(gap("missing_usage"));
    if (id !== undefined && source.events?.retried.has(id)) observations.push(gap("missing_usage"));
  }
  return observations;
}

function childObservations(task: TaskCall, resumes: number[], children: ReadonlyMap<string, unknown[]>, source: UsageSourceScope): UsageObservation[] {
  if (task.sessionId === undefined || task.start === undefined) return [gap("scope_unverified")];
  const child = children.get(task.sessionId);
  if (child === undefined || resumes.filter(value => value === task.start).length > 1) return [gap("scope_unverified")];
  const end = Math.min(...resumes.filter(value => value > task.start!), Number.POSITIVE_INFINITY);
  const session = child.filter(isRecord);
  const window: Message[] = [];
  let unknown = false;
  for (const message of session) {
    const time = messageInfo(message).time;
    const created = isRecord(time) ? count(time.created) : undefined;
    if (created === undefined) { unknown = true; continue; }
    if (created >= task.start && created < end && messageInfo(message).role === "assistant") window.push(message);
  }
  const result = stepObservations(window, session, source);
  if (source.events && window.some(message => {
    const id = messageId(message);
    return id === undefined || !source.events!.hasCompleted(id);
  })) result.push(gap("scope_unverified"));
  if (unknown || window.length === 0 || window.some(message => taskCalls(message).length > 0)) result.push(gap("scope_unverified"));
  if (source.events?.unattributedRetries.has(task.sessionId)) result.push(gap("scope_unverified"));
  return result;
}

/** Invocation-owned step snapshots. Late background results retain their
 * original prompt owner; incomplete work leaves gaps rather than erasing
 * trustworthy settled observations. Core owns complete sums and pricing.
 */
export function extractInvocationUsage(messages: unknown, prompt: string, prompts: ReadonlySet<string>, children: ReadonlyMap<string, unknown[]> = new Map(), source: UsageSourceScope = {}): AgentUsage {
  if (!Array.isArray(messages)) return usageFromObservations([gap("scope_unverified")]);
  const records = messages.filter(isRecord);
  const owned = ownedAssistants(records, prompt, prompts);
  const observations = stepObservations(owned.messages, records, source);
  if (source.unfinished) observations.push(gap("unfinished_work"));
  if (owned.ambiguous || source.eventCoverage === false) observations.push(gap("scope_unverified"));
  const resumes = new Map<string, number[]>();
  for (const task of records.flatMap(taskCalls)) {
    if (task.sessionId !== undefined && task.start !== undefined) resumes.set(task.sessionId, [...(resumes.get(task.sessionId) ?? []), task.start]);
  }
  for (const task of owned.messages.flatMap(taskCalls)) {
    if (task.sessionId === undefined && task.status === "error") continue;
    if (task.background) { observations.push(gap("unfinished_work")); continue; }
    observations.push(...childObservations(task, resumes.get(task.sessionId ?? "") ?? [], children, source));
  }
  const rootSession = records.map(messageInfo).find(info => typeof info.sessionID === "string")?.sessionID;
  if (typeof rootSession === "string" && source.events?.unattributedRetries.has(rootSession)) observations.push(gap("scope_unverified"));
  return usageFromObservations(observations);
}

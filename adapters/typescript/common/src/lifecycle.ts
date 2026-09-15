// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared process-adapter lifecycle host. It validates newline-delimited
// lifecycle requests, owns one adapter runtime, dispatches start, invoke, and
// stop operations, and returns normalized responses while keeping diagnostics
// off the protocol output stream.

import { randomBytes, timingSafeEqual } from "node:crypto";
import { createRequire } from "node:module";
import { createServer, type Server, type Socket } from "node:net";
import { createInterface } from "node:readline";
import type { Readable, Writable } from "node:stream";

import type { ValidateFunction } from "ajv";
import type {
  AgentConfig,
  AdapterHealthRequest,
  AdapterHealthResult,
  AgentRunRequest,
  AgentRunResult,
  JsonObject,
  RuntimeContext,
} from "nemo-fabric-adapter-contract";

export interface AdapterStartInput {
  agentName: string;
  baseDir: string;
  config: AgentConfig;
  runtimeContext: RuntimeContext;
  capabilityPlan?: JsonObject;
  telemetryPlan?: JsonObject;
}

export interface AdapterRuntime {
  start(input: AdapterStartInput): Promise<void>;
  invoke(request: AgentRunRequest, context: RuntimeContext): Promise<AgentRunResult>;
  health?(request: AdapterHealthRequest): Promise<AdapterHealthResult>;
  stop(): Promise<void>;
}

export type AdapterRuntimeFactory = () => AdapterRuntime | Promise<AdapterRuntime>;

export interface LifecycleHostOptions {
  input?: Readable;
  output?: Writable;
  diagnostics?: Writable;
}

interface HostState {
  runtime?: AdapterRuntime;
  runtimeId?: string;
  failed: boolean;
  invoking: boolean;
  stopping: boolean;
  healthServer?: Server;
  healthSockets?: Set<Socket>;
  healthToken?: string;
}

interface LifecycleRequest {
  operation: string;
  payload: Record<string, unknown>;
}

interface LifecycleFailure {
  stage: string;
  code: string;
  message: string;
  retryable: boolean;
  metadata?: JsonObject;
}

interface LifecycleResponse {
  operation: string;
  outcome:
    | { status: "succeeded"; output: unknown }
    | { status: "failed"; error: LifecycleFailure };
}

export class LifecycleError extends Error {
  readonly code: string;
  readonly retryable: boolean;
  readonly metadata?: JsonObject;

  constructor(code: string, message: string, options: { retryable?: boolean; metadata?: JsonObject } = {}) {
    super(message);
    this.name = "LifecycleError";
    this.code = code;
    this.retryable = options.retryable ?? false;
    this.metadata = options.metadata;
  }
}

class AdapterCallError extends LifecycleError {}

const require = createRequire(import.meta.url);
const Ajv2020 = (
  require("ajv/dist/2020.js") as { default: typeof import("ajv/dist/2020.js").default }
).default;
const ajv = new Ajv2020({ allErrors: true, strict: false });
ajv.addFormat("uint32", {
  type: "number",
  validate: (value: number) => Number.isInteger(value) && value >= 0 && value <= 0xffff_ffff,
});
ajv.addFormat("uint64", {
  type: "number",
  validate: (value: number) => Number.isSafeInteger(value) && value >= 0,
});
ajv.addFormat("uint128", {
  type: "number",
  validate: (value: number) => Number.isSafeInteger(value) && value >= 0,
});
ajv.addFormat("double", {
  type: "number",
  validate: (value: number) => Number.isFinite(value),
});
const validateAgentConfig = compileSchema("agent-config");
const validateAgentRunRequest = compileSchema("agent-run-request");
const validateAgentRunResult = compileSchema("agent-run-result");
const validateRuntimeContext = compileSchema("runtime-context");
const validateAdapterHealthRequest = compileSchema("adapter-health-request");
const validateAdapterHealthResult = compileSchema("adapter-health-result");

const HEALTH_CONTROL_HOST = "127.0.0.1";
const HEALTH_CONTROL_PROTOCOL = "fabric.health/v1alpha1";
const HEALTH_REQUEST_LIMIT = 1024 * 1024;
const HEALTH_REQUEST_TIMEOUT_MILLIS = 10_000;
const HEALTH_RESPONSE_RESERVE_MILLIS = 50;

function compileSchema(name: string): ValidateFunction {
  const schema = require(`nemo-fabric-adapter-contract/schemas/${name}`) as object;
  return ajv.compile(schema);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireRecord(value: unknown, code: string, message: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new LifecycleError(code, message);
  }
  return value;
}

function validate<T>(validator: ValidateFunction, value: unknown, code: string, message: string): T {
  if (!validator(value)) {
    throw new LifecycleError(code, message);
  }
  return value as T;
}

function runtimeId(operation: string, payload: Record<string, unknown>): string | undefined {
  const value =
    operation === "stop"
      ? payload.runtime_id
      : isRecord(payload.runtime_context)
        ? payload.runtime_context.runtime_id
        : undefined;
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function decodeRequest(value: unknown): LifecycleRequest {
  const message = requireRecord(value, "lifecycle_invalid_request", "Lifecycle request must be an object");
  const operation = message.operation;
  if (operation !== "start" && operation !== "invoke" && operation !== "stop") {
    throw new LifecycleError("lifecycle_invalid_operation", "Unknown lifecycle operation");
  }
  return {
    operation,
    payload: requireRecord(message.payload, "lifecycle_invalid_payload", "Lifecycle payload must be an object"),
  };
}

function decodeStart(payload: Record<string, unknown>): AdapterStartInput {
  if (typeof payload.agent_name !== "string" || payload.agent_name.length === 0) {
    throw new LifecycleError("lifecycle_invalid_start", "Start payload is missing an agent name");
  }
  if (typeof payload.base_dir !== "string" || payload.base_dir.length === 0) {
    throw new LifecycleError("lifecycle_invalid_start", "Start payload is missing a base directory");
  }
  const config = validate<AgentConfig>(
    validateAgentConfig,
    payload.config,
    "lifecycle_invalid_config",
    "Adapter config does not match its typed contract",
  );
  const context = validate<RuntimeContext>(
    validateRuntimeContext,
    payload.runtime_context,
    "lifecycle_invalid_context",
    "Runtime context does not match its typed contract",
  );
  const capabilityPlan = payload.capability_plan;
  const telemetryPlan = payload.telemetry_plan;
  if (capabilityPlan !== undefined && !isRecord(capabilityPlan)) {
    throw new LifecycleError("lifecycle_invalid_start", "Capability plan must be an object");
  }
  if (telemetryPlan !== undefined && !isRecord(telemetryPlan)) {
    throw new LifecycleError("lifecycle_invalid_start", "Telemetry plan must be an object");
  }
  return {
    agentName: payload.agent_name,
    baseDir: payload.base_dir,
    config,
    runtimeContext: context,
    capabilityPlan: capabilityPlan as JsonObject | undefined,
    telemetryPlan: telemetryPlan as JsonObject | undefined,
  };
}

function decodeInvocation(payload: Record<string, unknown>): {
  request: AgentRunRequest;
  context: RuntimeContext;
} {
  return {
    request: validate<AgentRunRequest>(
      validateAgentRunRequest,
      payload.request,
      "lifecycle_invalid_request",
      "Invocation request does not match its typed contract",
    ),
    context: validate<RuntimeContext>(
      validateRuntimeContext,
      payload.runtime_context,
      "lifecycle_invalid_context",
      "Runtime context does not match its typed contract",
    ),
  };
}

async function callAdapter<T>(operation: string, call: () => T | Promise<T>): Promise<T> {
  try {
    return await call();
  } catch (error) {
    if (error instanceof LifecycleError) {
      throw new AdapterCallError(error.code, error.message, {
        retryable: error.retryable,
        metadata: error.metadata,
      });
    }
    throw new AdapterCallError(
      `lifecycle_adapter_${operation}_failed`,
      `Adapter failed during lifecycle ${operation}`,
    );
  }
}

async function stopQuietly(runtime: AdapterRuntime, diagnostics: Writable): Promise<void> {
  try {
    await callAdapter("stop", () => runtime.stop());
  } catch (error) {
    diagnostics.write(`Adapter cleanup failed: ${error instanceof Error ? error.message : "unknown error"}\n`);
  }
}

function nowMillis(): number {
  return Date.now();
}

function healthCheck(
  name: string,
  status: "ok" | "failed" | "unknown" | "unsupported",
  reasonCode: string,
): NonNullable<AdapterHealthResult["checks"]>[number] {
  return {
    name,
    status,
    reason_code: reasonCode,
    observed_at_millis: nowMillis(),
    age_millis: 0,
  };
}

async function closeHealthServer(state: HostState): Promise<void> {
  const server = state.healthServer;
  const sockets = state.healthSockets;
  state.healthServer = undefined;
  state.healthSockets = undefined;
  state.healthToken = undefined;
  if (server === undefined) {
    return;
  }
  for (const socket of sockets ?? []) {
    socket.destroy();
  }
  await new Promise<void>((resolve) => server.close(() => resolve()));
}

async function startHealthServer(state: HostState): Promise<JsonObject> {
  const token = randomBytes(32).toString("base64url");
  const sockets = new Set<Socket>();
  const server = createServer((socket) => {
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
    void handleHealthConnection(state, socket);
  });
  await new Promise<void>((resolve, reject) => {
    const onError = (error: Error): void => reject(error);
    server.once("error", onError);
    server.listen(0, HEALTH_CONTROL_HOST, () => {
      server.off("error", onError);
      resolve();
    });
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    server.close();
    throw new Error("health control server did not expose a TCP address");
  }
  state.healthServer = server;
  state.healthSockets = sockets;
  state.healthToken = token;
  return {
    health_control: {
      protocol_version: HEALTH_CONTROL_PROTOCOL,
      host: HEALTH_CONTROL_HOST,
      port: address.port,
      token,
    },
  };
}

function readHealthLine(socket: Socket): Promise<string> {
  return new Promise((resolve, reject) => {
    let encoded = "";
    const cleanup = (): void => {
      socket.off("data", onData);
      socket.off("error", onError);
      socket.off("end", onEnd);
      socket.off("timeout", onTimeout);
    };
    const onData = (chunk: Buffer): void => {
      encoded += chunk.toString("utf8");
      if (Buffer.byteLength(encoded) > HEALTH_REQUEST_LIMIT) {
        cleanup();
        reject(new Error("health request exceeds the size limit"));
        return;
      }
      const newline = encoded.indexOf("\n");
      if (newline >= 0) {
        cleanup();
        resolve(encoded.slice(0, newline));
      }
    };
    const onError = (error: Error): void => {
      cleanup();
      reject(error);
    };
    const onEnd = (): void => {
      cleanup();
      reject(new Error("health request ended before a complete record"));
    };
    const onTimeout = (): void => {
      cleanup();
      reject(new Error("health request timed out"));
    };
    socket.on("data", onData);
    socket.once("error", onError);
    socket.once("end", onEnd);
    socket.once("timeout", onTimeout);
    socket.setTimeout(HEALTH_REQUEST_TIMEOUT_MILLIS);
  });
}

function tokensEqual(actual: unknown, expected: string | undefined): boolean {
  if (typeof actual !== "string" || expected === undefined) {
    return false;
  }
  const actualBytes = Buffer.from(actual);
  const expectedBytes = Buffer.from(expected);
  return actualBytes.length === expectedBytes.length && timingSafeEqual(actualBytes, expectedBytes);
}

async function handleHealthConnection(state: HostState, socket: Socket): Promise<void> {
  let responseSent = false;
  try {
    const message = requireRecord(
      JSON.parse(await readHealthLine(socket)) as unknown,
      "lifecycle_invalid_health_request",
      "Health request must be an object",
    );
    if (!tokensEqual(message.token, state.healthToken)) {
      return;
    }
    if (message.protocol_version !== HEALTH_CONTROL_PROTOCOL) {
      return;
    }
    const request = validate<AdapterHealthRequest>(
      validateAdapterHealthRequest,
      {
        runtime_id: message.runtime_id,
        timeout_millis: message.timeout_millis,
      },
      "lifecycle_invalid_health_request",
      "Health request does not match its typed contract",
    );
    if (request.runtime_id !== state.runtimeId || state.runtime === undefined) {
      return;
    }
    const result = await adapterHealth(state, request);
    socket.end(
      `${JSON.stringify({
        protocol_version: HEALTH_CONTROL_PROTOCOL,
        runtime_id: request.runtime_id,
        result,
      })}\n`,
    );
    responseSent = true;
  } catch {
    // Invalid and unauthenticated health requests fail closed without details.
  } finally {
    if (!responseSent) {
      socket.destroy();
    }
  }
}

class HealthHookTimeout extends Error {}

async function adapterHealth(
  state: HostState,
  request: AdapterHealthRequest,
): Promise<AdapterHealthResult> {
  const checks: NonNullable<AdapterHealthResult["checks"]> = [
    healthCheck("dependency.inference", "unsupported", "inference_probe_prohibited"),
  ];
  if (state.failed) {
    return { readiness: { state: "not_ready", reason_code: "runtime_failed" }, checks };
  }
  if (state.stopping) {
    return { readiness: { state: "not_ready", reason_code: "stop_in_progress" }, checks };
  }

  let readiness: NonNullable<AdapterHealthResult["readiness"]> = state.invoking
    ? { state: "not_ready", reason_code: "invocation_in_progress" }
    : { state: "ready", reason_code: "ready" };
  const runtime = state.runtime;
  const hook = runtime?.health;
  if (hook === undefined) {
    checks.push(healthCheck("adapter.health", "unsupported", "check_unsupported"));
    return { readiness, checks };
  }

  const hookBudgetMillis = Math.max(
    0,
    request.timeout_millis - HEALTH_RESPONSE_RESERVE_MILLIS,
  );
  if (hookBudgetMillis === 0) {
    checks.push(healthCheck("adapter.health", "unknown", "adapter_health_timed_out"));
    return { readiness, checks };
  }
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    const result = await Promise.race([
      callAdapter("health", () => hook.call(runtime, request)),
      new Promise<never>((_resolve, reject) => {
        timeout = setTimeout(() => reject(new HealthHookTimeout()), hookBudgetMillis);
      }),
    ]);
    validate<AdapterHealthResult>(
      validateAdapterHealthResult,
      result,
      "lifecycle_invalid_health_response",
      "Adapter health hook returned an invalid result",
    );
    checks.push(...(result.checks ?? []));
    if (!state.invoking && result.readiness !== undefined && result.readiness !== null) {
      readiness = result.readiness;
    }
  } catch (error) {
    checks.push(
      healthCheck(
        "adapter.health",
        "unknown",
        error instanceof HealthHookTimeout ? "adapter_health_timed_out" : "adapter_health_failed",
      ),
    );
  } finally {
    if (timeout !== undefined) {
      clearTimeout(timeout);
    }
  }
  return { readiness, checks };
}

function success(operation: string, output: unknown = null): LifecycleResponse {
  return { operation, outcome: { status: "succeeded", output } };
}

function failure(operation: string, error: LifecycleError): LifecycleResponse {
  const stage = operation === "invoke" ? "invoke" : operation;
  const detail: LifecycleFailure = {
    stage,
    code: error.code,
    message: error.message,
    retryable: error.retryable,
  };
  if (error.metadata !== undefined) {
    detail.metadata = error.metadata;
  }
  return { operation, outcome: { status: "failed", error: detail } };
}

async function dispatch(
  state: HostState,
  factory: AdapterRuntimeFactory,
  request: LifecycleRequest,
  diagnostics: Writable,
): Promise<LifecycleResponse> {
  const messageRuntimeId = runtimeId(request.operation, request.payload);
  if (messageRuntimeId === undefined) {
    throw new LifecycleError("lifecycle_invalid_runtime", "Lifecycle payload is missing a runtime ID");
  }

  if (request.operation === "start") {
    if (state.runtime !== undefined) {
      throw new LifecycleError("lifecycle_already_started", "Lifecycle host already owns a runtime");
    }
    let candidate: AdapterRuntime | undefined;
    try {
      candidate = await callAdapter("start", factory);
      const active = candidate;
      await callAdapter("start", () => active.start(decodeStart(request.payload)));
      state.runtime = active;
      state.runtimeId = messageRuntimeId;
      state.failed = false;
      state.invoking = false;
      state.stopping = false;
      const output = await startHealthServer(state);
      return success("start", output);
    } catch (error) {
      await closeHealthServer(state);
      state.runtime = undefined;
      state.runtimeId = undefined;
      if (candidate !== undefined) {
        await stopQuietly(candidate, diagnostics);
      }
      throw error;
    }
  }

  if (state.runtime === undefined || state.runtimeId === undefined) {
    throw new LifecycleError("lifecycle_not_started", "Lifecycle host has not started a runtime");
  }
  if (state.runtimeId !== messageRuntimeId) {
    throw new LifecycleError("lifecycle_runtime_mismatch", "Lifecycle payload does not match the active runtime");
  }

  if (request.operation === "stop") {
    const active = state.runtime;
    state.stopping = true;
    await closeHealthServer(state);
    try {
      await callAdapter("stop", () => active.stop());
      state.runtime = undefined;
      state.runtimeId = undefined;
      state.failed = false;
      return success("stop");
    } finally {
      state.stopping = false;
    }
  }

  if (state.failed) {
    throw new LifecycleError("lifecycle_runtime_failed", "Lifecycle runtime cannot accept another invocation");
  }
  const { request: invocation, context } = decodeInvocation(request.payload);
  state.invoking = true;
  try {
    const result = await callAdapter("invoke", () => state.runtime!.invoke(invocation, context));
    validate<AgentRunResult>(
      validateAgentRunResult,
      result,
      "lifecycle_invalid_response",
      "Adapter returned an invalid AgentRunResult",
    );
    return success("invoke", result);
  } catch (error) {
    if (
      error instanceof AdapterCallError ||
      (error instanceof LifecycleError && error.code === "lifecycle_invalid_response")
    ) {
      state.failed = true;
    }
    throw error;
  } finally {
    state.invoking = false;
  }
}

function parseLine(line: string): unknown {
  try {
    return JSON.parse(line) as unknown;
  } catch {
    throw new LifecycleError("lifecycle_invalid_request", "Lifecycle request is not valid JSON");
  }
}

/**
 * Serve the persistent, newline-delimited lifecycle protocol for one adapter runtime.
 *
 * The host reads one JSON request per input line, validates normalized contract
 * payloads, and writes exactly one normalized response per output line. A valid
 * start request creates the runtime through `factory`; subsequent invoke and
 * stop requests must carry the same runtime ID. The host validates adapter
 * results, converts adapter failures into stable lifecycle errors, and prevents
 * further invocation after an unexpected adapter or response-encoding failure.
 *
 * When the protocol uses process stdout, other stdout writes are redirected to
 * stderr so logs cannot corrupt the response stream. The active runtime is
 * cleaned up after a stop request, a failed start, or input termination. Tests
 * can provide isolated input, output, and diagnostic streams through `options`.
 *
 * @param factory Creates the adapter-owned runtime for a valid start request.
 * @param options Overrides the process streams, primarily for embedding and tests.
 */
export async function serve(factory: AdapterRuntimeFactory, options: LifecycleHostOptions = {}): Promise<void> {
  const input = options.input ?? process.stdin;
  const output = options.output ?? process.stdout;
  const diagnostics = options.diagnostics ?? process.stderr;
  const protocolWrite = output.write.bind(output);
  const originalStdoutWrite = process.stdout.write;
  const redirectsStdout = output === process.stdout;
  if (redirectsStdout) {
    process.stdout.write = process.stderr.write.bind(process.stderr) as typeof process.stdout.write;
  }
  const lines = createInterface({ input, crlfDelay: Infinity });
  const state: HostState = { failed: false, invoking: false, stopping: false };

  try {
    for await (const line of lines) {
      let operation = "start";
      let shouldStop = false;
      let response: LifecycleResponse;
      try {
        const parsed = parseLine(line);
        if (isRecord(parsed) && typeof parsed.operation === "string") {
          operation = parsed.operation;
        }
        const request = decodeRequest(parsed);
        operation = request.operation;
        response = await dispatch(state, factory, request, diagnostics);
        shouldStop = operation === "stop";
      } catch (error) {
        const lifecycleError =
          error instanceof LifecycleError
            ? error
            : new LifecycleError("lifecycle_invalid_request", "Invalid lifecycle request");
        response = failure(operation, lifecycleError);
        shouldStop = operation === "start" || operation === "stop";
      }

      let encoded: string;
      try {
        encoded = JSON.stringify(response);
      } catch {
        if (operation === "invoke") {
          state.failed = true;
        }
        response = failure(
          operation,
          new LifecycleError(
            "lifecycle_invalid_response",
            "Adapter response could not be encoded as lifecycle JSON",
          ),
        );
        encoded = JSON.stringify(response);
      }
      protocolWrite(`${encoded}\n`);
      if (shouldStop) {
        break;
      }
    }
  } finally {
    lines.close();
    state.stopping = true;
    await closeHealthServer(state);
    if (state.runtime !== undefined) {
      await stopQuietly(state.runtime, diagnostics);
    }
    if (redirectsStdout) {
      process.stdout.write = originalStdoutWrite;
    }
  }
}

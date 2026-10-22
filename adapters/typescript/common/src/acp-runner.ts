// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFile, spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { join } from "node:path";
import { Readable, Writable } from "node:stream";

import {
  PROTOCOL_VERSION,
  client,
  methods,
  ndJsonStream,
  type ActiveSession,
  type ClientConnection,
  type InitializeResponse,
  type McpServer,
  type PermissionOptionKind,
  type RequestPermissionRequest,
  type RequestPermissionResponse,
  type SessionConfigOption,
  type SetSessionConfigOptionRequest,
  type SetSessionConfigOptionResponse,
  type StopReason,
  type Usage,
} from "@agentclientprotocol/sdk";

export type AcpPermissionDecision = "allow_once" | "deny";
export type AcpMcpServer = McpServer;
export type AcpPermissionResolver = (
  request: RequestPermissionRequest,
) => AcpPermissionDecision | Promise<AcpPermissionDecision>;

export interface AcpConfigOption {
  id: string;
  value: string | boolean;
}

export interface AcpSessionReadinessNotification {
  method: string;
  isReady(params: unknown, sessionId: string): boolean;
}

export interface AcpSessionOptions {
  command: string;
  args?: string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
  mcpServers?: McpServer[];
  modeId?: string;
  configOptions?: AcpConfigOption[];
  readinessNotification?: AcpSessionReadinessNotification;
  permissionResolver?: AcpPermissionResolver;
  startupTimeoutMs?: number;
  promptTimeoutMs?: number;
  shutdownTimeoutMs?: number;
  clientName?: string;
  clientVersion?: string;
}

export interface AcpPromptResult {
  text: string;
  stopReason: StopReason;
  usage?: Usage;
}

export interface AcpSession {
  readonly sessionId: string;
  readonly initializeResponse: InitializeResponse;
  prompt(text: string, timeoutMs?: number): Promise<AcpPromptResult>;
  stop(): Promise<void>;
}

export class AcpRunnerError extends Error {
  readonly code: string;
  readonly retryable: boolean;

  constructor(code: string, message: string, options: { retryable?: boolean } = {}) {
    super(message);
    this.name = "AcpRunnerError";
    this.code = code;
    this.retryable = options.retryable ?? false;
  }
}

const DEFAULT_STARTUP_TIMEOUT_MS = 30_000;
const DEFAULT_PROMPT_TIMEOUT_MS = 300_000;
const DEFAULT_SHUTDOWN_TIMEOUT_MS = 5_000;

function timeout<T>(promise: Promise<T>, milliseconds: number, error: AcpRunnerError): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const expired = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => reject(error), milliseconds);
  });
  return Promise.race([promise, expired]).finally(() => {
    if (timer !== undefined) clearTimeout(timer);
  });
}

function waitForSpawn(child: ChildProcessWithoutNullStreams): Promise<void> {
  if (child.pid !== undefined) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const spawned = () => {
      child.off("error", failed);
      resolve();
    };
    const failed = () => {
      child.off("spawn", spawned);
      reject(new AcpRunnerError("acp_process_unavailable", "The ACP agent process could not be started"));
    };
    child.once("spawn", spawned);
    child.once("error", failed);
  });
}

function waitForExit(child: ChildProcessWithoutNullStreams): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
  return new Promise((resolve) => child.once("exit", () => resolve()));
}

async function terminateProcessTree(child: ChildProcessWithoutNullStreams, timeoutMs: number): Promise<void> {
  const pid = child.pid;
  if (pid === undefined) return;
  if (process.platform === "win32") {
    if (child.exitCode !== null || child.signalCode !== null) return;
    const taskkill = process.env.SystemRoot === undefined
      ? "taskkill"
      : join(process.env.SystemRoot, "System32", "taskkill.exe");
    await new Promise<void>((resolve) => {
      execFile(taskkill, ["/pid", String(pid), "/T", "/F"], { timeout: timeoutMs, windowsHide: true }, () => resolve());
    });
    if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
    await timeout(
      waitForExit(child),
      timeoutMs,
      new AcpRunnerError("acp_shutdown_timeout", "The ACP agent did not stop before the deadline"),
    );
    return;
  }
  const signalGroup = (signal: NodeJS.Signals): void => {
    try {
      process.kill(-pid, signal);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
    }
  };
  // Each agent has its own process group, including tool and MCP children.
  signalGroup("SIGTERM");
  try {
    await timeout(
      waitForExit(child),
      timeoutMs,
      new AcpRunnerError("acp_shutdown_timeout", "The ACP agent did not stop before the deadline"),
    );
  } catch (error) {
    if (!(error instanceof AcpRunnerError) || error.code !== "acp_shutdown_timeout") throw error;
  } finally {
    // The parent can exit before its descendants. Clean up the group even then.
    signalGroup("SIGKILL");
  }
  await timeout(
    waitForExit(child),
    timeoutMs,
    new AcpRunnerError("acp_shutdown_timeout", "The ACP agent did not stop before the deadline"),
  );
}

function selectedPermission(
  request: RequestPermissionRequest,
  decision: AcpPermissionDecision,
): RequestPermissionResponse {
  const preferredKinds: PermissionOptionKind[] =
    decision === "allow_once" ? ["allow_once"] : ["reject_once", "reject_always"];
  for (const kind of preferredKinds) {
    const option = request.options.find((candidate) => candidate.kind === kind);
    if (option !== undefined) {
      return { outcome: { outcome: "selected", optionId: option.optionId } };
    }
  }
  return { outcome: { outcome: "cancelled" } };
}

function validateMcpCapabilities(servers: McpServer[], response: InitializeResponse): void {
  for (const server of servers) {
    if ("type" in server && server.type === "http" && response.agentCapabilities?.mcpCapabilities?.http !== true) {
      throw new AcpRunnerError("acp_mcp_transport_unsupported", "The ACP agent does not support HTTP MCP servers");
    }
    if ("type" in server && server.type === "sse" && response.agentCapabilities?.mcpCapabilities?.sse !== true) {
      throw new AcpRunnerError("acp_mcp_transport_unsupported", "The ACP agent does not support SSE MCP servers");
    }
    if ("type" in server && server.type === "acp" && response.agentCapabilities?.mcpCapabilities?.acp !== true) {
      throw new AcpRunnerError("acp_mcp_transport_unsupported", "The ACP agent does not support ACP MCP servers");
    }
  }
}

function selectValues(option: SessionConfigOption): Array<string | boolean> {
  if (option.type === "boolean") return [true, false];
  return option.options.flatMap((entry) => ("group" in entry ? entry.options.map((item) => item.value) : [entry.value]));
}

function validateConfigOption(options: SessionConfigOption[], configured: AcpConfigOption): void {
  const advertised = options.find((option) => option.id === configured.id);
  if (advertised === undefined || !selectValues(advertised).includes(configured.value)) {
    throw new AcpRunnerError(
      "acp_config_option_unsupported",
      `The ACP agent does not support the configured ${configured.id} value`,
    );
  }
}

class SpawnedAcpSession implements AcpSession {
  readonly sessionId: string;
  readonly initializeResponse: InitializeResponse;

  private stopped = false;
  private readonly child: ChildProcessWithoutNullStreams;
  private readonly connection: ClientConnection;
  private readonly activeSession: ActiveSession;
  private readonly promptTimeoutMs: number;
  private readonly shutdownTimeoutMs: number;

  constructor(
    child: ChildProcessWithoutNullStreams,
    connection: ClientConnection,
    activeSession: ActiveSession,
    initializeResponse: InitializeResponse,
    options: AcpSessionOptions,
  ) {
    this.child = child;
    this.connection = connection;
    this.activeSession = activeSession;
    this.sessionId = activeSession.sessionId;
    this.initializeResponse = initializeResponse;
    this.promptTimeoutMs = options.promptTimeoutMs ?? DEFAULT_PROMPT_TIMEOUT_MS;
    this.shutdownTimeoutMs = options.shutdownTimeoutMs ?? DEFAULT_SHUTDOWN_TIMEOUT_MS;
  }

  async prompt(text: string, timeoutMs = this.promptTimeoutMs): Promise<AcpPromptResult> {
    if (this.stopped) {
      throw new AcpRunnerError("acp_session_stopped", "The ACP session is already stopped");
    }
    const exchange = async (): Promise<AcpPromptResult> => {
      const response = await this.activeSession.prompt(text);
      const output = await this.activeSession.readText();
      return {
        text: output,
        stopReason: response.stopReason,
        ...(response.usage === undefined || response.usage === null ? {} : { usage: response.usage }),
      };
    };
    try {
      return await timeout(
        exchange(),
        timeoutMs,
        new AcpRunnerError("acp_prompt_timeout", "The ACP agent did not finish the prompt before the deadline"),
      );
    } catch (error) {
      if (error instanceof AcpRunnerError && error.code === "acp_prompt_timeout") {
        try {
          await this.connection.agent.notify(methods.agent.session.cancel, { sessionId: this.sessionId });
        } catch {
          // Cleanup will close the connection and terminate the child.
        }
      }
      throw error;
    }
  }

  async stop(): Promise<void> {
    if (this.stopped) return;
    this.stopped = true;
    this.activeSession.dispose();

    if (this.initializeResponse.agentCapabilities?.sessionCapabilities?.close != null) {
      try {
        await timeout(
          this.connection.agent.request(methods.agent.session.close, { sessionId: this.sessionId }),
          this.shutdownTimeoutMs,
          new AcpRunnerError("acp_shutdown_timeout", "The ACP agent did not close the session before the deadline"),
        );
      } catch {
        // Process termination below is the final cleanup boundary.
      }
    }
    this.connection.close();
    await terminateProcessTree(this.child, this.shutdownTimeoutMs);
  }
}

async function stopPartial(
  child: ChildProcessWithoutNullStreams,
  connection: ClientConnection | undefined,
  activeSession: ActiveSession | undefined,
  shutdownTimeoutMs: number,
): Promise<void> {
  activeSession?.dispose();
  connection?.close();
  await terminateProcessTree(child, shutdownTimeoutMs);
}

export async function startAcpSession(options: AcpSessionOptions): Promise<AcpSession> {
  const startupTimeoutMs = options.startupTimeoutMs ?? DEFAULT_STARTUP_TIMEOUT_MS;
  const shutdownTimeoutMs = options.shutdownTimeoutMs ?? DEFAULT_SHUTDOWN_TIMEOUT_MS;
  const child = spawn(options.command, options.args ?? [], {
    cwd: options.cwd,
    env: options.env,
    stdio: ["pipe", "pipe", "pipe"],
    detached: process.platform !== "win32",
  });
  child.stderr.resume();

  let connection: ClientConnection | undefined;
  let activeSession: ActiveSession | undefined;
  try {
    await timeout(
      waitForSpawn(child),
      startupTimeoutMs,
      new AcpRunnerError("acp_startup_timeout", "The ACP agent process did not start before the deadline"),
    );
    const permissionResolver = options.permissionResolver ?? (() => "deny" as const);
    const app = client({ name: options.clientName ?? "nemo-fabric" });
    const pendingReadinessNotifications: unknown[] = [];
    let resolveReadiness: (() => void) | undefined;
    let rejectReadiness: ((error: unknown) => void) | undefined;
    const readiness =
      options.readinessNotification === undefined
        ? undefined
        : new Promise<void>((resolve, reject) => {
            resolveReadiness = resolve;
            rejectReadiness = reject;
          });
    // Notifications can reject readiness before session configuration finishes.
    // Keep the original promise so the await below still observes the failure.
    void readiness?.catch(() => { });
    app.onRequest(methods.client.session.requestPermission, async ({ params }) =>
      selectedPermission(params, await permissionResolver(params)),
    );
    if (options.readinessNotification !== undefined) {
      app.onNotification(
        options.readinessNotification.method,
        (params: unknown) => params,
        ({ params }) => {
          if (activeSession === undefined) {
            pendingReadinessNotifications.push(params);
            return;
          }
          try {
            if (options.readinessNotification?.isReady(params, activeSession.sessionId) === true) {
              resolveReadiness?.();
            }
          } catch (error) {
            rejectReadiness?.(error);
          }
        },
      );
    }
    const stream = ndJsonStream(
      Writable.toWeb(child.stdin) as WritableStream<Uint8Array>,
      Readable.toWeb(child.stdout) as ReadableStream<Uint8Array>,
    );
    connection = app.connect(stream);
    const initializeResponse = await timeout(
      connection.agent.request(methods.agent.initialize, {
        protocolVersion: PROTOCOL_VERSION,
        clientCapabilities: { session: { configOptions: {} } },
        clientInfo: {
          name: options.clientName ?? "nemo-fabric",
          version: options.clientVersion ?? "0.6.0",
        },
      }),
      startupTimeoutMs,
      new AcpRunnerError("acp_startup_timeout", "The ACP agent did not initialize before the deadline"),
    );
    if (initializeResponse.protocolVersion !== PROTOCOL_VERSION) {
      throw new AcpRunnerError("acp_protocol_unsupported", "The ACP agent selected an unsupported protocol version");
    }
    const mcpServers = options.mcpServers ?? [];
    validateMcpCapabilities(mcpServers, initializeResponse);
    activeSession = await timeout(
      connection.agent.buildSession({ cwd: options.cwd, mcpServers }).start(),
      startupTimeoutMs,
      new AcpRunnerError("acp_startup_timeout", "The ACP agent did not create a session before the deadline"),
    );
    if (options.readinessNotification !== undefined) {
      for (const params of pendingReadinessNotifications) {
        if (options.readinessNotification.isReady(params, activeSession.sessionId)) {
          resolveReadiness?.();
          break;
        }
      }
    }

    if (options.modeId !== undefined) {
      const modes = activeSession.modes?.availableModes ?? [];
      if (!modes.some((mode) => mode.id === options.modeId)) {
        throw new AcpRunnerError("acp_mode_unsupported", "The ACP agent does not support the configured mode");
      }
      await timeout(
        connection.agent.request(methods.agent.session.setMode, {
          sessionId: activeSession.sessionId,
          modeId: options.modeId,
        }),
        startupTimeoutMs,
        new AcpRunnerError("acp_startup_timeout", "The ACP agent did not set the session mode before the deadline"),
      );
    }

    let advertisedOptions = activeSession.newSessionResponse.configOptions ?? [];
    for (const configured of options.configOptions ?? []) {
      validateConfigOption(advertisedOptions, configured);
      const request: SetSessionConfigOptionRequest =
        typeof configured.value === "boolean"
          ? {
              sessionId: activeSession.sessionId,
              configId: configured.id,
              value: configured.value,
              type: "boolean",
            }
          : {
              sessionId: activeSession.sessionId,
              configId: configured.id,
              value: configured.value,
            };
      const response = await timeout(
        connection.agent.request<SetSessionConfigOptionResponse, SetSessionConfigOptionRequest>(
          methods.agent.session.setConfigOption,
          request,
        ),
        startupTimeoutMs,
        new AcpRunnerError("acp_startup_timeout", "The ACP agent did not apply session configuration before the deadline"),
      );
      advertisedOptions = response.configOptions;
    }

    if (readiness !== undefined) {
      await timeout(
        readiness,
        startupTimeoutMs,
        new AcpRunnerError("acp_startup_timeout", "The ACP session did not become ready before the deadline"),
      );
    }

    return new SpawnedAcpSession(child, connection, activeSession, initializeResponse, options);
  } catch (error) {
    await stopPartial(child, connection, activeSession, shutdownTimeoutMs);
    if (error instanceof AcpRunnerError) throw error;
    throw new AcpRunnerError("acp_start_failed", "The ACP session could not be started", { retryable: true });
  }
}

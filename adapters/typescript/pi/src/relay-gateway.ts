// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFile, spawn, type ChildProcess } from "node:child_process";
import { constants } from "node:fs";
import { access, mkdir, open, realpath, stat } from "node:fs/promises";
import { createServer } from "node:net";
import { delimiter, dirname, extname, isAbsolute, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";

export const RELAY_HEALTH_TIMEOUT_MS = 10_000;
export const RELAY_STOP_TIMEOUT_MS = 5_000;
export const RELAY_VERSION_TIMEOUT_MS = 5_000;

export interface RelayGatewayLaunch {
  executable: string;
  configPath: string;
  bind: string;
  url: string;
  logPath: string;
  openaiBaseUrl?: string;
  anthropicBaseUrl?: string;
}

export interface RelayCliContract {
  version: [number, number, number];
}

export class RelayGatewayError extends Error {
  constructor(message: string, options: ErrorOptions = {}) {
    super(message, options);
    this.name = "RelayGatewayError";
  }
}

async function executable(path: string): Promise<string | undefined> {
  try {
    const candidate = await realpath(path);
    if (!(await stat(candidate)).isFile()) {
      return undefined;
    }
    await access(candidate, process.platform === "win32" ? constants.F_OK : constants.X_OK);
    return candidate;
  } catch {
    return undefined;
  }
}

function commandCandidates(command: string, pathValue: string | undefined): string[] {
  const directories = (pathValue ?? "").split(delimiter).filter(Boolean);
  if (process.platform !== "win32" || extname(command).length > 0) {
    return directories.map((directory) => resolve(directory, command));
  }
  const extensions = (process.env.PATHEXT ?? ".COM;.EXE;.BAT;.CMD").split(";").filter(Boolean);
  return directories.flatMap((directory) =>
    extensions.map((extension) => resolve(directory, `${command}${extension}`)),
  );
}

export async function resolveRelayCommand(baseDir: string, value: string): Promise<string> {
  const hasPath = value.includes("/") || value.includes("\\");
  const candidates = hasPath
    ? [isAbsolute(value) ? value : resolve(baseDir, value)]
    : commandCandidates(value, process.env.PATH);
  for (const candidate of candidates) {
    const resolved = await executable(candidate);
    if (resolved !== undefined) {
      return resolved;
    }
  }
  throw new RelayGatewayError("NeMo Relay CLI executable was not found");
}

export async function findAvailableTcpPort(host = "127.0.0.1"): Promise<number> {
  const listener = createServer();
  try {
    await new Promise<void>((resolvePromise, reject) => {
      listener.once("error", reject);
      listener.listen(0, host, resolvePromise);
    });
    const address = listener.address();
    if (address === null || typeof address === "string") {
      throw new RelayGatewayError("NeMo Relay gateway port could not be allocated");
    }
    return address.port;
  } finally {
    await new Promise<void>((resolvePromise) => listener.close(() => resolvePromise()));
  }
}

type VersionRunner = (
  executable: string,
  args: string[],
  timeoutMs: number,
) => Promise<{ stdout: string; exitCode: number }>;

const runVersion: VersionRunner = (executablePath, args, timeoutMs) =>
  new Promise((resolvePromise, reject) => {
    execFile(executablePath, args, { encoding: "utf8", timeout: timeoutMs }, (error, stdout) => {
      if (error !== null) {
        reject(error);
        return;
      }
      resolvePromise({ stdout, exitCode: 0 });
    });
  });

export async function relayCliContract(
  executablePath: string,
  runner: VersionRunner = runVersion,
): Promise<RelayCliContract> {
  let completed: { stdout: string; exitCode: number };
  try {
    completed = await runner(executablePath, ["--version"], RELAY_VERSION_TIMEOUT_MS);
  } catch (error) {
    throw new RelayGatewayError("NeMo Relay CLI version could not be determined", { cause: error });
  }
  const match = /\b(\d+)\.(\d+)\.(\d+)(-[0-9A-Za-z][0-9A-Za-z.-]*)?(?:\+[0-9A-Za-z][0-9A-Za-z.-]*)?(?=\s|$)/.exec(
    completed.stdout,
  );
  if (completed.exitCode !== 0 || match === null) {
    throw new RelayGatewayError("NeMo Relay CLI version could not be determined");
  }
  const version: [number, number, number] = [Number(match[1]), Number(match[2]), Number(match[3])];
  if (match[4] !== undefined || version[0] !== 0 || version[1] !== 9) {
    throw new RelayGatewayError(
      `unsupported NeMo Relay CLI version ${version.join(".")}; NeMo Fabric Pi requires >=0.9.0,<0.10.0`,
    );
  }
  return { version };
}

function processExited(child: ChildProcess): boolean {
  return child.exitCode !== null || child.signalCode !== null;
}

export async function waitForRelayGateway(
  child: ChildProcess,
  healthUrl: string,
  options: {
    timeoutMs?: number;
    fetch?: typeof fetch;
    pollIntervalMs?: number;
  } = {},
): Promise<void> {
  const timeoutMs = options.timeoutMs ?? RELAY_HEALTH_TIMEOUT_MS;
  const fetchRequest = options.fetch ?? fetch;
  const pollIntervalMs = options.pollIntervalMs ?? 100;
  const deadline = performance.now() + timeoutMs;
  while (performance.now() < deadline) {
    if (processExited(child)) {
      throw new RelayGatewayError(
        `NeMo Relay gateway exited with status ${child.exitCode ?? child.signalCode} before becoming ready`,
      );
    }
    try {
      const response = await fetchRequest(healthUrl, {
        method: "HEAD",
        signal: AbortSignal.timeout(1_000),
      });
      const ready = response.ok;
      await response.body?.cancel();
      if (ready) {
        return;
      }
    } catch {
      // Retry until the gateway exits or the readiness deadline expires.
    }
    await delay(Math.min(pollIntervalMs, Math.max(0, deadline - performance.now())));
  }
  throw new RelayGatewayError(`NeMo Relay gateway did not become ready at ${healthUrl}`);
}

async function waitForExit(child: ChildProcess, timeoutMs: number): Promise<boolean> {
  if (processExited(child)) {
    return true;
  }
  return new Promise((resolvePromise) => {
    const timer = setTimeout(() => {
      child.off("exit", onExit);
      resolvePromise(false);
    }, timeoutMs);
    const onExit = () => {
      clearTimeout(timer);
      resolvePromise(true);
    };
    child.once("exit", onExit);
  });
}

export async function stopRelayGateway(child: ChildProcess, timeoutMs = RELAY_STOP_TIMEOUT_MS): Promise<void> {
  if (processExited(child)) {
    return;
  }
  try {
    child.kill("SIGTERM");
  } catch (error) {
    if (processExited(child)) {
      return;
    }
    throw new RelayGatewayError("NeMo Relay gateway could not be terminated", {
      cause: error,
    });
  }
  if (await waitForExit(child, timeoutMs)) {
    return;
  }
  try {
    child.kill("SIGKILL");
  } catch (error) {
    if (processExited(child)) {
      return;
    }
    throw new RelayGatewayError("NeMo Relay gateway could not be killed", {
      cause: error,
    });
  }
  if (!(await waitForExit(child, timeoutMs))) {
    throw new RelayGatewayError("NeMo Relay gateway did not stop after kill");
  }
}

type SpawnGateway = typeof spawn;

export async function startRelayGateway(
  launch: RelayGatewayLaunch,
  cwd: string,
  options: {
    spawn?: SpawnGateway;
    fetch?: typeof fetch;
    healthTimeoutMs?: number;
    stopTimeoutMs?: number;
  } = {},
): Promise<ChildProcess> {
  try {
    if (!(await stat(launch.configPath)).isFile()) {
      throw new Error("not a file");
    }
  } catch (error) {
    throw new RelayGatewayError("NeMo Relay gateway configuration was not generated", { cause: error });
  }
  await mkdir(dirname(launch.logPath), { recursive: true });
  const args = ["--config", launch.configPath, "--bind", launch.bind];
  if (launch.openaiBaseUrl !== undefined) {
    args.push("--openai-base-url", launch.openaiBaseUrl);
  }
  if (launch.anthropicBaseUrl !== undefined) {
    args.push("--anthropic-base-url", launch.anthropicBaseUrl);
  }

  const log = await open(launch.logPath, "w");
  let child: ChildProcess;
  try {
    child = (options.spawn ?? spawn)(launch.executable, args, {
      cwd,
      env: { ...process.env },
      stdio: ["ignore", log.fd, log.fd],
    });
    await new Promise<void>((resolvePromise, reject) => {
      child.once("spawn", resolvePromise);
      child.once("error", reject);
    });
  } catch (error) {
    throw new RelayGatewayError(`NeMo Relay gateway could not start; see ${launch.logPath}`, {
      cause: error,
    });
  } finally {
    await log.close();
  }

  try {
    await waitForRelayGateway(child, `${launch.url.replace(/\/+$/, "")}/healthz`, {
      ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
      ...(options.healthTimeoutMs === undefined ? {} : { timeoutMs: options.healthTimeoutMs }),
    });
  } catch (error) {
    try {
      await stopRelayGateway(child, options.stopTimeoutMs);
    } catch (stopError) {
      throw new RelayGatewayError(
        `NeMo Relay gateway failed to become ready and could not be stopped; see ${launch.logPath}`,
        {
          cause: new AggregateError([error, stopError], "NeMo Relay gateway startup and cleanup failed"),
        },
      );
    }
    throw new RelayGatewayError(`NeMo Relay gateway failed to become ready; see ${launch.logPath}`, {
      cause: error,
    });
  }
  return child;
}

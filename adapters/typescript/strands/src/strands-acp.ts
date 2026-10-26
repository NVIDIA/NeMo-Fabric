// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  access,
  mkdir,
  mkdtemp,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";

import type { AgentUsage } from "nemo-fabric-adapter-contract";
import {
  AcpRunnerError,
  LifecycleError,
  startAcpSession,
  type AcpPromptResult,
  type AcpSession,
  type AdapterStartInput,
} from "nemo-fabric-adapters-common";

import {
  selectMcpServers,
  selectModel,
  selectSettings,
  systemInstructions,
  workspacePath,
} from "./configuration.js";

export interface StrandsPromptOutcome {
  text: string;
  stopReason: AcpPromptResult["stopReason"];
  usage?: AgentUsage;
}

export interface StrandsSessionHandle {
  readonly id: string;
  prompt(text: string, timeoutMs?: number): Promise<StrandsPromptOutcome>;
  stop(): Promise<void>;
}

export interface StrandsSessionFactory {
  create(input: AdapterStartInput): Promise<StrandsSessionHandle>;
}

const INHERITED_ENVIRONMENT_NAMES = [
  "APPDATA",
  "COMSPEC",
  "ComSpec",
  "HOME",
  "HTTP_PROXY",
  "HTTPS_PROXY",
  "LANG",
  "LC_ALL",
  "LC_CTYPE",
  "LOCALAPPDATA",
  "NO_PROXY",
  "PATH",
  "PATHEXT",
  "Path",
  "PathExt",
  "SHELL",
  "SSL_CERT_DIR",
  "SSL_CERT_FILE",
  "SYSTEMROOT",
  "SystemRoot",
  "TEMP",
  "TMP",
  "TMPDIR",
  "USER",
  "USERPROFILE",
  "http_proxy",
  "https_proxy",
  "no_proxy",
  "AWS_ACCESS_KEY_ID",
  "AWS_SECRET_ACCESS_KEY",
  "AWS_SESSION_TOKEN",
  "AWS_PROFILE",
  "AWS_REGION",
  "AWS_DEFAULT_REGION",
  "AWS_CONFIG_FILE",
  "AWS_SHARED_CREDENTIALS_FILE",
  "AWS_CA_BUNDLE",
  "AWS_SDK_LOAD_CONFIG",
  "AWS_ROLE_ARN",
  "AWS_ROLE_SESSION_NAME",
  "AWS_WEB_IDENTITY_TOKEN_FILE",
  "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
  "AWS_CONTAINER_CREDENTIALS_FULL_URI",
  "AWS_CONTAINER_AUTHORIZATION_TOKEN",
  "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
  "AWS_EC2_METADATA_DISABLED",
  "AWS_BEARER_TOKEN_BEDROCK",
];

function childEnvironment(input: AdapterStartInput): NodeJS.ProcessEnv {
  const environment: NodeJS.ProcessEnv = {};
  for (const name of INHERITED_ENVIRONMENT_NAMES) {
    if (process.env[name] !== undefined) environment[name] = process.env[name];
  }
  Object.assign(environment, input.runtimeContext.environment.env ?? {});
  return environment;
}

function translateRunnerError(
  error: unknown,
  stage: "start" | "prompt",
): LifecycleError {
  if (error instanceof LifecycleError) return error;
  if (
    error instanceof AcpRunnerError &&
    error.code === "acp_process_unavailable"
  ) {
    return new LifecycleError(
      "strands_harness_unavailable",
      "The installed Strands CLI could not be started",
    );
  }
  if (error instanceof AcpRunnerError && error.code === "acp_prompt_timeout") {
    return new LifecycleError(
      "strands_prompt_timeout",
      "Strands did not finish the invocation before the deadline",
    );
  }
  // Do not expose target error text: it may contain credentials or prompt content.
  return new LifecycleError(
    stage === "start" ? "strands_start_failed" : "strands_session_failed",
    stage === "start"
      ? "Strands ACP initialization failed"
      : "Strands ACP session communication failed",
  );
}

function validCounter(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

export function normalizeUsage(
  usage: AcpPromptResult["usage"],
): AgentUsage | undefined {
  if (
    !usage ||
    ![usage.inputTokens, usage.outputTokens, usage.totalTokens].every(
      validCounter,
    )
  )
    return undefined;
  const {
    inputTokens: input,
    outputTokens: output,
    totalTokens: total,
  } = usage;
  if (input + output !== total) return undefined;
  const cache = usage.cachedReadTokens;
  const write = usage.cachedWriteTokens;
  if (
    (cache != null && !validCounter(cache)) ||
    (write != null && !validCounter(write))
  )
    return undefined;
  if ((cache ?? 0) + (write ?? 0) > input) return undefined;
  // Strands CLI 0.2.1 defaults missing cache counts to zero; only positive counts are known.
  // Keep inclusive input/output totals even when cache details are unknown.
  return {
    input_tokens: input,
    output_tokens: output,
    total_tokens: total,
    input_tokens_include_cache: true,
    ...((cache ?? 0) > 0 ? { cached_input_tokens: cache } : {}),
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
        input_tokens_include_cache: true,
        ...((cache ?? 0) > 0 ? { cached_input_tokens: cache } : {}),
        ...((write ?? 0) > 0 ? { cache_write_input_tokens: write } : {}),
      },
    },
  };
}

class StrandsAcpSession implements StrandsSessionHandle {
  readonly id: string;
  constructor(
    private readonly session: AcpSession,
    private readonly home: string,
  ) {
    this.id = session.sessionId;
  }

  async prompt(
    text: string,
    timeoutMs?: number,
  ): Promise<StrandsPromptOutcome> {
    try {
      const result = await this.session.prompt(text, timeoutMs);
      const usage = normalizeUsage(result.usage);
      return {
        text: result.text,
        stopReason: result.stopReason,
        ...(usage === undefined ? {} : { usage }),
      };
    } catch (error) {
      throw translateRunnerError(error, "prompt");
    }
  }

  async stop(): Promise<void> {
    try {
      await this.session.stop();
    } finally {
      await rm(this.home, { recursive: true, force: true });
    }
  }
}

async function command(
  environment: NodeJS.ProcessEnv,
): Promise<{ command: string; args: string[] }> {
  if (process.platform === "win32") {
    // npm creates .cmd shims on Windows; ACP uses direct spawn without a shell.
    for (const bin of (environment.PATH ?? environment.Path ?? "").split(
      delimiter,
    )) {
      for (const candidate of [
        join(bin, "node_modules/@strands-agents/cli/bin/strands.js"),
        resolve(bin, "../@strands-agents/cli/bin/strands.js"),
      ]) {
        try {
          await access(candidate);
          return { command: process.execPath, args: [candidate] };
        } catch {
          /* Try the next npm installation location. */
        }
      }
    }
  }
  return { command: "strands", args: [] };
}

export class StrandsAcpSessionFactory implements StrandsSessionFactory {
  async create(input: AdapterStartInput): Promise<StrandsSessionHandle> {
    const workspace = await workspacePath(input);
    const settings = selectSettings(input.config);
    const model = selectModel(input.config, settings);
    const instructions = systemInstructions(input.config);
    const configured = input.runtimeContext.environment.env ?? {};
    const mcpServers = selectMcpServers(input.config, configured, process.env);
    const environment = childEnvironment(input);
    if (model?.endpoint) {
      const key =
        model.keyEnvironment === undefined
          ? "unused"
          : (configured[model.keyEnvironment] ??
            process.env[model.keyEnvironment]);
      if (!key)
        throw new LifecycleError(
          "strands_credential_missing",
          "The configured Strands model credential is unavailable",
        );
      environment.LITELLM_BASE_URL = model.endpoint;
      environment.LITELLM_API_KEY = key;
    }
    const home = await mkdtemp(join(tmpdir(), "fabric-strands-"));
    try {
      const sourceHome =
        environment.HOME ?? environment.USERPROFILE ?? homedir();
      // Keep the standard AWS SDK credential chain (including SSO caches) while isolating CLI state.
      try {
        await access(join(sourceHome, ".aws"));
        await symlink(
          resolve(sourceHome, ".aws"),
          join(home, ".aws"),
          process.platform === "win32" ? "junction" : "dir",
        );
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      }
      environment.HOME = home;
      environment.USERPROFILE = home;
      environment.NO_COLOR = "1";
      environment.STRANDS_CLI_TELEMETRY = "off";
      environment.STRANDS_CLI_MCP_DISCOVERY = "off";
      await mkdir(join(home, ".strands/cli"), { recursive: true });
      await writeFile(
        join(home, ".strands/cli/config.json"),
        JSON.stringify({
          settings: {
            mcpDiscovery: false,
            skillDiscovery: false,
            telemetry: false,
          },
        }),
        { mode: 0o600 },
      );
      const executable = await command(environment);
      const session = await startAcpSession({
        command: executable.command,
        args: [
          ...executable.args,
          "--acp-server",
          "--set",
          "agentConfig.backgroundTasks=false",
          "--session",
          "off",
          "--memory",
          "off",
          "--skills",
          "off",
          "--builtin-plugins",
          "environment",
          "--builtin-tools",
          settings.builtinTools.join(","),
          ...(model === undefined ? [] : ["--model", model.name]),
          ...(instructions === undefined
            ? []
            : [`--instructions=${instructions}`]),
        ],
        cwd: workspace,
        env: environment,
        mcpServers,
        clientName: "nemo-fabric-strands",
        clientVersion: "0.6.0",
      });
      return new StrandsAcpSession(session, home);
    } catch (error) {
      await rm(home, { recursive: true, force: true });
      throw translateRunnerError(error, "start");
    }
  }
}

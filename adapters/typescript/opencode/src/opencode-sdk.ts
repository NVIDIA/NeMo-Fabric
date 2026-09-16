// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// OpenCode v2 SDK boundary. It owns dynamic harness loading, the embedded host,
// one native session, and the temporary credential environment required by the
// upstream SDK.

import type { AgentUsage } from "nemo-fabric-adapter-contract";
import type { AdapterStartInput } from "nemo-fabric-adapters-common";
import { LifecycleError } from "nemo-fabric-adapters-common";

import { selectModel } from "./configuration.js";
import { ModelEndpointProxy } from "./model-endpoint-proxy.js";
import type { OpenCodePromptOutcome, OpenCodeSessionFactory, OpenCodeSessionHandle } from "./runtime.js";

type OpenCodeClient = Awaited<ReturnType<(typeof import("@opencode/sdk"))["OpenCode"]["create"]>>;
type OpenCodeEmbedOptions = { overrides?: unknown[] };
type OpenCodeEmbedOptionsLoader = (configContent: string) => Promise<OpenCodeEmbedOptions>;
type ModuleResolver = (specifier: string) => string;
type ModuleLoader = (specifier: string) => Promise<unknown>;
interface EndpointProxy {
  readonly url: string;
  close(): Promise<void>;
}
type EndpointProxyFactory = (baseUrl: string) => Promise<EndpointProxy>;
type EmbeddedOpenCodeCreate = (
  options: Parameters<(typeof import("@opencode/sdk"))["OpenCode"]["create"]>[0],
  embedOptions: OpenCodeEmbedOptions,
) => Promise<OpenCodeClient>;

/**
 * OpenCode's Promise SDK accepts embedding overrides. They set the
 * location-specific configuration and instruction services to OpenCode's
 * documented `project: false, global: false` profile while leaving the session
 * location at the Fabric workspace.
 */
async function loadIsolatedEmbedOptions(configContent: string): Promise<OpenCodeEmbedOptions> {
  const [{ Config }, { InstructionDiscovery }] = await Promise.all([
    import("@opencode/core/config"),
    import("@opencode/core/instruction-discovery"),
  ]);
  return {
    overrides: [
      Config.node.replace(Config.configured({ project: false, global: false, content: configContent })),
      InstructionDiscovery.node.replace(InstructionDiscovery.configured({ project: false, global: false })),
    ],
  };
}

function hostConfigContent(model: ReturnType<typeof selectModel>, baseUrl = model.baseUrl): string {
  const provider = {
    settings: { apiKey: `{env:${model.apiKeyEnv}}` },
  };
  if (baseUrl !== undefined) {
    Object.assign(provider, {
      package: "aisdk:@ai-sdk/openai-compatible",
      settings: { ...provider.settings, baseURL: baseUrl },
      models: { [model.model]: {} },
    });
  }
  return JSON.stringify({
    providers: {
      [model.provider]: provider,
    },
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function usage(message: Record<string, unknown>): AgentUsage | undefined {
  const tokens = isRecord(message.tokens) ? message.tokens : undefined;
  const input = tokens?.input;
  const output = tokens?.output;
  const cost = message.cost;
  if (typeof input !== "number" && typeof output !== "number" && typeof cost !== "number") {
    return undefined;
  }
  const result: AgentUsage = {};
  if (typeof input === "number") {
    result.input_tokens = input;
  }
  if (typeof output === "number") {
    result.output_tokens = output;
  }
  if (typeof input === "number" && typeof output === "number") {
    result.total_tokens = input + output;
  }
  if (typeof cost === "number") {
    result.cost_usd = cost;
  }
  return result;
}

function totalUsage(messages: Record<string, unknown>[]): AgentUsage | undefined {
  let inputTokens = 0;
  let outputTokens = 0;
  let costUsd = 0;
  let hasInputTokens = false;
  let hasOutputTokens = false;
  let hasCostUsd = false;
  for (const message of messages) {
    const current = usage(message);
    if (current?.input_tokens !== undefined && current.input_tokens !== null) {
      inputTokens += current.input_tokens;
      hasInputTokens = true;
    }
    if (current?.output_tokens !== undefined && current.output_tokens !== null) {
      outputTokens += current.output_tokens;
      hasOutputTokens = true;
    }
    if (current?.cost_usd !== undefined && current.cost_usd !== null) {
      costUsd += current.cost_usd;
      hasCostUsd = true;
    }
  }
  if (!hasInputTokens && !hasOutputTokens && !hasCostUsd) {
    return undefined;
  }
  return {
    ...(hasInputTokens ? { input_tokens: inputTokens } : {}),
    ...(hasOutputTokens ? { output_tokens: outputTokens } : {}),
    ...(hasInputTokens && hasOutputTokens ? { total_tokens: inputTokens + outputTokens } : {}),
    ...(hasCostUsd ? { cost_usd: costUsd } : {}),
  };
}

function patch(diffs: unknown): string | undefined {
  if (!Array.isArray(diffs)) {
    return undefined;
  }
  const patches = diffs
    .filter((diff): diff is Record<string, unknown> => isRecord(diff))
    .map((diff) => diff.patch)
    .filter((value): value is string => typeof value === "string" && value.length > 0);
  return patches.length === 0 ? undefined : patches.join("\n");
}

function messageIds(messages: unknown[]): Set<string> {
  return new Set(
    messages
      .filter((message): message is Record<string, unknown> => isRecord(message))
      .map((message) => message.id)
      .filter((id): id is string => typeof id === "string"),
  );
}

/** Extract the terminal assistant outcome after OpenCode reports a session idle. */
export function extractOpenCodePromptOutcome(
  messages: unknown[],
  existingMessageIds?: ReadonlySet<string>,
): OpenCodePromptOutcome {
  const assistantMessages: Record<string, unknown>[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.type !== "assistant") {
      continue;
    }
    if (
      existingMessageIds !== undefined &&
      (typeof message.id !== "string" || existingMessageIds.has(message.id))
    ) {
      continue;
    }
    assistantMessages.push(message);
  }
  const terminal = assistantMessages.at(-1);
  if (terminal === undefined) {
    return {};
  }
  const invocationUsage = totalUsage(assistantMessages);
  if (terminal.finish === "error" || terminal.error !== undefined) {
    return {
      errorMessage: "OpenCode model invocation failed",
      ...(invocationUsage === undefined ? {} : { usage: invocationUsage }),
    };
  }
  if (!Array.isArray(terminal.content)) {
    return {};
  }
  const text = terminal.content
    .filter((part): part is Record<string, unknown> => isRecord(part) && part.type === "text")
    .map((part) => part.text)
    .filter((part): part is string => typeof part === "string")
    .join("");
  return {
    ...(text.length === 0 ? {} : { text }),
    ...(invocationUsage === undefined ? {} : { usage: invocationUsage }),
  };
}

export async function loadOpenCodeSdk(
  resolveModule: ModuleResolver = (specifier) => import.meta.resolve(specifier),
  loadModule: ModuleLoader = (specifier) => import(specifier),
): Promise<typeof import("@opencode/sdk")> {
  let coreConfig: string;
  let sdk: string;
  try {
    coreConfig = resolveModule("@opencode/core/config");
    sdk = resolveModule("@opencode/sdk");
  } catch {
    throw new LifecycleError(
      "opencode_harness_unavailable",
      "The OpenCode v2 harness is not installed. Install compatible packages with: npm install @opencode/core@2.0.3 @opencode/sdk@2.0.3",
    );
  }
  try {
    const [, loadedSdk] = await Promise.all([loadModule(coreConfig), loadModule(sdk)]);
    return loadedSdk as typeof import("@opencode/sdk");
  } catch {
    throw new LifecycleError("opencode_harness_load_failed", "The installed OpenCode v2 SDK could not be loaded");
  }
}

async function loadEmbeddedOpenCodeCreate(): Promise<EmbeddedOpenCodeCreate> {
  // @opencode/sdk@2.0.3 exposes its two-argument Promise SDK entry point as a
  // packaged file but does not export it from the package root. Resolve the
  // package root first so this remains valid when npm hoists dependencies. The
  // adapter exact-pins its Core and SDK peers while this private-file workaround
  // is required.
  const sdkRoot = import.meta.resolve("@opencode/sdk");
  const { create } = await import(new URL("./promise.js", sdkRoot).href);
  return create as EmbeddedOpenCodeCreate;
}

interface EnvironmentLease {
  release(): void;
}

function leaseCredential(input: AdapterStartInput, name: string): EnvironmentLease {
  const value = input.runtimeContext.environment.env?.[name] ?? process.env[name];
  if (value === undefined || value.length === 0) {
    throw new LifecycleError("opencode_credential_missing", "The configured OpenCode credential is not available");
  }
  const hadValue = Object.hasOwn(process.env, name);
  const previous = process.env[name];
  process.env[name] = value;
  return {
    release() {
      if (hadValue) {
        process.env[name] = previous;
      } else {
        delete process.env[name];
      }
    },
  };
}

class OpenCodeSdkSessionHandle implements OpenCodeSessionHandle {
  readonly id: string;
  private readonly client: OpenCodeClient;
  private readonly credential: EnvironmentLease;
  private readonly endpointProxy?: EndpointProxy;
  private stopped = false;

  constructor(id: string, client: OpenCodeClient, credential: EnvironmentLease, endpointProxy?: EndpointProxy) {
    this.id = id;
    this.client = client;
    this.credential = credential;
    this.endpointProxy = endpointProxy;
  }

  async prompt(text: string): Promise<OpenCodePromptOutcome> {
    let promptMayHaveBeenSubmitted = false;
    try {
      const historyBeforePrompt = await this.client.sessions.context({ sessionID: this.id });
      const existingMessageIds = messageIds(historyBeforePrompt);
      promptMayHaveBeenSubmitted = true;
      await this.client.sessions.prompt({ sessionID: this.id, text });
      await this.client.sessions.wait({ sessionID: this.id });
      const historyAfterPrompt = await this.client.sessions.context({ sessionID: this.id });
      const outcome = extractOpenCodePromptOutcome(historyAfterPrompt, existingMessageIds);
      if (outcome.errorMessage !== undefined) {
        return outcome;
      }
      try {
        const sessionPatch = patch(await this.client.sessions.diff({ sessionID: this.id }));
        return { ...outcome, ...(sessionPatch === undefined ? {} : { patch: sessionPatch }) };
      } catch {
        return outcome;
      }
    } catch {
      throw new LifecycleError("opencode_session_failed", "OpenCode session communication failed", {
        retryable: !promptMayHaveBeenSubmitted,
      });
    }
  }

  async stop(): Promise<void> {
    if (this.stopped) {
      return;
    }
    this.stopped = true;
    try {
      await this.client.sessions.remove({ sessionID: this.id });
    } finally {
      try {
        await this.client.close();
      } finally {
        try {
          await this.endpointProxy?.close();
        } finally {
          this.credential.release();
        }
      }
    }
  }
}

export class OpenCodeSdkSessionFactory implements OpenCodeSessionFactory {
  private readonly sdkLoader: typeof loadOpenCodeSdk;
  private readonly embedOptionsLoader: OpenCodeEmbedOptionsLoader;
  private readonly endpointProxyFactory: EndpointProxyFactory;

  constructor(
    sdkLoader: typeof loadOpenCodeSdk = loadOpenCodeSdk,
    embedOptionsLoader: OpenCodeEmbedOptionsLoader =
      sdkLoader === loadOpenCodeSdk ? loadIsolatedEmbedOptions : async () => ({}),
    endpointProxyFactory: EndpointProxyFactory = ModelEndpointProxy.create,
  ) {
    this.sdkLoader = sdkLoader;
    this.embedOptionsLoader = embedOptionsLoader;
    this.endpointProxyFactory = endpointProxyFactory;
  }

  async create(input: AdapterStartInput): Promise<OpenCodeSessionHandle> {
    const model = selectModel(input.config);
    const workspace = input.runtimeContext.environment.workspace ?? input.baseDir;
    const credential = leaseCredential(input, model.apiKeyEnv);
    let client: OpenCodeClient | undefined;
    let endpointProxy: EndpointProxy | undefined;
    try {
      const sdk = await this.sdkLoader();
      if (model.baseUrl !== undefined) {
        endpointProxy = await this.endpointProxyFactory(model.baseUrl);
      }
      const configContent = hostConfigContent(model, endpointProxy?.url);
      const embedOptions = await this.embedOptionsLoader(configContent);
      // The OpenCode convenience namespace silently ignores the embed argument.
      // Use the v2.0.3 Promise SDK entry point for production and retain the
      // public convenience API only for injected unit-test doubles.
      const createEmbeddedClient =
        this.sdkLoader === loadOpenCodeSdk
          ? await loadEmbeddedOpenCodeCreate()
          : (sdk.OpenCode.create as EmbeddedOpenCodeCreate);
      client = await createEmbeddedClient({
        config: { directory: workspace, project: false, content: configContent },
        fs: { filewatcher: false },
        models: { fetch: false },
      }, embedOptions);
      const session = await client.sessions.create({
        location: { directory: workspace },
        model: { providerID: model.provider, id: model.model },
      });
      return new OpenCodeSdkSessionHandle(session.id, client, credential, endpointProxy);
    } catch (error) {
      try {
        if (client !== undefined) {
          await client.close();
        }
      } catch {
        // Preserve the startup failure while still releasing the credential lease.
      } finally {
        try {
          await endpointProxy?.close();
        } finally {
          credential.release();
        }
      }
      if (error instanceof LifecycleError) {
        throw error;
      }
      throw new LifecycleError("opencode_start_failed", "OpenCode could not start an embedded session");
    }
  }
}

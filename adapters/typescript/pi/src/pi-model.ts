// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Resolves every configured NeMo Fabric model role to a Pi model. A role's
// settings.model_metadata is one Pi models.json model entry, so Pi's own loader
// defines and validates models that are not in its catalog.

import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { Api, CredentialStore, Model } from "@earendil-works/pi-ai";
import type { ModelRuntime } from "@earendil-works/pi-coding-agent";
import type { AgentModelConfig, JsonObject } from "nemo-fabric-adapter-contract";
import { LifecycleError } from "nemo-fabric-adapters-common";

export interface PiModelModules {
  ModelRuntime: typeof import("@earendil-works/pi-coding-agent").ModelRuntime;
  InMemoryModelsStore: typeof import("@earendil-works/pi-ai").InMemoryModelsStore;
}

export interface PiModelOptions {
  /** Keep the selected role on its catalog provider for NeMo Relay redirects. */
  relayEnabled: boolean;
  credential(name: string): string | undefined;
}

export interface ConfiguredModels {
  modelRuntime: ModelRuntime;
  /** Pi model for each configured role, starting with the selected role. */
  roles: Map<string, Model<Api>>;
  cleanup(): Promise<void>;
}

interface RoleModel {
  role: string;
  config: AgentModelConfig;
  providerId: string;
  metadata?: JsonObject;
}

export function withCustomBaseUrl<T extends { api: string; baseUrl: string; compat?: object }>(
  catalogModel: T,
  baseUrl: string | null | undefined,
  overrideBaseUrl = true,
): T {
  if (!baseUrl) {
    return catalogModel;
  }
  const model = overrideBaseUrl ? { ...catalogModel, baseUrl } : catalogModel;
  if (catalogModel.api !== "openai-completions") {
    return model;
  }
  return {
    ...model,
    // Generic OpenAI-compatible proxies may reject provider-specific
    // reasoning_content fields when Pi replays an assistant tool call.
    compat: { ...catalogModel.compat, requiresThinkingAsText: true },
  };
}

function modelMetadata(config: AgentModelConfig): JsonObject | undefined {
  const value = config.settings?.model_metadata;
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value : undefined;
}

/** Order roles with the selected role first and give each a Pi provider ID. */
function roleModels(models: Record<string, AgentModelConfig>, selectedRole: string): RoleModel[] {
  const ordered = Object.entries(models).sort(([left], [right]) =>
    left === selectedRole ? -1 : right === selectedRole ? 1 : 0,
  );
  const resolved: RoleModel[] = [];
  for (const [role, config] of ordered) {
    const metadata = modelMetadata(config);
    const identical = resolved.find((entry) => JSON.stringify(entry.config) === JSON.stringify(config));
    // The selected role keeps its provider name, as does any catalog model.
    // Another role that defines its own model gets a role-specific provider.
    const providerId =
      identical?.providerId ??
      (role === selectedRole || metadata === undefined ? config.provider : `${config.provider}-${role}`);
    resolved.push({ role, config, providerId, metadata });
  }
  return resolved;
}

function modelsJson(roles: RoleModel[]): JsonObject | undefined {
  const providers: Record<string, JsonObject> = {};
  for (const { role, config, providerId, metadata } of roles) {
    if (metadata === undefined) {
      continue;
    }
    if (config.api != null && metadata.api !== undefined && metadata.api !== config.api) {
      throw new LifecycleError(
        "pi_model_api_conflict",
        `models.${role}.api conflicts with its model_metadata.api`,
        { metadata: { field: `models.${role}.settings.model_metadata.api` } },
      );
    }
    const api = config.api ?? metadata.api;
    const baseUrl = config.base_url ?? undefined;
    providers[providerId] = {
      ...(baseUrl === undefined ? {} : { baseUrl }),
      models: [
        {
          ...metadata,
          ...(api === undefined ? {} : { api }),
          ...(baseUrl === undefined ? {} : { baseUrl }),
          id: config.model,
        },
      ],
    };
  }
  return Object.keys(providers).length === 0 ? undefined : { providers };
}

async function createModelRuntime(
  pi: PiModelModules,
  credentials: CredentialStore,
  models: JsonObject | undefined,
): Promise<{ modelRuntime: ModelRuntime; directory?: string }> {
  const options = { credentials, allowModelNetwork: false, refreshOnCreate: false };
  if (models === undefined) {
    return { modelRuntime: await pi.ModelRuntime.create({ ...options, modelsPath: null }) };
  }
  const directory = await mkdtemp(join(tmpdir(), "fabric-pi-models-"));
  try {
    const modelsPath = join(directory, "models.json");
    await writeFile(modelsPath, JSON.stringify(models), { mode: 0o600 });
    const modelRuntime = await pi.ModelRuntime.create({
      ...options,
      modelsPath,
      modelsStore: new pi.InMemoryModelsStore(),
    });
    const error = modelRuntime.getError();
    if (error) {
      throw new LifecycleError("pi_model_invalid", error);
    }
    return { modelRuntime, directory };
  } catch (error) {
    await rm(directory, { recursive: true, force: true });
    throw error;
  }
}

async function setCredentials(modelRuntime: ModelRuntime, roles: RoleModel[], options: PiModelOptions) {
  const keys = new Map<string, string>();
  for (const { role, config, providerId } of roles) {
    const name = config.api_key_env;
    if (!name) {
      throw new LifecycleError("pi_api_key_env_required", `The Pi model role ${role} requires api_key_env`, {
        metadata: { field: `models.${role}.api_key_env` },
      });
    }
    const key = options.credential(name);
    if (!key) {
      throw new LifecycleError("pi_credential_missing", `Credential environment variable ${name} is not set`);
    }
    if (keys.has(providerId) && keys.get(providerId) !== key) {
      throw new LifecycleError(
        "pi_credential_conflict",
        `Pi model roles use different credentials for provider ${providerId}`,
        { metadata: { field: `models.${role}.api_key_env` } },
      );
    }
    keys.set(providerId, key);
  }
  for (const [providerId, key] of keys) {
    await modelRuntime.setRuntimeApiKey(providerId, key);
  }
}

/** Resolve every model role in one Pi ModelRuntime. */
export async function loadConfiguredModels(
  pi: PiModelModules,
  credentials: CredentialStore,
  models: Record<string, AgentModelConfig>,
  selectedRole: string,
  options: PiModelOptions,
): Promise<ConfiguredModels> {
  const roles = roleModels(models, selectedRole);
  const { modelRuntime, directory } = await createModelRuntime(pi, credentials, modelsJson(roles));
  const cleanup = async () => {
    if (directory !== undefined) {
      await rm(directory, { recursive: true, force: true });
    }
  };
  try {
    await setCredentials(modelRuntime, roles, options);
    const selected = roles[0];
    if (selected === undefined) {
      throw new LifecycleError("pi_model_required", "The Pi adapter requires one configured model");
    }
    const relayProvider = options.relayEnabled && selected.metadata === undefined && selected.config.base_url;
    if (relayProvider) {
      // Configure the provider before Relay loads so its provider-wide redirect
      // sees a consistent catalog instead of one overlaid selected model.
      modelRuntime.registerProvider(selected.providerId, { baseUrl: relayProvider });
    }
    const resolved = new Map<string, Model<Api>>();
    for (const { role, config, providerId, metadata } of roles) {
      const found = modelRuntime.getModel(providerId, config.model);
      if (found === undefined) {
        throw new LifecycleError(
          "pi_model_unknown",
          `The Pi catalog does not contain ${config.provider}/${config.model}; declare it in models.${role}.settings.model_metadata`,
          { metadata: { field: `models.${role}.model` } },
        );
      }
      let model = found;
      if (metadata === undefined) {
        if (config.api != null && config.api !== model.api) {
          model = { ...model, api: config.api };
        }
        model = withCustomBaseUrl(model, config.base_url, !(relayProvider && role === selected.role));
      }
      resolved.set(role, model);
    }
    return { modelRuntime, roles: resolved, cleanup };
  } catch (error) {
    await cleanup();
    throw error;
  }
}

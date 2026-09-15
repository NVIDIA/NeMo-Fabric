// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Translate only the model configuration declared by the OpenCode descriptor.

import type { AgentConfig, AgentModelConfig } from "nemo-fabric-adapter-contract";
import { LifecycleError } from "nemo-fabric-adapters-common";

export interface OpenCodeModel {
  provider: string;
  model: string;
  apiKeyEnv: string;
  baseUrl?: string;
}

function loopbackHostname(hostname: string): boolean {
  return hostname === "localhost" || hostname === "::1" || hostname === "[::1]" || /^127(?:\.\d{1,3}){3}$/u.test(hostname);
}

function validEndpoint(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === "https:" || (url.protocol === "http:" && loopbackHostname(url.hostname));
  } catch {
    return false;
  }
}

function validModel(model: AgentModelConfig): OpenCodeModel {
  if (
    typeof model.provider !== "string" ||
    model.provider.length === 0 ||
    typeof model.model !== "string" ||
    model.model.length === 0 ||
    typeof model.api_key_env !== "string" ||
    model.api_key_env.length === 0
  ) {
    throw new LifecycleError("opencode_invalid_model", "OpenCode model configuration does not match the adapter schema");
  }
  if (
    model.base_url !== undefined &&
    model.base_url !== null &&
    (typeof model.base_url !== "string" || !validEndpoint(model.base_url))
  ) {
    throw new LifecycleError("opencode_invalid_model", "OpenCode model configuration does not match the adapter schema");
  }
  return {
    provider: model.provider,
    model: model.model,
    apiKeyEnv: model.api_key_env,
    ...(typeof model.base_url === "string" ? { baseUrl: model.base_url } : {}),
  };
}

export function selectModel(config: AgentConfig): OpenCodeModel {
  const entries = Object.entries(config.models ?? {});
  if (entries.length === 0) {
    throw new LifecycleError("opencode_model_required", "The OpenCode adapter requires one configured model");
  }
  const selected = config.models?.default ?? (entries.length === 1 ? entries[0]?.[1] : undefined);
  if (selected === undefined) {
    throw new LifecycleError(
      "opencode_model_ambiguous",
      "Configure a default model role when the OpenCode adapter receives multiple models",
    );
  }
  return validModel(selected);
}

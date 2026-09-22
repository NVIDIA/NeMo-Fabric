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
  sampling?: {
    temperature?: number;
    topP?: number;
  };
}

const ENVIRONMENT_VARIABLE_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/u;

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

function sampling(model: AgentModelConfig, baseUrl: string | undefined): OpenCodeModel["sampling"] {
  if (model.max_tokens !== undefined && model.max_tokens !== null) {
    throw new LifecycleError(
      "opencode_max_tokens_unsupported",
      "OpenCode does not support models.max_tokens",
    );
  }
  if (model.settings !== undefined) {
    throw new LifecycleError(
      "opencode_model_settings_unsupported",
      "OpenCode does not support provider-specific model settings",
    );
  }
  const hasTemperature = model.temperature !== undefined && model.temperature !== null;
  const hasTopP = model.top_p !== undefined && model.top_p !== null;
  if (!hasTemperature && !hasTopP) {
    return undefined;
  }
  if (baseUrl === undefined) {
    throw new LifecycleError(
      "opencode_sampling_requires_base_url",
      "OpenCode sampling settings require an OpenAI-compatible models.base_url endpoint",
    );
  }
  if (
    (hasTemperature && (typeof model.temperature !== "number" || !Number.isFinite(model.temperature))) ||
    (hasTopP &&
      (typeof model.top_p !== "number" || !Number.isFinite(model.top_p) || model.top_p < 0 || model.top_p > 1))
  ) {
    throw new LifecycleError("opencode_invalid_model", "OpenCode model configuration does not match the adapter schema");
  }
  return {
    ...(typeof model.temperature === "number" ? { temperature: model.temperature } : {}),
    ...(typeof model.top_p === "number" ? { topP: model.top_p } : {}),
  };
}

function validModel(model: AgentModelConfig): OpenCodeModel {
  if (
    typeof model.provider !== "string" ||
    model.provider.length === 0 ||
    typeof model.model !== "string" ||
    model.model.length === 0 ||
    typeof model.api_key_env !== "string" ||
    !ENVIRONMENT_VARIABLE_NAME.test(model.api_key_env)
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
  const baseUrl = typeof model.base_url === "string" ? model.base_url : undefined;
  const configuredSampling = sampling(model, baseUrl);
  return {
    provider: model.provider,
    model: model.model,
    apiKeyEnv: model.api_key_env,
    ...(baseUrl === undefined ? {} : { baseUrl }),
    ...(configuredSampling === undefined ? {} : { sampling: configuredSampling }),
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

export function selectSystemInstruction(config: AgentConfig): string | undefined {
  const instruction = config.instructions?.system;
  if (instruction === undefined || instruction === null) {
    return undefined;
  }
  if (typeof instruction.content !== "string" || instruction.content.trim().length === 0) {
    throw new LifecycleError(
      "opencode_invalid_system_instruction",
      "OpenCode system instructions must contain non-empty text",
    );
  }
  if (instruction.mode !== undefined && instruction.mode !== "replace") {
    throw new LifecycleError(
      "unsupported_system_instruction_mode",
      `OpenCode does not support instructions.system.mode=${JSON.stringify(instruction.mode)}; supported modes: replace`,
      {
        metadata: {
          field: "instructions.system.mode",
          mode: instruction.mode,
          supported_modes: ["replace"],
        },
      },
    );
  }
  return instruction.content;
}

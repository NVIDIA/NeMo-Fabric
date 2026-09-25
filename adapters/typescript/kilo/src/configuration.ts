// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentConfig, AgentModelConfig } from "nemo-fabric-adapter-contract";
import { LifecycleError } from "nemo-fabric-adapters-common";

export interface KiloModel {
  provider: string;
  model: string;
  apiKeyEnv: string;
  baseUrl?: string;
  temperature?: number;
  topP?: number;
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

function validateModel(model: AgentModelConfig): KiloModel {
  if (model.max_tokens != null) {
    throw new LifecycleError("kilo_max_tokens_unsupported", "Kilo Code does not support models.max_tokens through this adapter");
  }
  if (model.settings !== undefined) {
    throw new LifecycleError("kilo_model_settings_unsupported", "Kilo Code does not support provider-specific model settings through this adapter");
  }
  if (
    typeof model.provider !== "string" || model.provider.length === 0 ||
    typeof model.model !== "string" || model.model.length === 0 ||
    typeof model.api_key_env !== "string" || !ENVIRONMENT_VARIABLE_NAME.test(model.api_key_env)
  ) {
    throw new LifecycleError("kilo_invalid_model", "Kilo Code model configuration does not match the adapter schema");
  }
  if (model.base_url != null && (typeof model.base_url !== "string" || !validEndpoint(model.base_url))) {
    throw new LifecycleError("kilo_invalid_model", "Kilo Code model configuration does not match the adapter schema");
  }
  if (model.temperature != null && (typeof model.temperature !== "number" || !Number.isFinite(model.temperature))) {
    throw new LifecycleError("kilo_invalid_model", "Kilo Code model configuration does not match the adapter schema");
  }
  if (model.top_p != null && (typeof model.top_p !== "number" || !Number.isFinite(model.top_p) || model.top_p < 0 || model.top_p > 1)) {
    throw new LifecycleError("kilo_invalid_model", "Kilo Code model configuration does not match the adapter schema");
  }
  return {
    provider: model.provider,
    model: model.model,
    apiKeyEnv: model.api_key_env,
    ...(typeof model.base_url === "string" ? { baseUrl: model.base_url } : {}),
    ...(typeof model.temperature === "number" ? { temperature: model.temperature } : {}),
    ...(typeof model.top_p === "number" ? { topP: model.top_p } : {}),
  };
}

export function selectModel(config: AgentConfig): KiloModel {
  const entries = Object.entries(config.models ?? {});
  if (entries.length === 0) {
    throw new LifecycleError("kilo_model_required", "The Kilo Code adapter requires one configured model");
  }
  const selected = config.models?.default ?? (entries.length === 1 ? entries[0]?.[1] : undefined);
  if (selected === undefined) {
    throw new LifecycleError("kilo_model_ambiguous", "Configure a default model role when the Kilo Code adapter receives multiple models");
  }
  return validateModel(selected);
}

export function selectSystemInstruction(config: AgentConfig): string | undefined {
  const instruction = config.instructions?.system;
  if (instruction == null) return undefined;
  if (typeof instruction.content !== "string" || instruction.content.trim().length === 0) {
    throw new LifecycleError("kilo_invalid_system_instruction", "Kilo Code system instructions must contain non-empty text");
  }
  if (instruction.mode !== undefined && instruction.mode !== "replace") {
    throw new LifecycleError("unsupported_system_instruction_mode", "Kilo Code supports only replacement system instructions", {
      metadata: {field: "instructions.system.mode", mode: instruction.mode, supported_modes: ["replace"]},
    });
  }
  return instruction.content;
}

export function selectMaxTurns(config: AgentConfig): number | undefined {
  const value = config.runtime?.max_turns;
  if (value == null) return undefined;
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new LifecycleError("kilo_invalid_max_turns", "Kilo Code runtime.max_turns must be a positive integer");
  }
  return value;
}

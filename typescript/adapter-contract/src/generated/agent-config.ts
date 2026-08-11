// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical adapter-contract JSON Schemas.
// Do not edit it directly; run `npm run generate` instead.

import type { JsonObject } from "../json.js";

/**
 * Configuration projected southbound to one adapter target.
 */
export interface AgentConfig {
  /**
   * Adapter-owned fields validated against the selected adapter descriptor.
   */
  extensions?: JsonObject;
  /**
   * Adapter-owned target settings.
   */
  harness?: AgentHarnessConfig | null;
  /**
   * Normalized instructions applied by the adapter target.
   */
  instructions?: AgentInstructionsConfig | null;
  /**
   * MCP servers routed to the adapter target.
   */
  mcp?: AgentMcpConfig | null;
  /**
   * Named model roles applied by the adapter target.
   */
  models?: {
    [k: string]: AgentModelConfig;
  };
  /**
   * Adapter-applied runtime behavior.
   */
  runtime?: AgentRuntimeConfig | null;
  /**
   * Skills made available to the adapter target.
   */
  skills?: AgentSkillConfig | null;
  /**
   * Named tool definitions and effective tool policy.
   */
  tools?: AgentToolsConfig | null;
  /**
   * Custom agent or workflow selection and construction settings.
   */
  workflow?: AgentWorkflowConfig | null;
}
/**
 * Adapter-owned target settings projected from `FabricConfig.harness`.
 */
export interface AgentHarnessConfig {
  /**
   * Adapter-owned harness fields.
   */
  extensions?: JsonObject;
  /**
   * Target-specific settings validated by the selected adapter descriptor.
   */
  settings?: JsonObject;
}
/**
 * Normalized instructions projected to an adapter target.
 */
export interface AgentInstructionsConfig {
  /**
   * Adapter-owned instruction categories.
   */
  extensions?: JsonObject;
  /**
   * System instructions for the selected adapter target.
   */
  system?: AgentInstructionConfig | null;
}
/**
 * One normalized instruction value projected to an adapter target.
 */
export interface AgentInstructionConfig {
  /**
   * Instruction text.
   */
  content: string;
  /**
   * Adapter-owned instruction fields.
   */
  extensions?: JsonObject;
  /**
   * How the instruction is applied.
   */
  mode?: "replace";
}
/**
 * Named MCP servers routed to an adapter target.
 */
export interface AgentMcpConfig {
  /**
   * Adapter-owned MCP fields.
   */
  extensions?: JsonObject;
  /**
   * MCP servers keyed by normalized server name.
   */
  servers?: {
    [k: string]: AgentMcpServerConfig;
  };
}
/**
 * One MCP server routed to an adapter target.
 */
export interface AgentMcpServerConfig {
  /**
   * MCP tool names to expose. `None` exposes every discovered tool.
   */
  allowed_tools?: string[] | null;
  /**
   * Command-line arguments passed to an MCP stdio process.
   */
  args?: string[];
  /**
   * MCP tool names blocked after applying the optional allowlist.
   */
  blocked_tools?: string[];
  /**
   * Environment variables passed to an MCP stdio process.
   */
  env?: {
    [k: string]: string;
  };
  /**
   * Adapter-owned MCP server fields.
   */
  extensions?: JsonObject;
  /**
   * MCP transport identifier.
   */
  transport: string;
  /**
   * MCP server URL for network transports or executable for stdio.
   */
  url: string;
}
/**
 * Configuration for one named model role projected to an adapter target.
 */
export interface AgentModelConfig {
  /**
   * Environment variable containing the provider credential.
   */
  api_key_env?: string | null;
  /**
   * Optional provider API base URL.
   */
  base_url?: string | null;
  /**
   * Adapter-owned model fields.
   */
  extensions?: JsonObject;
  /**
   * Provider model identifier.
   */
  model: string;
  /**
   * Model provider identifier.
   */
  provider: string;
  /**
   * Provider-specific model settings.
   */
  settings?: JsonObject;
  /**
   * Optional model temperature.
   */
  temperature?: number | null;
}
/**
 * Runtime behavior applied by an adapter target.
 */
export interface AgentRuntimeConfig {
  /**
   * Adapter-owned runtime fields.
   */
  extensions?: JsonObject;
  /**
   * Maximum number of agent turns allowed for one invocation.
   */
  max_turns?: number | null;
}
/**
 * Skill paths made available to an adapter target.
 */
export interface AgentSkillConfig {
  /**
   * Adapter-owned skill fields.
   */
  extensions?: JsonObject;
  /**
   * Skill paths resolved for the task environment.
   */
  paths?: string[];
}
/**
 * Named tool definitions and effective adapter-target tool policy.
 */
export interface AgentToolsConfig {
  /**
   * Named tools to block.
   */
  blocked?: string[];
  /**
   * Tool and tool-group definitions keyed by normalized name.
   */
  definitions?: {
    [k: string]: AgentToolDefinition;
  };
  /**
   * Named tools to expose. `None` preserves the adapter-target default.
   */
  enabled?: string[] | null;
  /**
   * Adapter-owned tool fields.
   */
  extensions?: JsonObject;
}
/**
 * One named tool or tool-group definition resolved by an adapter.
 */
export interface AgentToolDefinition {
  /**
   * Adapter-owned tool-definition fields.
   */
  extensions?: JsonObject;
  /**
   * Resolution semantics declared by the selected adapter descriptor.
   */
  kind: string;
  /**
   * Executable or factory reference interpreted under `kind`.
   */
  ref: string;
  /**
   * Definition-specific construction settings.
   */
  settings?: JsonObject;
}
/**
 * Custom agent or workflow selection and construction settings.
 */
export interface AgentWorkflowConfig {
  entrypoint: AgentWorkflowEntrypointConfig;
  /**
   * Adapter-owned workflow fields.
   */
  extensions?: JsonObject;
  /**
   * Agent-specific construction settings.
   */
  settings?: JsonObject;
}
/**
 * Entry point resolved by the selected adapter.
 */
export interface AgentWorkflowEntrypointConfig {
  /**
   * Adapter-owned entry-point fields.
   */
  extensions?: JsonObject;
  /**
   * Resolution semantics declared by the selected adapter descriptor.
   */
  kind: string;
  /**
   * Executable or factory reference interpreted under `kind`.
   */
  ref: string;
}

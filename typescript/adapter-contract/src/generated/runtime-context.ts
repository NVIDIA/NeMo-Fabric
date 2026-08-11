// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical adapter-contract JSON Schemas.
// Do not edit it directly; run `npm run generate` instead.

import type { JsonObject } from "../json.js";

/**
 * Context generated for one invocation of a started runtime.
 */
export interface RuntimeContext {
  artifacts: ArtifactManifest;
  environment: EnvironmentHandle;
  /**
   * Invocation handle id.
   */
  invocation_id: string;
  /**
   * Request id.
   */
  request_id: string;
  /**
   * Runtime handle id.
   */
  runtime_id: string;
  /**
   * Runtime telemetry context generated for this invocation.
   */
  telemetry?: RuntimeTelemetryContext | null;
}
/**
 * Artifact manifest visible to the adapter at invocation start.
 */
export interface ArtifactManifest {
  /**
   * Artifact entries.
   */
  artifacts?: ArtifactRef[];
  /**
   * Artifact root directory.
   */
  root?: string | null;
}
/**
 * Reference to one artifact.
 */
export interface ArtifactRef {
  /**
   * Artifact kind.
   */
  kind: string;
  /**
   * Optional media type.
   */
  media_type?: string | null;
  /**
   * Artifact-specific metadata preserved across the Rust and Python SDK boundary.
   */
  metadata?: JsonObject;
  /**
   * Logical artifact name.
   */
  name: string;
  /**
   * Artifact path.
   */
  path: string;
}
/**
 * Prepared execution environment.
 */
export interface EnvironmentHandle {
  /**
   * Artifact root visible to the harness runtime.
   */
  artifacts?: string | null;
  /**
   * Provider connection metadata.
   */
  connection?: JsonObject;
  /**
   * Where NeMo Fabric control code runs.
   */
  control_location: "external_control" | "in_env_control";
  /**
   * Environment variables visible to the harness and its tools.
   */
  env?: {
    [k: string]: string;
  };
  /**
   * Environment handle id.
   */
  environment_id: string;
  /**
   * Provider-specific metadata.
   */
  metadata?: JsonObject;
  /**
   * Whether NeMo Fabric owns the environment resource.
   */
  ownership: "caller_owned" | "fabric_owned";
  /**
   * Environment provider.
   */
  provider: string;
  /**
   * Workspace visible to the harness runtime.
   */
  workspace?: string | null;
}
/**
 * Runtime telemetry config passed to adapters.
 */
export interface RuntimeTelemetryContext {
  /**
   * Generated Relay config path for this invocation.
   */
  config_path?: string | null;
  /**
   * Environment variables NeMo Fabric applies while invoking the adapter.
   */
  env?: {
    [k: string]: string;
  };
  /**
   * Additional telemetry metadata surfaced to consumers and adapters.
   */
  metadata?: JsonObject;
  /**
   * Whether Relay is enabled for this invocation.
   */
  relay_enabled: boolean;
}

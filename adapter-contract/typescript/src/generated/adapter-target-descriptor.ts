// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is generated from the canonical NeMo Fabric JSON Schemas.
// Do not edit it directly; run `npm run generate` instead.

import type { JsonObject } from "../json.js";

/**
 * Independently registered target implemented by an adapter.
 */
export type AdapterTargetDescriptor = {
  /**
   * Adapter implementation selected by this target.
   */
  adapter_id: string;
  /**
   * Adapter Contract version shared with Adapter Descriptor.
   */
  contract_version: "fabric.adapter/v1alpha2";
  /**
   * Unique registered target id.
   */
  id: string;
} & JsonObject & {
  spec: WorkflowTargetSpec;
  type: "workflow";
} & JsonObject;

/**
 * Workflow-specific Adapter Target Descriptor fields.
 */
export type WorkflowTargetSpec = {
  entrypoint: WorkflowEntrypointConfig;
  /**
   * JSON Schema for `FabricConfig.workflow.settings`.
   */
  settings_schema?: JsonObject | null;
} & JsonObject;
/**
 * Entry point projected southbound to the adapter.
 */
export type WorkflowEntrypointConfig = {
  /**
   * Adapter-defined entry-point resolution semantics.
   */
  kind: string;
  /**
   * Adapter-defined workflow reference.
   */
  ref: string;
} & JsonObject;

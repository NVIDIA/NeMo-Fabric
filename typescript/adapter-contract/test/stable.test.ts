// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ADAPTER_CONTRACT_VERSION } from "../src/index.js";
import type {
  AdapterConfigInput,
  AdapterContractVersion,
  AdapterDescriptor,
  AdapterKind,
  AdapterTelemetryProviderSupport,
  AgentConfig,
  ControlLocation,
  EnvironmentOwnership,
  InstructionMode,
  JsonValue,
  RuntimeContext,
  TelemetryProvider,
} from "../src/index.js";
// @ts-expect-error invocation types are available only from the preview entry point
import type { AgentRunResult } from "../src/index.js";

const descriptor: AdapterDescriptor = {
  adapter_id: "pi",
  adapter_kind: "process",
  contract_version: ADAPTER_CONTRACT_VERSION,
  custom_extension: { enabled: true },
  extension_schemas: {
    agent_config: { type: "object", additionalProperties: true },
  },
  harness: "pi",
};

const config: AgentConfig = {
  extensions: { enabled: true, nested: [1, null, "value"] },
  models: {
    default: {
      model: "example-model",
      provider: "example-provider",
      temperature: null,
    },
  },
  tools: { enabled: null },
};

const context: RuntimeContext = {
  artifacts: { artifacts: [], root: null },
  environment: {
    control_location: "external_control",
    environment_id: "env-1",
    ownership: "caller_owned",
    provider: "local",
  },
  invocation_id: "invocation-1",
  request_id: "request-1",
  runtime_id: "runtime-1",
};

const jsonValues: JsonValue[] = [
  null,
  true,
  1,
  "text",
  ["nested"],
  { nested: [false, null] },
];

const supportTypes: [
  AdapterConfigInput,
  AdapterContractVersion,
  AdapterKind,
  ControlLocation,
  EnvironmentOwnership,
  InstructionMode,
  TelemetryProvider,
  AdapterTelemetryProviderSupport,
] = [
  "agent_config",
  ADAPTER_CONTRACT_VERSION,
  "process",
  "external_control",
  "caller_owned",
  "replace",
  "relay",
  { integration_modes: ["native"] },
];

void descriptor;
void config;
void context;
void jsonValues;
void supportTypes;

const wrongVersion: AdapterDescriptor = {
  adapter_id: "pi",
  adapter_kind: "process",
  // @ts-expect-error contract_version is the exact negotiated literal
  contract_version: "fabric.adapter/v1alpha1",
  harness: "pi",
};
void wrongVersion;

const wrongExtensionPoint: AdapterDescriptor = {
  adapter_id: "pi",
  adapter_kind: "process",
  contract_version: ADAPTER_CONTRACT_VERSION,
  // @ts-expect-error extension_schemas accepts only canonical extension points
  extension_schemas: { unknown_location: {} },
  harness: "pi",
};
void wrongExtensionPoint;

const wrongTelemetryProvider: AdapterDescriptor = {
  adapter_id: "pi",
  adapter_kind: "process",
  contract_version: ADAPTER_CONTRACT_VERSION,
  harness: "pi",
  telemetry: {
    providers: {
      // @ts-expect-error telemetry provider keys are the Rust enum values
      custom: {},
    },
  },
};
void wrongTelemetryProvider;

// @ts-expect-error required descriptor fields cannot be omitted
const incompleteDescriptor: AdapterDescriptor = { adapter_id: "pi" };
void incompleteDescriptor;

const invalidFlattenedExtension: AdapterDescriptor = {
  adapter_id: "pi",
  adapter_kind: "process",
  contract_version: ADAPTER_CONTRACT_VERSION,
  // @ts-expect-error flattened descriptor extensions must be JSON-compatible
  custom_hook: () => "not JSON",
  harness: "pi",
};
void invalidFlattenedExtension;

const invalidClosedConfig: AgentConfig = {
  models: {
    default: {
      model: "example-model",
      provider: "example-provider",
      // @ts-expect-error closed contract objects reject unknown keys
      unexpected: true,
    },
  },
};
void invalidClosedConfig;

const invalidTelemetryProvider: AdapterDescriptor = {
  adapter_id: "pi",
  adapter_kind: "process",
  contract_version: ADAPTER_CONTRACT_VERSION,
  harness: "pi",
  telemetry: {
    providers: {
      // @ts-expect-error telemetry provider names come from the canonical schema
      custom: {},
    },
  },
};
void invalidTelemetryProvider;

// @ts-expect-error undefined is not JSON
const undefinedJson: JsonValue = undefined;
void undefinedJson;

// @ts-expect-error bigint is not JSON
const bigintJson: JsonValue = 1n;
void bigintJson;

// @ts-expect-error functions are not JSON
const functionJson: JsonValue = () => "not JSON";
void functionJson;

// @ts-expect-error Date instances are not JSON objects
const dateJson: JsonValue = new Date();
void dateJson;

// @ts-expect-error Map instances are not JSON objects
const mapJson: JsonValue = new Map<string, JsonValue>();
void mapJson;

// @ts-expect-error symbols are not JSON
const symbolJson: JsonValue = Symbol("not JSON");
void symbolJson;

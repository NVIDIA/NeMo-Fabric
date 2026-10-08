// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ADAPTER_CONTRACT_VERSION } from "../src/index.js";
import type {
  AdapterContractVersion,
  AdapterDescriptor,
  AdapterKind,
  AdapterTargetDescriptor,
  AdapterTelemetryProviderSupport,
  AgentConfig,
  AgentUsage,
  ControlLocation,
  EnvironmentOwnership,
  InstructionMode,
  JsonValue,
  McpAuthenticationConfig,
  OAuthTokenEndpointAuthMethod,
  RuntimeContext,
  TelemetryProvider,
} from "../src/index.js";

const descriptor: AdapterDescriptor = {
  adapter_id: "pi",
  adapter_kind: "process",
  contract_version: ADAPTER_CONTRACT_VERSION,
  custom_extension: { enabled: true },
  extension_schemas: {
    agent_config: { type: "object", additionalProperties: true },
  },
};

const usage: AgentUsage = {
  input_tokens: 20,
  cached_input_tokens: 5,
  input_tokens_include_cache: true,
};
const invalidCacheSemantics: AgentUsage = {
  // @ts-expect-error cache semantics require an explicit boolean, not a numeric flag
  input_tokens_include_cache: 1,
};
void usage;
void invalidCacheSemantics;

const target: AdapterTargetDescriptor = {
  adapter_id: "pi",
  contract_version: ADAPTER_CONTRACT_VERSION,
  id: "pi.workflow",
  spec: { entrypoint: { kind: "factory", ref: "fabric.agent.react" } },
  type: "workflow",
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
  mcp: {
    servers: {
      protected: {
        authentication: {
          client_id: "fabric-client",
          scopes: ["tools:read"],
          token_endpoint_auth_method: "client_secret_basic",
          type: "oauth2",
        },
        custom_headers: { "X-Fabric-Client": "typescript" },
        transport: "streamable_http",
        url: "https://mcp.example.com",
      },
    },
  },
  tools: { enabled: null },
};

const serviceAccountAuthentication: McpAuthenticationConfig = {
  client_id: "fabric-service",
  client_secret_env: "FABRIC_CLIENT_SECRET",
  token_endpoint_auth_method: "client_secret_post",
  token_url: "https://auth.example.com/token",
  type: "service_account",
};

const tokenEndpointAuthMethod: OAuthTokenEndpointAuthMethod = "none";

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
  AdapterContractVersion,
  AdapterKind,
  ControlLocation,
  EnvironmentOwnership,
  InstructionMode,
  TelemetryProvider,
  AdapterTelemetryProviderSupport,
] = [
  ADAPTER_CONTRACT_VERSION,
  "process",
  "external_control",
  "caller_owned",
  "replace",
  "relay",
  { integration_modes: ["native"] },
];

void descriptor;
void target;
void config;
void serviceAccountAuthentication;
void tokenEndpointAuthMethod;
void context;
void jsonValues;
void supportTypes;

const wrongVersion: AdapterDescriptor = {
  adapter_id: "pi",
  adapter_kind: "process",
  // @ts-expect-error contract_version is the exact negotiated literal
  contract_version: "fabric.adapter/v1alpha1",
};
void wrongVersion;

const wrongExtensionPoint: AdapterDescriptor = {
  adapter_id: "pi",
  adapter_kind: "process",
  contract_version: ADAPTER_CONTRACT_VERSION,
  // @ts-expect-error extension_schemas accepts only canonical extension points
  extension_schemas: { unknown_location: {} },
};
void wrongExtensionPoint;

const wrongTelemetryProvider: AdapterDescriptor = {
  adapter_id: "pi",
  adapter_kind: "process",
  contract_version: ADAPTER_CONTRACT_VERSION,
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

// @ts-expect-error service-account authentication requires client credentials and a token URL
const incompleteServiceAccountAuthentication: McpAuthenticationConfig = {
  type: "service_account",
};
void incompleteServiceAccountAuthentication;

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

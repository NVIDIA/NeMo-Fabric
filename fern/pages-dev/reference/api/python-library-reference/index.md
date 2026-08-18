---
title: "Python SDK Reference"
slug: "/reference/api/python-library-reference"
description: "Complete reference for the public NeMo Fabric Python SDK."
---
{/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 */}

# API Overview

## Modules

- [`nemo_fabric.client`](./nemo_fabric.client.md#module-nemo_fabricclient): Native Python client for resolving and running NVIDIA NeMo Fabric agents.
- [`nemo_fabric.runtime`](./nemo_fabric.runtime.md#module-nemo_fabricruntime): Runtime lifecycle support for the NVIDIA NeMo Fabric Python SDK.
- [`nemo_fabric.streaming`](./nemo_fabric.streaming.md#module-nemo_fabricstreaming): NVIDIA NeMo Relay streaming support for the NVIDIA NeMo Fabric Python SDK.
- [`nemo_fabric.openai_streaming`](./nemo_fabric.openai_streaming.md#module-nemo_fabricopenai_streaming): Adapter-native OpenAI streaming for the NVIDIA NeMo Fabric Python SDK.
- [`nemo_fabric.models`](./nemo_fabric.models.md#module-nemo_fabricmodels): Pydantic SDK models for NVIDIA NeMo Fabric configuration and requests.
- [`nemo_fabric.types`](./nemo_fabric.types.md#module-nemo_fabrictypes): Public data contracts for the NeMo Fabric Python SDK.
- [`nemo_fabric.errors`](./nemo_fabric.errors.md#module-nemo_fabricerrors): Public exception hierarchy for the NeMo Fabric Python SDK.

## Classes

- [`client.Fabric`](./nemo_fabric.client.md#class-fabric): Primary Python entrypoint for NeMo Fabric.
- [`runtime.Runtime`](./nemo_fabric.runtime.md#class-runtime): One logical, stateful harness execution.
- [`runtime.RuntimeStatus`](./nemo_fabric.runtime.md#class-runtimestatus): Lifecycle state of a runtime.
- [`streaming.InvokeStream`](./nemo_fabric.streaming.md#class-invokestream): Async iterator of raw ATOF records for one runtime invocation.
- [`openai_streaming.OpenAIInvokeStream`](./nemo_fabric.openai_streaming.md#class-openaiinvokestream): Async iterator of OpenAI chat-completion chunks for one invocation.
- [`models.DiscoveryConfig`](./nemo_fabric.models.md#class-discoveryconfig): Explicit local descriptor discovery paths.
- [`models.EnvironmentConfig`](./nemo_fabric.models.md#class-environmentconfig): Execution environment configuration supplied by the consumer.
- [`models.FabricBaseModel`](./nemo_fabric.models.md#class-fabricbasemodel): Base class for SDK-facing Pydantic models.
- [`models.FabricConfig`](./nemo_fabric.models.md#class-fabricconfig): SDK-facing typed NeMo Fabric agent configuration.
- [`models.HarnessConfig`](./nemo_fabric.models.md#class-harnessconfig): Harness adapter selection plus adapter-owned settings.
- [`models.InstructionConfig`](./nemo_fabric.models.md#class-instructionconfig): One portable instruction value.
- [`models.InstructionsConfig`](./nemo_fabric.models.md#class-instructionsconfig): Harness-neutral agent instructions.
- [`models.McpAuthenticationConfig`](./nemo_fabric.models.md#class-mcpauthenticationconfig): MCP server authentication configuration.
- [`models.McpConfig`](./nemo_fabric.models.md#class-mcpconfig): MCP capability configuration.
- [`models.McpServerConfig`](./nemo_fabric.models.md#class-mcpserverconfig): MCP server configuration.
- [`models.MetadataConfig`](./nemo_fabric.models.md#class-metadataconfig): Human-readable agent identity.
- [`models.ModelConfig`](./nemo_fabric.models.md#class-modelconfig): Configuration for one model role.
- [`models.RelayAtifConfig`](./nemo_fabric.models.md#class-relayatifconfig): NeMo Relay ATIF export configuration.
- [`models.RelayAtofConfig`](./nemo_fabric.models.md#class-relayatofconfig): NeMo Relay ATOF export configuration.
- [`models.RelayAtofFileSinkConfig`](./nemo_fabric.models.md#class-relayatoffilesinkconfig): NeMo Relay ATOF file sink configuration.
- [`models.RelayAtofStreamSinkConfig`](./nemo_fabric.models.md#class-relayatofstreamsinkconfig): NeMo Relay ATOF stream sink configuration.
- [`models.RelayComponentConfig`](./nemo_fabric.models.md#class-relaycomponentconfig): Generic NeMo Relay plugin component configuration.
- [`models.RelayConfig`](./nemo_fabric.models.md#class-relayconfig): First-class NeMo Relay integration configuration.
- [`models.RelayConfigPolicy`](./nemo_fabric.models.md#class-relayconfigpolicy): NeMo Relay config validation policy.
- [`models.RelayHttpStorageConfig`](./nemo_fabric.models.md#class-relayhttpstorageconfig): NeMo Relay ATIF HTTP storage configuration.
- [`models.RelayObservabilityConfig`](./nemo_fabric.models.md#class-relayobservabilityconfig): NeMo Relay observability component configuration.
- [`models.RelayOpenTelemetryConfig`](./nemo_fabric.models.md#class-relayopentelemetryconfig): NeMo Relay typed OpenTelemetry destination configuration.
- [`models.RelayOpenTelemetryEndpointConfig`](./nemo_fabric.models.md#class-relayopentelemetryendpointconfig): One typed NeMo Relay OpenTelemetry destination.
- [`models.RelayS3StorageConfig`](./nemo_fabric.models.md#class-relays3storageconfig): NeMo Relay ATIF S3 storage configuration.
- [`models.RunRequest`](./nemo_fabric.models.md#class-runrequest): One validated NeMo Fabric invocation request.
- [`models.RuntimeConfig`](./nemo_fabric.models.md#class-runtimeconfig): Invocation runtime contract.
- [`models.SkillConfig`](./nemo_fabric.models.md#class-skillconfig): Skill capability configuration.
- [`models.TelemetryConfig`](./nemo_fabric.models.md#class-telemetryconfig): Telemetry configuration.
- [`models.TelemetryProviderConfig`](./nemo_fabric.models.md#class-telemetryproviderconfig): Provider-specific telemetry configuration.
- [`models.ToolDefinitionConfig`](./nemo_fabric.models.md#class-tooldefinitionconfig): One named normalized tool or tool-group definition.
- [`models.ToolsConfig`](./nemo_fabric.models.md#class-toolsconfig): Harness-neutral tool capability configuration.
- [`models.WorkflowConfig`](./nemo_fabric.models.md#class-workflowconfig): Registered workflow target and immutable construction settings.
- [`types.AdapterInfo`](./nemo_fabric.types.md#class-adapterinfo): Resolved adapter identity attached to a run plan.
- [`types.ArtifactManifest`](./nemo_fabric.types.md#class-artifactmanifest): Normalized collection of artifacts produced by a run.
- [`types.ArtifactRef`](./nemo_fabric.types.md#class-artifactref): Reference to one artifact produced by a run.
- [`types.DoctorCheck`](./nemo_fabric.types.md#class-doctorcheck): One diagnostic check in a ``DoctorReport``.
- [`types.DoctorReport`](./nemo_fabric.types.md#class-doctorreport): Aggregate preflight diagnostics for a resolved run plan.
- [`types.ErrorInfo`](./nemo_fabric.types.md#class-errorinfo): Structured failure returned inside a normalized ``RunResult``.
- [`types.FabricEvent`](./nemo_fabric.types.md#class-fabricevent): One normalized lifecycle or invocation event.
- [`types.RunOutput`](./nemo_fabric.types.md#class-runoutput): Normalized adapter output.
- [`types.RunPlan`](./nemo_fabric.types.md#class-runplan): Immutable execution plan produced before a runtime is started.
- [`types.RunResult`](./nemo_fabric.types.md#class-runresult): Normalized terminal result from one NeMo Fabric invocation.
- [`types.RunUsage`](./nemo_fabric.types.md#class-runusage): Normalized invocation usage reported by an adapter target.
- [`types.RuntimeCapabilities`](./nemo_fabric.types.md#class-runtimecapabilities): Operations declared by the resolved runtime and adapter.
- [`types.RuntimeHandle`](./nemo_fabric.types.md#class-runtimehandle): Opaque identity and binding for one started runtime.
- [`types.TelemetryRef`](./nemo_fabric.types.md#class-telemetryref): Reference to external or persisted telemetry for a run.
- [`errors.FabricCapabilityError`](./nemo_fabric.errors.md#class-fabriccapabilityerror): Operation rejected by resolved runtime capabilities or implementation status.
- [`errors.FabricConfigError`](./nemo_fabric.errors.md#class-fabricconfigerror): Invalid SDK input, request shape, factory, or resolved config.
- [`errors.FabricError`](./nemo_fabric.errors.md#class-fabricerror): Base class for structured SDK-level NeMo Fabric errors.
- [`errors.FabricNativeUnavailableError`](./nemo_fabric.errors.md#class-fabricnativeunavailableerror): SDK call requires the PyO3 extension, but it is not installed or importable.
- [`errors.FabricRuntimeError`](./nemo_fabric.errors.md#class-fabricruntimeerror): Failure while starting, invoking, stopping, or otherwise driving a runtime.
- [`errors.FabricStateError`](./nemo_fabric.errors.md#class-fabricstateerror): Operation rejected because a local runtime is in the wrong state.

## Functions

- No functions


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

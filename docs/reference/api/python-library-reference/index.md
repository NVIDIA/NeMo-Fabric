---
title: "Python SDK Reference"
slug: "/reference/api/python-library-reference"
description: "Complete reference for the public NeMo Fabric Python SDK."
---
{/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 */}

# API Overview

## Modules

- [`nemo_fabric.client`](./nemo_fabric.client.md#module-nemo_fabricclient): Native Python client for resolving and running NeMo Fabric agents.
- [`nemo_fabric.runtime`](./nemo_fabric.runtime.md#module-nemo_fabricruntime): Runtime lifecycle support for the Fabric Python SDK.
- [`nemo_fabric.models`](./nemo_fabric.models.md#module-nemo_fabricmodels): Pydantic SDK models for NeMo Fabric configuration and requests.
- [`nemo_fabric.types`](./nemo_fabric.types.md#module-nemo_fabrictypes): Public data contracts for the NeMo Fabric Python SDK.
- [`nemo_fabric.errors`](./nemo_fabric.errors.md#module-nemo_fabricerrors): Public exception hierarchy for the NeMo Fabric Python SDK.

## Classes

- [`client.Fabric`](./nemo_fabric.client.md#class-fabric): Primary Python entrypoint for NeMo Fabric.
- [`runtime.Runtime`](./nemo_fabric.runtime.md#class-runtime): One logical, stateful harness execution.
- [`runtime.RuntimeStatus`](./nemo_fabric.runtime.md#class-runtimestatus): Lifecycle state of a runtime.
- [`models.EnvironmentConfig`](./nemo_fabric.models.md#class-environmentconfig): Execution environment configuration supplied by the consumer.
- [`models.FabricBaseModel`](./nemo_fabric.models.md#class-fabricbasemodel): Base class for SDK-facing Pydantic models.
- [`models.FabricConfig`](./nemo_fabric.models.md#class-fabricconfig): SDK-facing typed Fabric agent configuration.
- [`models.HarnessConfig`](./nemo_fabric.models.md#class-harnessconfig): Harness adapter selection plus adapter-owned settings.
- [`models.McpConfig`](./nemo_fabric.models.md#class-mcpconfig): MCP capability configuration.
- [`models.McpServerConfig`](./nemo_fabric.models.md#class-mcpserverconfig): MCP server configuration.
- [`models.MetadataConfig`](./nemo_fabric.models.md#class-metadataconfig): Human-readable agent identity.
- [`models.ModelConfig`](./nemo_fabric.models.md#class-modelconfig): Model alias configuration.
- [`models.RelayAtifConfig`](./nemo_fabric.models.md#class-relayatifconfig): NeMo Relay ATIF export configuration.
- [`models.RelayAtofConfig`](./nemo_fabric.models.md#class-relayatofconfig): NeMo Relay ATOF export configuration.
- [`models.RelayAtofFileSinkConfig`](./nemo_fabric.models.md#class-relayatoffilesinkconfig): NeMo Relay ATOF file sink configuration.
- [`models.RelayAtofStreamSinkConfig`](./nemo_fabric.models.md#class-relayatofstreamsinkconfig): NeMo Relay ATOF stream sink configuration.
- [`models.RelayComponentConfig`](./nemo_fabric.models.md#class-relaycomponentconfig): Generic NeMo Relay plugin component configuration.
- [`models.RelayConfig`](./nemo_fabric.models.md#class-relayconfig): First-class NeMo Relay integration configuration.
- [`models.RelayConfigPolicy`](./nemo_fabric.models.md#class-relayconfigpolicy): NeMo Relay config validation policy.
- [`models.RelayHttpStorageConfig`](./nemo_fabric.models.md#class-relayhttpstorageconfig): NeMo Relay ATIF HTTP storage configuration.
- [`models.RelayObservabilityConfig`](./nemo_fabric.models.md#class-relayobservabilityconfig): NeMo Relay observability component configuration.
- [`models.RelayOtlpConfig`](./nemo_fabric.models.md#class-relayotlpconfig): NeMo Relay OTLP export configuration for OpenTelemetry/OpenInference.
- [`models.RelayS3StorageConfig`](./nemo_fabric.models.md#class-relays3storageconfig): NeMo Relay ATIF S3 storage configuration.
- [`models.RunRequest`](./nemo_fabric.models.md#class-runrequest): One validated Fabric invocation request.
- [`models.RuntimeConfig`](./nemo_fabric.models.md#class-runtimeconfig): Runtime input/output contract.
- [`models.SkillConfig`](./nemo_fabric.models.md#class-skillconfig): Skill capability configuration.
- [`models.TelemetryConfig`](./nemo_fabric.models.md#class-telemetryconfig): Telemetry configuration.
- [`models.TelemetryProviderConfig`](./nemo_fabric.models.md#class-telemetryproviderconfig): Provider-specific telemetry configuration.
- [`models.ToolsConfig`](./nemo_fabric.models.md#class-toolsconfig): Harness-neutral tool capability configuration.
- [`types.AdapterInfo`](./nemo_fabric.types.md#class-adapterinfo): Resolved adapter identity attached to a run plan.
- [`types.ArtifactManifest`](./nemo_fabric.types.md#class-artifactmanifest): Normalized collection of artifacts produced by a run.
- [`types.ArtifactRef`](./nemo_fabric.types.md#class-artifactref): Reference to one artifact produced by a run.
- [`types.DoctorCheck`](./nemo_fabric.types.md#class-doctorcheck): One diagnostic check in a ``DoctorReport``.
- [`types.DoctorReport`](./nemo_fabric.types.md#class-doctorreport): Aggregate preflight diagnostics for a resolved run plan.
- [`types.ErrorInfo`](./nemo_fabric.types.md#class-errorinfo): Structured failure returned inside a normalized ``RunResult``.
- [`types.FabricEvent`](./nemo_fabric.types.md#class-fabricevent): One normalized lifecycle or invocation event.
- [`types.RunOutput`](./nemo_fabric.types.md#class-runoutput): Normalized adapter output.
- [`types.RunPlan`](./nemo_fabric.types.md#class-runplan): Immutable execution plan produced before a runtime is started.
- [`types.RunResult`](./nemo_fabric.types.md#class-runresult): Normalized terminal result from one Fabric invocation.
- [`types.RuntimeCapabilities`](./nemo_fabric.types.md#class-runtimecapabilities): Operations declared by the resolved runtime and adapter.
- [`types.RuntimeHandle`](./nemo_fabric.types.md#class-runtimehandle): Opaque identity and binding for one started runtime.
- [`types.TelemetryRef`](./nemo_fabric.types.md#class-telemetryref): Reference to external or persisted telemetry for a run.
- [`errors.FabricCapabilityError`](./nemo_fabric.errors.md#class-fabriccapabilityerror): Operation rejected by resolved runtime capabilities or implementation status.
- [`errors.FabricConfigError`](./nemo_fabric.errors.md#class-fabricconfigerror): Invalid SDK input, request shape, factory, or resolved config.
- [`errors.FabricError`](./nemo_fabric.errors.md#class-fabricerror): Base class for structured SDK-level Fabric errors.
- [`errors.FabricNativeUnavailableError`](./nemo_fabric.errors.md#class-fabricnativeunavailableerror): SDK call requires the PyO3 extension, but it is not installed or importable.
- [`errors.FabricRuntimeError`](./nemo_fabric.errors.md#class-fabricruntimeerror): Failure while starting, invoking, stopping, or otherwise driving a runtime.
- [`errors.FabricStateError`](./nemo_fabric.errors.md#class-fabricstateerror): Operation rejected because a local runtime is in the wrong state.

## Functions

- No functions


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

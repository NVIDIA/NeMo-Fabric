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
- [`nemo_fabric.session`](./nemo_fabric.session.md#module-nemo_fabricsession): Session lifecycle support for the Fabric Python SDK.
- [`nemo_fabric.types`](./nemo_fabric.types.md#module-nemo_fabrictypes): Public data contracts for the NeMo Fabric Python SDK.
- [`nemo_fabric.errors`](./nemo_fabric.errors.md#module-nemo_fabricerrors): Public exception hierarchy for the NeMo Fabric Python SDK.

## Classes

- [`client.Fabric`](./nemo_fabric.client.md#class-fabric): Primary Python entrypoint for NeMo Fabric.
- [`session.Session`](./nemo_fabric.session.md#class-session): One ordered multi-turn conversation over a Fabric runtime.
- [`session.SessionStatus`](./nemo_fabric.session.md#class-sessionstatus): Lifecycle state of a session runtime.
- [`types.AdapterInfo`](./nemo_fabric.types.md#class-adapterinfo): Resolved adapter identity attached to a run plan.
- [`types.ArtifactManifest`](./nemo_fabric.types.md#class-artifactmanifest): Normalized collection of artifacts produced by a run.
- [`types.ArtifactRef`](./nemo_fabric.types.md#class-artifactref): Reference to one artifact produced by a run.
- [`types.DoctorCheck`](./nemo_fabric.types.md#class-doctorcheck): One diagnostic check in a ``DoctorReport``.
- [`types.DoctorReport`](./nemo_fabric.types.md#class-doctorreport): Aggregate preflight diagnostics for a resolved run plan.
- [`types.EffectiveConfig`](./nemo_fabric.types.md#class-effectiveconfig): Immutable result of config loading and ordered profile application.
- [`types.EnvironmentConfig`](./nemo_fabric.types.md#class-environmentconfig): Execution environment configuration.
- [`types.ErrorInfo`](./nemo_fabric.types.md#class-errorinfo): Structured failure returned inside a normalized ``RunResult``.
- [`types.FabricConfig`](./nemo_fabric.types.md#class-fabricconfig): Mutable typed representation of a Fabric agent configuration.
- [`types.FabricEvent`](./nemo_fabric.types.md#class-fabricevent): One normalized lifecycle or invocation event.
- [`types.HarnessConfig`](./nemo_fabric.types.md#class-harnessconfig): Harness adapter selection and adapter-owned settings.
- [`types.McpConfig`](./nemo_fabric.types.md#class-mcpconfig): MCP capability configuration with authoring helpers.
- [`types.MetadataConfig`](./nemo_fabric.types.md#class-metadataconfig): Agent identity and human-readable metadata.
- [`types.RunPlan`](./nemo_fabric.types.md#class-runplan): Immutable execution plan produced before a runtime is started.
- [`types.RunRequest`](./nemo_fabric.types.md#class-runrequest): One normalized invocation request.
- [`types.RunResult`](./nemo_fabric.types.md#class-runresult): Normalized terminal result from one Fabric invocation.
- [`types.RuntimeCapabilities`](./nemo_fabric.types.md#class-runtimecapabilities): Operations declared by the resolved runtime and adapter.
- [`types.RuntimeConfig`](./nemo_fabric.types.md#class-runtimeconfig): Runtime input/output contract.
- [`types.RuntimeHandle`](./nemo_fabric.types.md#class-runtimehandle): Opaque identity and binding for one started runtime.
- [`types.RuntimeUpdate`](./nemo_fabric.types.md#class-runtimeupdate): Capability-gated update requested for a running session.
- [`types.RuntimeUpdateResult`](./nemo_fabric.types.md#class-runtimeupdateresult): Normalized outcome of a runtime update request.
- [`types.SessionHandle`](./nemo_fabric.types.md#class-sessionhandle): Caller-facing identity and lifecycle snapshot for one session.
- [`types.SkillConfig`](./nemo_fabric.types.md#class-skillconfig): Skill capability configuration.
- [`types.TelemetryConfig`](./nemo_fabric.types.md#class-telemetryconfig): Telemetry configuration with authoring helpers.
- [`types.TelemetryRef`](./nemo_fabric.types.md#class-telemetryref): Reference to external or persisted telemetry for a run.
- [`errors.FabricCapabilityError`](./nemo_fabric.errors.md#class-fabriccapabilityerror): Operation rejected by resolved runtime capabilities or implementation status.
- [`errors.FabricConfigError`](./nemo_fabric.errors.md#class-fabricconfigerror): Invalid SDK input, request shape, profile stack, or resolved config.
- [`errors.FabricError`](./nemo_fabric.errors.md#class-fabricerror): Base class for structured SDK-level Fabric errors.
- [`errors.FabricNativeUnavailableError`](./nemo_fabric.errors.md#class-fabricnativeunavailableerror): SDK call requires the PyO3 extension, but it is not installed or importable.
- [`errors.FabricRuntimeError`](./nemo_fabric.errors.md#class-fabricruntimeerror): Failure while starting, invoking, stopping, or otherwise driving a runtime.
- [`errors.FabricStateError`](./nemo_fabric.errors.md#class-fabricstateerror): Operation rejected because a local session handle is in the wrong state.

## Functions

- No functions


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

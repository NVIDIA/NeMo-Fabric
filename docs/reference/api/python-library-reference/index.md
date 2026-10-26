---
title: "Python SDK Reference"
slug: "/reference/api/python-library-reference"
description: "Complete reference for the public NeMo Fabric Python SDK."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# API Overview

## Modules

- [`nemo_fabric.accounting`](./nemo_fabric.accounting.md#module-nemo_fabricaccounting): Typed model accounting and invocation summaries with explicit coverage.
- [`nemo_fabric.reporting`](./nemo_fabric.reporting.md#module-nemo_fabricreporting): Inspect saved run journals and legacy results without importing traces.
- [`nemo_fabric.comparisons`](./nemo_fabric.comparisons.md#module-nemo_fabriccomparisons): Compare caller-defined task attempts using saved NeMo Fabric invocation evidence.
- [`nemo_fabric.client`](./nemo_fabric.client.md#module-nemo_fabricclient): Native Python client for resolving and running NVIDIA NeMo Fabric agents.
- [`nemo_fabric.capabilities`](./nemo_fabric.capabilities.md#module-nemo_fabriccapabilities): Metadata-only adapter inspection for host-side admission.
- [`nemo_fabric.runtime`](./nemo_fabric.runtime.md#module-nemo_fabricruntime): Runtime lifecycle support for the NVIDIA NeMo Fabric Python SDK.
- [`nemo_fabric.service`](./nemo_fabric.service.md#module-nemo_fabricservice): Long-lived service lifecycle support for the NeMo Fabric Python SDK.
- [`nemo_fabric.streaming`](./nemo_fabric.streaming.md#module-nemo_fabricstreaming): NVIDIA NeMo Relay streaming support for the NVIDIA NeMo Fabric Python SDK.
- [`nemo_fabric.openai_streaming`](./nemo_fabric.openai_streaming.md#module-nemo_fabricopenai_streaming): Adapter-native OpenAI streaming for the NVIDIA NeMo Fabric Python SDK.
- [`nemo_fabric.models`](./nemo_fabric.models.md#module-nemo_fabricmodels): Pydantic SDK models for NVIDIA NeMo Fabric configuration and requests.
- [`nemo_fabric.types`](./nemo_fabric.types.md#module-nemo_fabrictypes): Public data contracts for the NeMo Fabric Python SDK.
- [`nemo_fabric.errors`](./nemo_fabric.errors.md#module-nemo_fabricerrors): Public exception hierarchy for the NeMo Fabric Python SDK.

## Classes

- [`accounting.DimensionSummary`](./nemo_fabric.accounting.md#class-dimensionsummary): An aggregate together with its explicit invocation coverage.
- [`accounting.UsageSummary`](./nemo_fabric.accounting.md#class-usagesummary): Coverage of an explicit invocation manifest, including missing results.
- [`client.Fabric`](./nemo_fabric.client.md#class-fabric): Primary Python entrypoint for NeMo Fabric.
- [`capabilities.AdapterCapabilityProfile`](./nemo_fabric.capabilities.md#class-adaptercapabilityprofile): Declared support for a standalone runtime, not observed execution provenance.
- [`runtime.Runtime`](./nemo_fabric.runtime.md#class-runtime): One logical, stateful harness execution.
- [`runtime.RuntimeStatus`](./nemo_fabric.runtime.md#class-runtimestatus): Lifecycle state of a runtime.
- [`service.Service`](./nemo_fabric.service.md#class-service): One prepared or attached long-lived adapter service.
- [`service.ServiceStatus`](./nemo_fabric.service.md#class-servicestatus): Lifecycle state of a prepared or attached service.
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
- [`models.PricingConfig`](./nemo_fabric.models.md#class-pricingconfig): Frozen Relay-compatible price-catalog override, read during planning.
- [`models.RecordConfig`](./nemo_fabric.models.md#class-recordconfig): Opt-in local request journal.
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
- [`models.SystemStatsConfig`](./nemo_fabric.models.md#class-systemstatsconfig): Native invocation stats selection, independent of Relay and recording.
- [`models.SystemStatsSamplingConfig`](./nemo_fabric.models.md#class-systemstatssamplingconfig): Opt-in periodic collection; bounds include both endpoint samples.
- [`models.TelemetryConfig`](./nemo_fabric.models.md#class-telemetryconfig): Telemetry configuration.
- [`models.TelemetryProviderConfig`](./nemo_fabric.models.md#class-telemetryproviderconfig): Provider-specific telemetry configuration.
- [`models.ToolDefinitionConfig`](./nemo_fabric.models.md#class-tooldefinitionconfig): One named normalized tool or tool-group definition.
- [`models.ToolsConfig`](./nemo_fabric.models.md#class-toolsconfig): Harness-neutral tool capability configuration.
- [`models.WorkflowConfig`](./nemo_fabric.models.md#class-workflowconfig): Registered workflow target and immutable construction settings.
- [`types.AdapterInfo`](./nemo_fabric.types.md#class-adapterinfo): Resolved adapter identity attached to a run plan.
- [`types.AppliedPricing`](./nemo_fabric.types.md#class-appliedpricing): Selected price entry and integer nano-USD rates per million tokens.
- [`types.ArtifactManifest`](./nemo_fabric.types.md#class-artifactmanifest): Normalized collection of artifacts produced by a run.
- [`types.ArtifactRef`](./nemo_fabric.types.md#class-artifactref): Reference to one artifact produced by a run.
- [`types.CgroupObservation`](./nemo_fabric.types.md#class-cgroupobservation): Native cgroup snapshot or explicit unavailable/unsupported reason.
- [`types.CgroupSnapshot`](./nemo_fabric.types.md#class-cgroupsnapshot): Native adapter-host cgroup measurements, separate from process RSS.
- [`types.CgroupStats`](./nemo_fabric.types.md#class-cgroupstats): Cgroup endpoint observations and counter deltas over an invocation.
- [`types.CostAmount`](./nemo_fabric.types.md#class-costamount): Complete and observed amounts in integer nano-USD, never an invoice guarantee.
- [`types.DoctorCheck`](./nemo_fabric.types.md#class-doctorcheck): One diagnostic check in a ``DoctorReport``.
- [`types.DoctorReport`](./nemo_fabric.types.md#class-doctorreport): Aggregate preflight diagnostics for a resolved run plan.
- [`types.EnvironmentHandle`](./nemo_fabric.types.md#class-environmenthandle): Durable identity and provider binding for a prepared or attached environment.
- [`types.EnvironmentReference`](./nemo_fabric.types.md#class-environmentreference): Provider-specific identity for an existing execution environment.
- [`types.ErrorInfo`](./nemo_fabric.types.md#class-errorinfo): Structured failure returned inside a normalized ``RunResult``.
- [`types.FabricEvent`](./nemo_fabric.types.md#class-fabricevent): One normalized lifecycle or invocation event.
- [`types.GaugeSummary`](./nemo_fabric.types.md#class-gaugesummary): Sampled gauge extrema, hold-last-value mean, and covered time.
- [`types.InvocationSystemStats`](./nemo_fabric.types.md#class-invocationsystemstats): System measurements for one invocation in its executing runner.
- [`types.Measurement`](./nemo_fabric.types.md#class-measurement): Measured numeric value, an unlimited limit, or explicit missing evidence.
- [`types.ModelUsage`](./nemo_fabric.types.md#class-modelusage): Complete normalized tokens, observations, gaps and separately typed costs.
- [`types.PerModelUsage`](./nemo_fabric.types.md#class-permodelusage): Normalized counts for a source-established accounting model/provider.
- [`types.PricingBasis`](./nemo_fabric.types.md#class-pricingbasis): Validated comparison basis for persisted catalog estimates.
- [`types.ResourceObservation`](./nemo_fabric.types.md#class-resourceobservation): Adapter host process-tree reading, or an explicit collection failure.
- [`types.RunOutput`](./nemo_fabric.types.md#class-runoutput): Normalized adapter output.
- [`types.RunPlan`](./nemo_fabric.types.md#class-runplan): Immutable execution plan produced before a runtime is started.
- [`types.RunResult`](./nemo_fabric.types.md#class-runresult): Normalized terminal result from one NeMo Fabric invocation.
- [`types.RunUsage`](./nemo_fabric.types.md#class-runusage): Normalized invocation-local usage reported by an adapter target.
- [`types.RuntimeCapabilities`](./nemo_fabric.types.md#class-runtimecapabilities): Operations declared by the resolved runtime and adapter.
- [`types.RuntimeHandle`](./nemo_fabric.types.md#class-runtimehandle): Opaque identity and binding for one started runtime.
- [`types.RuntimeResourceSnapshot`](./nemo_fabric.types.md#class-runtimeresourcesnapshot): Timestamped resource usage of one runtime's local adapter host process tree.
- [`types.SampleSummary`](./nemo_fabric.types.md#class-samplesummary): Retained native series, reductions, and collection-quality diagnostics.
- [`types.ServiceHandle`](./nemo_fabric.types.md#class-servicehandle): Opaque identity and binding for one prepared or attached service.
- [`types.ServiceReference`](./nemo_fabric.types.md#class-servicereference): Reference to an already-running caller-owned service.
- [`types.TelemetryRef`](./nemo_fabric.types.md#class-telemetryref): Reference to external or persisted telemetry for a run.
- [`types.TokenCost`](./nemo_fabric.types.md#class-tokencost): Provider-reported amounts and separate catalog estimates with their basis.
- [`types.TokenCounts`](./nemo_fabric.types.md#class-tokencounts): Independently nullable tokens in one fixed invocation/model scope.
- [`types.UsageGap`](./nemo_fabric.types.md#class-usagegap): Dimension-specific incompleteness with an optional stable diagnostic.
- [`errors.FabricCapabilityError`](./nemo_fabric.errors.md#class-fabriccapabilityerror): Operation rejected by resolved runtime capabilities or implementation status.
- [`errors.FabricConfigError`](./nemo_fabric.errors.md#class-fabricconfigerror): Invalid SDK input, request shape, factory, or resolved config.
- [`errors.FabricError`](./nemo_fabric.errors.md#class-fabricerror): Base class for structured SDK-level NeMo Fabric errors.
- [`errors.FabricNativeUnavailableError`](./nemo_fabric.errors.md#class-fabricnativeunavailableerror): SDK call requires the PyO3 extension, but it is not installed or importable.
- [`errors.FabricRuntimeError`](./nemo_fabric.errors.md#class-fabricruntimeerror): Failure while starting, invoking, stopping, or otherwise driving a runtime.
- [`errors.FabricStateError`](./nemo_fabric.errors.md#class-fabricstateerror): Operation rejected because a local runtime is in the wrong state.
- [`errors.ResourceObservationUnavailableError`](./nemo_fabric.errors.md#class-resourceobservationunavailableerror): Runtime resources cannot be read now.
- [`errors.ResourceObservationUnsupportedError`](./nemo_fabric.errors.md#class-resourceobservationunsupportederror): Runtime resources have no reader.

## Functions

- [`accounting.summarize_usage`](./nemo_fabric.accounting.md#function-summarize_usage): Summarize invocations against an explicit expected identity manifest.
- [`reporting.main`](./nemo_fabric.reporting.md#function-main): Inspect a saved run journal or legacy result as readable text or JSON.
- [`reporting.read_evidence`](./nemo_fabric.reporting.md#function-read_evidence): Read a run journal or legacy result without opening referenced artifacts.
- [`reporting.read_invocation_record`](./nemo_fabric.reporting.md#function-read_invocation_record): Read legacy start/end evidence; a truncated tail never creates a result.
- [`reporting.read_result`](./nemo_fabric.reporting.md#function-read_result): Read one saved full result without requiring task labels or a manifest.
- [`reporting.render_invocation_record_summary`](./nemo_fabric.reporting.md#function-render_invocation_record_summary): Render legacy invocation evidence without claiming one-shot completion.
- [`reporting.render_result_summary`](./nemo_fabric.reporting.md#function-render_result_summary): Render concise measurement values; details add retained history and sources.
- [`reporting.summarize_result`](./nemo_fabric.reporting.md#function-summarize_result): Describe one returned invocation, preserving independent usable evidence.
- [`comparisons.main`](./nemo_fabric.comparisons.md#function-main): Print an explicit comparison manifest as a table or JSON without running agents.
- [`comparisons.read_comparison_manifest`](./nemo_fabric.comparisons.md#function-read_comparison_manifest): Read a bounded manifest; evidence paths are relative to its explicit root.
- [`comparisons.render_comparison_summary`](./nemo_fabric.comparisons.md#function-render_comparison_summary): Render per-metric coverage; details retain identities and pricing evidence.
- [`comparisons.summarize_attempts`](./nemo_fabric.comparisons.md#function-summarize_attempts): Summarize an explicit inventory of scheduled custom-task attempts.
- [`capabilities.inspect_adapter`](./nemo_fabric.capabilities.md#function-inspect_adapter): Inspect a standalone runtime using descriptor metadata without loading its SDK.


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

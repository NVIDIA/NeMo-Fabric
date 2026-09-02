<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA-labs Object Oriented Agents (NOOA) Adapters for NVIDIA NeMo Fabric

This directory provides two ways to run NOOA agents through the NeMo Fabric
lifecycle contract. Choose the integration that matches the agent interface:

| Integration | Use It For |
| --- | --- |
| [InteractiveAgent adapter](docs/interactive-agent.md) | Registered `InteractiveAgent` implementations, including `CodingAgent` and `ArcSolverBase` targets |
| [BenchAgent adapter](docs/bench-agent.md) | `nooa_bench.BenchAgent` tasks and Harbor evaluations |

The shared `nvidia.fabric.nooa` adapter owns the common NOOA queue dispatcher.
It also maps normalized skill paths and whole MCP servers into compatible
registered targets.
The dedicated `nvidia.fabric.nooa.bench-agent` adapter maps the benchmark-native
`BenchAgent` task contract directly into a NeMo Fabric invocation.

## Install from Source

This directory has no package metadata. Use one Python environment that
contains the following components:

- NeMo Fabric, `nemo-fabric-adapter-contract`, and
  `nemo-fabric-adapters-common`.
- NOOA core and the package that provides the selected agent.
- The source adapter in this directory.

NOOA requires Python 3.12 or 3.13. Expose the adapter source from the NVIDIA
NeMo Fabric repository root:

```bash
export PYTHONPATH="$PWD/external/nooa/src${PYTHONPATH:+:$PYTHONPATH}"
```

During source development, include `external/nooa` and the selected target
descriptor's directory in `FabricConfig.discovery.local_paths`. The BenchAgent
Harbor guide instead packages the harness descriptor in the uploaded NeMo Fabric
configuration bundle.

## Configure Relay

Relay is optional. Install NOOA core and CLI without their optional Relay
extra, then install the compatible Relay Python package in the adapter
environment:

```bash
pip install "nemo-relay>=0.7.2,<0.8"
```

Both adapters declare Relay outputs for Agent Trajectory Interchange Format
(ATIF), OpenTelemetry, and OpenInference. NeMo Fabric supplies the generated
`FABRIC_RELAY_CONFIG_PATH`; the adapters reject ambient user or project plugin
configuration and activate only the generated document.

A Relay setup failure prevents agent execution and returns a failed result. If
artifact finalization fails after the agent completes, the adapter preserves
the functional result and marks telemetry as degraded. A leaked Relay scope
quarantines telemetry on later invocations.

Relay-backed `Runtime.invoke_stream()` returns raw Agent Trajectory
Observability Format (ATOF) records and a separate terminal result. Neither
adapter claims native model-response streaming.

## Choose an Environment

These adapters do not provide a sandbox. Agents run model-generated code and
shell commands with the permissions of their NeMo Fabric environment. Select an
environment provider with the isolation required by your workload.

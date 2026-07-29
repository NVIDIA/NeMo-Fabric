<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Hermes Agent Adapter

This adapter runs Hermes Agent through its Python SDK.

## Install

Hermes Agent and this adapter require Python 3.11 through 3.13. The following
table shows which components each installation provides:

| Installation | Runtime | Adapter | Harness | NeMo Relay Python Package |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[hermes-agent]"` | Yes | Yes | Yes | No |
| `pip install "nemo-fabric[hermes-agent,relay]"` | Yes | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-hermes[harness]"` | No | Yes | Yes | No |
| `pip install "nemo-fabric-adapters-hermes[full]"` | No | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-hermes[relay]"` | No | Yes | No | Yes |
| `pip install nemo-fabric-adapters-hermes` | No | Yes | No | No |

For an environment-managed harness, use `hermes-agent>=0.17.0`. For split
runtime and adapter environments, configure `ADAPTER_PYTHON` or
`harness.settings.python` and use matching NeMo Fabric release versions. Refer
to the [installation guide](https://nvidia-nemo-fabric.docs.buildwithfern.com/nemo/fabric/getting-started/install#install-an-adapter-and-harness-without-the-runtime).

Relay is optional for ordinary runs. Relay telemetry and
`Runtime.invoke_stream()` require one of the installations in the table that
includes the NeMo Relay Python package.

## What It Maps

The adapter receives a normalized payload from NeMo Fabric and materializes a native Hermes Agent configuration for:

- selected model provider, model name, base URL, and temperature through
  `models`;
- `instructions.system` and `runtime.max_turns`;
- workspace and explicit environment variables through `environment`;
- invocation timeout through `runtime.timeout_seconds`;
- NeMo Fabric skills as external skill directories for Hermes Agent;
- NeMo Fabric MCP servers as Hermes Agent MCP server config;
- `tools.enabled` and `tools.blocked` as Hermes-native toolset selection and
  blocking policy;
- optional NeMo Relay telemetry plugin configuration.

Tool selectors are Hermes toolset names because that is the native policy
surface Hermes exposes. Keep Hermes-specific controls such as
terminal timeout, reasoning configuration, and plugin configuration in
`harness.settings`. The adapter derives Hermes state from the NeMo Fabric
artifact root and creates a child under `runtimes/<runtime_id>`, so invocations
in one NeMo Fabric runtime share state without sharing config or the session
database with another runtime.

## Execution Model

Each NeMo Fabric runtime starts one local adapter host, constructs one Hermes Agent
`AIAgent`, and opens one `SessionDB`. Ordered `Runtime.invoke(...)` calls reuse
those native objects and pass the prior turn's returned transcript back to
`run_conversation(...)`. Runtime stop calls the agent's idempotent `close()`
method, closes the session database, and releases the Relay plugin context when
enabled.

Hermes Agent Relay telemetry is finalized after each NeMo Fabric invocation so its ATOF
and ATIF artifacts are complete when that invocation returns. This telemetry
boundary does not recreate the `AIAgent` or `SessionDB`.

## Maintaining The Adapter

Keep `fabric-adapter.json` aligned with the Python implementation:

- `contract_version` must match the adapter contract supported by NeMo Fabric core.
- `adapter_id` is the stable id selected by `harness.adapter_id`.
- `adapter_kind` is `python` because NeMo Fabric can invoke it through Python.
- `runner.module` names the persistent host module that NeMo Fabric invokes with
  `python -m`.
- `requirements` supplies dependency checks to NeMo Fabric diagnostics; keep
  required env vars, binaries, or packages current.
- `config.accepts` must match the NeMo Fabric sections this adapter maps into Hermes Agent.
- `telemetry.providers` declares provider-specific outputs and integration modes
  the adapter can produce or forward.

Do not put end-user agent settings in this directory. Users vary harness,
model, skills, MCP, tools, telemetry, and runtime behavior through complete
typed `FabricConfig` values and ordinary Python composition. The adapter
descriptor describes adapter capabilities; it is not an agent configuration.
Add descriptor fields only when NeMo Fabric core or the SDK actually uses them.

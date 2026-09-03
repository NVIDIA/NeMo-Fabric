<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Hermes Agent Adapter

This adapter runs Hermes Agent through its Python SDK.

## Install

Hermes Agent and this adapter require Python 3.11 through 3.13. Hermes Agent
0.20 and later is not installable from PyPI. Install Hermes Agent by following
the [Hermes Agent installation guide](https://hermes-agent.nousresearch.com/docs/installation),
then install the NeMo Fabric packages into the Python environment that runs
Hermes Agent.

The following table shows which NeMo Fabric components each package expression
provides. None of these expressions installs Hermes Agent:

| Installation | Runtime | Adapter | Harness | NeMo Relay Python Package |
| --- | --- | --- | --- | --- |
| `pip install nemo-fabric nemo-fabric-adapters-hermes` | Yes | Yes | No | No |
| `pip install "nemo-fabric[relay]" nemo-fabric-adapters-hermes` | Yes | Yes | No | Yes |
| `pip install "nemo-fabric-adapters-hermes[full]"` | No | Yes | No | Yes |
| `pip install "nemo-fabric-adapters-hermes[relay]"` | No | Yes | No | Yes |
| `pip install nemo-fabric-adapters-hermes` | No | Yes | No | No |

For local development from this repository, check out and install the pinned
Hermes Agent source into the project environment:

```bash
just install-hermes-agent
```

For split runtime and adapter environments, configure `ADAPTER_PYTHON` and use
matching NeMo Fabric release versions. Refer to the
[installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install#install-an-adapter-and-harness-without-the-runtime).

Relay is optional for ordinary runs. Relay telemetry and
`Runtime.invoke_stream()` require one of the installations in the table that
includes the NeMo Relay Python package.

## What It Maps

The adapter receives a normalized payload from NeMo Fabric and materializes a native Hermes Agent configuration for:

- selected model provider, model name, base URL, temperature, `top_p`, and
  per-model `max_tokens` through `models`;
- replacement `instructions.system` and `runtime.max_turns`;
- workspace and explicit environment variables through `environment`;
- invocation timeout through `runtime.timeout_seconds`;
- NeMo Fabric skills as external skill directories for Hermes Agent;
- NeMo Fabric MCP servers as Hermes Agent MCP server config;
- `tools.enabled` and `tools.blocked` as Hermes-native toolset selection and
  blocking policy;
- optional NeMo Relay telemetry plugin configuration.

Tool selectors are Hermes toolset names because that is the native policy
surface Hermes exposes.

During run-plan resolution, NeMo Fabric Core validates the
`models.<role>.top_p` (from 0 through 1) and positive
`models.<role>.max_tokens` extensions against the Hermes descriptor before
runtime startup. Hermes reads these prevalidated values from `AgentConfig`. A
model-level `max_tokens` overrides the harness-level default for the selected
model role.

The descriptor validates the following `harness.settings` fields:

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `reasoning_config` | object | `{"effort": "none"}` | Configures Hermes model reasoning. The closed object accepts an optional `enabled` boolean and an optional `effort` value of `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`. |
| `plugins_enabled` | array of nonempty strings | `[]` | Enables Hermes plugins by identifier. NeMo Fabric adds `observability/nemo_relay` when Relay telemetry is enabled. |
| `save_trajectories` | boolean | `false` | Enables Hermes-native JSONL conversation trajectory saving. This is separate from normalized NeMo Fabric telemetry. |
| `max_tokens` | positive integer | `512` | Limits the number of tokens in each Hermes model response. |
| `terminal_timeout` | positive number | `60` | Limits a Hermes terminal operation in seconds. |

For example:

```python
from nemo_fabric import HarnessConfig

harness = HarnessConfig(
    adapter_id="nvidia.fabric.hermes",
    settings={
        "reasoning_config": {"enabled": True, "effort": "medium"},
        "plugins_enabled": ["disk-cleanup"],
        "save_trajectories": True,
        "max_tokens": 1024,
        "terminal_timeout": 90,
    },
)
```

Use `runtime.max_turns` to set the Hermes agent-loop budget. The adapter maps
that normalized field to `AIAgent.max_iterations`; `max_iterations` is not a
Hermes harness setting.

The adapter derives Hermes state from the NeMo Fabric artifact root and creates
a child under `runtimes/<runtime_id>`, so invocations in one NeMo Fabric runtime
share state without sharing config or the session database with another
runtime.

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

Keep `hermes.fabric-adapter.json` aligned with the Python implementation:

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

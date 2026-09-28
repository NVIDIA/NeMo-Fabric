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
| `pip install "nemo-fabric-adapters-hermes[full]"` | No | Yes | No | No |
| `pip install nemo-fabric-adapters-hermes` | No | Yes | No | No |

Hermes Agent v2026.9.24 requires Relay 0.8.x. Keep it in a separate Python
environment from NeMo Fabric Relay 0.9 integrations. For local development,
create an isolated environment with the pinned released Hermes source:

```bash
just install-hermes-agent
export ADAPTER_PYTHON="$PWD/.venv-hermes/bin/python"
```

For split runtime and adapter environments, configure `ADAPTER_PYTHON` and use
matching NeMo Fabric release versions. Refer to the
[installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install#install-an-adapter-and-harness-without-the-runtime).

Hermes Relay telemetry and `Runtime.invoke_stream()` are temporarily
unavailable. Ordinary `Runtime.invoke()` calls remain supported.

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

Tool selectors are Hermes toolset names because that is the native policy
surface Hermes exposes.

During run-plan resolution, NeMo Fabric Core validates the
`models.<role>.top_p` (from 0 through 1) and positive
`models.<role>.max_tokens` fields against the Hermes descriptor before
runtime startup. Hermes reads these prevalidated values from `AgentConfig`. A
model-level `max_tokens` overrides the harness-level default for the selected
model role.

The descriptor validates the following `harness.settings` fields:

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `reasoning_config` | object | `{"effort": "none"}` | Configures Hermes model reasoning. The closed object accepts an optional `enabled` boolean and an optional `effort` value of `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`. |
| `plugins_enabled` | array of nonempty strings | `[]` | Enables Hermes plugins by identifier. |
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
method and closes the session database.

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

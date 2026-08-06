<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run NVIDIA NeMo Fabric Agents with Harbor

These examples keep Harbor in control of tasks, containers, verification,
rewards, retries, concurrency, and job layout while `FabricAgent` translates
Harbor options into one final typed `FabricConfig`. Complete the shared host
setup below, then use any walkthrough that matches the integration behavior
you want to exercise.

## Walkthroughs

| Walkthrough | What it demonstrates |
| --- | --- |
| [Calculator walkthrough](calculator/README.md) | Validate the complete integration and Harbor reward with a deterministic, credential-free smoke test, then optionally run the same task with the LLM-backed Hermes Agent or Claude harness. |
| [SWE-Bench walkthrough](swebench/README.md) | Run Hermes Agent and Claude experiments with skills, MCP servers, tool policy, Relay telemetry, and SWE-Bench verification. |

The calculator's scripted run is useful for validating a new checkout or
environment without calling an LLM. Its Hermes Agent and Claude runs exercise real
model integrations on the same small task. SWE-Bench exercises a real coding
task and supports comparisons across configuration variations.

## Execution Model

```mermaid
flowchart LR
    subgraph Host["<b>Harbor host process</b>"]
        direction TB
        Inputs["Harbor task and agent options"]:::data
        Start["Start Harbor run"]
        Build["FabricAgent builds<br/>final FabricConfig"]

        Inputs --> Start --> Build
    end

    subgraph Container["<b>Harbor task container</b>"]
        direction TB
        Prepare["Prepare task environment"]
        Invoke["Invoke NeMo Fabric runner"]
        Resolve["Fabric.run(): resolve<br/>adapter and assets"]
        Execute["Run selected harness<br/>in task workspace"]
        Verify["Verify task"]
        Result["Result and artifacts<br/>returned to host"]:::data
        Reward["Reward"]:::data

        Prepare --> Invoke --> Resolve --> Execute --> Verify --> Reward
        Execute --> Result
    end

    Host -- "FabricConfig + RunRequest + base_dir" --> Container

    classDef data fill:transparent,stroke:transparent
    style Host stroke-width:2px
    style Container stroke-width:2px
```

`FabricAgent` and `FabricConfig` construction run in the Harbor host process.
The pinned NeMo Fabric package, adapter discovery, harness execution, workspace, and
verifier run inside the isolated task container. Constructing the config does
not read task paths; adapter and asset resolution is deferred to
`Fabric.run()` with the task-local `base_dir`.

## Install the Host and Task Environments

Use the following package requirements for the two-environment model. Pin the
host and task packages to the same NeMo Fabric release. These examples use
version `0.1.0`.

| Environment | Required Dependencies | Purpose |
| --- | --- | --- |
| Harbor host | `nemo-fabric[harbor]==0.1.0` | Harbor CLI, `FabricAgent`, and typed `FabricConfig` construction |
| Claude task without Relay | `nemo-fabric[claude]==0.1.0` | NeMo Fabric runner, Claude adapter, and supported Claude harness |
| Claude task with Relay | `nemo-fabric[claude]==0.1.0` plus a NeMo Relay 0.7.x CLI on `PATH` | NeMo Fabric runner, Claude adapter and harness, and the adapter-managed Relay gateway and hooks |
| Hermes Agent task with Relay | `nemo-fabric[hermes-agent,relay]==0.1.0` | NeMo Fabric runner, Hermes Agent adapter and harness, and the NeMo Relay Python package |

The `nemo-fabric` package installs the runtime. The `relay` extra installs the
NeMo Relay Python package, not the CLI required by Claude.

## How Harbor Inputs Become FabricConfig

`FabricAgent` starts with the selected adapter and Harbor task workspace, then
applies every run input through typed NeMo Fabric models before crossing the task
container boundary:

| Harbor input | `FabricConfig` field |
| --- | --- |
| `--ak fabric_adapter_id=...` | `harness.adapter_id` |
| `--model` | `models.default` |
| `--skill` | `skills.paths` |
| `--mcp-config` | `mcp.servers` |
| `--ak fabric_telemetry=relay` | `telemetry.providers.relay` and `relay.observability` |
| `--ak fabric_model_base_url=<url>` | `models.default.base_url` |
| `--ak fabric_system_instruction=<text>` | `instructions.system` |
| `--ak fabric_max_turns=<count>` | `runtime.max_turns` |
| `--ak fabric_runtime_timeout_seconds=<seconds>` | `runtime.timeout_seconds` |
| `--ak fabric_environment_env='{...}'` | `environment.env` |
| `--ak fabric_blocked_tools='[...]'` | `tools.blocked` |
| `--ak fabric_enabled_tools='[...]'` | `tools.enabled` |
| `--ak fabric_harness_settings='{...}'` | Merged into `harness.settings`; planning rejects non-empty settings when the selected descriptor does not declare `settings_schema` |

The result is the complete `FabricConfig` uploaded with the `RunRequest` and
task-local `base_dir`. The container-side runner deserializes that payload and
passes it to `Fabric.run()` without adding configuration policy. The task,
verifier, and `FabricAgent` stay fixed, so each experiment changes only the
named Harbor input and its resulting evidence remains attributable.

## Shared Host Setup

Run every command from the repository root on an x86_64 Linux host with Python
3.12, `uv`, Docker, and the Docker Compose plugin. Create the host environment
and verify the relevant entry points:

```bash
cd "$(git rev-parse --show-toplevel)"
uv sync --python 3.12 --extra harbor
uv run --extra harbor harbor --version
uv run --extra harbor python -c \
  'from nemo_fabric.integrations.harbor import FabricAgent; print(FabricAgent.import_path())'
docker version
docker compose version
```

The Harbor command must report 0.18.x, and the Python command must print
`nemo_fabric.integrations.harbor.fabric_agent:FabricAgent`.

### Docker Installed with Snap

The Snap build of Docker sees a private `/tmp`, while Harbor creates temporary
Docker Compose overlays in the host temporary directory. If `command -v docker`
prints `/snap/bin/docker`, run this in every shell used for Harbor:

```bash
mkdir -p "$HOME/harbor-tmp"
export TMPDIR="$HOME/harbor-tmp"
uv run --extra harbor python -c \
  'import tempfile; print(tempfile.gettempdir())'
```

The final command must print a path under `$HOME/harbor-tmp`. Keep this shell
open and continue with either the [calculator](calculator/README.md) or
[SWE-Bench](swebench/README.md) guide.

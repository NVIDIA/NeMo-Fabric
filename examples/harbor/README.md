<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Example

This example shows how to use the installed
[`FabricAgent`](../../python/src/nemo_fabric/integrations/harbor/fabric_agent.py)
to run a Fabric harness inside a Harbor task environment.

Harbor owns task and dataset materialization, container lifecycle, verification,
rewards, retries, concurrency, and job layout. Fabric owns config validation,
harness lifecycle, normalized results, artifacts, and telemetry references. One
Harbor agent run creates one independent Fabric runtime.

```text
Harbor task
  -> FabricAgent
  -> HarborRunSpec JSON
  -> sandbox-local Fabric.run()
  -> selected harness
  -> RunResult JSON
  -> Harbor metadata and verifier
```

## Where the Fabric SDK runs

`FabricAgent` runs on Harbor's host side, so it does not open the Fabric config
or task workspace itself. It packages the instruction and Harbor-owned inputs
into a `HarborRunSpec`, uploads that file, and starts the Fabric runner inside
the task environment.

The runner can access the task files. It loads the YAML as a `FabricConfig`,
applies Harbor's model, MCP server, and skill inputs to a copy, and calls
`Fabric().run(...)`. That call handles the complete start, invoke, and stop
lifecycle for one independent Fabric runtime. The integration therefore does
not call `start_runtime()` directly.

## Install

Install the Harbor dependency with Fabric:

```bash
python3 -m pip install "nemo-fabric[harbor]"
```

For a source checkout:

```bash
python3 -m pip install -e .
python3 -m pip install -e ../harbor
```

Fabric, the selected adapter, and the config file must also be available inside
the Harbor task environment.

## Prepare a Fabric config

Create one complete config for the execution path. For example:

```yaml
schema_version: fabric.agent/v1alpha1

metadata:
  name: harbor-review-agent

harness:
  adapter_id: nvidia.fabric.hermes.cli
  resolution: preinstalled
  settings:
    cwd: /app
    base_url: https://integrate.api.nvidia.com/v1

models:
  default:
    provider: nvidia
    model: nvidia/nemotron-3-nano-30b-a3b

runtime:
  input_schema: text
  output_schema: message
  artifacts: /logs/agent/fabric-artifacts

environment:
  provider: local
  workspace: /app
  artifacts: /logs/agent/fabric-artifacts

telemetry:
  enabled: false
```

Copy the config into the task image, for example at
`/opt/fabric/configs/hermes.yaml`. `fabric_config_path` always refers to the
path inside the task environment, not the host checkout.

## Run the task

Pass one complete Fabric config path through Harbor's agent arguments:

```bash
harbor run --path <dataset-or-task-dir> \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak fabric_config_path=/opt/fabric/configs/hermes.yaml \
  --ae NVIDIA_API_KEY="$NVIDIA_API_KEY"
```

`fabric_config_path` is resolved inside the task container. The config selects
the harness, runtime, environment, and telemetry behavior. Harbor supplies the
task instruction and, when present, its model, MCP servers, and skill directory.

## Config composition

The sandbox runner validates the YAML as `FabricConfig`, makes a deep copy, and
then applies Harbor-owned inputs:

- `--model` replaces `models.default` with a `ModelConfig`;
- Harbor MCP servers replace the config's MCP section through
  `add_mcp_server()`;
- Harbor's skill directory replaces the config's skill section through
  `add_skill_path()`.

The final config is passed directly to `Fabric.run()` with a `RunRequest`.
Harbor scheduling values and job IDs do not enter the Fabric config or runtime.

## Inspect the result

The host writes a validated `HarborRunSpec` to the Harbor log directory and
uploads it to a unique task-environment path. The sandbox runner writes one
normalized `RunResult` to a unique result path. The host validates that result
before copying summary fields into `AgentContext.metadata["fabric"]`.

The Harbor agent log directory contains `fabric-run-<id>.json` and
`fabric-result-<id>.json`. The result includes Fabric status, harness and
adapter identity, runtime and invocation IDs, artifacts, telemetry references,
and structured errors. Use Harbor's viewer to inspect the trial and reward:

```bash
harbor view <jobs-directory>
```

## Optional agent arguments

`FabricAgent` accepts these additional constructor arguments:

- `fabric_python`: Python executable used to start the sandbox runner;
- `fabric_cwd`: working directory for installation and execution;
- `fabric_install_command`: environment bootstrap command;
- `fabric_timeout_sec`: timeout for bootstrap and execution.

## Demo and tests

The [multi-harness demo](demo/README.md) provides complete configs and commands
for a credential-free smoke run, Hermes CLI, Hermes with Relay, and Codex CLI.

Run the lightweight integration tests with:

```bash
pytest tests/python/test_harbor_integration.py \
  tests/integrations/test_harbor_runner.py
```

The Docker-backed SWE-Bench check is opt-in:

```bash
RUN_FABRIC_HARBOR_SWEBENCH_DOCKER=1 \
pytest tests/e2e/test_harbor_swebench_task.py
```

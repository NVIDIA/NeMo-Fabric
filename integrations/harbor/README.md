<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Integration

Use `nemo_fabric.integrations.harbor:FabricAgent` to run a Fabric harness inside
a Harbor task environment.

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

## Use FabricAgent

Pass one complete Fabric config path through Harbor's agent arguments:

```bash
harbor run --path <dataset-or-task-dir> \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak fabric_config_path=/opt/fabric/configs/hermes.yaml \
  --ae NVIDIA_API_KEY="$NVIDIA_API_KEY"
```

`fabric_config_path` is resolved inside the task container. The config selects
the harness, runtime, environment, and telemetry behavior for the run.

`FabricAgent` accepts these Fabric-specific constructor arguments:

- `fabric_config_path`: complete Fabric YAML config inside the task environment;
- `fabric_python`: Python executable used to start the sandbox runner;
- `fabric_cwd`: optional working directory for Fabric commands;
- `fabric_install_command`: optional environment bootstrap command;
- `fabric_timeout_sec`: optional timeout for bootstrap and execution.

## Config Composition

The sandbox runner validates the YAML as `FabricConfig`, makes a deep copy, and
then applies Harbor-owned inputs:

- `--model` replaces `models.default` with a `ModelConfig`;
- Harbor MCP servers replace the config's MCP section through
  `add_mcp_server()`;
- Harbor's skill directory replaces the config's skill section through
  `add_skill_path()`.

The final config is passed directly to `Fabric.run()` with a `RunRequest`.
Harbor scheduling values and job IDs do not enter the Fabric config or runtime.

## Exchange Boundaries

The host writes a validated `HarborRunSpec` to the Harbor log directory and
uploads it to a unique task-environment path. The sandbox runner writes one
normalized `RunResult` to a unique result path. The host validates that result
before copying summary fields into `AgentContext.metadata["fabric"]`.

## Demo and Tests

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

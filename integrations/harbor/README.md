<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Integration

Fabric provides a Harbor `BaseAgent` wrapper at
`nemo_fabric.integrations.harbor:FabricAgent`.

Use this when Harbor should keep ownership of evaluation semantics while Fabric
owns the selected agent harness invocation.

## Ownership

Harbor owns:

- task and dataset materialization;
- environment/container lifecycle;
- verifier execution and reward calculation;
- Harbor job, trial, log, and artifact layout.

Fabric owns:

- Fabric agent config/profile resolution;
- selected harness invocation, such as Hermes SDK or CLI;
- normalized `RunRequest` / `RunResult` handling;
- Fabric artifacts, logs, patch metadata, and telemetry references.

The integration shape is:

```text
Harbor task/env -> FabricAgent -> Fabric SDK runner -> harness runtime -> Fabric result -> Harbor metadata/verifier
```

## Install

Install the Harbor extra when the environment does not already provide Harbor:

```bash
pip install "nemo-fabric[harbor]"
```

For local checkout development:

```bash
python3 -m pip install -e .
python3 -m pip install -e ../harbor
```

## Using FabricAgent

`FabricAgent` follows Harbor 0.16.1's external-agent contract. The runner and
config files must be installed or copied into the task environment:

```bash
harbor run --path <dataset-or-task-dir> \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak fabric_config_path=/opt/fabric/agent.yaml \
  --ak 'fabric_profile_paths=["/opt/fabric/profiles/hermes.yaml"]' \
  --ae NVIDIA_API_KEY="$NVIDIA_API_KEY"
```

Important kwargs:

- `fabric_config_path`: YAML config path visible inside the Harbor environment.
- `fabric_profile_paths`: YAML profile path or ordered profile-path list.
- `fabric_python`: Python command used for the sandbox-local SDK runner.
- `fabric_cwd`: optional working directory for Fabric commands.
- `fabric_install_command`: optional explicit install/bootstrap command.
- `fabric_timeout_sec`: optional timeout for Fabric install/run commands.

Harbor passes the task instruction to Fabric as `RunRequest.input`. Harbor
metadata such as model name, skills directory, and MCP server definitions are
included under `RunRequest.context`. The sandbox-local runner loads YAML into
`FabricConfig` and `FabricProfileConfig`, then calls `Fabric.run()`.
The normalized result is saved as `fabric-result.json`, and summary fields are
copied into `context.metadata["fabric"]`.

## Multi-Harness Demo

The runnable MVP demo includes explicit Harbor CLI commands for a
credential-free pipeline check plus real Hermes, Hermes-with-Relay, and Codex
variants. See [`demo/README.md`](demo/README.md) for the commands and recording
flow.

## Local Smoke

The lightweight smoke uses a fake Harbor environment and validates command
construction plus metadata propagation:

```bash
python3 python/tests/smoke_harbor_integration.py
```

## SWE-Bench Smoke

The Docker-backed SWE-Bench smoke is opt-in because it requires Docker, a local
SWE-Bench image, and a Harbor-generated task directory. Harbor still owns task
materialization and verification; Fabric only invokes the configured harness and
captures artifacts.

```bash
RUN_FABRIC_HARBOR_SWEBENCH_DOCKER=1 python3 tests/smoke_harbor_swebench_task.py
```

To run the verifier path as well:

```bash
RUN_FABRIC_HARBOR_SWEBENCH_DOCKER=1 \
RUN_FABRIC_HARBOR_SWEBENCH_VERIFY=1 \
python3 tests/smoke_harbor_swebench_task.py
```

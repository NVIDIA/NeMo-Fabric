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
Harbor task/env -> FabricAgent -> fabric run -> harness runtime -> Fabric result -> Harbor metadata/verifier
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

`FabricAgent` is selected like any other custom Harbor agent:

```bash
harbor run --path <dataset-or-task-dir> \
  --agent-import-path nemo_fabric.integrations.harbor:FabricAgent \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ak fabric_agent_path=/workspace/code-review-agent \
  --ak fabric_profiles=hermes_sdk \
  --ak fabric_cli=fabric
```

Important kwargs:

- `fabric_agent_path`: path to the Fabric agent package or `agent.yaml` visible
  inside the Harbor environment.
- `fabric_profiles`: profile name or profile list applied in order.
- `fabric_cli`: Fabric CLI command visible inside the Harbor environment.
- `fabric_cwd`: optional working directory for Fabric commands.
- `fabric_install_command`: optional explicit install/bootstrap command.
- `fabric_timeout_sec`: optional timeout for Fabric install/run commands.

Harbor passes the task instruction to Fabric as `RunRequest.input`. Harbor
metadata such as model name, skills directory, and MCP server definitions are
included under `RunRequest.context`. Fabric writes the normalized result to the
Harbor logs directory as `fabric-result.json`, and summary fields are copied into
`context.metadata["fabric"]`.

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

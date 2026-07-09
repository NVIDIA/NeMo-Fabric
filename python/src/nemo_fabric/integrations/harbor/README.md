<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Integration

The public Harbor entrypoint is:

```text
nemo_fabric.integrations.harbor:FabricAgent
```

`FabricAgent` builds a `HarborRunSpec`, uploads it with
`BaseEnvironment.upload_file()`, and invokes
`nemo_fabric.integrations.harbor.runner` inside the task environment. The
runner validates one complete YAML config as `FabricConfig`, clones it, applies
Harbor's model, MCP, and skill inputs, and calls `Fabric.run()` directly.

The sandbox writes a normalized `RunResult`. The host downloads and validates
that result before populating `AgentContext.metadata["fabric"]`. Each run uses
unique specification and result paths.

## Package layout

- `models.py` defines `HarborRunSpec` and `HarborMcpServer`.
- `fabric_agent.py` implements the Harbor `BaseAgent` wrapper, transport, and
  result handling.
- `runner.py` composes the final Fabric config and owns the sandbox-local SDK
  call.
- `__init__.py` exports `FabricAgent` as the package entrypoint.

## Constructor arguments

- `fabric_config_path`: complete Fabric YAML config inside the task environment;
- `fabric_python`: Python executable used to start the runner;
- `fabric_cwd`: optional working directory for installation and execution;
- `fabric_install_command`: optional environment bootstrap command;
- `fabric_timeout_sec`: optional timeout for bootstrap and execution.

See [`examples/harbor/README.md`](../../../../../examples/harbor/README.md)
for installation and usage, and
[`examples/harbor/demo/README.md`](../../../../../examples/harbor/demo/README.md)
for runnable Harbor commands.

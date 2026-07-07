<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Integration

This package lets Harbor use Fabric as its agent execution layer. The public
Harbor entrypoint is:

```text
nemo_fabric.integrations.harbor:FabricAgent
```

Harbor continues to own task materialization, the task environment, verification,
rewards, and job layout. Fabric resolves its config and profiles, invokes the
selected harness, and returns a normalized result.

## Execution Flow

```text
harbor run
  -> FabricAgent on the host
  -> Harbor BaseEnvironment.exec(...)
  -> python -m nemo_fabric.integrations.harbor.runner in the task environment
  -> Fabric.run(...)
  -> selected Fabric harness adapter
```

`FabricAgent` writes a JSON run specification into the Harbor environment. The
runner loads the referenced YAML files as `FabricConfig` plus ordered profile
mappings, applies Harbor's model selection as the final profile, and invokes the
Fabric Python SDK. This path does not invoke the Fabric CLI.

The runner must execute inside the task environment because that is where the
harness reads and modifies the task workspace. Fabric, its adapter, and all
referenced config/profile paths must therefore be available there.

## Package Layout

- `__init__.py` implements the Harbor `BaseAgent` integration, command
  construction, and result propagation.
- `runner.py` is the sandbox-side SDK entrypoint.

The normalized Fabric result is written to `/logs/agent/fabric-result.json` by
default, downloaded to the Harbor agent log directory, and summarized in
`AgentContext.metadata["fabric"]`. A failed setup, runner invocation, or result
transfer fails the Harbor agent run rather than producing partial metadata.

## Configuration

`FabricAgent` accepts these Fabric-specific constructor arguments through
Harbor's `--ak` flags:

- `fabric_config_path`: Fabric YAML config path inside the task environment.
- `fabric_profile_paths`: one profile path or an ordered list of profile paths.
- `fabric_python`: Python executable used to start the runner.
- `fabric_cwd`: optional working directory for installation and execution.
- `fabric_install_command`: optional environment bootstrap command.
- `fabric_timeout_sec`: optional timeout for bootstrap and execution.
- `fabric_spec_path` and `fabric_result_path`: internal exchange-file paths.

See [`integrations/harbor/README.md`](../../../../../integrations/harbor/README.md)
for installation and usage, and
[`integrations/harbor/demo/README.md`](../../../../../integrations/harbor/demo/README.md)
for runnable Harbor CLI examples.

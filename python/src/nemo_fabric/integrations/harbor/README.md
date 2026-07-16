<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Integration

Use `nemo_fabric.integrations.harbor:FabricAgent` to run Fabric adapters in
Harbor tasks. Harbor options select the model, harness, skills, MCP servers,
tool policy, and telemetry; `FabricAgent` translates them into one typed
`FabricConfig` for the task run.

See the [Harbor example](../../../../../examples/harbor/README.md) for runnable
SWE-Bench commands, configuration variations, reward checks, and Relay
artifacts.

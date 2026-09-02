<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric mini-SWE-agent Adapter

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Fabric?color=green)](https://github.com/NVIDIA/NeMo-Fabric/releases)

`nemo-fabric-adapters-mini-swe-agent` translates the NVIDIA NeMo Fabric
adapter contract into mini-SWE-agent's native shell-only execution loop.

## Install

The following table shows which components each installation provides:

| Installation | Runtime | Adapter | Harness | Relay Python |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[mini-swe-agent]"` | Yes | Yes | Yes | No |
| `pip install "nemo-fabric[mini-swe-agent,relay]"` | Yes | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-mini-swe-agent[harness]"` | No | Yes | Yes | No |
| `pip install "nemo-fabric-adapters-mini-swe-agent[relay]"` | No | Yes | No | Yes |
| `pip install "nemo-fabric-adapters-mini-swe-agent[full]"` | No | Yes | Yes | Yes |
| `pip install nemo-fabric-adapters-mini-swe-agent` | No | Yes | No | No |

The optional Relay integration emits live model, tool, step, and invocation
telemetry and supports Relay-backed ATOF streaming. It does not enable native
OpenAI token streaming.

For configuration and behavior, see the
[mini-SWE-agent adapter README](https://github.com/NVIDIA/NeMo-Fabric/tree/main/adapters/mini-swe-agent/README.md).

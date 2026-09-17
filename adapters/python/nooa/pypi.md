<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric NOOA Adapters

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Fabric?color=green)](https://github.com/NVIDIA/NeMo-Fabric/releases)

`nemo-fabric-adapters-nooa` provides NVIDIA NeMo Fabric adapters for
[NVIDIA-labs Object Oriented Agents](https://github.com/nvidia-nemo/labs-OO-Agents)
(NOOA) `InteractiveAgent` targets and `nooa_bench.BenchAgent` evaluations.

## Install

| Installation | Runtime | Adapter | NOOA | Relay Python |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[nooa]"` | Yes | Yes | Yes | No |
| `pip install "nemo-fabric[nooa,relay]"` | Yes | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-nooa[harness]"` | No | Yes | Yes | No |
| `pip install "nemo-fabric-adapters-nooa[relay]"` | No | Yes | No | Yes |
| `pip install "nemo-fabric-adapters-nooa[full]"` | No | Yes | Yes | Yes |
| `pip install nemo-fabric-adapters-nooa` | No | Yes | No | No |

For configuration, supported targets, and behavior, refer to the
[NOOA adapter README](https://github.com/NVIDIA/NeMo-Fabric/tree/main/adapters/python/nooa/README.md).

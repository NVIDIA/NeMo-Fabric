<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric LangChain Deep Agents Adapter

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Fabric?color=green)](https://github.com/NVIDIA/NeMo-Fabric/releases)

![Diagram showing NeMo Fabric connecting applications, evaluations, and reinforcement learning rollouts to Hermes, Codex, Claude, and Deep Agents, with results, artifacts, and telemetry as outputs.](https://raw.githubusercontent.com/NVIDIA/NeMo-Fabric/refs/heads/main/assets/fabric-hero-option2.png)

Provides a NeMo Fabric adapter for use with [LangChain Deep Agents](https://www.langchain.com/deep-agents).

## Install

Installation can be performed using the `nemo-fabric` meta package with the `deepagents` extra, or by installing the adapter and harness packages separately. The following table shows which components each installation provides:

| Installation | Runtime | Adapter | Harness | NeMo Relay Python Package |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[deepagents]"` | Yes | Yes | Yes | No |
| `pip install "nemo-fabric[deepagents,relay]"` | Yes | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-deepagents[harness]"` | No | Yes | Yes | No |
| `pip install "nemo-fabric-adapters-deepagents[full]"` | No | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-deepagents[relay]"` | No | Yes | No | Yes |
| `pip install nemo-fabric-adapters-deepagents` | No | Yes | No | No |

NeMo Relay is optional for ordinary runs. NeMo Relay telemetry and streaming
require one of the installations in the table that includes the NeMo Relay
Python package.

Refer to the [installation guide](https://nvidia-nemo-fabric.docs.buildwithfern.com/nemo/fabric/getting-started/install) for more details.

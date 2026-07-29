<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Hermes Agent Adapter

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Fabric?color=green)](https://github.com/NVIDIA/NeMo-Fabric/releases)

Provides a NeMo Fabric adapter for use with [Hermes Agent](https://hermes-agent.nousresearch.com/).

## Install

Hermes Agent and this adapter require Python versions 3.11 through 3.13.
Installation can be performed using the `nemo-fabric` meta package with the `hermes-agent` extra, or by installing the adapter and harness packages separately. The following table shows which components each installation provides:

| Installation | Runtime | Adapter | Harness | NeMo Relay Python Package |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[hermes-agent]"` | Yes | Yes | Yes | No |
| `pip install "nemo-fabric[hermes-agent,relay]"` | Yes | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-hermes[harness]"` | No | Yes | Yes | No |
| `pip install "nemo-fabric-adapters-hermes[full]"` | No | Yes | Yes | Yes |
| `pip install "nemo-fabric-adapters-hermes[relay]"` | No | Yes | No | Yes |
| `pip install nemo-fabric-adapters-hermes` | No | Yes | No | No |

NeMo Relay is optional for ordinary runs. NeMo Relay telemetry and streaming
require one of the installations in the table that includes the NeMo Relay
Python package.

Refer to the [installation guide](https://nvidia-nemo-fabric.docs.buildwithfern.com/nemo/fabric/getting-started/install) for more details.


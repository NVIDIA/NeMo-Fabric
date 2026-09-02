<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Fabric?color=green)](https://github.com/NVIDIA/NeMo-Fabric/releases)

![Diagram showing NeMo Fabric connecting applications, evaluation systems, and reinforcement learning rollouts to harnesses and custom agents, with results, artifacts, and telemetry as outputs.](https://raw.githubusercontent.com/NVIDIA/NeMo-Fabric/refs/heads/main/assets/fabric-hero.png)

NeMo Fabric gives users one configurable, observable way to run applications
across agent harnesses and custom agents. It standardizes configuration,
lifecycle management, and results without requiring a separate integration for
every Adapter Target.

NeMo Fabric lets you change Adapter Targets without rebuilding each
integration, isolate conflicting runtime dependencies, and manage target
configuration, execution, and observability consistently. Every run returns
normalized results, artifacts, and telemetry for downstream systems to consume.

It provides:

- a versioned, typed configuration contract;
- ordinary Python composition for experiment variants;
- adapter integrations for target-specific launch and control;
- a Python SDK backed by the Rust core;
- normalized run results, artifact manifests, and telemetry references.

## Install

Install the core runtime and Python SDK:

```bash
pip install nemo-fabric
```

### Supported Python Versions

NeMo Fabric supports Python 3.11 through 3.14. However, some harnesses and
integrations have more restrictive requirements. Hermes Agent requires
Python 3.11 through 3.13, and the Harbor integration requires Python 3.12 or later.


### Supported Harnesses

The following table shows the install target for each supported agent harness:

| Agent Harness | Runtime, Adapter, and Harness | Adapter and Harness | Adapter Only |
| --- | --- | --- | --- |
| [Claude Code](https://pypi.org/project/nemo-fabric-adapters-claude/) | `nemo-fabric[claude]` | `nemo-fabric-adapters-claude[harness]` | `nemo-fabric-adapters-claude` |
| [Codex](https://pypi.org/project/nemo-fabric-adapters-codex/) | `nemo-fabric[codex]` | `nemo-fabric-adapters-codex[harness]` | `nemo-fabric-adapters-codex` |
| [Hermes Agent](https://pypi.org/project/nemo-fabric-adapters-hermes/) | Install Hermes Agent separately, then install `nemo-fabric` and `nemo-fabric-adapters-hermes` | Install Hermes Agent separately, then install `nemo-fabric-adapters-hermes` | `nemo-fabric-adapters-hermes` |
| [LangChain Deep Agents](https://pypi.org/project/nemo-fabric-adapters-deepagents/) | `nemo-fabric[deepagents]` | `nemo-fabric-adapters-deepagents[harness]` | `nemo-fabric-adapters-deepagents` |
| [mini-SWE-agent](https://pypi.org/project/nemo-fabric-adapters-mini-swe-agent/) | `nemo-fabric[mini-swe-agent]` | `nemo-fabric-adapters-mini-swe-agent[harness]` | `nemo-fabric-adapters-mini-swe-agent` |


To install the NeMo Fabric runtime, adapter, and supported harness in one
environment, choose one of the following `nemo-fabric` harness extras:

```bash
pip install "nemo-fabric[claude]"
pip install "nemo-fabric[codex]"
pip install "nemo-fabric[deepagents]"
pip install "nemo-fabric[mini-swe-agent]"
```

Hermes Agent 0.20 and later is not installable from PyPI. For this reason the Hermes Agent adapter does not provide a `harness` extra. Follow the
[Hermes Agent installation guide](https://hermes-agent.nousresearch.com/docs/installation),
then install `nemo-fabric` and `nemo-fabric-adapters-hermes` into the Python
environment that runs Hermes Agent.

To install an adapter and its harness without the NeMo Fabric runtime, choose
one of the following adapter package `harness` extras:

```bash
pip install "nemo-fabric-adapters-claude[harness]"
pip install "nemo-fabric-adapters-codex[harness]"
pip install "nemo-fabric-adapters-deepagents[harness]"
pip install "nemo-fabric-adapters-mini-swe-agent[harness]"
```

Every adapter package also provides an adapter-scoped `full` extra, which does
not install the NeMo Fabric runtime. For Claude, Codex, and mini-SWE-agent, `full` installs the same dependencies as `harness`. For LangChain Deep Agents, `full` also installs the NeMo Relay Python package. The Hermes adapter's `full` extra installs NeMo Relay but does not install Hermes Agent.

If the environment already manages a compatible harness, choose one of the
following bare adapter packages:

```bash
pip install nemo-fabric-adapters-claude
pip install nemo-fabric-adapters-codex
pip install nemo-fabric-adapters-deepagents
pip install nemo-fabric-adapters-hermes
pip install nemo-fabric-adapters-mini-swe-agent
```

The adapter distribution contains only adapter-owned runtime dependencies. It
does not install the NeMo Fabric runtime. For package-installable harnesses,
select `harness` or `full` to install the harness. The Hermes adapter never
installs Hermes Agent. If the runtime shares an environment with an existing
compatible harness, install `nemo-fabric` and the bare adapter package together.


### Integrations

#### Harbor Integration

Install the Harbor integration with the `harbor` extra:

```bash
pip install "nemo-fabric[harbor]"
```

#### NeMo Relay Integration

Install a compatible NeMo Relay package with the `relay` extra:

```bash
pip install "nemo-fabric[relay]"
```

This installs a version of the
[NeMo Relay Python package](https://docs.nvidia.com/nemo/relay) known to be
compatible with the installed version of NeMo Fabric.

The LangChain Deep Agents and Hermes Agent adapter packages also provide
`relay` and `full` extras for environments that do not install `nemo-fabric`.
Claude and Codex require the
[`nemo-relay` CLI](https://crates.io/crates/nemo-relay-cli) instead of the NeMo
Relay Python package. They do not provide a `relay` extra. Refer to the
[NeMo Relay CLI](https://docs.nvidia.com/nemo/fabric/getting-started/install#nemo-relay-cli)
install guide for instructions on installing the CLI tool.

## Learn More

Refer to the [NVIDIA NeMo Fabric documentation](https://docs.nvidia.com/nemo/fabric)
for installation, configuration, and usage guidance. Source code is available
in the [NVIDIA NeMo Fabric repository](https://github.com/NVIDIA/NeMo-Fabric/).

<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Fabric)](https://github.com/NVIDIA/NeMo-Fabric/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Fabric/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Fabric?color=green)](https://github.com/NVIDIA/NeMo-Fabric/releases)
[![PyPI](https://img.shields.io/pypi/v/nemo-fabric?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-fabric/)
[![Crates.io](https://img.shields.io/crates/v/nemo-fabric-core?label=nemo-fabric-core&color=B7410E&logo=rust)](https://crates.io/crates/nemo-fabric-core)
[![Crates.io](https://img.shields.io/crates/v/nemo-fabric-cli?label=nemo-fabric-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-fabric-cli)

NeMo Fabric gives applications one configurable, observable way to run applications
across multiple agent harnesses. It standardizes configuration, lifecycle
management, and results without requiring a separate integration for every harness.

NeMo Fabric lets you change harnesses without rebuilding each integration,
isolate conflicting runtime dependencies, and manage harness configuration,
execution, and observability consistently. Every run returns normalized
results, artifacts, and telemetry for downstream systems to consume.

It provides:

- a versioned, typed configuration contract;
- ordinary Python composition for experiment variants;
- adapter integrations for harness-specific launch and control;
- a Python SDK backed by the Rust core;
- normalized run results, artifact manifests, and telemetry references.

## Install

NeMo Fabric supports Python 3.11 through 3.14. Use Python 3.11 through 3.13 for
Hermes Agent; the Harbor integration requires Python 3.12 or later.

Install the core runtime and Python SDK:

```bash
pip install nemo-fabric
```

To install the NeMo Fabric runtime, adapter, and supported harness in one
environment, choose one of the following `nemo-fabric` harness extras:

```bash
pip install "nemo-fabric[claude]"
pip install "nemo-fabric[codex]"
pip install "nemo-fabric[deepagents]"
pip install "nemo-fabric[hermes-agent]"
```

To install an adapter and its harness without the NeMo Fabric runtime, choose
one of the following adapter package `harness` extras:

```bash
pip install "nemo-fabric-adapters-claude[harness]"
pip install "nemo-fabric-adapters-codex[harness]"
pip install "nemo-fabric-adapters-deepagents[harness]"
pip install "nemo-fabric-adapters-hermes[harness]"
```

Every adapter package also provides an adapter-scoped `full` extra, which does
not install the NeMo Fabric runtime. For Claude and Codex, `full` installs the
same dependencies as `harness`. For LangChain Deep Agents and Hermes Agent,
`full` also installs the NeMo Relay Python package.

If the environment already manages a compatible harness, choose one of the
following bare adapter packages:

```bash
pip install nemo-fabric-adapters-claude
pip install nemo-fabric-adapters-codex
pip install nemo-fabric-adapters-deepagents
pip install nemo-fabric-adapters-hermes
```

The adapter distribution contains only adapter-owned runtime dependencies. It
does not install the NeMo Fabric runtime. Select `harness` or `full` to install
the harness. If the runtime shares an environment with an existing compatible
harness, install `nemo-fabric` and the bare adapter package together.

NeMo Fabric supports running an agent harness in a different virtual
environment from the NeMo Fabric runtime. This separation can isolate harnesses
that have conflicting dependencies. Use matching NeMo Fabric release versions
for the runtime and adapter package unless a different pairing has been
explicitly validated.

### Integrations

#### Harbor Integration

```bash
pip install "nemo-fabric[harbor]"
```

#### Relay Integration

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
[NeMo Relay CLI](https://nvidia-nemo-fabric.docs.buildwithfern.com/nemo/fabric/getting-started/install#install-nemo-relay)
install guide for instructions on installing the CLI tool.

## Core Concepts

- **Typed configuration:** Construct a complete `FabricConfig` in Python and
  use ordinary functions to create experiment variants.
- **Adapters:** Select harness-specific integrations with
  `harness.adapter_id`.
- **Artifacts:** Receive normalized output, logs, patches, and telemetry
  references through an `ArtifactManifest`.

The experimental `nemo-fabric` CLI is distributed separately from the Python
package. It selects complete typed configs from built-in presets and maintained
examples. Applications that need a stable integration surface should use the
Python SDK.

## Learn More

Refer to the [NVIDIA NeMo Fabric documentation](https://nvidia-nemo-fabric.docs.buildwithfern.com/nemo/fabric)
for installation, configuration, and usage guidance. Source code is available
in the [NVIDIA NeMo Fabric repository](https://github.com/NVIDIA/nemo-fabric/).

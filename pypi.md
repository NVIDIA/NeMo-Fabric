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

Install the core runtime and Python SDK:

```bash
pip install nemo-fabric
```

To use a supported agent harness, install its adapter extra:

```bash
pip install "nemo-fabric[claude]"
pip install "nemo-fabric[codex]"
pip install "nemo-fabric[deepagents]"
pip install "nemo-fabric[hermes]"
```

NeMo Fabric supports running an agent harness in a different virtual environment than the one used to run NeMo Fabric itself. This is useful for running agents that have conflicting dependencies with NeMo Fabric or other agents.

The adapter must be installed into the virtual environment that the harness is installed in. For this reason adapters intentionally have minimal dependencies.

### Integrations

#### Harbor Integration

```bash
pip install "nemo-fabric[harbor]"
```

#### Relay Integration

```bash
pip install "nemo-fabric[relay]"
```

This installs a version of [NeMo Relay](https://docs.nvidia.com/nemo/relay) Python library known to be compatible with the installed version of NeMo Fabric.

Some adapters, such as Claude and Codex, require the
[`nemo-relay` CLI](https://crates.io/crates/nemo-relay-cli) tool instead of the
NeMo Relay Python library. Refer to the
[NeMo Relay CLI](https://docs.nvidia.com/nemo/fabric/getting-started/install#nemo-relay-cli) install guide for instructions on installing the CLI tool.

### Python Versions

NeMo Fabric supports Python versions 3.11-3.14, however some of the integrations and adapters may have additional requirements. Specifically Hermes Agent doesn't support Python 3.14 yet, and the Harbor integration requires Python 3.12 or later.

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

<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric

NVIDIA NeMo Fabric is a runtime execution layer for agents. It turns multiple
agent harnesses into one configurable, observable lifecycle surface.

NeMo Fabric standardizes how applications configure, launch, invoke, and
collect artifacts from agent harnesses. It provides:

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

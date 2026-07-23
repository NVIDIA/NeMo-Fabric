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

When the runtime and adapter share an environment, replace `<adapter>` with
`claude`, `codex`, `deepagents`, or `hermes`. Repeat the command for each harness
you use:

```bash
pip install "nemo-fabric[<adapter>]"
```

These extras install the adapter and its adapter-owned runtime dependencies.
They do not install the corresponding agent harness.

NeMo Fabric supports two environment layouts:

- **Co-located runtime and harness:** Install one of the `nemo-fabric` adapter
  extras shown above and a compatible harness in the same virtual environment.
- **Separate runtime and harness:** Install `nemo-fabric` in the runtime
  environment. In the harness environment, install both the corresponding
  `nemo-fabric-adapters-*` distribution and a compatible harness.

Separate environments are useful when a harness has dependencies that conflict
with NeMo Fabric or another harness. Refer to the
[adapter compatibility reference](https://github.com/NVIDIA/NeMo-Fabric/tree/main/adapters)
for exact package names, supported harness versions, and installation
instructions.

For a complete co-located Claude environment, the `claude-agent` convenience
extra includes the adapter and the exact SDK version tested by this repository.
The `hermes-agent` convenience extra includes the Hermes adapter and its
declared Hermes Agent dependency:

```bash
pip install "nemo-fabric[claude-agent]"
pip install "nemo-fabric[hermes-agent]"
```

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

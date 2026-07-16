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
- profile-based configuration for evaluation and ablation runs;
- adapter integrations for harness-specific launch and control;
- a Python SDK backed by a Rust core;
- normalized run results, artifact manifests, and telemetry references.

## Install

Install the core runtime and Python SDK:

```bash
pip install "nemo-fabric[runtime]"
```

To use a supported agent harness, install its adapter extra:

```bash
pip install "nemo-fabric[claude]"
pip install "nemo-fabric[codex]"
pip install "nemo-fabric[deepagents]"
pip install "nemo-fabric[hermes]"
```

Fabric supports running an agent harness in a different virtual environment than the one used to run Fabric itself. This is useful for running agents that have conflicting dependencies with Fabric or other agents.

The adapter must be installed into the virtual environment that the harness is installed in. For this reason adapters intentionally have minimal dependencies.

### Integrations

Harbor integration:

```bash
pip install "nemo-fabric[harbor]"
```

Relay integration:

```bash
pip install "nemo-fabric[relay]"
```

### Python Versions

NeMo Fabric supports Python versions 3.11-3.14, however some of the integrations and adapters may have additional requirements. Specifically the Hermes adapter doesn't support Python 3.14 yet, and the Harbor integration requires Python 3.12 or later.

## Core Concepts

- **Agent source:** Provide either an agent package path or a typed
  `FabricConfig`.
- **Typed configuration:** Construct configuration in memory with the Python
  SDK, or use `agent.yaml` as a portable representation.
- **Profiles:** Vary the harness, model, MCP servers, tools, skills, telemetry,
  or environment without editing the base configuration.
- **Adapters:** Select harness-specific integrations with
  `harness.adapter_id`.
- **Artifacts:** Receive normalized output, logs, patches, and telemetry
  references through an `ArtifactManifest`.

## Learn More

Refer to the [NVIDIA NeMo Fabric documentation](https://nvidia-nemo-fabric.docs.buildwithfern.com/nemo/fabric)
for installation, configuration, and usage guidance. Source code is available
in the [NVIDIA NeMo Fabric repository](https://github.com/NVIDIA/nemo-fabric/).

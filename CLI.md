<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Fabric Experimentation CLI

`nemo-fabric` is a maintained developer experimentation interface over the
NeMo Fabric SDK. It provides a fast way to try harnesses, run examples, inspect
plans, diagnose integrations, and execute user-owned Python configurations.
The SDK remains the canonical configuration and execution contract.

The CLI is intentionally an experimentation surface. Its command grammar may
evolve as workflows mature; typed SDK applications should be used when API
stability or production integration is required.

## Intent

Keep the CLI as a thin SDK-backed runner:

- every run starts from a complete, typed `FabricConfig`;
- the CLI selects a preset, example variant, or Python factory;
- planning and execution go through the public Python SDK;
- `base_dir` only resolves relative resource paths; and
- each selector returns a complete config without implicit merging.

## Supported Inputs

### 1. Preset

```bash
nemo-fabric run --preset hermes --input "Say hello"
```

A small, complete config maintained with the CLI. Useful for smoke tests and
quick harness probes. Presets do not inherit or merge.

### 2. Example

```bash
nemo-fabric run \
  --example examples.code_review_agent \
  --variant hermes \
  --input "Review this workspace"
```

A runnable SDK example with complete variants. Examples are Python source and
assets that people can read, copy, and edit. Each variant is an independent
factory.

Copy the bundled example into an editable Python package with:

```bash
nemo-fabric example init examples.code_review_agent my_agent
```

### 3. User factory

```bash
nemo-fabric run \
  --factory my_agent.config:build_config \
  --base-dir . \
  --input "Review this workspace"
```

The unrestricted customization path. The callable takes no arguments and
returns a complete `FabricConfig`.

The `--factory` name makes the executable Python contract explicit.

All three selectors feed the same commands:

```text
plan   doctor   run   chat
```

## Designed For

- quick harness and model experiments;
- smoke tests and troubleshooting;
- comparing complete preset or example variants;
- copying and editing Python examples;
- invoking custom `FabricConfig` factories; and
- inspecting plans and diagnostics.

## Not Designed For

- production deployment or scheduling;
- persistent configuration or secrets management;
- implicit configuration merging or inheritance;
- arbitrary CLI configuration mutation;
- evaluation orchestration or experiment tracking; or
- replacing direct SDK applications.

## Boundaries

- Presets, examples, and factories each return a complete `FabricConfig`.
- No preset inheritance or core merge semantics.
- No implicit user/project/system config discovery.
- JSON request input is still valid; it describes an invocation, not an agent.
- Adapter-generated harness files remain valid implementation details.
- Private serialization across Python/Rust or Harbor process boundaries is not
  a public authoring format.

## Design Bias

Start with the three explicit selectors. Keep presets tiny, make examples the
editable learning surface, and use Python factories for real customization.
Keep registration package-owned for now; consider plugin discovery only after
the workflows prove useful.

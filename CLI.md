<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Fabric Experimentation CLI

`nemo-fabric` is a small interface for trying harnesses, inspecting plans, and
starting from maintained examples. It is intentionally for experimentation;
applications that need a stable integration should use a language API directly.

The command implementation and catalogs live in the Rust `fabric-cli` crate.
The `nemo-fabric-runtime` Python package exposes the same Rust implementation
through a thin console-script bridge, so package and standalone installs do not
maintain separate CLIs.

## Variations

Use a preset for the shortest complete probe:

```bash
nemo-fabric run --preset hermes --input "Say hello"
```

Use a maintained example to exercise a complete workflow and variant:

```bash
nemo-fabric run --example code-review --variant hermes \
  --input "Review the workspace"
```

Generate ordinary editable application code when the catalog is no longer
enough:

```bash
nemo-fabric example init code-review my-agent --language python --variant hermes
nemo-fabric example init code-review my-agent-rs --language rust --variant hermes
```

The Python scaffold calls the Python SDK; the Rust scaffold calls
`fabric-core`. Neither is loaded back into the central CLI.

## Boundaries

- Every preset and example variant constructs a complete typed `FabricConfig`.
- Examples reuse preset defaults and one shared workspace/skill asset tree.
- Variants do not inherit, merge, or behave like profiles.
- There is no `--config`, `--profile`, or `--factory` input.
- Fabric does not discover or persist YAML, TOML, or JSON agent configuration.
- JSON request payloads and harness-generated files remain valid; they are not
  Fabric configuration sources.
- The CLI is not a scheduler, evaluation framework, or production deployment
  interface.

Current lifecycle commands are `plan`, `doctor`, and `run`. Catalog discovery
is available through `preset list/show` and `example list/show`.

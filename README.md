<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric

Fabric is a runtime execution layer for agents. It turns multiple agent
harnesses into one configurable, observable lifecycle surface.

<p align="center">
  <img src="assets/fabric-hero.png" alt="NeMo Fabric connects deployment platforms, evaluation harnesses, and RL rollout harnesses to multiple agent runtimes through one observable execution surface." width="1000">
</p>

## Architecture

NeMo Fabric standardizes how applications configure, launch, invoke, and collect
artifacts from agent harnesses.

Fabric provides:

- a versioned typed config contract, with `agent.yaml` as the portable file
  format;
- profile-based config variation for evaluation and ablation runs;
- adapter descriptors for harness-specific launch and control;
- a Rust core with a CLI and Python bindings;
- JSON Schema snapshots for the public config and runtime contract;
- normalized run results, artifact manifests, and telemetry references.

```mermaid
flowchart TB
  Consumer["Consumer\nCLI | Python SDK | integrations"]
  Config["Agent source\nagent.yaml or FabricConfig"]
  Core["Fabric Rust core\nresolve | plan | create | invoke | destroy"]
  Adapter["Selected Fabric adapter"]
  Harness["Agent harness runtime\nHermes | Codex | custom"]
  Artifacts["Artifact manifest\noutput | logs | patches | telemetry refs"]
  Relay["NeMo Relay\nATOF / ATIF when enabled"]

  Consumer --> Core
  Config --> Core
  Core --> Adapter
  Adapter --> Harness
  Harness --> Artifacts
  Core --> Artifacts
  Core -. telemetry config .-> Relay
  Harness -. harness telemetry .-> Relay
```

## Building and Installing

This path installs Fabric, installs Hermes Agent in a separate Python environment,
and runs one input through the Hermes Agent adapter.

Prerequisites:

- Rust and Cargo
- Python 3.11+ for Fabric
- Python 3.11-3.13 for Hermes Agent
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- `just` 1.50.0+
- `NVIDIA_API_KEY` for NVIDIA-hosted model access

Install `just` if not already installed.
```bash
cargo install just --locked
```

Ensure the local Cargo bin directory is in your `PATH`, if not set it with:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

Refer to the [official installation guide](https://just.systems/man/en/installation.html) for more details.

Install Fabric from the source checkout:

```bash
just build-all
just wheels
```

## Quick Start: Hermes Agent

Install Hermes into its own environment (the nemo-fabric[hermes] extra will install Hermes Agent, and the Hermes Agent adapter but not Fabric itself):

```bash
# Use any Python 3.11-3.13 interpreter for Hermes.
python3 -m venv .tmp/hermes-venv
.tmp/hermes-venv/bin/python -m pip install --find-links dist "nemo-fabric[hermes]"
```

If you are working from a local Hermes checkout, replace the final install line
with:

```bash
.tmp/hermes-venv/bin/python -m pip install -e ../hermes-agent
.tmp/hermes-venv/bin/python -m pip install --find-links <path-fabric-repo>/dist nemo-fabric-adapters-hermes
```

Run the code-review example:

```bash
export NVIDIA_API_KEY=...
export ADAPTER_PYTHON="$PWD/.tmp/hermes-venv/bin/python"

.venv/bin/python -m examples.code_review_agent \
  --input "Reply with exactly: fabric works"
```

`ADAPTER_PYTHON` selects the interpreter used to launch any Python adapter.
An explicit `harness.settings.python` or `harness.settings.python_env` takes
precedence. If none is configured and `ADAPTER_PYTHON` is unset, Fabric falls
back to `python3`. 

Use `ADAPTER_PYTHON` when the harness is installed in a separate environment from Fabric. The environment must have the adapter package installed, the adapters tend to be small and self-contained with minimal dependencies.

The run returns a normalized `RunResult` JSON payload and writes logs/artifacts
under `examples/code_review_agent/artifacts/hermes/`. Its complete base
config and clone-based variants live in
`examples/code_review_agent/config.py`.

### Next Steps
- Follow the [Example Notebooks](examples/notebooks/README.md) for a guided tour of the Python SDK.
- Refer to the [Python SDK guide](docs/sdk/python.mdx): typed configuration, planning,
  diagnostics, requests, multi-turn runtimes, parallelism, results, and errors.

## Claude Adapter

Build the local wheels and install Fabric with the independent Claude adapter:

```bash
just wheels
python -m pip install --find-links dist "nemo-fabric[claude]"
```

Refer to the [Claude adapter guide](adapters/claude/README.md) for
typed configuration, normalized tools, MCP and skills, multi-turn resume,
authentication, and execution details.

## Core Concepts

- **Agent source:** callers provide a typed `FabricConfig`. Start with `examples/code_review_agent/config.py` for the application-facing Pydantic pattern.
- **Tools policy:** call `FabricConfig.block_tools()` to add harness-neutral
  blocked tool names or toolsets to a typed configuration:

  ```python
  config.block_tools("browser", "shell")
  ```

  The selected adapter interprets these names: Hermes Agent maps them to disabled
  toolsets, Claude maps them to `disallowed_tools`, Deep Agents enforces them
  with middleware, and adapters without a native deny mechanism route the
  policy as unsupported.
- **Adapters:** harness-specific integrations selected by `harness.adapter_id`.
  The Hermes Agent adapter lives under `adapters/hermes/`; the Codex adapter lives under `adapters/codex/`; the
  [Claude adapter](adapters/claude/README.md)
  lives under `adapters/claude/`; the LangChain Deep Agents adapter lives under
  `adapters/deepagents/`. Harness Agent-specific extensions belong under
  `harness.settings` so the normalized contract can remain stable.
- **Artifacts:** normalized output, logs, patches, and telemetry references
  returned through an `ArtifactManifest`.

Fabric applies profiles in caller order and validates the final effective config
before planning or running.

Path sources select profiles by name. Typed `FabricConfig` sources usually
compose the final config in Python; `FabricProfileConfig` values are available
for callers that need ordered file-style overlays. The SDK rejects raw profile
mappings and mixed profile stacks. See the
[Python SDK guide](docs/sdk/python.mdx) for the complete public API,
type definitions, lifecycle semantics, and error behavior.

`run(...)` owns the complete start, invoke, and stop lifecycle. For typed
in-memory configuration, planning and diagnostics, explicit requests,
multi-turn runtimes, application-owned parallelism, results, and errors, see
the [Python SDK guide](docs/sdk/python.mdx). Exact signatures are in the
[generated Python API reference](docs/reference/api/python-library-reference/index.md).

## More Workflows

- [Example Notebooks](examples/notebooks/README.md) provide a guided tour of the Python SDK.
- [Python SDK guide](docs/sdk/python.mdx): typed configuration, planning,
  diagnostics, requests, multi-turn runtimes, parallelism, results, and errors.
- [Consumer integration skills](skills/README.md): repository-local coding-agent
  skills for integrating Fabric into an application through the Python SDK.
- [Getting Started overview](docs/about-nemo-fabric/overview.mdx): interface
  selection and the end-to-end Fabric workflow.
- [Harbor examples](examples/harbor/README.md): validate the integration with a
  deterministic, credential-free calculator smoke, optionally run the same
  task with Hermes or Claude, and evaluate real coding tasks with SWE-Bench.
- Adapter guides: [Hermes](adapters/hermes/README.md),
  [Codex SDK](adapters/codex/README.md), and
  [Deep Agents](adapters/deepagents/README.md).

## Tests

To run the full test suite, bootstrap a virtual environment with the optional dependencies.

```bash
uv venv --seed .venv --python 3.12
source .venv/bin/activate
uv sync --all-groups --all-extras
```

Build Fabric and the Python extension. Because the virtual environment is
already bootstrapped, pass `no_uv=true` to avoid reinstalling dependencies.

```bash
just no_uv=true build-all
```

Run both Rust and Python tests:

```bash
just no_uv=true test-all
```

Run just the Rust tests:

```bash
just no_uv=true test-rust
```

Run just the Python tests:

```bash
just no_uv=true test-python
```

Running `pytest` directly:

```bash
pytest
```

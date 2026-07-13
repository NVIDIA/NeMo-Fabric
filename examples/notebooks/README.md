<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Onboarding notebooks

A guided, hands-on tour of the NeMo Fabric Python SDK -- the fastest way to
learn how to configure, run, inspect, and vary an agent.

| Notebook | What it covers |
| --- | --- |
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | The full lifecycle on one harness: describe an agent as a typed `FabricConfig`, inspect it with `plan()`, diagnose the environment with `doctor()`, `run()` one request, read the normalized `RunResult`, and continue across turns with a stateful runtime. |
| [`02_variations.ipynb`](02_variations.ipynb) | Why Fabric exists: take that same agent and run it on every harness whose prerequisites are present (Hermes, Deep Agents, Codex, Claude), add and remove skills and MCP servers, and turn on NeMo Relay to emit trace files. |

Read them in order. The quickstart teaches the mental model; the variations
notebook shows how one agent runs across harnesses and how evaluation and
ablation runs sweep its configuration.

## Prerequisites

- Build the SDK and native extension from the repo root: `just build-all`. This
  alone is enough to run both notebooks top to bottom.
- To actually *run* a harness (rather than just inspect its config), that
  harness's adapter and credentials must be present:
  - **Hermes** (both notebooks): install Hermes in its own environment (the repo
    README's [Hermes SDK quick start](../../README.md#quick-start-hermes-sdk))
    and set `NVIDIA_API_KEY`. The setup cell auto-detects `.tmp/hermes-venv`.
  - **Deep Agents, Codex, Claude** (variations notebook): the matching adapter
    installed in the Fabric environment, plus that harness's credentials
    (`NVIDIA_API_KEY` for Deep Agents; an authenticated `codex` and
    `OPENAI_API_KEY` for Codex; `ANTHROPIC_API_KEY` for Claude).
- `NVIDIA_API_KEY` is loaded from a gitignored `.env` at the repo root if present.

Every live cell is guarded. With only `just build-all` done, both notebooks
still execute end to end: the variations notebook runs each harness whose
prerequisites are met and, for the rest, inspects the resolved config with
`plan()` and prints exactly what to provide to run it for real.

## Launch

```bash
just notebooks
```

This opens Jupyter Lab in this directory using the project interpreter, fetching
Jupyter on demand (it is not added to the project lockfile). To use an existing
Jupyter, run it with the `.venv` interpreter so `nemo_fabric` is importable:

```bash
.venv/bin/jupyter lab examples/notebooks
```

Committed notebooks are kept output-free. Run artifacts land under gitignored
`artifacts/` directories: the quickstart writes to `examples/notebooks/artifacts/`,
and the variations notebook reuses the code-review example's builders, so its runs
(including Relay traces) write under `examples/code_review_agent/artifacts/`.

For the complete API, see the [Python SDK guide](../../docs/sdk/python.mdx).

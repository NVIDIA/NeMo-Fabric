<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Onboarding notebooks

A guided, hands-on tour of the NeMo Fabric Python SDK -- the fastest way to
learn how to configure, run, inspect, and vary an agent. It's possible to run these workflows in Google Colab with no setup. Click here to open the notebooks in Google Colab [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/NeMo-Fabric/).

| Notebook | What it covers |
| --- | --- |
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | **Fully self-contained.** The full lifecycle on one harness, every agent built inline: describe an agent as a typed `FabricConfig`, inspect it with `plan()`, diagnose the environment with `doctor()`, `run()` one request, read the normalized `RunResult`, and continue across turns with a stateful runtime. |
| [`02_variations.ipynb`](02_variations.ipynb) | **Advanced composition on the maintained [code-review example](../code_review_agent/README.md).** Build on its `base_config()` to run the same agent across harnesses (Hermes Agent, Deep Agents, Codex, Claude) and to vary configuration — skills, MCP servers, models, and NeMo Relay telemetry. |

Read them in order. The quickstart teaches the mental model standalone; the
variations notebook shows advanced composition against a real, maintained agent.

## Prerequisites

- Build the SDK and native extension from the repo root: `just build-all`. This
  alone is enough to run both notebooks top to bottom.
- To actually *run* a harness (rather than just inspect its config), that
  harness's adapter and credentials must be present:
  - **Hermes Agent** (both notebooks): install Hermes Agent in its own environment (the repo
    README's [Hermes Agent quick start](../../README.md#quick-start-hermes-agent))
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
`artifacts/` directories: the quickstart and the variations notebook's Relay
traces write to `examples/notebooks/artifacts/`, while the variations notebook's
harness runs reuse the code-review example's builders and write under
`examples/code_review_agent/artifacts/`.


## Next Steps
- Refer to the [Python SDK guide](../../docs/sdk/python.mdx): typed configuration, planning, diagnostics, requests, multi-turn runtimes, parallelism, results, and errors.
- Other examples in this repo [`examples/`](../README.md#more-workflows).

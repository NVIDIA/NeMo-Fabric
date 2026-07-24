<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Onboarding notebooks

A guided, hands-on tour of the NeMo Fabric Python SDK -- the fastest way to
learn how to configure, run, inspect, and vary an agent.

| Notebook | What it covers |
| --- | --- |
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | **Fully self-contained.** The full lifecycle on one harness, every agent built inline: describe an agent as a typed `FabricConfig`, inspect it with `plan()`, diagnose the environment with `doctor()`, `run()` one request, read the normalized `RunResult`, and continue across turns with a stateful runtime. |
| [`02_variations.ipynb`](02_variations.ipynb) | **Advanced composition on the maintained [code-review example](../code_review_agent/README.md).** Build on its `base_config()` to run the same agent across harnesses (Hermes Agent, Deep Agents, Codex, Claude) and to vary configuration — skills, MCP servers, models, and NeMo Relay telemetry. |

Read them in order. The quickstart teaches the mental model standalone; the
variations notebook shows advanced composition against a real, maintained agent.

## Prerequisites

- Build the SDK and native extension from the repo root: `just build-all`. This
  alone is enough to execute both notebooks top to bottom with unavailable live
  harnesses skipped.
- To actually *run* a harness (rather than just inspect its config), that
  harness's adapter and credentials must be present:
  - **Hermes Agent** (both notebooks): follow the
    [Hermes Agent adapter setup](../../adapters/hermes/README.md) with Python
    3.11-3.13 and set `NVIDIA_API_KEY`. The setup cells auto-detect
    `.tmp/hermes-venv`.
  - **Deep Agents, Codex, Claude** (variations notebook): the matching adapter
    installed in the NeMo Fabric environment, plus that harness's credentials
    (`NVIDIA_API_KEY` for Deep Agents; an existing ChatGPT or provisioned API-key
    login for Codex; `ANTHROPIC_API_KEY` for the documented Claude run).
    Relay-enabled Hermes Agent and Deep Agents runs also need the `nemo-relay`
    Python package in the selected adapter environment.
- `NVIDIA_API_KEY` is loaded from a gitignored `.env` at the repo root if present.

Every live cell checks the prerequisites it can inspect. Missing key-based
credentials skip their harness and still show its resolved plan. Codex
authentication is validated only after its adapter starts, so an attempted
Codex authentication failure is collected as an execution failure. After every
variant is attempted, the notebook raises on any attempted-run failure. The
Relay cell also requires a succeeded result, a Relay telemetry reference, and a
nonempty, parseable ATOF trace.

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
- Other examples in this repo [`examples/`](../README.md).

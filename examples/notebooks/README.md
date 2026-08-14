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
  is enough to execute the setup, planning, and configuration-inspection cells.
  Live runs require the prerequisites below.
- To actually *run* a harness (rather than just inspect its config), that
  harness's adapter and credentials must be present:
  - **Hermes Agent** (both notebooks): from the repository root, run
    `just install-hermes-agent`, then set `NVIDIA_API_KEY`. The recipe checks out
    the pinned Hermes Agent source and synchronizes it into the project
    environment. End users installing outside a source checkout must follow the
    [Hermes Agent installation guide](https://hermes-agent.nousresearch.com/docs/installation);
    Hermes Agent 0.20 and later is no longer installable from PyPI.
  - **Deep Agents, Codex, Claude** (variations notebook): the matching adapter
    installed in the NeMo Fabric environment, plus that harness's credentials
    (`NVIDIA_API_KEY` for Deep Agents; an existing ChatGPT or provisioned API-key
    login for Codex; `ANTHROPIC_API_KEY` for the documented Claude run).
    Relay-enabled Hermes Agent and Deep Agents runs also need the `nemo-relay`
    Python package in the selected adapter environment.
- API keys are loaded from a gitignored `.env` at the repo root if present.
  Each notebook securely prompts for any missing keys it uses. Press Enter to
  skip a key and the live harness calls that require it.

Every live cell checks the prerequisites it can inspect. Missing key-based
credentials skip their harness and still show its resolved plan. Codex
authentication is validated only after its adapter starts, so an attempted
Codex authentication failure is collected as an execution failure. After every
variant is attempted, the notebook raises on any attempted-run failure. The
Relay cell also requires a succeeded result, a Relay telemetry reference, and a
nonempty, parseable ATOF trace.

## Google Colab

To open these notebooks in a Google Colab environment, you can click the following link: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/NeMo-Fabric)


## Jupyter Lab

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

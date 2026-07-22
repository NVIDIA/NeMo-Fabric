<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Code Review Agent

This example reviews the repository under `repos/my-service`. It constructs a
complete `FabricConfig` with the public Pydantic models and passes it directly
to the Python SDK. Variants are independent deep copies of that config.

Each variant is an independent Python factory that returns a complete config.

## Set up

Run commands from the repository root. Build NeMo Fabric and install its Python SDK
into the project virtual environment:

```bash
just build-all
```

The default variant uses Hermes Agent with an NVIDIA-hosted model. Follow the
[Hermes Agent quick start](../../README.md#quick-start-hermes-agent) to install
Hermes Agent, then set `NVIDIA_API_KEY` and `ADAPTER_PYTHON` as described there.

The config also demonstrates a harness-native GitHub MCP server. Set
`GITHUB_MCP_URL` when you want to use that server; the review prompt below does
not require it.

## Inspect the plan

Resolve the default config without starting a runtime or calling a model:

```bash
.venv/bin/python -m examples.code_review_agent --plan
```

The JSON output shows the selected adapter, resolved workspace, capabilities,
environment, and telemetry plan.

## Run the agent

Run one request through the default Hermes Agent variant:

```bash
.venv/bin/python -m examples.code_review_agent \
  --input "Reply with exactly: fabric works"
```

The command prints a normalized `RunResult` and writes runtime artifacts under
`examples/code_review_agent/artifacts/hermes/`.

## Choose a variant

The entrypoint exposes complete harness configs defined in
[`config.py`](./config.py):

| Variant | Command option | Additional setup |
| --- | --- | --- |
| Hermes Agent | `--variant hermes` | Installed [Hermes Agent adapter requirements](../../adapters/hermes/README.md) and `NVIDIA_API_KEY`|
| Codex | `--variant codex` | Installed [Codex adapter](../../adapters/codex/README.md) and an existing ChatGPT or API key login |
| Claude | `--variant claude` | Installed [Claude adapter requirements](../../adapters/claude/README.md) and `ANTHROPIC_API_KEY` |
| Deep Agents | `--variant deepagents` | Installed [Deep Agents adapter requirements](../../adapters/deepagents/README.md) and `NVIDIA_API_KEY` |

Add `--relay` to any variant to enable the Relay ATOF and ATIF configuration:

Relay requirements depend on the selected adapter. The Codex and Claude
adapters require an external `nemo-relay` CLI in the supported `0.6.x` range;
the Python package named `nemo-relay` does not install that command.

```bash
.venv/bin/python -m examples.code_review_agent \
  --variant hermes \
  --relay \
  --input "Review calculator.py"
```

Use `--plan` with these options to inspect a variant before running it.
Use `--show-output` to print the adapter's `output.response` value on the final
line after the normalized result.

## Compose configs in Python

The config module also provides environment, MCP, and telemetry functions for
application-owned composition:

```python
from examples.code_review_agent import (
    BASE_DIR,
    hermes_config,
    with_opensandbox,
    with_relay,
)

config = hermes_config()
relay_config = with_relay(config)
sandbox_config = with_opensandbox(config)
```

Each function returns a deep copy. `config`, `relay_config`, and
`sandbox_config` can therefore be planned or run independently with
`base_dir=BASE_DIR`.

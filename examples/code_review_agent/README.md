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
[Hermes Agent quick start](../../README.md#quick-start-hermes-agent) through the
environment installation steps, which create `.tmp/hermes-venv`, then set
`NVIDIA_API_KEY`. Select that interpreter only for Hermes commands:

```bash
ADAPTER_PYTHON="$PWD/.tmp/hermes-venv/bin/python" \
  .venv/bin/python -m examples.code_review_agent \
  --input "Reply with exactly: fabric works"
```

Do not export a Hermes-only `ADAPTER_PYTHON` globally before running the Codex,
Claude, or Deep Agents variants. Those variants use the project interpreter
unless you select another interpreter explicitly.

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
ADAPTER_PYTHON="$PWD/.tmp/hermes-venv/bin/python" \
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
| Hermes Agent | `--variant hermes` | Created the environment from the [Hermes Agent quick start](../../README.md#quick-start-hermes-agent) and set `NVIDIA_API_KEY` |
| Codex | `--variant codex` | Installed [Codex adapter](../../adapters/codex/README.md) and an existing ChatGPT or API key login |
| Claude | `--variant claude` | Installed [Claude adapter requirements](../../adapters/claude/README.md) and `ANTHROPIC_API_KEY` |
| Deep Agents | `--variant deepagents` | Installed [Deep Agents adapter requirements](../../adapters/deepagents/README.md) and `NVIDIA_API_KEY` |

Add `--relay` to any variant to enable the Relay ATOF and ATIF configuration:

Relay requirements depend on the selected adapter. The Codex and Claude
adapters require an external `nemo-relay` CLI in the supported `0.6.x` range;
the Python package named `nemo-relay` does not install that command. Hermes
Agent and Deep Agents require the Relay Python package in their selected
adapter environment. Refer to the
[installation guide](../../docs/getting-started/install.mdx#install-nemo-relay)
for the current compatibility requirements.

```bash
ADAPTER_PYTHON="$PWD/.tmp/hermes-venv/bin/python" \
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
    with_github_mcp,
    with_opensandbox,
    with_relay,
)

config = hermes_config()
relay_config = with_relay(config)
sandbox_config = with_opensandbox(config)
github_config = with_github_mcp(config)
```

Each function returns a deep copy. The four configs can therefore be planned or
run independently with `base_dir=BASE_DIR`. Set `GITHUB_MCP_URL` before running
`github_config`; it maps the server into the selected harness's native MCP
configuration. The default smoke does not configure or contact that server.

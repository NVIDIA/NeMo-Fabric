<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Code Review Agent

This example reviews the repository under `repos/my-service`. It constructs a
complete `FabricConfig` with the public Pydantic models and passes it directly
to the Python SDK. Variants are independent deep copies of that config.

Each variant is an independent Python factory that returns a complete config.
The Pi and Deep Agents variants intentionally share the same model, code-review
instruction, workspace, and default skill. This makes it possible to compare
the harnesses while keeping the agent intent fixed.

## Set up

Run commands from the repository root. Build NeMo Fabric and its maintained
language packages:

```bash
just build-all
```

The default variant uses Hermes Agent. Check out the pinned Hermes Agent source
and synchronize it into the project environment:

```bash
just install-hermes-agent
```

Set `NVIDIA_API_KEY`, then run the default variant with the project interpreter:

```bash
.venv/bin/python -m examples.code_review_agent \
  --input "Reply with exactly: NeMo Fabric works"
```

Hermes Agent 0.20 and later is no longer installable from PyPI. End users
installing outside a source checkout must follow the
[Hermes Agent installation guide](https://hermes-agent.nousresearch.com/docs/installation).
If Hermes Agent uses a separate environment, set `ADAPTER_PYTHON` to that
environment's Python interpreter only for Hermes Agent commands.

The Pi variant uses the source-built Pi SDK adapter and requires Node.js
22.19.0 or newer. The preceding `just build-all` command builds it. To rebuild
only the TypeScript packages, run:

```bash
just build-typescript
```

The example discovers `adapters/typescript/pi/pi.fabric-adapter.json` directly
from the source checkout. Set `NVIDIA_API_KEY` before running the Pi variant.

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
  --input "Reply with exactly: NeMo Fabric works"
```

The command prints a normalized `RunResult` and writes runtime artifacts under
`examples/code_review_agent/artifacts/hermes/`.

## Choose a variant

The entrypoint exposes complete harness configs defined in
[`config.py`](./config.py):

| Variant | Command option | Additional setup |
| --- | --- | --- |
| Hermes Agent | `--variant hermes` | Ran `just install-hermes-agent` and set `NVIDIA_API_KEY` |
| Codex | `--variant codex` | Installed [Codex adapter](../../adapters/codex/README.md) and an existing ChatGPT or API key login |
| Claude | `--variant claude` | Installed [Claude adapter requirements](../../adapters/claude/README.md) and `ANTHROPIC_API_KEY` |
| Deep Agents | `--variant deepagents` | Installed [Deep Agents adapter requirements](../../adapters/deepagents/README.md) and `NVIDIA_API_KEY` |
| Pi | `--variant pi` | Built the [Pi adapter](../../adapters/typescript/pi/README.md) with Node.js 22.19 or newer and set `NVIDIA_API_KEY` |

Relay is available only for supported variants. Requirements depend on the
selected adapter. The Codex and Claude
adapters require a `nemo-relay` CLI in the `>=0.7.2,<0.8` range. NeMo Fabric's
`relay` extra does not install the CLI. Hermes Agent and Deep Agents require the
Relay Python package in their selected adapter environment. Refer to the
[installation guide](../../docs/getting-started/install.mdx#install-nemo-relay)
for the current compatibility requirements.

For example, run the Hermes Agent variant with Relay:

```bash
.venv/bin/python -m examples.code_review_agent \
  --variant hermes \
  --relay \
  --input "Review calculator.py"
```

Use `--plan` with these options to inspect a variant before running it.
Use `--show-output` to print the adapter's `output.response` value on the final
line after the normalized result.

Run the Pi variant with:

```bash
.venv/bin/python -m examples.code_review_agent \
  --variant pi \
  --show-output \
  --input "Read calculator.py and review it for correctness risks. Cite the file and line you inspected."
```

The Pi variant loads the example's explicit `skills/code-review` skill. Its
example-specific tool policy supports inspecting files without enabling shell
or editing capabilities. MCP and Relay are also not enabled; passing `--relay`
with `--variant pi` is rejected until the adapter supports that integration.

## Vary skills

Pi and Deep Agents load `skills/code-review` by default. Remove the default
skill without changing the rest of the variant:

```bash
.venv/bin/python -m examples.code_review_agent \
  --variant pi \
  --no-skills \
  --show-output \
  --input "Read calculator.py and review it for correctness risks. Cite the file and line you inspected."
```

Replace the defaults with one or more skill directories by repeating
`--skill-path`. Relative paths resolve from `examples/code_review_agent`:

```bash
.venv/bin/python -m examples.code_review_agent \
  --variant pi \
  --skill-path ./skills/code-review \
  --skill-path ../../tests/fixtures/alternate \
  --plan
```

The options are intentionally generic rather than Pi-specific, so the same
skill selection can be used with another variant that supports Fabric skills.
`--skill-path` and `--no-skills` cannot be combined.

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
    with_skill_paths,
)

config = hermes_config()
relay_config = with_relay(config)
sandbox_config = with_opensandbox(config)
github_config = with_github_mcp(config)
no_skills_config = with_skill_paths(config)
```

Each function returns a deep copy. The configs can therefore be planned or run
independently with `base_dir=BASE_DIR`. Set `GITHUB_MCP_URL` before running
`github_config`; it maps the server into the selected harness's native MCP
configuration. The default smoke does not configure or contact that server.

`with_native_otel` supports the Codex and Deep Agents adapters. It raises
`ValueError` for Hermes Agent and Claude, whose native telemetry contracts do
not accept this configuration.

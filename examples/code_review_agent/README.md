<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Code Review Agent

This example uses NVIDIA NeMo Fabric to review the sample repository in
`repos/my-service`. Start with the default Hermes Agent, add the capabilities
you need, and then run the same task with another agent harness.

The example builds each configuration from the public Pydantic models. The
factory and composition functions return independent copies, so changing one
configuration does not affect another.

## Run the Default Demo

Run commands from the repository root. Build NeMo Fabric and its maintained
language packages, and then install the pinned Hermes Agent source:

```bash
just build-all
just install-hermes-agent
```

Set `NVIDIA_API_KEY`, then run the example:

```bash
.venv/bin/python -m examples.code_review_agent \
  --input "Review calculator.py" \
  --show-output
```

Since Hermes Agent is the default, the command does not need `--variant hermes`.
It prints the normalized run result followed by the agent response and writes
artifacts under `examples/code_review_agent/artifacts/hermes/`.

Hermes Agent 0.20 and later is not available from PyPI. For installation
outside this source checkout, follow the
[Hermes Agent installation guide](https://hermes-agent.nousresearch.com/docs/installation).
If Hermes Agent runs in a separate environment, set `ADAPTER_PYTHON` to that
environment's Python interpreter.

## Vary the Capabilities

The following options work with the default Hermes Agent. You can combine them
with another harness unless its subsection notes an exception.

### Inspect the Plan

Use `--plan` to inspect the resolved adapter, workspace, capabilities,
environment, and telemetry without starting a runtime:

```bash
.venv/bin/python -m examples.code_review_agent --plan
```

### Change the Skills

The default configuration loads `skills/code-review`. Remove the skill without
changing the rest of the configuration:

```bash
.venv/bin/python -m examples.code_review_agent \
  --no-skills \
  --plan
```

Use `--skill-path <PATH?` to replace the default with another skill. Repeat the
option to add multiple directories. Relative paths resolve from
`examples/code_review_agent`; `--skill-path` and `--no-skills` cannot be
combined.

### Enable Relay Telemetry and Streaming

Install Relay as described in the
[Relay installation guide](../../docs/getting-started/install.mdx#install-nemo-relay),
and then enable Relay telemetry to collect Agent Trajectory Observability
Format (ATOF) stream records:

```bash
.venv/bin/python -m examples.code_review_agent \
  --relay \
  --stream \
  --input "Review calculator.py"
```

The command collects Relay ATOF records and then prints one JSON document with
the records and the separate terminal result after the stream completes. Omit
`--stream` to retain Relay artifacts without including the records in console
output.

### Compose Capabilities in Python

Use the helpers in [`config.py`](./config.py) when your application owns the
configuration:

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
skill_config = with_skill_paths(config, "./skills/code-review")
mcp_config = with_github_mcp(config)
relay_config = with_relay(config)
sandbox_config = with_opensandbox(config)
```

Set `GITHUB_MCP_URL` before running a configuration that uses the GitHub MCP
server. The default demo does not contact that server. Pass `base_dir=<BASE_DIR>`
when planning or running these configurations.

The module also includes Relay OpenTelemetry and OpenInference examples.
Adapter-native OpenTelemetry is available for Codex and Deep Agents.

## Vary the Agent Harness

Use the same entry point, workspace, and review request with another harness by
adding `--variant`. The value for each harness appears in parentheses below.
Keep any capability options from the previous section that the selected
harness supports.

Codex and Claude omit the default code-review skill; add
`--skill-path ./skills/code-review` to retain it. For Relay, Codex and Claude
require the NeMo Relay CLI, while Hermes Agent and Deep Agents use the Relay
Python package. Additional requirements appear in the corresponding
subsections.

For example, after installing Deep Agents, this command keeps the default skill
and Relay configuration while changing the harness:

```bash
.venv/bin/python -m examples.code_review_agent \
  --variant deepagents \
  --relay \
  --input "Review calculator.py" \
  --show-output
```

### Hermes Agent (`hermes`)

Hermes Agent is the baseline used by the default demo. Specify the variant only
when an explicit configuration is useful, such as `--variant hermes --plan`.
Hermes supports the example's skills, MCP, and Relay configurations.

### Codex (`codex`)

Install and authenticate the [Codex adapter](../../adapters/python/codex/README.md).
This variant uses GPT-5.4.

### Claude (`claude`)

Install the [Claude adapter requirements](../../adapters/python/claude/README.md) and
set `ANTHROPIC_API_KEY`.

### Deep Agents (`deepagents`)

Install the
[Deep Agents adapter requirements](../../adapters/python/deepagents/README.md). This
variant uses the `NVIDIA_API_KEY` configured for the default demo.

### NVIDIA-labs Object Oriented Agents (NOOA) CodingAgent (`nooa`)

Follow the
[NOOA InteractiveAgent source instructions](../../external/nooa/docs/interactive-agent.md#run-from-source).
CodingAgent is a workflow target rather than a harness, but the example selects
it through the same `--variant` option. The variant discovers
`nvidia.nooa.coding-agent` and uses the `NVIDIA_API_KEY` configured for the
default demo.

Its Relay integration requires `nemo-relay>=0.7.2,<0.8`. The `--stream` option
collects Relay ATOF records; it is not native model-response streaming.

### Pi (`pi`)

Install Node.js 22.19 or later, and follow the
[Pi adapter source instructions](../../adapters/typescript/pi/README.md). The
initial `just build-all` command builds the Pi adapter.

This variant adds an explicit `read` tool to the default code-review skill and
uses `NVIDIA_API_KEY`. Pi does not currently support Relay or MCP, so do not
use `--relay` or a configuration created by `with_github_mcp`.

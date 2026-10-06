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
export ADAPTER_PYTHON="$PWD/.venv-hermes/bin/python"
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
  --variant deepagents \
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
    deepagents_config,
    hermes_config,
    with_github_mcp,
    with_opensandbox,
    with_relay,
    with_skill_paths,
)

config = hermes_config()
skill_config = with_skill_paths(config, "./skills/code-review")
mcp_config = with_github_mcp(config)
relay_config = with_relay(deepagents_config())
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
`--skill-path ./skills/code-review` to retain it. Relay configurations for
Codex, Claude, and Pi require a NeMo Relay CLI in the `>=0.9,<0.10` range.
Pi also requires its Relay Pi extension. Hermes Agent and Deep Agents require
the `nemo-relay>=0.9,<0.10` Python package. Cline does not support Relay
telemetry in this initial adapter.
Additional requirements appear in the corresponding subsections.

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
Hermes supports the example's skills, MCP, and Relay configurations when
installed from the merged upstream revision used by `just install-hermes-agent`.

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

Install `nemo-fabric[nooa]` and follow the
[NOOA InteractiveAgent instructions](../../adapters/python/nooa/docs/interactive-agent.md).
CodingAgent is a workflow target rather than a harness, but the example selects
it through the same `--variant` option. The variant discovers
`nvidia.nooa.coding-agent` and uses the `NVIDIA_API_KEY` configured for the
default demo.

Its Relay integration requires `nemo-relay>=0.9,<0.10`. The `--stream` option
collects Relay ATOF records; it is not native model-response streaming.

### OpenClaw (`openclaw`)

Install Node.js and OpenClaw, then install the [OpenClaw adapter](../../adapters/python/openclaw/README.md). This variant uses the `NVIDIA_API_KEY` configured for the default demo and retains the default code-review skill. OpenClaw does not currently support Relay telemetry.

### OpenHands (`openhands`)

On Python 3.12 or later, install the tested OpenHands SDK and tools, then install
the NeMo Fabric extra:

```bash
pip install "openhands-sdk==1.50.0" "openhands-tools==1.50.0"
pip install "nemo-fabric[openhands]"
```

The NeMo Fabric extra installs the adapter, but not the OpenHands packages. This
variant uses the `NVIDIA_API_KEY` configured for the default demo, maps the
terminal and file editor tools, and retains the default code-review skill.
The 1.50.0 package pair is validated with NVIDIA NIM. OpenHands does not
currently support Relay telemetry.

When you run the source checkout without activating its virtual environment,
select the same interpreter for the adapter subprocess:

```bash
export ADAPTER_PYTHON="$PWD/.venv/bin/python"
```

### Cline (`cline`)

Install Node.js 22.19 or newer and follow the
[Cline adapter installation instructions](../../adapters/typescript/cline/README.md).
The source-tree adapter build is included in `just build-all`, but that command
does not install the caller-owned Cline SDK. Install it separately from the
repository root:

```bash
npm install --prefix adapters/typescript \
  --workspace nemo-fabric-adapters-cline \
  --include-workspace-root \
  --no-save \
  --package-lock=false \
  --ignore-scripts \
  --no-audit \
  --no-fund \
  @cline/sdk@0.0.83
```

This variant maps the default skill and the `read_files`, `search_codebase`,
and `skills` Cline tools, and uses `NVIDIA_API_KEY`.

Inspect its plan with:

```bash
.venv/bin/python -m examples.code_review_agent --variant cline --plan
```

Run the review with an absolute path because Cline's `read_files` tool requires
one:

```bash
.venv/bin/python -m examples.code_review_agent \
  --variant cline \
  --input "Review $(realpath examples/code_review_agent/repos/my-service/calculator.py)" \
  --show-output
```

The variant uses `instructions.system.mode: replace`, which discards Cline's
native system prompt. Applications that need workspace-location context must
include it in the replacement instruction or provide absolute paths in user
input.

The initial Cline adapter does not support Relay telemetry or streaming.

### Kilo Code (`kilo`)

Install Node.js 22.19 or later, install the source adapter dependencies, and
then install the caller-owned Kilo CLI:

```bash
just install-typescript-kilo
just build-typescript
npm install --prefix adapters/typescript \
  --workspace nemo-fabric-adapters-kilo \
  --include-workspace-root \
  --no-save \
  --package-lock=false \
  --ignore-scripts \
  --no-audit \
  --no-fund \
  @kilocode/cli@7.7.12
```

The variant maps the NVIDIA model endpoint, replacement review instruction,
read-oriented tool policy, maximum turns, and default code-review skill. Kilo
Code does not currently support Relay through this adapter.

To use a different OpenAI-compatible model endpoint, override the model ID,
URL, and credential environment variable. For a trusted private IPv4 HTTP
server, add the explicit opt-in shown below. If the server does not require
authentication,
set the environment variable to any nonempty placeholder:

```bash
LOCAL_MODEL_KEY=local .venv/bin/python -m examples.code_review_agent \
  --variant kilo \
  --model "my-org/my-model" \
  --base-url "http://192.168.1.10:8000/v1" \
  --api-key-env LOCAL_MODEL_KEY \
  --allow-insecure-http-model-endpoint \
  --input "Review calculator.py" --show-output
```

The opt-in sends the credential and review content over an unencrypted network
connection. Use a dedicated credential for that server, or a placeholder when
it requires none. Public IP addresses and hostnames remain blocked; omit the
opt-in for HTTPS or loopback endpoints.

### Pi (`pi`)

Install Node.js 22.19 or later, and follow the
[Pi adapter source instructions](../../adapters/typescript/pi/README.md). The
initial `just build-all` command builds the Pi adapter.

This variant adds an explicit `read` tool to the default code-review skill and
uses `NVIDIA_API_KEY`. Pi supports normalized stdio and streamable HTTP MCP
servers.

For Relay telemetry, install `nemo-relay-cli-bin>=0.9.0,<0.10.0` as described in the
[Pi adapter instructions](../../adapters/typescript/pi/README.md#install-nemo-relay)
and pass the Relay Pi extension path explicitly:

```bash
.venv/bin/python -m examples.code_review_agent \
  --variant pi \
  --relay \
  --stream \
  --pi-relay-extension-path /path/to/NeMo-Relay/crates/cli/assets/pi-extension \
  --input "Review calculator.py"
```

The Pi variant uses the default embedded collector to collect per-invocation
model-turn ATOF records for successful Relay redirects, then prints one JSON
document containing `atof_records` and the separate terminal `result`. Relay
retains redirect-decision marks in configured ATOF artifacts, while Pi's startup
`model_redirect` marks are not included in `atof_records`. Omit `--stream` to
retain Relay artifacts without collecting records for that JSON output.

### Qwen Code (`qwen`)

Install Node.js 22.19 or later and the pinned
[Qwen adapter dependencies](../../adapters/typescript/qwen/README.md), then build
the adapter:

```bash
just install-typescript-qwen
npm run build --prefix adapter-contract/typescript
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-common
npm run build --prefix adapters/typescript --workspace nemo-fabric-adapters-qwen
.venv/bin/python -m examples.code_review_agent --variant qwen --plan
```

Set `NVIDIA_API_KEY` to run the example. The variant keeps the code-review skill,
uses Qwen's non-interactive default approval mode, and blocks shell and edit
tools. The variant does not configure an MCP server or Relay telemetry.

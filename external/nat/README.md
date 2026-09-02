<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Agent Toolkit Reference Adapter for NVIDIA NeMo Fabric

This source-only adapter runs a NeMo Agent Toolkit workflow behind
the NeMo Fabric lifecycle contract. It is a third-party adapter reference, not a
bundled NeMo Fabric adapter or a published package.

The implementation constructs NeMo Agent Toolkit configuration in memory from
the typed southbound `AgentConfig` dataclass; it does not read a NeMo Agent
Toolkit YAML file, parse the northbound `FabricConfig`, or depend on Pydantic for
its contract boundary.

## Configuration Boundary

NeMo Fabric owns portable configuration. `FabricConfig.workflow.target_id`
selects a registered target. Planning projects that target's entry point and
the consumer's settings into `AgentConfig.workflow`. `tools.definitions`
supplies named functions and function groups that the adapter resolves as
installed NeMo Agent Toolkit components.

| `AgentConfig` field | NeMo Agent Toolkit configuration |
| --- | --- |
| `models.<role>` | `llms.<role>`; every NeMo Fabric model-role name is preserved |
| `instructions.system` | `append` only; maps to the built-in `react_agent` workflow `additional_instructions`. Other modes and workflow types are rejected. |
| `workflow.entrypoint.kind=factory` | Resolve a NeMo Fabric-defined agent intent |
| `workflow.entrypoint.ref=fabric.agent.react` | NeMo Agent Toolkit `react_agent` workflow factory |
| `workflow.settings` | Remaining `workflow` component fields |
| `tools.definitions.<name>` with `kind=function` | NeMo Agent Toolkit `functions.<name>`; `ref` becomes `_type` |
| `tools.definitions.<name>` with `kind=function_group` | NeMo Agent Toolkit `function_groups.<name>`; `ref` becomes `_type` |
| `mcp.servers.<name>` | Generated `mcp_client` function group named `<name>` |
| `tools.enabled`, `tools.blocked` | Effective native NeMo Agent Toolkit workflow tool selection |

Configurations that set `instructions.system` must now select `mode="append"`
explicitly. Earlier adapter versions accepted an omitted mode while mapping the
instruction to additive `additional_instructions`; requiring the explicit mode
removes that ambiguity. The adapter rejects the contract default, `replace`,
because the built-in `react_agent` workflow does not expose a native replacement
seam.

The adapter loads installed `nat.components` entry points before validating the
generated configuration with NeMo Agent Toolkit. A custom function or function
group is supplied as an installed NeMo Agent Toolkit component package and
selected by `tools.definitions.ref`. No Python callable crosses the configuration
contract. A target package can publish a different workflow settings schema
without changing this shared NeMo Agent Toolkit adapter.

At runtime, `start` loads components, enters one `WorkflowBuilder`, creates a
`SessionManager` with that shared builder, and retains both resources. Each
`invoke` opens a session from the retained manager, enters `session.run(...)`,
and awaits `runner.result()`. That runner path already invokes NeMo Agent Toolkit
functions asynchronously while preserving the toolkit's concurrency, context,
tracing, and lifecycle behavior; the adapter does not call a workflow function's
lower-level `ainvoke()` method directly.

Native OpenAI streaming is available when the retained `SessionManager` reports
`ChatResponseChunk` as the workflow's streaming output schema. The adapter is
tested against NeMo Agent Toolkit 1.7.0 and 1.8.0; both versions' shared ReAct
registrations declare that schema. The adapter checks the schema rather than
branching on a workflow name, opens one NeMo Agent Toolkit session and one run,
and consumes `runner.result_stream(to_type=ChatResponseChunk)` exactly once. It
serializes and forwards the OpenAI Chat Completions chunks in order. `invoke`
serializes the terminal NeMo Agent Toolkit `runner.result()` object into
`result.output["response"]`. By contrast, the terminal NeMo Fabric
`invoke_openai_stream` result contains the concatenated string `delta.content`
values for choice index `0` in
`result.output["response"]`; empty and usage-only streams complete with an empty
response. The adapter does not add SSE framing, a `[DONE]` marker, or a synthetic
finish chunk. A workflow without the required schema returns
`nat_openai_stream_unsupported_schema` before NeMo Agent Toolkit opens a session.

`stop` shuts down the session manager and exits the builder context. This first
reference does not claim cancellation, service, or live-update support.

## MCP Tool Filters

The adapter consumes the `AgentConfig.mcp.servers` entries, including the
normalized per-server filters. NeMo Fabric MCP tool names remain bare
server-local names; NeMo Agent Toolkit exposes a selected member as
`<server>__<tool>`.

For `stdio` servers, NeMo Fabric expands `$VAR` references in `url` and forwards
the result as the literal command without shell parsing. Structured `args` and
`env` are forwarded to NeMo Agent Toolkit. Only `url` supports variable
expansion; `args` and `env` values are literal.

| NeMo Fabric server policy | Generated NeMo Agent Toolkit function group |
| --- | --- |
| `allowed_tools` omitted and `blocked_tools=[]` | No `include` or `exclude`; expose all discovered tools |
| Nonempty `allowed_tools` only | `include=allowed_tools` |
| `allowed_tools` omitted and nonempty `blocked_tools` | `exclude=blocked_tools` |
| Both lists configured | `include=allowed_tools`; NeMo Fabric requires `blocked_tools` to be disjoint, so those names are already outside the allowlist |
| `allowed_tools=[]` | Omit the generated group; expose no tools from that server |

NeMo Agent Toolkit rejects a function group that sets both `include` and
`exclude`, so the adapter emits only `include` whenever an allowlist is present.
NeMo Fabric rejects blank names and an allow/block overlap before adapter
startup. A nonempty generated MCP group is added to workflows that expose
`tool_names`; callers do not repeat portable MCP servers in `harness.settings`.
A workflow implementation that requires at least one tool can still reject a
configuration whose effective tool set is empty.

Per-server MCP filters and root `tools.enabled` or `tools.blocked` solve
different problems. MCP filters select members within one server. Root tool
policy selects across the effective native NeMo Agent Toolkit tool surface.

## Development Bootstrap

This directory intentionally has no package metadata. For this source-only
example, use one Python environment for NeMo Fabric, the common adapter host,
NeMo Agent Toolkit, and every NeMo Agent Toolkit component referenced by the
config:

```bash
uv pip install \
  nemo-fabric-adapter-contract \
  nemo-fabric-adapters-common \
  nvidia-nat-core \
  nvidia-nat-langchain \
  nvidia-nat-mcp
export PYTHONPATH="$PWD/external/nat/src${PYTHONPATH:+:$PYTHONPATH}"
```

`PYTHONPATH` is a development bootstrap limitation, not the target installation
contract. The example config discovers `nat.fabric-adapter.json` and the
adjacent target records through an explicit local path.

The calculator example starts its source-only MCP server over stdio, so it does
not require a separately managed endpoint. Run the typed `FabricConfig` example:

```bash
uv run python external/nat/examples/calculator.py \
  --base-dir "$PWD/.tmp/nat-reference"
```

The email-phishing example uses the NeMo Agent Toolkit example component
registered by `nat_email_phishing_analyzer`. Install that component from a NeMo
Agent Toolkit checkout, then run the example:

```bash
uv pip install -e \
  "<path-to-nat-checkout>/examples/evaluation_and_profiling/email_phishing_analyzer"
uv run python external/nat/examples/email_phishing.py \
  --base-dir "$PWD/.tmp/nat-reference"
```

Both examples accept `--plan` to inspect the resolved plan without starting the
runtime. They use Python `FabricConfig` objects only; no YAML is involved.

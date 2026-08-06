<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric LangGraph Reference Adapter

This source-only reference adapter exercises the public NVIDIA NeMo Fabric
adapter contract without becoming a bundled adapter or a published package. It
defines a factory entry-point contract that a third-party adapter can adopt and
extend.

It translates the supported normalized model, system-instruction, tool-policy,
MCP, and MCP-filter fields into explicit factory arguments. Graph topology,
application-owned tools, checkpointers, and state remain application concerns.
It does not claim skill support because it does not define how a skill path
becomes LangGraph behavior.

## Workflow Contract

The descriptor owns the `workflow` schema. Planning validates the factory kind,
the `module:factory` reference syntax, and JSON-compatible factory settings. It
does not import application code or compile a graph.

| Field | Contract |
| --- | --- |
| `workflow.entrypoint.kind` | `langgraph_factory` |
| `workflow.entrypoint.ref` | An importable `module:factory` reference resolved from `base_dir` at runtime |
| `workflow.settings` | Application keyword arguments, plus the `llm_name` and `tool_names` selectors |

The adapter reserves three factory arguments and rejects them in
`workflow.settings`: `model_config`, `system_instruction`, and `mcp_servers`.
It builds them from the normalized configuration before calling the selected
factory.

| NeMo Fabric input | Factory argument |
| --- | --- |
| `models.<role>` selected by `workflow.settings.llm_name` | `model_config` |
| `instructions.system` | `system_instruction` |
| Harness-native `mcp.servers.<name>` | `mcp_servers` |
| `tools.enabled`, `tools.blocked` | Effective `tool_names` |

`workflow.settings.tool_names` declares the complete set of application tool
names the factory may use. When a root tool policy is configured, the adapter
validates every selector against that list and passes only the effective names.
It also passes only MCP servers whose names remain selected. Per-server MCP
allowlists and blocklists remain on the injected server entries for the factory
to enforce while it discovers MCP tools.

The root `tools` contract is a selection policy; it is not a registry of Python
tool definitions. A factory can own an application tool such as
`current_timezone`, but this reference does not invent a portable definition
format for it. Similarly, `skills.paths` remains unsupported and is rejected at
planning time.

At runtime start, the adapter imports the configured factory, calls it once,
and calls `compile()` on the returned uncompiled graph. The compiled graph is
retained for the NVIDIA NeMo Fabric runtime and receives the raw `input` value
for each invocation. It must expose `ainvoke(input)` or `invoke(input)` and
return JSON-serializable output.

The factory owns graph topology, turns `model_config` into its native model
client, creates any application tools, opens MCP clients from `mcp_servers`,
and owns checkpointers and per-user state. A third-party adapter can define a
different kind, reference syntax, settings schema, or capability mapping in its
own descriptor.

## Development Bootstrap

This directory intentionally has no package metadata or discovery wiring. Use
one environment for NVIDIA NeMo Fabric, the common adapter host, LangGraph, and
the dependencies imported by the selected application factory:

```bash
uv pip install \
  nemo-fabric-adapters-common \
  langgraph \
  langchain-mcp-adapters \
  langchain-openai
export PYTHONPATH="$PWD/external/langgraph/src${PYTHONPATH:+:$PYTHONPATH}"
```

Stage the descriptor in the current agent-local discovery location:

```bash
mkdir -p .tmp/langgraph-reference/adapters/langgraph
cp external/langgraph/fabric-adapter.json \
  .tmp/langgraph-reference/adapters/langgraph/fabric-adapter.json
cp external/langgraph/examples/calculator.py \
  external/langgraph/examples/email_phishing.py \
  .tmp/langgraph-reference/
```

`PYTHONPATH` is a development bootstrap limitation, not a third-party adapter
installation contract.

## Calculator MCP Example

The calculator graph receives its `mcp_math` server through normalized
`mcp.servers`, opens its own Streamable HTTP MCP client, and enforces that
server's tool filters while it discovers tools. Its factory receives
`tool_names` after Fabric applies the root tool policy. The input carries the
application user ID, so application code can use it for state or authorization
without adding a Fabric-specific user contract.

Start the included MCP server at `http://127.0.0.1:9901/mcp`:

```bash
uv run python external/langgraph/examples/calculator_mcp.py
```

Then inspect the plan or invoke the graph:

```bash
uv run python external/langgraph/examples/calculator.py \
  --base-dir "$PWD/.tmp/langgraph-reference" \
  --plan

uv run python external/langgraph/examples/calculator.py \
  --base-dir "$PWD/.tmp/langgraph-reference"
```

The factory is `calculator:build_graph`. Its configuration selects `mcp_math`
in `workflow.settings.tool_names`; the server URL and transport remain in
portable `mcp.servers` rather than in application settings.

## Email-Phishing Analyzer Example

The email-phishing graph receives `model_config` for the `nim_llm` model role
and `system_instruction` from portable configuration. It owns the
OpenAI-compatible model client, reads only the credential variable named by the
selected model, and uses the NVIDIA Inference endpoint only after runtime
startup.

```bash
export NVIDIA_API_KEY=...
uv run python external/langgraph/examples/email_phishing.py \
  --base-dir "$PWD/.tmp/langgraph-reference" \
  --plan

uv run python external/langgraph/examples/email_phishing.py \
  --base-dir "$PWD/.tmp/langgraph-reference"
```

The `--plan` command verifies the descriptor-owned workflow contract and the
adapter capability claims without importing the factory or contacting the model
service. The second command imports and compiles the graph, then invokes the
configured model. The factory adapts the raw text input to the `EmailState`
mapping required by its graph.

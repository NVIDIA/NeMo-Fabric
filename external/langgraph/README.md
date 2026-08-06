<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric LangGraph Reference Adapter

This source-only reference adapter exercises the public NVIDIA NeMo Fabric
adapter contract without becoming a bundled adapter or a published package. It
defines a context-aware factory entry-point contract that a third-party adapter
can adopt and extend.

The adapter calls `build_graph(context)` with application-owned settings and
supported normalized resources. Graph topology, application-owned tools,
checkpointers, and state remain application concerns.

## Workflow Contract

The descriptor owns the `workflow` schema. Planning validates the factory kind,
the `module:factory` reference syntax, and JSON-compatible factory settings. It
does not import application code or compile a graph.

| Field | Contract |
| --- | --- |
| `workflow.entrypoint.kind` | `langgraph_factory` |
| `workflow.entrypoint.ref` | An importable `module:factory` reference resolved from `base_dir` at runtime |
| `workflow.settings` | Application-owned factory configuration, passed unchanged as `context.workflow_settings` |

The factory accepts one `context` argument. No setting name is reserved for
normalized resource delivery, so a graph can use names such as `model_config`
or `skill_paths` for its own configuration without colliding with the adapter.

| NeMo Fabric input | Factory context field |
| --- | --- |
| `models` | `context.models` |
| `instructions.system` | `context.instructions.system` |
| Harness-native `mcp.servers.<name>` | `context.mcp_servers` |
| `tools.enabled`, `tools.blocked` | `context.tools` |
| `skills.paths` | `context.skills` |

`workflow.settings.tool_names` declares the complete set of application tool
names the factory may use. When a root tool policy is configured, the adapter
validates every selector against that list and exposes only the effective names
in `context.tools`. It also exposes only MCP servers whose names remain
selected in `context.mcp_servers`. Per-server MCP allowlists and blocklists
remain on the server entries for the factory to enforce while it discovers MCP
tools.

The root `tools` contract is a selection policy; it is not a registry of Python
tool definitions. A factory can own an application tool such as
`current_timezone`, but this reference does not invent a portable definition
format for it. The email example treats each `context.skills` entry as either a
`SKILL.md` file or a directory containing one, then appends its content to the
system instructions. Other factories can define a different LangGraph-specific
skill representation.

At runtime start, the adapter imports the configured factory, calls it once,
and calls `compile()` on the returned uncompiled graph. The compiled graph is
retained for the NVIDIA NeMo Fabric runtime and receives the raw `input` value
for each invocation. It must expose `ainvoke(input)` or `invoke(input)` and
return JSON-serializable output.

The factory owns graph topology, selects a model from `context.models`, creates
any application tools, opens MCP clients from `context.mcp_servers`, interprets
`context.skills`, and owns checkpointers and per-user state. A third-party
adapter can define a different kind, reference syntax, settings schema, or
capability mapping in its own descriptor.

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
cp -R external/langgraph/examples/skills .tmp/langgraph-reference/skills
```

`PYTHONPATH` is a development bootstrap limitation, not a third-party adapter
installation contract.

## Calculator MCP Example

The calculator graph receives its `mcp_math` server through normalized
`mcp.servers`, opens its own Streamable HTTP MCP client, and enforces that
server's tool filters while it discovers tools. Its factory reads the selected
server from `context.mcp_servers` and the effective tool names from
`context.tools`. The input carries the application user ID, so application code
can use it for state or authorization without adding a Fabric-specific user
contract.

Start the included MCP server at `http://127.0.0.1:9901/mcp`:

```bash
uv run python external/langgraph/examples/calculator_mcp.py
```

This server blocks the terminal and must remain running. In a second terminal,
inspect the plan or invoke the graph:

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

The email-phishing graph selects `nim_llm` from `context.models` using its
application-owned `workflow.settings.llm_name`, reads
`context.instructions.system`, and loads the resolved `context.skills` paths
into its system instructions. It owns the OpenAI-compatible model client, reads
only the credential variable named by the selected model, and uses the NVIDIA
Inference endpoint only after runtime startup.

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

<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric LangGraph Reference Adapter

This source-only reference adapter exercises the public NVIDIA NeMo Fabric
adapter contract without becoming a bundled adapter or a published package. It
defines a factory entry-point contract that a third-party adapter can adopt and
extend.

The reference has no model, tool, MCP, skill, or telemetry translation. Those
are application-owned graph concerns. Its descriptor consequently declares no
normalized configuration support instead of accepting fields that it cannot
enforce.

## Workflow Contract

The descriptor owns the `workflow` schema. Planning validates the factory kind,
the `module:factory` reference syntax, and JSON-compatible factory settings. It
does not import application code or compile a graph.

| Field | Contract |
| --- | --- |
| `workflow.entrypoint.kind` | `langgraph_factory` |
| `workflow.entrypoint.ref` | An importable `module:factory` reference resolved from `base_dir` at runtime |
| `workflow.settings` | Keyword arguments forwarded unchanged to the factory |

At runtime start, the adapter imports the configured factory, calls it once,
and calls `compile()` on the returned uncompiled graph. The compiled graph is
retained for the NVIDIA NeMo Fabric runtime and receives the raw `input` value
for each invocation. It must expose `ainvoke(input)` or `invoke(input)` and
return JSON-serializable output.

The factory owns graph topology, model clients, tools, MCP connections,
checkpointers, and per-user state. A third-party adapter can define a different
kind, reference syntax, settings schema, or capability mapping in its own
descriptor.

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
cp external/langgraph/examples/calculator_graph.py \
  external/langgraph/examples/email_phishing_graph.py \
  .tmp/langgraph-reference/
```

`PYTHONPATH` is a development bootstrap limitation, not a third-party adapter
installation contract.

## Calculator MCP Example

The calculator graph is application code. It opens its own Streamable HTTP MCP
client when it invokes the selected tool. The input carries the application
user ID, so application code can use it for state or authorization without
adding a Fabric-specific user contract.

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

The factory is `calculator_graph:build_graph`; its `mcp_url` setting stays
inside `workflow.settings` because the reference does not translate a portable
MCP configuration.

## Email-Phishing Analyzer Example

The email-phishing graph owns its OpenAI-compatible model client. It reads the
credential variable named by its factory setting and uses the configured NVIDIA
Inference endpoint only after runtime startup.

```bash
export NVIDIA_API_KEY=...
uv run python external/langgraph/examples/email_phishing.py \
  --base-dir "$PWD/.tmp/langgraph-reference" \
  --plan

uv run python external/langgraph/examples/email_phishing.py \
  --base-dir "$PWD/.tmp/langgraph-reference"
```

The `--plan` command verifies the descriptor-owned workflow contract without
importing the factory or contacting the model service. The second command
imports and compiles the graph, then invokes the configured model.

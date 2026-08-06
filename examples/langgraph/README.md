<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LangGraph Examples

This directory contains two native LangGraph examples:

- `calculator_mcp.py` implements a per-user ReAct agent with a local
  `current_timezone` tool and a per-user streamable-HTTP MCP calculator client.
- `email_phishing_analyzer.py` implements a purpose-built state graph that returns
  a structured phishing assessment.

They use `meta/llama-3.1-70b-instruct` through the OpenAI-compatible NIM endpoint.
NVIDIA NeMo Fabric does not yet ship a LangGraph adapter. The YAML files are
validated by the examples and intentionally use `langgraph:` entry points; they
are not accepted by `FabricConfig` yet.

## Set Up the Python Environment

From the repository root, install the adapter and test dependency groups that
provide LangGraph, LangChain MCP adapters, and the local MCP server:

```bash
uv sync --no-default-groups --group adapters --group adapter-tests --group test
```

## Run the Calculator Example

Set an NVIDIA API key and start the included MCP server in one terminal:

```bash
export NVIDIA_API_KEY=<your-api-key>
.venv/bin/python -m examples.langgraph.mcp_math_server --port 9901
```

Run the calculator graph for a user in another terminal:

```bash
.venv/bin/python -m examples.langgraph.calculator_mcp \
  --user-id alice \
  --input "What is 9 multiplied by 7?"
```

Each user ID creates a separate graph, `InMemorySaver`, and
`MultiServerMCPClient`. Reusing a user ID resumes only that user's conversation.
The example passes `verbose` to LangGraph's `debug` setting.
`retry_parsing_errors` has no direct LangGraph equivalent, so it is retained in
the source configuration as an adapter requirement rather than silently applied.

## Run the Phishing Analyzer Example

Set an NVIDIA API key, then run the structured-output graph:

```bash
export NVIDIA_API_KEY=<your-api-key>
.venv/bin/python -m examples.langgraph.email_phishing_analyzer \
  --input "Provide your account and routing numbers to receive a refund."
```

The workflow has application-owned state (`body` and `assessment`) and returns a
JSON-safe result. The NIM binding uses OpenAI-compatible function calling to
produce the structured result. It does not use MCP tools or per-user
checkpointers, which makes it the contrasting workflow required to define a
reusable adapter boundary.

## Validate the Examples

Run the offline checks without an API key or a live MCP server:

```bash
.venv/bin/python -m pytest tests/examples/test_langgraph_examples.py
```

## Work Needed for a Full Adapter

The examples validate the two workflow shapes, but a generic adapter needs the
following additional work before it can support them through NVIDIA NeMo Fabric:

1. Add a LangGraph adapter package, descriptor, installation extra, wheel data,
   and lifecycle host. The package needs a `langgraph_factory` entry point, not an
   arbitrary compiled-graph import.
2. Define static and dynamic workflow contracts. A contract must validate
   `workflow.settings`, declare each accepted normalized capability, define model,
   instruction, local-tool, and MCP injection points, and contribute a digest to
   the run plan.
3. Map a selected Fabric model alias to an NIM `ChatOpenAI` binding. The adapter
   must validate the NIM base URL and credential environment variable before
   starting a runtime.
4. Build a policy-aware complete tool inventory. It must retain MCP origins such
   as `mcp_math__calculator__multiply`, enforce `tools.enabled` and
   `tools.blocked` for local and MCP tools, and reject unknown selectors.
5. Create MCP clients and graph/checkpoint resources with the correct lifetime.
   The calculator shows per-user state, but Fabric currently scopes a runtime to a
   Fabric runtime ID rather than an end-user identity. A full adapter needs an
   explicit, authenticated user/session contract before claiming per-user support.
6. Define the phishing graph's input/output projection, structured-output failure
   behavior, and retry semantics. `retry_parsing_errors` needs a documented
   LangGraph-equivalent policy that avoids retrying side-effecting tool calls.
7. Add conformance coverage for contract resolution, capability rejection,
   multi-invocation state isolation, MCP filtering and tool policy, normalized
   failures, cleanup, and opt-in NIM/MCP end-to-end tests.

These examples show why the factory-and-contract boundary is necessary: the
workflows share model and tool injection but do not share state, output projection,
retry behavior, or resource lifetime.

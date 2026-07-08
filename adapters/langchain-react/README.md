<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LangChain ReAct Adapter

Runs Fabric-native ReAct agents through LangGraph without NeMo Agent Toolkit
registration or config models. This adapter backs AALGO-277 optimize parity for
`react-optimize` and `calculator-optimize` equivalents.

Install Fabric with the adapter dependency before running it:

```bash
python3 -m pip install -e ".[langchain-react]"
```

## Platform Qwen / IGW smoke (no API key)

Point at a platform Inference Gateway OpenAI-compatible deployment:

```bash
cd NeMo-Fabric
uv sync --all-groups && uv pip install -e ".[langchain-react]"
export FABRIC_LANGCHAIN_PYTHON="$(pwd)/.venv/bin/python"

# Simple ReAct answer
cargo run -q -p fabric-cli -- run examples/react-optimize-agent \
  --profile qwen-igw-local \
  --input "In one short sentence, what is the capital of France?" \
  --show-output

# Calculator via text-mode ReAct (Qwen vLLM may not support native tool calling)
cargo run -q -p fabric-cli -- run examples/react-optimize-agent \
  --profile qwen-igw-local \
  --profile qwen-calculator-text \
  --input "What is 12 multiplied by 8? Use the multiply tool." \
  --show-output

# Full opt-in smoke
RUN_FABRIC_LANGCHAIN_REACT_E2E=1 .venv/bin/python tests/smoke_langchain_react_qwen.py
```

Override the gateway via `examples/react-optimize-agent/profiles/qwen-igw-local.yaml`
(`models.default.base_url`, `models.default.model`). Set `allow_empty_api_key: true`
and `api_key: not-used` when the gateway does not require auth.

Use `qwen-calculator-text` (not `calculator-native-tools`) when the deployed model
does not have vLLM `--enable-auto-tool-choice` enabled.

## Fabric Agent Package Shape

```yaml
schema_version: fabric.agent/v1alpha1
harness:
  adapter_id: nvidia.fabric.langchain.react
  settings:
    workflow:
      tool_names: [wiki, clock]
      llm_name: default
      parse_agent_response_max_retries: 3
      max_tool_calls: 15
      use_native_tool_calling: false
    tools:
      wiki:
        kind: wiki_search
      clock:
        kind: current_datetime
models:
  default:
    provider: openai
    model: nvidia-nemotron-3-nano-30b-a3b
    temperature: 0.0
    top_p: 1.0
    api_key_env: NVIDIA_API_KEY
```

Calculator optimize parity uses `use_native_tool_calling: true` and expands the
`calculator` function group into `add`, `subtract`, `multiply`, `divide`, and
`compare` tools.

## Per-Trial Model Overrides

The optimize harness applies Optuna suggestions through `RunRequest.overrides`
before the adapter constructs the OpenAI-compatible client:

```json
{
  "request_id": "trial-3-row-1",
  "input": "Who invented the telephone?",
  "overrides": {
    "temperature": 0.4,
    "top_p": 0.85
  }
}
```

Nested overrides under `models.default` are also accepted.

## Built-In Tools

These mirror the NAT `nvidia_nat_langchain` tool surface (plus NAT core datetime tools):

| Kind | NAT equivalent | Notes |
|------|----------------|-------|
| `wiki_search` | `wiki_search` | WikipediaLoader search |
| `exa_internet_search` | `exa_internet_search` | Requires `EXA_API_KEY` / `langchain-exa` |
| `code_generation` | `code_generation` | Uses configured workflow/tool LLM |
| `current_datetime` | NAT core `current_datetime` | Honors `RunRequest.context.timezone` |
| `current_timezone` | NAT core `current_timezone` | IANA timezone name |
| `function_group` / `calculator` | calculator function group | add/subtract/multiply/divide/compare |
| `tavily_internet_search` | removed in NAT 1.8 | Fails with migration message |

## Runtime Parity

Compared to NAT `react_agent` registration, the adapter also supports:

- `trim_messages` history limiting via `workflow.max_history`
- multi-turn `request.input` as chat `messages`
- per-trial `RunRequest.overrides` for `temperature` / `top_p`
- `GraphRecursionError` handling aligned with NAT streaming fallback text

## Telemetry

When `FABRIC_RELAY_ENABLED=true` (set by Fabric when the `relay` profile is
active and `runtime.artifacts` / `environment.artifacts` are configured), the
adapter:

1. Runs the LangGraph agent inside the NeMo Relay plugin context.
2. Attaches `NemoRelayCallbackHandler` so LLM/tool scopes are recorded.
3. Collects `relay_artifacts` **after** plugin shutdown so finalized ATIF files
   are included (`kind: atif` and `kind: atof`).
4. Returns those paths in the normalized adapter output; Fabric promotes them
   into `RunResult.artifacts` as `relay_atif` / `relay_atof`.

Use the bundled `profiles/relay.yaml` (includes artifact roots required for
Relay config materialization). Opt-in smoke:

```bash
uv pip install -e ".[langchain-react,relay]"
RUN_FABRIC_LANGCHAIN_REACT_ATIF_E2E=1 python tests/smoke_langchain_react_atif_qwen.py
```

## Maintaining The Adapter

Keep `fabric-adapter.json` aligned with `nemo_fabric_adapters.langchain_react.adapter:run`.
The ReAct runtime under `react/` is a NAT-free port of the relevant
`nvidia_nat_langchain` graph, parser, and prompt pieces.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Hermes Agent Adapter

This adapter runs Hermes Agent through its Python SDK.

## Install

The adapter supports Python 3.11 through 3.14. The pinned upstream Hermes
revision requires Python 3.14 and Relay 0.10 on supported platforms. Hermes
Agent 0.20 and later is not installable from PyPI. Follow the
[Hermes Agent installation guide](https://hermes-agent.nousresearch.com/docs/installation),
then install the NeMo Fabric packages into the same Python environment.

The following table shows which NeMo Fabric components each package expression
provides. None of these expressions installs Hermes Agent:

| Installation | Runtime | Adapter | Harness | NeMo Relay Python Package |
| --- | --- | --- | --- | --- |
| `pip install nemo-fabric nemo-fabric-adapters-hermes` | Yes | Yes | No | No |
| `pip install "nemo-fabric[hermes-agent]" "nemo-fabric-adapters-hermes[relay]"` | Yes | Yes | No | Yes |
| `pip install "nemo-fabric-adapters-hermes[full]"` | No | Yes | No | Yes |
| `pip install "nemo-fabric-adapters-hermes[relay]"` | No | Yes | No | Yes |
| `pip install nemo-fabric-adapters-hermes` | No | Yes | No | No |

Released Hermes Agent v2026.9.24 requires Relay 0.8.x. NeMo Fabric tests Relay
telemetry with the merged, unreleased upstream Hermes revision on Python 3.14.
Install the adapter's `relay` or `full` extra only with a Hermes checkout that
supports Relay 0.9 or 0.10; those extras conflict with the published Hermes
release. The adapter extras support `nemo-relay>=0.9,<0.11`. The pinned upstream
Hermes revision narrows that range to `nemo-relay>=0.10,<0.11` on supported
Python 3.14 platforms.
The adapter rejects an incompatible Hermes or Relay installation at startup
when Relay telemetry is enabled.

Create its isolated development environment:

```bash
just install-hermes-agent
export ADAPTER_PYTHON="$PWD/.venv-hermes/bin/python"
```

For split runtime and adapter environments, configure `ADAPTER_PYTHON` and use
matching NeMo Fabric release versions. Refer to the
[installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install#install-an-adapter-and-harness-without-the-runtime).

Relay telemetry and `Runtime.invoke_stream()` require the merged Hermes revision
above until an upstream release supports Relay 0.9 or 0.10. Ordinary `Runtime.invoke()`
also works with released Hermes in a separate environment.

## What It Maps

The adapter receives a normalized payload from NeMo Fabric and materializes a native Hermes Agent configuration for:

- selected model provider, model name, base URL, temperature, `top_p`, and
  per-model `max_tokens` through `models`;
- replacement `instructions.system` and `runtime.max_turns`;
- workspace and explicit environment variables through `environment`;
- invocation timeout through `runtime.timeout_seconds`;
- NeMo Fabric skills as external skill directories for Hermes Agent;
- NeMo Fabric MCP servers as Hermes Agent MCP server config;
- `tools.enabled` and `tools.blocked` as Hermes-native toolset selection and
  blocking policy;
- optional NeMo Relay telemetry plugin configuration.

Tool selectors are Hermes toolset names because that is the native policy
surface Hermes exposes.

During run-plan resolution, NeMo Fabric Core validates the
`models.<role>.top_p` (from 0 through 1) and positive
`models.<role>.max_tokens` fields against the Hermes descriptor before
runtime startup. Hermes reads these prevalidated values from `AgentConfig`. A
model-level `max_tokens` overrides the harness-level default for the selected
model role.

The descriptor validates the following `harness.settings` fields:

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `reasoning_config` | object | `{"effort": "none"}` | Configures Hermes model reasoning. The closed object accepts an optional `enabled` boolean and an optional `effort` value of `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`. |
| `plugins_enabled` | array of nonempty strings | `[]` | Enables Hermes plugins by identifier. Relay telemetry uses the generated `plugins.toml`; do not list the removed `observability/nemo_relay` plugin. |
| `save_trajectories` | boolean | `false` | Enables Hermes-native JSONL conversation trajectory saving. This is separate from normalized NeMo Fabric telemetry. |
| `max_tokens` | positive integer | `512` | Limits the number of tokens in each Hermes model response. |
| `terminal_timeout` | positive number | `60` | Limits a Hermes terminal operation in seconds. |

For example:

```python
from nemo_fabric import HarnessConfig

harness = HarnessConfig(
    adapter_id="nvidia.fabric.hermes",
    settings={
        "reasoning_config": {"enabled": True, "effort": "medium"},
        "plugins_enabled": ["disk-cleanup"],
        "save_trajectories": True,
        "max_tokens": 1024,
        "terminal_timeout": 90,
    },
)
```

Use `runtime.max_turns` to set the Hermes agent-loop budget. The adapter maps
that normalized field to `AIAgent.max_iterations`; `max_iterations` is not a
Hermes harness setting.

The adapter derives Hermes state from the NeMo Fabric artifact root and creates
a child under `runtimes/<runtime_id>`, so invocations in one NeMo Fabric runtime
share state without sharing config or the session database with another
runtime.

## Token Usage

Hermes counts its main model's tokens per session, and the adapter keeps one
session per runtime. The adapter reads the agent's session counters right before
each turn runs and again after it, so each result's `usage` is that turn's
change: `input_tokens` with cache reads and cache writes included,
`cached_input_tokens`, `output_tokens`, and `total_tokens`. A failure result
reports its own tokens too, and an invocation that raised or was cancelled
never adds its tokens to the next one. A counter that is missing, that is not
a non-negative integer below 2^64, or that went down because Hermes reset its
session, is left out for that invocation. Delegation is one of Hermes' default
toolsets: a subagent is its own agent with its own counters, and Hermes runs a
top-level delegation in the background, so the subagent's calls can finish
after the invocation returns. An invocation whose agent built a subagent has no
`usage`; later invocations report only their own turns. Hermes also leaves out
of its counters the summary request it makes when a turn reaches its iteration
limit.
The adapter adds the counts that this summary response reported, before Hermes
reads its text, so they count even when Hermes cannot read the response and
answers with its fallback text. An invocation whose summary response carried no
usage, or whose summary request failed before any response, has no `usage`.

Hermes keeps auxiliary model calls, such as context-compression summaries and
vision analysis, out of these counters. The adapter adds the usage of each
auxiliary response that Hermes' auxiliary client returns to the invocation
whose turn made the call, using the same rules as a main-loop response. A
response without usage, or one whose usage left out a count, leaves the
affected counts unknown. An auxiliary request that raised, including a
response the client could not parse, may have been billed, so the invocation
has no `usage`. So does a request that returned no response, and one that
had more than one observed dispatch, such as a streamed request Hermes retried
without streaming. The adapter counts Hermes' auxiliary dispatch
notifications. Hermes disables SDK retries for its OpenAI and Anthropic
auxiliary clients. Native transport shim, SDK and gateway retries below that
boundary are not observed.
The same holds for a request still running, or a response
Hermes has not recorded yet, when the invocation returns, and that usage is
never added to a later invocation.
Mixture-of-agents calls are already in the session counters and are not
counted twice. An invocation that delegates has no `usage`, because a
background subagent's main and auxiliary calls can finish after the
invocation returns; the adapter never adds a subagent's auxiliary calls to
the parent. Hermes' Anthropic auxiliary client drops the cache counts when it
converts a response, so the adapter reads the raw Anthropic usage before the
conversion. If that response reaches Hermes' usage recorder as another
object, such as one a Relay execution intercept replaced, the invocation has
no `usage`. Hermes' Codex Responses and Copilot ACP auxiliary clients fill
omitted counts with 0 when they convert the usage, so a response from one of
them leaves every count unknown. Auxiliary calls on the native Bedrock and
Gemini transports follow the same raw-usage check as main-loop calls. A plugin context engine may call
a model outside the auxiliary client, so an invocation in which one compressed
has no `usage`. The adapter sets
`auxiliary.title_generation.model_upgrade_enabled` to false, so Hermes keeps its
locally derived session title and makes no title model call; with Relay
enabled, title generation is off entirely. Fabric disables Hermes' automatic
post-turn memory and skill review with `skip_background_review`. If Hermes
nevertheless enters the background-review path during an invocation, the
adapter leaves that invocation's usage unknown, because the review forks an
agent that records its tokens outside the auxiliary client. The adapter wraps
`agent.aux_accounting.record_aux_usage`, the auxiliary client's
`_relay_sync_completion`, `_relay_async_completion`, `_unwrap_data_envelope`
and `_notify_aux_dispatch`,
`agent.anthropic_adapter.create_anthropic_message`, and the Anthropic
auxiliary client's `create` once per process, and the agent's
`_spawn_background_review` when it starts the agent. The runtime fails to
start if the installed Hermes no longer calls them by those names.

Hermes adds no tokens for a provider response that carried no usage, so an
invocation with such a response has no `usage`. For a response it rejects as
invalid, a reply that hit the output token limit, and a reply stopped by a
content filter, Hermes returns before adding the response's usage to its
counters. The adapter adds the counts that such a response reported itself,
using the provider and API mode that served it, and counts it only once when
Hermes records it after all. A streamed call that fails and is retried inside
Hermes' API call, such as an empty stream, never reaches these checks, so its
billed usage is not observed and the invocation can undercount. Hermes reads a
count that a response's usage left out as 0, so the adapter checks each
response's usage before Hermes normalizes it. A reported zero still counts as 0.
When a response left out its input count, the invocation's `usage` leaves out
`input_tokens` and `total_tokens`, and keeps `cached_input_tokens` when the
cache-read count is known. Anthropic Messages usage reports input without the
cache, so its input count also needs both cache fields. When a response left out
its output count, the `usage` leaves out `output_tokens` and `total_tokens`.
When a response sent no cache details, the `usage` leaves out
`cached_input_tokens`.

This check covers the usage that the OpenAI-compatible, Responses, and Anthropic
clients return. Hermes' native Bedrock Converse and Gemini transports fill a
count the provider left out with 0, so the adapter reads which counts their raw
usage reported before they do: Bedrock's input needs `inputTokens`, its cached
input `cacheReadInputTokens` and its output `outputTokens`, and a Converse
stream without its metadata event reports none; Gemini's input needs
`promptTokenCount`, its cached input `cachedContentTokenCount`, and its output
`candidatesTokenCount` or `thoughtsTokenCount`. Hermes' Anthropic transport
replaces a stream event's `usage: null`, which some Anthropic-compatible
gateways send, with zeros or with the start event's output count before the SDK
builds the final message, so the adapter cannot see it. A null start usage
leaves input and cached input unknown, because Hermes' filled usage has no cache
fields, but a null final usage leaves a wrong output count. The Anthropic API
itself always sends both. Hermes' Codex app-server runtime records usage outside
this path. Hermes passes a response's raw usage to no per-agent hook, so the
adapter wraps `record_response_usage` and the three handlers for invalid,
truncated, and refused responses in `agent.turn_response_check`, and the
iteration-limit summary's `_summary_text` and attempt builders in
`agent.chat_completion_helpers`, `tools.delegate_tool._build_child_agent`, and
the native usage readers in `agent.bedrock_adapter` and
`agent.gemini_native_adapter`, once per process. The runtime fails to start if
the installed Hermes no longer calls them by those names.

## Execution Model

Each NeMo Fabric runtime starts one local adapter host, constructs one Hermes Agent
`AIAgent`, and opens one `SessionDB`. Ordered `Runtime.invoke(...)` calls reuse
those native objects and pass the prior turn's returned transcript back to
`run_conversation(...)`. Runtime stop calls the agent's idempotent `close()`
method, closes the session database, and releases the Relay plugin context when
enabled. Relay subscriber work is flushed after each invocation so ATIF output
is complete before the result returns.

## Maintaining The Adapter

Keep `hermes.fabric-adapter.json` aligned with the Python implementation:

- `contract_version` must match the adapter contract supported by NeMo Fabric core.
- `adapter_id` is the stable id selected by `harness.adapter_id`.
- `adapter_kind` is `python` because NeMo Fabric can invoke it through Python.
- `runner.module` names the persistent host module that NeMo Fabric invokes with
  `python -m`.
- `requirements` supplies dependency checks to NeMo Fabric diagnostics; keep
  required env vars, binaries, or packages current.
- `config.accepts` must match the NeMo Fabric sections this adapter maps into Hermes Agent.
- `telemetry.providers` declares provider-specific outputs and integration modes
  the adapter can produce or forward.

Do not put end-user agent settings in this directory. Users vary harness,
model, skills, MCP, tools, telemetry, and runtime behavior through complete
typed `FabricConfig` values and ordinary Python composition. The adapter
descriptor describes adapter capabilities; it is not an agent configuration.
Add descriptor fields only when NeMo Fabric core or the SDK actually uses them.

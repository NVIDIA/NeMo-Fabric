<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric NAT Reference Adapter

This source-only adapter runs an NVIDIA NeMo Agent Toolkit (NAT) workflow behind
the NeMo Fabric lifecycle contract. It is a third-party adapter reference, not a
bundled NeMo Fabric adapter or a published package.

The implementation is generic. It constructs NAT configuration in memory from
`FabricConfig`; it does not read a NAT YAML file and does not hardcode the
calculator or email-phishing components.

## Configuration Boundary

NeMo Fabric owns portable configuration. `harness.settings` contains only
NAT-native component configuration that has no portable NeMo Fabric equivalent.

| NeMo Fabric input | NAT configuration |
| --- | --- |
| `models.<role>` | `llms.<role>`; every Fabric model-role name is preserved |
| `instructions.system` | Built-in `react_agent` workflow `additional_instructions`; other workflow types reject this field in the initial adapter |
| `harness.settings.workflow` | `workflow` |
| `harness.settings.functions` | `functions` |
| `harness.settings.function_groups` | `function_groups` |
| Harness-native `mcp.servers.<name>` | Generated `mcp_client` function group named `<name>` |
| `tools.enabled`, `tools.blocked` | Effective NAT-native workflow tool selection |

The adapter loads installed `nat.components` entry points before NAT validates
the generated configuration. A custom function, function group, or workflow is
therefore supplied as an installed NAT component package and selected by its
registered `_type`. No Python import path or callable crosses `FabricConfig`.

At runtime, `start` loads components, enters one `WorkflowBuilder`, creates a
`SessionManager` with that shared builder, and retains both resources. Each
`invoke` opens a session from the retained manager, enters `session.run(...)`,
and awaits `runner.result()`. `stop` shuts down the session manager and exits
the builder context. This first reference does not claim cancellation, service,
streaming, or live-update support.

## MCP Tool Filters

The adapter consumes the routed `capability_plan.native.mcp_servers` entries,
including the normalized per-server filters. Fabric MCP tool names remain bare
server-local names; NAT exposes a selected member as `<server>__<tool>`.

| Fabric server policy | Generated NAT function group |
| --- | --- |
| `allowed_tools` omitted and `blocked_tools=[]` | No `include` or `exclude`; expose all discovered tools |
| Nonempty `allowed_tools` only | `include=allowed_tools` |
| `allowed_tools` omitted and nonempty `blocked_tools` | `exclude=blocked_tools` |
| Both lists configured | `include=allowed_tools`; Fabric requires `blocked_tools` to be disjoint, so those names are already outside the allowlist |
| `allowed_tools=[]` | Omit the generated group; expose no tools from that server |

NAT rejects a function group that sets both `include` and `exclude`, so the
adapter emits only `include` whenever an allowlist is present. Fabric rejects
blank names and an allow/block overlap before adapter startup. A nonempty
generated MCP group is added to workflows that expose `tool_names`; callers do
not repeat portable MCP servers in `harness.settings`. A workflow implementation
that requires at least one tool can still reject a configuration whose effective
tool set is empty.

Per-server MCP filters and root `tools.enabled` or `tools.blocked` solve
different problems. MCP filters select members within one server. Root tool
policy selects across the effective NAT-native tool surface.

## Development Bootstrap

This directory intentionally has no package metadata or discovery wiring. Until
source resolution and third-party descriptor discovery are available, use one
Python environment for NeMo Fabric, the common adapter host, NAT, and every NAT
component referenced by the config:

```bash
uv pip install \
  nemo-fabric-adapters-common \
  nvidia-nat-core \
  nvidia-nat-langchain \
  nvidia-nat-mcp
export PYTHONPATH="$PWD/external/nat/src${PYTHONPATH:+:$PYTHONPATH}"
```

`PYTHONPATH` is a development bootstrap limitation, not the target installation
contract. Stage the descriptor in the current agent-local discovery location:

```bash
mkdir -p .tmp/nat-reference/adapters/nat
cp external/nat/fabric-adapter.json \
  .tmp/nat-reference/adapters/nat/fabric-adapter.json
```

The calculator example also requires a streamable-HTTP calculator MCP server.
Set its endpoint and run the typed `FabricConfig` example:

```bash
export CALCULATOR_MCP_URL=http://127.0.0.1:9901/mcp
uv run python external/nat/examples/calculator.py \
  --base-dir "$PWD/.tmp/nat-reference"
```

The email-phishing example uses the NAT example component registered by
`nat_email_phishing_analyzer`. Install that component from a NAT checkout, then
run the example:

```bash
uv pip install -e \
  "<path-to-nat-checkout>/examples/evaluation_and_profiling/email_phishing_analyzer"
uv run python external/nat/examples/email_phishing.py \
  --base-dir "$PWD/.tmp/nat-reference"
```

Both examples accept `--plan` to inspect the resolved plan without starting the
runtime. They use Python `FabricConfig` objects only; no YAML is involved.

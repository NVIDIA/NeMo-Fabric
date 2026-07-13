<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hermes CLI Adapter

This adapter runs Hermes through the installed `hermes` CLI. Fabric starts the
adapter as a Python module for local debugging, `fabric run`, and
environment-backed consumers. The adapter then invokes Hermes and captures
stdout, stderr, exit status, logs, and artifacts.

Fabric invokes `runner.module` with `python -m`. The module reads the Fabric
invocation from standard input, writes Hermes-native config, and then launches
the real Hermes CLI. `runner.callable` records the corresponding reusable
Python function.

## What It Maps

The adapter receives Fabric's normalized payload and writes Hermes-native config
before calling the CLI. It maps:

- model provider, model name, and base URL;
- workspace and terminal settings;
- Fabric skills as Hermes external skill directories;
- Fabric MCP servers as Hermes MCP server config;
- `tools.blocked` as Hermes disabled toolsets in the generated config, unioned
  with `harness.settings.disabled_toolsets`;
- selected CLI flags and environment variables from harness settings.

`hermes_home` configures a base directory. The adapter creates a child under
`runtimes/<runtime_id>` so invocations in one Fabric runtime share Hermes state
without sharing config or the session database with another runtime.

## Maintaining The Adapter

Keep `fabric-adapter.json` aligned with the Python implementation:

- `contract_version` must match the adapter contract supported by Fabric core.
- `adapter_id` is the stable id selected by `harness.adapter_id`.
- `adapter_kind` is `python` because Fabric invokes the adapter with Python.
- `runner.module` names the module that Fabric invokes with `python -m`.
  `runner.callable` names the equivalent reusable Python function.
- Harness settings should use `hermes_command` and `hermes_args` for the actual
  Hermes CLI command and arguments.
- `requirements` powers `fabric doctor`; keep required env vars and the `hermes`
  binary requirement current.
- `config.accepts` must match the Fabric sections this adapter maps into Hermes.
- `telemetry.supports` lists telemetry paths the adapter can produce or forward.

When Hermes CLI flags or config files change, update the mapping code and the
descriptor together. User-facing run variations belong in `agent.yaml` or
profiles, not in the adapter directory. Add descriptor fields only when Fabric
core or the SDK actually uses them.

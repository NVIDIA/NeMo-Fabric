<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hermes Agent Adapter

This adapter runs Hermes Agent through one of two Relay-compatible execution
strategies selected by `harness.settings.relay_launch_mode`:

- `native_plugin` (default) invokes the Hermes Python SDK and activates Relay
  through its Python API. When `relay.dynamic_plugins` is non-empty, the
  adapter retains Relay's owned dynamic-plugin host for the complete Hermes
  call.
- `cli_wrapper` invokes `nemo-relay run --agent hermes`. Relay owns the
  transient gateway, hook injection, child process, and cleanup. Dynamic
  plugins are provisioned through Relay's lifecycle in invocation-isolated
  directories before Hermes starts.

`cli_wrapper` requires Relay telemetry and Relay 0.6 or newer. Override the
executable with `harness.settings.relay_cli_command`; it defaults to
`nemo-relay`.

For a Relay configuration that should move unchanged from an evaluation into a
deployment, set `relay.plugin_config_path` to one canonical Relay
`plugins.toml`. Fabric loads its built-in components and standard
`[[plugins.dynamic]]` declarations for either launch strategy. Do not combine
that path with Fabric's inline Relay observability, component, dynamic-plugin,
or policy fields. Relay's separate `config.toml` remains invocation-owned
agent/gateway launch metadata in CLI-wrapper mode. CLI-wrapper mode can
provision lifecycle-managed Python worker environments from these declarations;
native mode can directly activate native dynamic plugins, while worker plugins
still require an existing Relay-managed `environment_ref`.

Fabric invokes the adapter module with `python -m` through the core runtime
lifecycle. The module entry point and the descriptor's callable route use the
same `run(payload: dict) -> dict` implementation.

## What It Maps

The adapter receives Fabric's normalized payload and materializes Hermes-native
configuration for:

- model provider, model name, base URL, temperature, and token settings;
- workspace and terminal settings;
- Fabric skills as Hermes external skill directories;
- Fabric MCP servers as Hermes MCP server config;
- `tools.blocked` as Hermes disabled toolsets, unioned with
  `harness.settings.disabled_toolsets`;
- optional NeMo Relay 0.6 observability, built-in component, and dynamic-plugin
  configuration.

`hermes_home` configures a base directory. The adapter creates a child under
`runtimes/<runtime_id>` so invocations in one Fabric runtime share Hermes state
without sharing config or the session database with another runtime.

## Maintaining The Adapter

Keep `fabric-adapter.json` aligned with the Python implementation:

- `contract_version` must match the adapter contract supported by Fabric core.
- `adapter_id` is the stable id selected by `harness.adapter_id`.
- `adapter_kind` is `python` because Fabric can invoke it through Python.
- `runner.module` names the module that Fabric invokes with `python -m`.
  `runner.callable` names the equivalent reusable Python function.
- `requirements` powers `fabric doctor`; keep required env vars, binaries, or
  packages current.
- `config.accepts` must match the Fabric sections this adapter maps into Hermes.
- `telemetry.providers` declares provider-specific outputs and integration modes
  the adapter can produce or forward.

Do not put end-user agent settings in this directory. Users vary harness,
model, skills, MCP, tools, telemetry, and runtime behavior through `agent.yaml`
and profiles. The adapter descriptor describes adapter capabilities; it is not a
profile. Add descriptor fields only when Fabric core or the SDK actually uses
them.

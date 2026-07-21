<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Hermes Agent Adapter

This adapter runs Hermes Agent through its Python SDK.

## Install

To install just the Hermes Agent adapter by itself:

```bash
pip install "nemo-fabric[hermes]"
```

To install the Hermes Agent adapter along with the NeMo Fabric Runtime:

```bash
pip install "nemo-fabric[hermes, runtime]"
```

To install the Hermes Agent adapter along with a compatible version of Hermes Agent:

```bash
pip install "nemo-fabric[hermes, hermes-agent]"
```

## What It Maps

The adapter receives a normalized payload from Fabric and materializes Hermes-native configuration for:

- model provider, model name, base URL, temperature, and token settings;
- workspace and terminal settings;
- Fabric skills as Hermes external skill directories;
- Fabric MCP servers as Hermes MCP server config;
- `tools.blocked` as Hermes disabled toolsets, unioned with
  `harness.settings.disabled_toolsets`;
- optional NeMo Relay telemetry plugin configuration.

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
model, skills, MCP, tools, telemetry, and runtime behavior through complete
typed `FabricConfig` values and ordinary Python composition. The adapter
descriptor describes adapter capabilities; it is not an agent configuration.
Add descriptor fields only when Fabric core or the SDK actually uses them.

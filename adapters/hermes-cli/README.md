<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hermes CLI Adapter

This adapter runs Hermes through the installed `hermes` CLI. It is the process
path for local debugging, `fabric run`, and environment-backed consumers that
want Fabric to invoke a command and capture stdout, stderr, exit code, logs, and
artifacts.

Unlike the Hermes SDK adapter, this adapter intentionally does not advertise
`runner.module` or `runner.callable`. Fabric invokes `runner.script` as a process
and the script then launches the Hermes CLI.

## What It Maps

The adapter receives Fabric's normalized payload and writes Hermes-native config
before calling the CLI. It maps:

- model provider, model name, and base URL;
- workspace and terminal settings;
- Fabric skills as Hermes external skill directories;
- Fabric MCP servers as Hermes MCP server config;
- selected CLI flags and environment variables from harness settings.

## Maintaining The Adapter

Keep `fabric-adapter.json` aligned with the process implementation:

- `adapter_id` is the stable id selected by `harness.adapter_id`.
- `adapter_kind` is currently `python` because the wrapper script is Python, but
  the execution model is process-backed.
- `runner.script` must point to the executable wrapper that reads Fabric payload
  JSON from stdin and writes JSON output to stdout.
- `requirements` powers `fabric doctor`; keep required env vars and the `hermes`
  binary requirement current.
- `config.accepts` must match the Fabric sections this adapter maps into Hermes.
- `telemetry.supports` lists telemetry paths the adapter can produce or forward.

When Hermes CLI flags or config files change, update the mapping code and the
descriptor together. User-facing run variations belong in `agent.yaml` or
profiles, not in the adapter directory. Add descriptor fields only when Fabric
core or the SDK actually uses them.

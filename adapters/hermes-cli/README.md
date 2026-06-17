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
`runner.module` or `runner.callable`. Fabric invokes a small launcher as a
process. The launcher reads the Fabric invocation from `FABRIC_INVOCATION`,
falls back to stdin for simple/debug runs, writes Hermes-native config, and
then launches the real Hermes CLI.

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
- `adapter_kind` is `process` because Fabric owns process supervision,
  stdout/stderr capture, exit status, logs, and artifacts.
- `runner.command` and `runner.script` define the launcher process. The script
  reads Fabric payload JSON from `FABRIC_INVOCATION`, with stdin as fallback,
  and writes JSON output to stdout.
- Harness settings should use `hermes_command` and `hermes_args` for the actual
  Hermes CLI command and arguments. Do not use `command` for Hermes itself;
  `command` belongs to the Fabric process runner.
- `requirements` powers `fabric doctor`; keep required env vars and the `hermes`
  binary requirement current.
- `config.accepts` must match the Fabric sections this adapter maps into Hermes.
- `telemetry.supports` lists telemetry paths the adapter can produce or forward.

When Hermes CLI flags or config files change, update the mapping code and the
descriptor together. User-facing run variations belong in `agent.yaml` or
profiles, not in the adapter directory. Add descriptor fields only when Fabric
core or the SDK actually uses them.

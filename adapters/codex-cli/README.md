<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Codex Adapter

Runs an installed Codex through Fabric's Python-adapter lifecycle. One
Fabric runtime maps to one Codex thread.

To install just the Codex adapter by itself:

```bash
pip install "nemo-fabric[codex]"
```

To install just the Codex adapter along with the NeMo Fabric Runtime:
```bash
pip install "nemo-fabric[codex, runtime]"
```

## Authentication and Codex Config

The adapter does not read, copy, or rewrite Codex credentials. The child Codex
process inherits `HOME`, `CODEX_HOME`, platform runtime variables, and proxy or
certificate settings, so an existing `codex login` session remains
authoritative. Additional variables must be provided through
`harness.settings.env`.

Codex continues to load its system, user, profile, and trusted project config.
Fabric adds only explicitly configured invocation overrides:

- `models.default.model` selects `--model`; omit it to use Codex's configured
  default.
- `environment.workspace` selects the process working directory.
- `harness.settings.sandbox` selects `read-only`, `workspace-write`, or
  `danger-full-access`.
- `harness.settings.codex_profile` selects a Codex profile.
- `harness.settings.config_overrides` emits repeated `--config key=value`
  arguments. Values may be TOML scalars or arrays; use dotted keys for nested
  Codex settings.
- `harness.settings.codex_args` is an escape hatch for additional CLI flags.
- `harness.settings.timeout_seconds` bounds each invocation and defaults to 1800.

`codex_command`, `codex_state_dir`, `cwd`, `env`, and
`skip_git_repo_check` are available for prepared environments and tests.

## Execution Paths

The first invocation records Codex's generated thread ID against the Fabric
runtime ID. Later invocations on the same runtime use
`codex exec resume <thread-id>`. Codex owns its transcript and authentication;
Fabric owns the runtime lifecycle and runtime-to-thread correlation record.
Both `fabric run` and stateful runtime paths accept text input.

Use `codex_cli_config()` from `examples.code_review_agent` for local one-shot
and multi-turn examples.

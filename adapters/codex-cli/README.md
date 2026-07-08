# Codex CLI Adapter

Runs an installed Codex CLI through Fabric's process-adapter lifecycle. One
Fabric runtime maps to one Codex thread.

Keep `fabric-adapter.json` aligned with the adapter implementation.
`contract_version` must match the adapter contract supported by Fabric core;
`adapter_id` is the stable id selected by `harness.adapter_id`.

Install Fabric with the adapter dependency before running it:

```bash
python3 -m pip install -e ".[codex]"
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

Use the `codex_cli` profile under `examples/code-review-agent/profiles/` for
local one-shot and `fabric chat` examples.

# Codex CLI Adapter

Runs an installed Codex CLI through Fabric's process-adapter lifecycle. The
same adapter supports one-shot and session runtime modes.

## Authentication and Codex Config

The adapter does not read, copy, or rewrite Codex credentials. The child Codex
process inherits `HOME`, `CODEX_HOME`, and its environment, so an existing
`codex login` session remains authoritative. Automation may provide
`CODEX_API_KEY` directly to the Codex process.

Codex continues to load its system, user, profile, and trusted project config.
Fabric adds only explicitly configured invocation overrides:

- `models.default.model` selects `--model`; omit it to use Codex's configured
  default.
- `environment.workspace` selects the process working directory.
- `harness.settings.sandbox` selects `read-only`, `workspace-write`, or
  `danger-full-access`.
- `harness.settings.codex_profile` selects a Codex profile.
- `harness.settings.config_overrides` emits repeated `--config key=value`
  arguments.
- `harness.settings.codex_args` is an escape hatch for additional CLI flags.

`codex_command`, `codex_state_dir`, `cwd`, `env`, and
`skip_git_repo_check` are available for prepared environments and tests.

## Runtime Modes

One-shot mode runs `codex exec --json --ephemeral` and returns the final agent
message, usage, thread ID, and Codex events in the normalized Fabric result.

Session mode omits `--ephemeral`. The first invocation records Codex's generated
thread ID against Fabric's session ID; later invocations use
`codex exec resume <thread-id>`. Codex owns its transcript and authentication;
Fabric owns the lifecycle and the session-to-thread correlation record.

Use the `codex_cli` and `codex_cli_session` profiles under
`examples/code-review-agent/profiles/` for local one-shot and `fabric chat`
examples.

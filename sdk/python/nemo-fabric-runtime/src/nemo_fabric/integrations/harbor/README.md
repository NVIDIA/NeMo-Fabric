<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Integration

Use `nemo_fabric.integrations.harbor:FabricAgent` to run NVIDIA NeMo Fabric
adapters in Harbor tasks. Harbor options select the model, harness, skills, MCP
servers, tool policy, and telemetry; `FabricAgent` translates them into one
typed `FabricConfig` for the task run.

Harbor's `skills_dir` is a task-side collection containing `<skill_name>/SKILL.md` directories. The task runner expands its immediate entries into individual NeMo Fabric skill paths in name order, preserving explicit `config.skills.paths` and avoiding duplicate paths. An empty collection adds no skills. Missing collections and entries without a regular `SKILL.md` file fail before harness execution; skill contents are still validated by the selected adapter. Collection paths are resolved inside the task environment, relative to `fabric_config_base_dir` when not absolute, never through the host filesystem. Use matching NeMo Fabric versions on the host and in the task environment for this transport contract.

Refer to the [Harbor example](../../../../../../../examples/harbor/README.md)
for runnable SWE-Bench commands, configuration variations, reward checks, and
Relay artifacts.

## Credential Transport

Use `fabric_model_api_key_env` to name the model credential variable. The name
must match Harbor's redaction policy, for example `MODEL_API_KEY`. Names such as
`MODEL_DSN` are rejected during options validation, including Harbor preflight.
Supply literal credentials through Harbor's managed `extra_env` (`--ae`).
In `fabric_environment_env`, sensitive entries must use host-variable references
such as `{"MODEL_API_KEY": "${MODEL_API_KEY}"}` without inline defaults.
The bridge resolves these references with Harbor's utilities and merges them into
Harbor's managed environment and redaction
inputs, rejects conflicting values, and transports only variable names in the
retained run specification. The task runner resolves those names from its
execution environment; a missing variable fails with a name-only diagnostic.
This keeps raw credentials out of Harbor's top-level job configuration, which
is outside its trial-directory scrubber.

Non-sensitive settings such as `RUN_MODE` can use arbitrary valid environment
variable names. Harbor does not scrub their values. Use names recognized by
Harbor's `is_sensitive_env_key` policy for all credentials, including credentials
other than the configured model key; the bridge cannot infer secrets from values.

Deploy matching bridge and runner versions: a runner predating name-only transport cannot read the new payload. Do not put credentials in harness settings or native configuration files.

## Execution Outcomes

The task runner writes normalized evidence before exiting nonzero for failed or cancelled runs. The bridge downloads that evidence before reporting a Harbor execution failure or cancellation, including when the runner exits nonzero. A completed answer can still receive verifier reward zero; verifier scoring is not an execution failure. A missing or malformed result does not hide an available process failure.

Lifecycle failures before a normalized result is available produce a separate
`runner_error` record. Core host-operation deadlines (`host_timeout`) use code
`timeout`. The bridge classifies only this canonical code as a deadline, whether
it appears in a runner error or normalized result, and raises `TimeoutError`,
which Harbor translates into `AgentTimeoutError`. Cancellation takes precedence.
An invocation with status `cancelled` becomes a trial execution error and retains
its normalized status; it does not interrupt the Harbor job. Actual orchestration
cancellation still propagates as `asyncio.CancelledError`.
The bridge does not
inspect harness IDs, native error-code lists, or diagnostic text. Other
adapter-specific timeout codes are not automatically reclassified.

## Finalized Telemetry Artifacts

The task runner validates telemetry after the one-shot NeMo Fabric runtime shuts down. For Relay trajectories finalized during shutdown, it reads the runtime-owned plugin configuration and collects local ATIF files from that runtime's configured output directory. It validates the trajectory before promoting it to Harbor's `agent/trajectory.json`; ambiguous or malformed artifacts remain telemetry failures rather than changing the verifier reward. Remote-only output and nested filename templates are not collected by this fallback. Adapter invocation results and multi-turn behavior are unchanged.

## Invocation Accounting

The integration projects normalized `RunResult.usage` into Harbor's `AgentContext` without requiring NeMo Relay or ATIF. Harbor input counts include cache: inclusive NeMo Fabric input is copied unchanged, while exclusive input is combined with cache only when both counts are known. If input semantics are unknown, Harbor input remains unknown unless the adapter explicitly reports zero cache tokens. Legacy results remain valid, but their input count needs explicit cache semantics or an ATIF fallback to populate Harbor's inclusive count; the original values remain in result metadata. Unknown counts and costs remain `None`; estimates are not promoted to reported cost. ATIF metrics fill only missing fields and are never added to normalized usage. The same collection path applies to unsuccessful results that contain usage.

## Host Admission And Task Execution

The `nemo-fabric[harbor]` extra installs a matching
`nemo-fabric-adapter-catalog` release on the Harbor host. The catalog contains
descriptor metadata, not executable adapters. The host does not need to install
each harness SDK to derive the agent's instance capabilities.

`FabricAgent` uses the shared `inspect_adapter()` API and the existing native
planner to validate the requested configuration. Skills and MCP support come
from the descriptor's accepted configuration sections. ATIF requires a selected
provider that declares ATIF output; Relay also requires an explicitly enabled,
unambiguous observability component. These are admission claims, not proof that
a trajectory was produced.

**Known limitation:** Released Harbor versions that gate trace export on the agent class's static ATIF flag cannot export NeMo Fabric trajectories, even when the selected instance supports ATIF and a trajectory exists. The class default remains conservative. To export through Harbor, use a build containing the fix in [Harbor PR #3528](https://github.com/harbor-framework/harbor/pull/3528), or a subsequent release that includes it. Recorded trajectory artifacts remain available independently of Harbor's exporter.

For an external adapter, pass `fabric_adapter_descriptor=/host/adapter.fabric-adapter.json`
with its canonical descriptor. Separately, use `fabric_discovery_paths` to
locate its executable descriptor inside the task. Host inspection does not read
task-local paths or add catalog resources to execution discovery.

When metadata is unavailable, basic execution remains possible with false
optional-feature claims. Requested skills, MCP, or telemetry fail clearly rather
than being silently dropped. Malformed metadata is an error. Registered workflows
require target-specific admission and are not supported by this standalone path.

The runner compares the host's normalized descriptor fingerprint with the
descriptor in its actual execution plan before starting the harness. A mismatch
fails with instructions to align the host catalog and task adapter releases or
external metadata. Pin compatible versions in both environments. This check
detects descriptor drift; it is not runtime-observed software provenance.

## Explicit Harness Recipes

The bridge does not inject harness-specific permissions, environment variables,
or turn and timeout limits. Specify those choices in the benchmark recipe through
`fabric_harness_settings`, `fabric_environment_env`, `fabric_max_turns`, and
`fabric_runtime_timeout_seconds`. Omitted values use the adapter's normal
behavior. The example commands include the sandbox and permission settings
needed by their tasks.

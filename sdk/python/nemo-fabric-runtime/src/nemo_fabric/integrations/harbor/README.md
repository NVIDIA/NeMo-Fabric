<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Integration

Use `nemo_fabric.integrations.harbor:FabricAgent` to run NVIDIA NeMo Fabric
adapters in Harbor tasks. Harbor options select the model, harness, skills, MCP
servers, tool policy, and telemetry; `FabricAgent` translates them into one
typed `FabricConfig` for the task run.

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

## Invocation Accounting

The integration projects normalized `RunResult.usage` into Harbor's `AgentContext` without requiring NeMo Relay or ATIF. Harbor input counts include cache: inclusive NeMo Fabric input is copied unchanged, while exclusive input is combined with cache only when both counts are known. If input semantics are unknown, Harbor input remains unknown unless the adapter explicitly reports zero cache tokens. Legacy results remain valid, but their input count needs explicit cache semantics or an ATIF fallback to populate Harbor's inclusive count; the original values remain in result metadata. Unknown counts and costs remain `None`; estimates are not promoted to reported cost. ATIF metrics fill only missing fields and are never added to normalized usage. The same collection path applies to unsuccessful results that contain usage.

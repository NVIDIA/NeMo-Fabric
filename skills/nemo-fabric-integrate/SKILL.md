---
name: nemo-fabric-integrate
description: Use this skill when integrating NeMo Fabric into a consumer application, service, evaluation harness, or platform through the typed Python SDK — translating the consumer's own application, job, or deployment config into an in-memory FabricConfig, choosing the single-invocation convenience API or an explicitly started runtime, validating with plan and doctor, and consuming normalized results, artifacts, and telemetry.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Integrate NeMo Fabric Through The Python SDK

Use this skill when a consumer codebase — an application, service, evaluation
harness, or platform — needs to run agent harnesses through NeMo Fabric's typed
Python SDK. The consumer owns its own configuration object and translates it
into an in-memory `FabricConfig`; NeMo Fabric owns adapter selection, the runtime
lifecycle, and normalized results.

Do not use this skill to author or modify NeMo Fabric adapters, change NeMo Fabric core or
its bindings, or maintain repository infrastructure. If you are contributing to
NeMo Fabric itself, use the maintainer skills in `.agents/skills/` instead.

## Integration Boundary

Stay on the public, in-memory contract. These rules keep a consumer integration
supported and upgrade-safe:

- Import only from the public `nemo_fabric` package. Never import `_native` or
  any adapter-internal module.
- Build configuration as a typed `FabricConfig` in memory and pass it directly to
  NeMo Fabric. Create every deployment or evaluation variant with ordinary Python
  functions and `model_copy(deep=True)`. A platform integration can serialize
  the typed config inside a private transient run specification when it crosses
  a process boundary; that transport is not a public authoring format.
- Let NeMo Fabric own harness control. Do not reimplement start, invoke, or stop
  logic, and do not manage adapter threads, sessions, or processes directly.
- Treat `runtime_id`, `invocation_id`, and `request_id` as opaque correlation
  strings, not parsable or reusable state.

See [config-mapping.md](references/config-mapping.md) for how to translate a
consumer config object into `FabricConfig`, and for the full list of mechanics
that stay hidden behind this boundary.

## Install And Set Up The Environment

The consumer or its execution environment owns installation; NeMo Fabric validates
runtime assumptions but never installs harnesses or credentials at run time.

- `uv pip install nemo-fabric[runtime]` (add the `harbor` extra for the Harbor integration). Refer to the
  [installation guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/getting-started/install.mdx).
- Select a harness adapter — the `adapter_id` set in `HarnessConfig`, for example
  `nvidia.fabric.hermes` — and install its extra the same way, for example
  `pip install "nemo-fabric[hermes]"`
  (available extras: `hermes`, `codex`, `deepagents`, `claude`), plus the
  corresponding harness binaries and dependencies.
- Provide model credentials through environment variables named by the config
  (`ModelConfig.api_key_env`), never as literals in code.
- Confirm the native extension is importable; SDK calls raise
  `FabricNativeUnavailableError` when it is missing.

## Build The Typed Config From Consumer Config

Map the consumer's application, job, or deployment object into a `FabricConfig`
with the public models and helper methods:

```python
from nemo_fabric import (
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    RuntimeConfig,
)


def to_fabric_config(job) -> FabricConfig:
    config = FabricConfig(
        metadata=MetadataConfig(name=job.name),
        harness=HarnessConfig(adapter_id=job.adapter_id, resolution="preinstalled"),
        models={
            "default": ModelConfig(
                provider=job.provider,
                model=job.model,
                api_key_env=job.api_key_env,
            )
        },
        runtime=RuntimeConfig(input_schema="chat", output_schema="message"),
    )
    config.add_skill_path(job.skill_dir)
    config.add_mcp_server(
        "github",
        transport="streamable-http",
        url="${GITHUB_MCP_URL}",
        exposure="harness_native",
    )
    return config
```

- Shape capabilities with `add_skill_path`, `remove_skill_path`,
  `add_mcp_server`, `remove_mcp_server`, and `enable_relay`.
- Create deployment or evaluation variants with `model_copy(deep=True)` and
  ordinary Python functions; each copy plans and runs independently.
- Pass `base_dir=...` to any `Fabric` call when the config uses relative paths,
  so skills, workspaces, and artifacts anchor to the consumer's own layout.

The repository [`code_review_agent` example](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/code_review_agent)
shows this pattern end to end with complete Hermes Agent, Codex, Deep Agents,
environment, MCP, and telemetry variants. Reuse it rather than duplicating config
construction.

## Choose A Lifecycle

Pick the smallest lifecycle the consumer needs:

- **Single invocation** — one input, no retained state after the call.
  `await Fabric().run(config, input=...)` runs the full start, invoke, and stop
  cycle and returns a `RunResult`. Pass
  `request=RunRequest(...)` instead of `input=...` when the invocation needs a
  caller-owned request ID or context (the two are mutually exclusive).
- **Stateful runtime** — ordered turns over one logical harness lifecycle. Start it with
  `start_runtime(...)` and use the returned `Runtime` as an async context
  manager so cleanup runs on exit — shutdown is attempted, not guaranteed
  (`stop()` can raise `FabricRuntimeError`; see Consume Results And Handle
  Errors). A runtime accepts one active invocation at a time; overlapping calls
  raise `FabricStateError`.

The selected adapter owns the execution topology. The bundled Claude, Codex,
Deep Agents, and Hermes Agent adapters retain their native client, graph/checkpointer,
or agent/database inside one local host for the full runtime. Local `process`
and `python` adapters use this host lifecycle; consumers do not select another
local execution mechanism in `FabricConfig`. Do not replay an invocation after
a runtime failure. Stop the failed runtime and explicitly start a new one
according to the application's retry policy.

The lifecycle fragment below shows both forms. It assumes the caller has already
set `config = to_fabric_config(job)` and chosen `base`, as described in the
configuration example above:

```python
import asyncio

from nemo_fabric import Fabric


async def main() -> None:
    fabric = Fabric()

    # Single invocation
    result = await fabric.run(config, base_dir=base, input="Review the changes.")

    # Multi-turn
    async with await fabric.start_runtime(config, base_dir=base) as runtime:
        first = await runtime.invoke(input="Inspect the repository")
        second = await runtime.invoke(input="Now review the latest patch")


asyncio.run(main())
```

NeMo Fabric owns no queue, worker pool, retry policy, or concurrency limit. For
parallel work, start independent runtimes and let the consumer decide how many.

## Validate Before Running

Resolve and diagnose before spending work on a runtime, especially in a new
environment or before relying on an optional capability:

```python
fabric = Fabric()
plan = fabric.plan(config, base_dir=base)             # sync: adapter + capabilities
report = await fabric.doctor(config, base_dir=base)   # async: preflight checks

print(plan.adapter.adapter_id, report.status)
```

- Use `plan(...)` to confirm adapter selection and capability routing before
  running.
- Use `doctor(...)` to check adapter availability, resolution, environment
  context, and declared requirements such as required environment variables. Its
  aggregate `status` is `pass`, `warn`, or `fail`. It does **not** validate the
  contents of `harness.settings`: an unknown or misspelled adapter setting still
  passes and is silently ignored unless the adapter reads it, so validate
  settings against the adapter's own docs and your integration tests.

## Consume Results And Handle Errors

Every invocation that reaches the adapter boundary returns a normalized
`RunResult`, even when the harness invocation itself failed. Inspect the failure
fields before reading output:

```python
result = await fabric.run(config, base_dir=base, input="Review the changes.")

if result.status == "succeeded":
    use_output(result.output, result.artifacts, result.telemetry)
else:
    handle_failure(result.status, result.error, result.events)  # failed, cancelled, ...
```

- Treat `status == "succeeded"` as the only success. Other terminal values
  (`failed`, `cancelled`) are unsuccessful, and `error` may be `None` even then,
  so branch on `status`, not on `error`. Read `status`, `error`, and `events`
  before processing `output`.
- Capture `artifacts` and `telemetry` references as the returned evidence for
  platforms and evaluations. Store and log `runtime_id`, `invocation_id`, and
  `request_id` separately as opaque strings.
- Catch `FabricError` subclasses for lifecycle failures that prevent a
  normalized result: `FabricConfigError`, `FabricCapabilityError`,
  `FabricRuntimeError`, `FabricStateError`, and `FabricNativeUnavailableError`.
- The consumer owns retries and failure policy; NeMo Fabric does not retry by
  default. `run(...)` and `async with` runtimes attempt cleanup automatically,
  so prefer them over manual `stop()` — but shutdown is not guaranteed: `stop()`,
  including the automatic call when an `async with` block exits, can raise
  `FabricRuntimeError`. On a normal exit that error propagates; after an
  invocation error the cleanup failure is attached to the original exception. Be
  ready to handle a shutdown failure.

See [results-and-errors.md](references/results-and-errors.md) for the full
result-field and error inventory, and
[sdk-api-inventory.md](references/sdk-api-inventory.md) for when to use each
`Fabric` and `Runtime` method.

## Test And Validate The Integration

- Write focused integration tests that build the consumer's `FabricConfig`,
  assert `plan(...)` selects the expected adapter and capabilities, and — where
  a harness and credentials are available — run one invocation and assert the
  `RunResult` status and evidence.
- `plan(...)` is credential-free — use it as the CI gate that validates adapter
  selection and capability routing without a model or secrets. `doctor(...)` also
  runs without calling a model, but it checks declared environment requirements
  (such as required API-key variables) and returns `fail` when they are unset, so
  run it where the environment is provisioned and read its per-check results.
- Run the consumer project's own build and test commands. For a source checkout
  of NeMo Fabric, `just build-all` rebuilds the native extension and
  `just test-python` runs the Python suite.
- Confirm the typed config is passed directly to NeMo Fabric and no non-public
  imports were added.

## Checklist

- [ ] The consumer config object is translated directly into an in-memory `FabricConfig`.
- [ ] Only public `nemo_fabric` symbols are imported; no `_native` or adapter internals.
- [ ] The consumer config is built in memory and passed directly to NeMo Fabric.
- [ ] The right lifecycle is chosen: `run(...)` for a single invocation, `start_runtime(...)` with `async with` for multi-turn.
- [ ] `plan(...)` and `doctor(...)` validate adapter selection, capabilities, and environment before execution.
- [ ] Installation, adapter dependencies, and credentials are owned by the environment, not consumer code.
- [ ] `RunResult` status, error, and events are inspected before output; artifacts and telemetry are captured.
- [ ] `FabricError` subclasses are handled, including a `FabricRuntimeError` raised by shutdown; cleanup is delegated to `run(...)` or `async with` (attempted, not guaranteed).
- [ ] Correlation IDs are stored and logged as opaque strings.
- [ ] Focused integration tests pass and NeMo Fabric validation (`plan`/`doctor`, tests) succeeds.

## Related Documentation

Link to these canonical sources instead of duplicating them:

- [Python SDK guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/python.mdx)
- [NeMo Fabric overview](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/about-nemo-fabric/overview.mdx) and
  [installation guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/getting-started/install.mdx)
- Generated API reference (public API index; the installed `nemo_fabric` type
  stubs are authoritative for exact signatures, fields, and defaults):
  [client](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.client.md),
  [runtime](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.runtime.md),
  [models](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.models.md),
  [types](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.types.md),
  [errors](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.errors.md)
- Canonical in-memory config example:
  [examples/code_review_agent](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/code_review_agent)
- Platform and evaluation-harness integration:
  [examples/harbor](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/harbor) and
  [nemo_fabric.integrations.harbor](https://github.com/NVIDIA/NeMo-Fabric/tree/main/python/src/nemo_fabric/integrations/harbor).
  Harbor constructs a typed config from explicit agent inputs and transports it
  inside a private transient run specification at the task-process boundary.
  Follow the code-review example for consumer integration code; Harbor's
  transport representation is an internal process-boundary contract.

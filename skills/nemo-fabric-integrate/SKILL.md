---
name: nemo-fabric-integrate
description: Use this skill when integrating NeMo Fabric into a consumer application, service, evaluation harness, or platform through the typed Python SDK — translating the consumer's own application, job, or deployment config into an in-memory FabricConfig, choosing one-shot run versus a stateful runtime, validating with plan and doctor, and consuming normalized results, artifacts, and telemetry.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Integrate NeMo Fabric Through The Python SDK

Use this skill when a consumer codebase — an application, service, evaluation
harness, or platform — needs to run agent harnesses through NeMo Fabric's typed
Python SDK. The consumer owns its own configuration object and translates it
into an in-memory `FabricConfig`; Fabric owns adapter selection, the runtime
lifecycle, and normalized results.

Do not use this skill to author or modify Fabric adapters, change Fabric core or
its bindings, or maintain repository infrastructure. If you are contributing to
Fabric itself, use the maintainer skills in `.agents/skills/` instead.

## Integration Boundary

Stay on the public, in-memory contract. These rules keep a consumer integration
supported and upgrade-safe:

- Import only from the public `nemo_fabric` package. Never import `_native`,
  `_config_sources`, or any adapter-internal module.
- Build configuration as a typed `FabricConfig` in memory. Do not write, read,
  or materialize `agent.yaml`, portable agent packages, or any intermediate
  config file from consumer integration code. (A platform integration such as
  Harbor may bake a config file into a task image at its deployment boundary —
  that is a deployment mechanic, not the in-memory consumer pattern here.)
- Do not serialize configs, apply file-backed profiles, or resolve profiles by
  name. Build every consumer variant with ordinary Python functions and
  `model_copy(deep=True)`; profile mechanics stay behind the integration
  boundary.
- Let Fabric own harness control. Do not reimplement start, invoke, or stop
  logic, and do not manage adapter threads, sessions, or processes directly.
- Treat `runtime_id`, `invocation_id`, and `request_id` as opaque correlation
  strings, not parsable or reusable state.

See [config-mapping.md](references/config-mapping.md) for how to translate a
consumer config object into `FabricConfig`, and for the full list of mechanics
that stay hidden behind this boundary.

## Install And Set Up The Environment

The consumer or its execution environment owns installation; Fabric validates
runtime assumptions but never installs harnesses or credentials at run time.

- Fabric is not published on PyPI yet. From a source checkout, `just build-all`
  builds the native extension and installs the SDK. To install into another
  environment, build wheels with `just wheels`, then
  `uv pip install --find-links <dist_dir> "nemo-fabric[runtime]"` (add the
  `harbor` extra for the Harbor integration). See the
  [installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install).
- Select a harness adapter — the `adapter_id` set in `HarnessConfig`, for example
  `nvidia.fabric.hermes` — and install its extra the same way, for example
  `uv pip install --find-links <dist_dir> "nemo-fabric[adapters-hermes]"`
  (available extras: `adapters-hermes`, `adapters-codex-cli`,
  `adapters-deepagents`, `adapters-claude`), plus the adapter's own harness
  binaries and dependencies.
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
shows this pattern end to end with complete Hermes, Codex CLI, Deep Agents,
environment, MCP, and telemetry variants. Reuse it rather than duplicating config
construction.

## Choose A Lifecycle

Pick the smallest lifecycle the consumer needs:

- **One-shot** — one input, no retained state. `await Fabric().run(config, input=...)`
  runs the full start, invoke, and stop cycle and returns a `RunResult`. Pass
  `request=RunRequest(...)` instead of `input=...` when the invocation needs a
  caller-owned request ID or context (the two are mutually exclusive).
- **Stateful runtime** — ordered turns over one live harness. Start it with
  `start_runtime(...)` and use the returned `Runtime` as an async context
  manager so shutdown is guaranteed. A runtime accepts one active invocation at
  a time; overlapping calls raise `FabricStateError`.

```python
from nemo_fabric import Fabric

fabric = Fabric()

# One-shot
result = await fabric.run(config, base_dir=base, input="Review the changes.")

# Multi-turn
async with await fabric.start_runtime(config, base_dir=base) as runtime:
    first = await runtime.invoke(input="Inspect the repository")
    second = await runtime.invoke(input="Now review the latest patch")
```

Fabric owns no queue, worker pool, retry policy, or concurrency limit. For
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
- Use `resolve(...)` when only the normalized effective config is needed, with
  no adapter resolution.

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
- The consumer owns retries and failure policy; Fabric does not retry by
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
- Prefer credential-free checks in CI: `plan(...)` and `doctor(...)` validate
  configuration and environment assumptions without calling a model.
- Run the consumer project's own build and test commands. For a source checkout
  of Fabric, `just build-all` rebuilds the native extension and
  `just test-python` runs the Python suite.
- Confirm no config files were written and no non-public imports were added.

## Checklist

- [ ] The consumer config object is translated directly into an in-memory `FabricConfig`.
- [ ] Only public `nemo_fabric` symbols are imported; no `_native`, `_config_sources`, or adapter internals.
- [ ] No `agent.yaml`, portable package, serialized config, or file profile is created or read.
- [ ] The right lifecycle is chosen: `run(...)` for one-shot, `start_runtime(...)` with `async with` for multi-turn.
- [ ] `plan(...)` and `doctor(...)` validate adapter selection, capabilities, and environment before execution.
- [ ] Installation, adapter dependencies, and credentials are owned by the environment, not consumer code.
- [ ] `RunResult` status, error, and events are inspected before output; artifacts and telemetry are captured.
- [ ] `FabricError` subclasses are handled, including a `FabricRuntimeError` raised by shutdown; cleanup is delegated to `run(...)` or `async with` (attempted, not guaranteed).
- [ ] Correlation IDs are stored and logged as opaque strings.
- [ ] Focused integration tests pass and Fabric validation (`plan`/`doctor`, tests) succeeds.

## Related Documentation

Link to these canonical sources instead of duplicating them:

- [Python SDK guide](https://docs.nvidia.com/nemo/fabric/sdk/python)
- [Getting started](https://docs.nvidia.com/nemo/fabric/getting-started/overview) and
  [installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install)
- Generated API reference — exact signatures and fields:
  [client](https://docs.nvidia.com/nemo/fabric/reference/api/python-library-reference/client),
  [runtime](https://docs.nvidia.com/nemo/fabric/reference/api/python-library-reference/runtime),
  [models](https://docs.nvidia.com/nemo/fabric/reference/api/python-library-reference/models),
  [types](https://docs.nvidia.com/nemo/fabric/reference/api/python-library-reference/types),
  [errors](https://docs.nvidia.com/nemo/fabric/reference/api/python-library-reference/errors)
- Canonical in-memory config example:
  [examples/code_review_agent](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/code_review_agent)
- Platform and evaluation-harness integration:
  [examples/harbor](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/harbor) and
  [nemo_fabric.integrations.harbor](https://github.com/NVIDIA/NeMo-Fabric/tree/main/python/src/nemo_fabric/integrations/harbor).
  Note the difference: Harbor bakes a config into the task image and passes it as
  a file-backed `fabric_config_path` (YAML) at the container boundary. That is a
  Harbor deployment mechanic, not the in-memory `FabricConfig` pattern this skill
  teaches — follow the code-review example for consumer integration code, and
  treat Harbor as platform context where a config crosses into a task container
  as a file.

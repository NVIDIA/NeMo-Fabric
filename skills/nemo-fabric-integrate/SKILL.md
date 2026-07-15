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
  config file from consumer code.
- Do not serialize configs, apply file-backed profiles, or resolve profiles by
  name. In-memory callers that genuinely need ordered overlays pass typed
  `FabricProfileConfig` values through `profiles=[...]`.
- Let Fabric own harness control. Do not reimplement start, invoke, or stop
  logic, and do not manage adapter threads, sessions, or processes directly.
- Treat `runtime_id`, `invocation_id`, and `request_id` as opaque correlation
  strings, not parsable or reusable state.

See `references/config-mapping.md` for how to translate a consumer config object
into `FabricConfig`, and for the full list of mechanics that stay hidden behind
this boundary.

## Install And Set Up The Environment

The consumer or its execution environment owns installation; Fabric validates
runtime assumptions but never installs harnesses or credentials at run time.

- Install the SDK with the extras the integration needs, for example
  `pip install "nemo-fabric[runtime]"`; add integration extras such as `harbor`
  when used. From a source checkout, `just build-all` builds the native
  extension and installs the SDK.
- Select a harness adapter (for example `nvidia.fabric.hermes`,
  `nvidia.fabric.codex.cli`, or `nvidia.fabric.langchain.deepagents`) and install
  that adapter's own dependencies and harness binaries.
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

The repository `examples/code_review_agent/` shows this pattern end to end with
complete Hermes, Codex CLI, Deep Agents, environment, MCP, and telemetry
variants. Reuse it rather than duplicating config construction.

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
- Use `doctor(...)` to catch missing dependencies, unsupported settings, and
  environment problems, including unsupported `harness.settings`. Its aggregate
  `status` is `pass`, `warn`, or `fail`.
- Use `resolve(...)` when only the normalized effective config is needed, with
  no adapter resolution.

## Consume Results And Handle Errors

Every invocation that reaches the adapter boundary returns a normalized
`RunResult`, even when the harness invocation itself failed. Inspect the failure
fields before reading output:

```python
result = await fabric.run(config, base_dir=base, input="Review the changes.")

if result.error is not None:
    handle_failure(result.status, result.error, result.events)
else:
    use_output(result.output, result.artifacts, result.telemetry)
```

- `error` is `None` on success; read `status`, `error`, and `events` before
  processing `output`.
- Capture `artifacts` and `telemetry` references as the returned evidence for
  platforms and evaluations. Store and log `runtime_id`, `invocation_id`, and
  `request_id` separately as opaque strings.
- Catch `FabricError` subclasses for lifecycle failures that prevent a
  normalized result: `FabricConfigError`, `FabricCapabilityError`,
  `FabricRuntimeError`, `FabricStateError`, and `FabricNativeUnavailableError`.
- The consumer owns retries and failure policy; Fabric does not retry by
  default. `run(...)` and `async with` runtimes attempt cleanup automatically,
  so prefer them over manual `stop()`.

See `references/results-and-errors.md` for the full result-field and error
inventory, and `references/sdk-api-inventory.md` for when to use each `Fabric`
and `Runtime` method.

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
- [ ] `FabricError` subclasses are handled and runtime cleanup is guaranteed through `run(...)` or `async with`.
- [ ] Correlation IDs are stored and logged as opaque strings.
- [ ] Focused integration tests pass and Fabric validation (`plan`/`doctor`, tests) succeeds.

## Related Documentation

Link to these canonical sources instead of duplicating them:

- Python SDK guide: `docs/sdk/python.mdx`
- Getting started and installation: `docs/getting-started/overview.mdx`
- Generated API reference (exact signatures): `docs/reference/api/python-library-reference/`
- Canonical in-memory config example: `examples/code_review_agent/`
- Platform and evaluation-harness integration: `examples/harbor/` and
  `python/src/nemo_fabric/integrations/harbor/`

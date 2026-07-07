# Python SDK Guide

<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

This guide describes the public Python SDK shape for NeMo Fabric. The generated
API reference remains the source for exact signatures; this page explains how the
pieces are intended to fit together.

## Principles

- `Fabric` is the primary SDK entrypoint.
- Python callers can use either an agent package path or a typed `FabricConfig`.
- Path-backed calls use profile names from the agent package.
- Typed-config calls use ordered profile mappings, not a public profile class.
- `run(...)` is the one-shot convenience path.
- `start_session(...)` is the reusable multi-turn path.
- Consumers own scheduling, retries, tenancy, and product-level orchestration.

## Agent Sources

Path-backed usage is best for CLI parity, examples, CI, and reproducibility:

```python
from nemo_fabric import Fabric

fabric = Fabric()
plan = fabric.plan(
    "examples/code-review-agent",
    profiles=["hermes_sdk", "mcp_github"],
)
```

Typed-config usage is best when Platform, Harbor, Gym, or another consumer
already owns the top-level job/deployment config:

```python
from nemo_fabric import Fabric, FabricConfig

config = FabricConfig.from_mapping(
    {
        "metadata": {"name": "code-review-agent"},
        "harness": {"adapter_id": "nvidia.fabric.hermes.sdk"},
        "models": {
            "default": {
                "provider": "nvidia",
                "model": "nvidia/nemotron-3-nano-30b-a3b",
            }
        },
        "runtime": {"input_schema": "chat", "output_schema": "message"},
    }
)
```

Raw dictionaries are not accepted directly as agent sources. Convert them with
`FabricConfig.from_mapping(...)` so validation and extension preservation are
explicit.

## Config Helpers

Typed config exposes small authoring helpers for common capability edits. These
helpers mutate the config before planning or starting a runtime; they do not
modify already-started runtimes.

```python
config.add_skill_path("./skills/code-review")
config.add_mcp_server(
    "github",
    transport="streamable-http",
    url="${GITHUB_MCP_URL}",
    exposure="harness_native",
)
config.enable_relay(output_dir="./artifacts/relay")
```

Use profile mappings for ordered variations:

```python
profiles = [
    {
        "name": "github_mcp",
        "mcp": {
            "servers": {
                "github": {
                    "transport": "streamable-http",
                    "url": "${GITHUB_MCP_URL}",
                    "exposure": "fabric_managed",
                }
            }
        },
    }
]
```

## Planning And Diagnostics

`resolve(...)` validates and normalizes config. `plan(...)` resolves adapter and
runtime capabilities. `doctor(...)` performs preflight checks.

```python
async with Fabric() as fabric:
    effective = fabric.resolve(config)
    plan = fabric.plan(config, profiles=profiles, base_dir="examples/code-review-agent")
    report = await fabric.doctor(
        config,
        profiles=profiles,
        base_dir="examples/code-review-agent",
    )
```

`base_dir` is only valid with typed config. It gives relative paths in the config
the same anchor an agent package directory would provide.

## One-Shot Runs

`run(...)` performs one complete lifecycle: plan, start runtime, invoke once,
collect result/artifacts, and stop runtime.

```python
from nemo_fabric import RunRequest

request = RunRequest(
    input="Review the workspace changes.",
    request_id="job-123-turn-1",
    context={"job_id": "job-123"},
    overrides={"max_iterations": 1},
)

async with Fabric() as fabric:
    result = await fabric.run(
        config,
        base_dir="examples/code-review-agent",
        session_id="job-123",
        request=request,
    )

print(result.status)
print(result.output)
print(result.artifacts)
```

`session_id` is a first-class convenience for passing a caller-owned stable
conversation/task key. It is encoded into request context and rejected if it
conflicts with an existing request context value.

## Sessions

`start_session(...)` starts one reusable runtime and returns a `Session`. The
session serializes turns by default and injects the stable `session_id` into each
request.

```python
async with await Fabric().start_session(
    "examples/code-review-agent",
    profiles=["hermes_session"],
    session_id="review-session-123",
) as session:
    first = await session.invoke(input="Inspect the repository")
    second = await session.invoke(input="Now review the latest patch")
```

Use `Session.stream(...)` for buffered event/result iteration. Use
`Session.update(...)` and `Session.cancel(...)` only when the selected adapter
advertises those capabilities; unsupported operations raise
`FabricCapabilityError`.

## Concurrency Boundary

Fabric does not schedule jobs, manage queues, or own rollout parallelism. Those
remain consumer responsibilities. Fabric keeps each runtime/session object safe
to call, permits multiple independent runtimes in one process, serializes
same-session invocations by default, and returns structured status/errors/logs
for the consumer to propagate.

## Errors

All public SDK errors inherit from `FabricError`.

- `FabricConfigError`: invalid source, config, profile, request, or override.
- `FabricCapabilityError`: selected adapter does not support the requested
  operation.
- `FabricRuntimeError`: runtime startup, invocation, or shutdown failed before a
  normalized result could be returned.
- `FabricStateError`: invalid session state transition.
- `FabricNativeUnavailableError`: native extension is not installed.

Consumers own job-level retries. Fabric reports structured failure metadata and
performs best-effort runtime cleanup for runtimes it starts.

## CLI Relationship

The CLI is file-first. It loads `agent.yaml`, applies profile files in order, and
executes through the same Rust core. The SDK is config-first for Python
consumers. A path-backed SDK call and an equivalent CLI call should resolve to
the same effective config and run plan.

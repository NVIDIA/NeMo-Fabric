# Python SDK Contract

## Scope and Status

This is the target public API. MVP includes typed sources, resolution, planning,
diagnostics, oneshot runs, sessions, typed results and errors, capability checks,
and a stable buffered `stream()` shape. Runtime updates, progressive streaming,
and service mode may follow MVP. Unsupported operations raise
`FabricCapabilityError`.

## Design

Fabric owns runtime execution; callers own orchestration, servers, tenancy,
persistence, and product workflows. The SDK uses one source abstraction, one
ordered `profiles` argument, and one method per lifecycle operation.

## Common Types

All values crossing the Python/native boundary are JSON-shaped.

```python
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Literal, overload

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
PathSource = str | os.PathLike[str]
AgentSource = PathSource | FabricConfig
```

Invalid JSON values raise `FabricConfigError` before native execution.

## Client and CLI

```python
class FabricClient:
    def __init__(self) -> None: ...
```

`FabricClient` is native-only. The CLI is a separate surface over the same core;
the same file-backed config and profiles produce equivalent contract data.

## Agent Sources and Profiles

Profile types follow the agent source:

```python
@overload
def plan(
    agent: PathSource,
    *,
    profiles: Sequence[str] | None = None,
) -> RunPlan: ...

@overload
def plan(
    agent: FabricConfig,
    *,
    profiles: Sequence[FabricProfileConfig] | None = None,
    base_dir: PathSource | None = None,
) -> RunPlan: ...
```

The same overload pattern applies to `resolve`, `doctor`, `run`,
`start_session`, and `start_service`.

- Strings are paths, never raw config, adapter IDs, or agent names.
- Paths use ordered profile names; `FabricConfig` uses ordered
  `FabricProfileConfig` objects. Mixed stacks and bare strings are rejected.
- `base_dir` applies only to `FabricConfig`.
- Raw mappings require explicit `from_mapping(...)` conversion.
- Equivalent file and typed sources produce equivalent configs and plans.

There is no public singular `profile` alias or public `plan_config`,
`run_config`, `doctor_config`, `start`, or `start_config` family.

## Typed Config

Typed config uses the same schema as `agent.yaml`.

```python
class MetadataConfig:
    name: str
    description: str | None
    extra_fields: Mapping[str, JSONValue]

class HarnessConfig:
    adapter_id: str
    resolution: str | None
    settings: Mapping[str, JSONValue]
    extra_fields: Mapping[str, JSONValue]

class RuntimeConfig:
    mode: Literal["oneshot", "session", "service"]
    transport: str | None
    input_schema: str | None
    output_schema: str | None
    artifacts: str | Path | None
    extra_fields: Mapping[str, JSONValue]

class EnvironmentConfig:
    provider: str
    workspace: str | Path | None
    artifacts: str | Path | None
    settings: Mapping[str, JSONValue]
    metadata: Mapping[str, JSONValue]
    extra_fields: Mapping[str, JSONValue]

class FabricConfig:
    schema_version: str
    metadata: MetadataConfig
    harness: HarnessConfig
    runtime: RuntimeConfig
    environment: EnvironmentConfig | None
    models: Mapping[str, Mapping[str, JSONValue]]
    mcp: Mapping[str, JSONValue] | None
    skills: Mapping[str, JSONValue] | None
    telemetry: Mapping[str, JSONValue] | None
    profiles: Mapping[str, JSONValue] | None
    tools: JSONValue
    extra_fields: Mapping[str, JSONValue]

    @classmethod
    def from_mapping(cls, value: Mapping[str, JSONValue]) -> FabricConfig: ...

    def to_mapping(self) -> dict[str, JSONValue]: ...

class FabricProfileConfig:
    schema_version: str
    name: str
    description: str | None
    harness: HarnessConfig | Mapping[str, JSONValue] | None
    runtime: RuntimeConfig | Mapping[str, JSONValue] | None
    environment: EnvironmentConfig | Mapping[str, JSONValue] | None
    models: Mapping[str, Mapping[str, JSONValue]] | None
    mcp: Mapping[str, JSONValue] | None
    skills: Mapping[str, JSONValue] | None
    telemetry: Mapping[str, JSONValue] | None
    tools: JSONValue
    extra_fields: Mapping[str, JSONValue]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, JSONValue],
    ) -> FabricProfileConfig: ...

    def to_mapping(self) -> dict[str, JSONValue]: ...
```

- `metadata` and `harness` are required; names, adapter IDs, and runtime mode
  are validated.
- Defaults are the v1alpha1 schemas, `oneshot`, and local environment.
  Omitted runtime transport and schemas resolve to `library`, `text`, and
  `text`.
- Constructors reject unknown keywords. Mapping conversion preserves unknown
  fields through `extra_fields` and returns deep copies.
- Profile sections are partial recursive overlays. They are validated as a
  complete config after merging with the base and earlier profiles.
- Config is mutable before resolution; plans and runtimes are snapshots.
- Unstable model, MCP, skill, telemetry, and tool shapes remain JSON mappings.
- `FabricConfig.profiles` controls discovery; lifecycle `profiles` selects
  overlays.

## Config Extension

Normalized fields represent cross-harness concepts. Adapter-only fields belong
in `HarnessConfig.settings`. Unknown fields are preserved but are not supported
until the SDK recognizes them.

## Inspection Types

Inspection and result models are typed, read-only mappings that preserve unknown
fields.

```python
class AdapterInfo:
    adapter_id: str
    harness: str
    adapter_kind: str
    metadata: Mapping[str, JSONValue]

class RuntimeCapabilities:
    session: bool
    service: bool
    streaming: bool
    updates: bool
    cancellation: bool
    concurrent_invocations: bool
    metadata: Mapping[str, JSONValue]

class EffectiveConfig:
    agent_name: str
    profiles: Sequence[str]
    agent_root: Path
    config_path: Path | None
    config_root: Path
    config: FabricConfig

class RunPlan:
    effective_config: EffectiveConfig
    agent_name: str
    profiles: Sequence[str]
    adapter: AdapterInfo
    capabilities: RuntimeCapabilities

class DoctorCheck:
    name: str
    status: Literal["pass", "warn", "fail"]
    message: str
    metadata: Mapping[str, JSONValue]

class DoctorReport:
    agent_name: str
    profiles: Sequence[str]
    status: Literal["pass", "warn", "fail"]
    checks: Sequence[DoctorCheck]
```

`harness` is the stable machine-readable harness identifier. `adapter_id`
identifies its Fabric adapter implementation, while `adapter_kind` identifies
the execution mechanism.

## Client API

These compact signatures use the source-specific overloads above.

```python
class FabricClient:
    def resolve(
        self,
        agent: AgentSource,
        *,
        profiles: Sequence[str] | Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
    ) -> EffectiveConfig: ...

    def plan(
        self,
        agent: AgentSource,
        *,
        profiles: Sequence[str] | Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
    ) -> RunPlan: ...

    async def doctor(
        self,
        agent: AgentSource,
        *,
        profiles: Sequence[str] | Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
    ) -> DoctorReport: ...

    async def run(
        self,
        agent: AgentSource,
        *,
        profiles: Sequence[str] | Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
        input: JSONValue = None,
        input_file: str | Path | None = None,
        request: RunRequest | Mapping[str, JSONValue] | None = None,
        request_file: str | Path | None = None,
        request_id: str | None = None,
        context: Mapping[str, JSONValue] | None = None,
        overrides: Mapping[str, JSONValue] | None = None,
    ) -> RunResult: ...

    async def start_session(
        self,
        agent: AgentSource,
        *,
        profiles: Sequence[str] | Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
        session_id: str | None = None,
        overrides: Mapping[str, JSONValue] | None = None,
    ) -> Session: ...

    async def start_service(
        self,
        agent: AgentSource,
        *,
        profiles: Sequence[str] | Sequence[FabricProfileConfig] | None = None,
        base_dir: PathSource | None = None,
        service_id: str | None = None,
        overrides: Mapping[str, JSONValue] | None = None,
    ) -> RuntimeService: ...
```

`resolve()` resolves config only; `plan()` resolves adapters and capabilities.

## Requests and Overrides

```python
class RunRequest:
    input: JSONValue
    request_id: str
    context: Mapping[str, JSONValue]
    overrides: Mapping[str, JSONValue] | None
    extra_fields: Mapping[str, JSONValue]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, JSONValue],
    ) -> RunRequest: ...

    def to_mapping(self) -> dict[str, JSONValue]: ...
```

At most one input source is accepted; none means empty text. File inputs apply
only to `run()`. Request IDs default automatically, context is caller-owned, and
unknown fields are preserved. Complete requests reject separate request fields.
There is no `from_text()` or `input_text` alias.

Merge precedence is:

```text
base config < ordered profiles < service < session < invocation
```

Objects merge recursively; later scalars, arrays, and `null` replace earlier
values. Lists are not concatenated. Runtime changes are capability-gated.

## Oneshot Runs

`run()` resolves, plans, creates, invokes, collects, and destroys one runtime.
Cleanup failure raises `FabricRuntimeError` even after a successful invocation.

## Sessions

A `Session` owns one runtime and orders turns unless concurrency is declared.

```python
class SessionInfo:
    session_id: str
    runtime_id: str
    agent_name: str
    profiles: Sequence[str]
    harness: str
    adapter_id: str
    adapter_kind: str
    status: Literal["active", "stopped", "failed"]
    capabilities: RuntimeCapabilities

class Session:
    session_id: str
    runtime_id: str
    info: SessionInfo

    async def invoke(
        self,
        *,
        input: JSONValue = None,
        request: RunRequest | Mapping[str, JSONValue] | None = None,
        request_id: str | None = None,
        context: Mapping[str, JSONValue] | None = None,
        overrides: Mapping[str, JSONValue] | None = None,
    ) -> RunResult: ...

    async def stream(
        self,
        *,
        input: JSONValue = None,
        request: RunRequest | Mapping[str, JSONValue] | None = None,
        request_id: str | None = None,
        context: Mapping[str, JSONValue] | None = None,
        overrides: Mapping[str, JSONValue] | None = None,
    ) -> AsyncIterator[FabricEvent | RunResult]: ...

    async def update(self, update: RuntimeUpdate) -> RuntimeUpdateResult: ...
    async def cancel(self) -> None: ...
    async def stop(self) -> None: ...
```

- `Session.info` copies plan and runtime identity; it never derives one identity
  field from another.
- `cancel()` targets the current invocation, leaves a supported runtime active,
  and raises `FabricCapabilityError` when unsupported.
- `stop()` rejects active work and destroys an idle runtime exactly once.
  Invoke, cancel, and stop transitions are serialized.

## Services

Service mode reuses one runtime. `RuntimeService` owns it; `ServiceSession` owns
only logical state. Callers retain serving, authentication, tenancy, persistence,
and scheduling.

```python
class ServiceInfo:
    service_id: str
    runtime_id: str
    agent_name: str
    profiles: Sequence[str]
    harness: str
    adapter_id: str
    adapter_kind: str
    status: Literal["active", "stopped", "failed"]
    capabilities: RuntimeCapabilities

class ServiceSessionInfo:
    service_id: str
    session_id: str
    runtime_id: str
    status: Literal["active", "closed", "failed"]

class ServiceSession:
    service_id: str
    session_id: str
    info: ServiceSessionInfo

    async def invoke(...) -> RunResult: ...
    async def stream(...) -> AsyncIterator[FabricEvent | RunResult]: ...
    async def update(self, update: RuntimeUpdate) -> RuntimeUpdateResult: ...
    async def cancel(self) -> None: ...
    async def close(self) -> None: ...

class RuntimeService:
    service_id: str
    runtime_id: str
    info: ServiceInfo

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        context: Mapping[str, JSONValue] | None = None,
        overrides: Mapping[str, JSONValue] | None = None,
    ) -> ServiceSession: ...

    async def get_session(self, session_id: str) -> ServiceSession: ...
    async def invoke(...) -> RunResult: ...
    async def stream(...) -> AsyncIterator[FabricEvent | RunResult]: ...
    async def cancel(self, request_id: str) -> None: ...
    async def update(self, update: RuntimeUpdate) -> RuntimeUpdateResult: ...
    async def close_session(self, session_id: str) -> None: ...
    async def stop(self) -> None: ...
```

Abbreviated invocation methods match `Session`. `ServiceSession.close()` releases
logical state; `RuntimeService.stop()` closes idle sessions and the runtime.
Direct service calls are stateless. IDs are correlation, not authorization.

## Streaming and Updates

`stream()` yields events and one terminal result. Adapters may buffer; callers
must not depend on buffering. Event kinds and metadata are additive.

```python
class RuntimeUpdate:
    overrides: Mapping[str, JSONValue]
    metadata: Mapping[str, JSONValue]

class RuntimeUpdateResult:
    status: Literal["applied", "partially_applied", "rejected"]
    applied: Mapping[str, JSONValue]
    rejected: Mapping[str, JSONValue]
    reason: str | None
```

The target determines update scope. Unsupported updates raise
`FabricCapabilityError`; supported updates report applied and rejected fields.

## Results and Identity

```python
class ErrorInfo:
    stage: str
    code: str
    message: str
    retryable: bool
    metadata: Mapping[str, JSONValue]

class ArtifactRef:
    name: str
    kind: str
    path: Path
    media_type: str | None
    metadata: Mapping[str, JSONValue]

class ArtifactManifest:
    root: Path | None
    artifacts: Sequence[ArtifactRef]

class TelemetryRef:
    provider: str
    kind: str
    uri: str | None
    trace_id: str | None
    metadata: Mapping[str, JSONValue]

class FabricEvent:
    event_id: str
    timestamp_millis: int
    kind: str
    message: str
    metadata: Mapping[str, JSONValue]

class RunResult:
    agent_name: str
    profiles: Sequence[str]
    harness: str
    adapter_kind: str
    adapter_id: str
    runtime_id: str
    invocation_id: str
    request_id: str
    status: Literal["succeeded", "failed", "cancelled"]
    output: JSONValue
    error: ErrorInfo | None
    artifacts: ArtifactManifest
    telemetry: Sequence[TelemetryRef]
    events: Sequence[FabricEvent]
    metadata: Mapping[str, JSONValue]
    extra_fields: Mapping[str, JSONValue]
```

`profiles` is the full ordered stack; no singular field exists. Harness, adapter,
and runtime identities stay distinct. Normalized harness failure returns a
failed result; lifecycle failure raises a typed exception.

## Errors

```python
class FabricError(RuntimeError):
    stage: str | None
    code: str | None
    retryable: bool
    details: Mapping[str, JSONValue]

class FabricConfigError(FabricError): ...
class FabricRuntimeError(FabricError): ...
class FabricStateError(FabricRuntimeError): ...
class FabricCapabilityError(FabricRuntimeError): ...
class FabricNativeUnavailableError(FabricRuntimeError): ...
```

Invalid input, unsupported operations, bad handle state, and lifecycle failure
map to the four specific errors above. Native exceptions never leak. Python task
cancellation remains `asyncio.CancelledError` with deterministic cleanup.

## Compatibility

- Unknown fields survive Python, native, adapter, and serialization boundaries.
- New optional fields and event kinds are additive.
- New required fields require a schema-version change.
- Capabilities declare support for session, service, streaming, updates,
  cancellation, and concurrency.
- Public symbols and signatures are covered by static type and API contract
  tests.
- Aliases are added only for migration from an actually released API.

## Non-Goals

The SDK does not own external server lifecycle, authentication, tenancy policy,
durable job persistence, UI state, evaluation scoring, or caller-specific
orchestration.

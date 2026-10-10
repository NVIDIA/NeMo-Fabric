---
title: "Client"
slug: "/reference/api/python-library-reference/client"
description: "Resolve, plan, diagnose, and run agents with NVIDIA NeMo Fabric."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.client`

Native Python client for resolving and running NVIDIA NeMo Fabric agents.



---


## <kbd>class</kbd> `Fabric`

Primary Python entrypoint for NeMo Fabric.

Every config-dependent lifecycle method accepts a complete, typed ``FabricConfig`` plus an optional ``base_dir`` used to resolve relative paths. Compose variants in Python before calling the SDK. The ``doctor()``, ``plan()``, and ``run()`` results are typed, read-only mapping models. ``start_runtime()`` returns an active local ``Runtime`` handle. Explicit environment users call either ``prepare_environment()`` or ``attach_environment()``, then ``start_runtime_in()`` and ``release_environment()`` separately.

``Fabric`` uses the native Rust extension. SDK calls raise ``FabricNativeUnavailableError`` when the native extension is not installed.

See the Getting Started overview for runnable single-invocation, typed-config, and multi-turn examples.


### <kbd>method</kbd> `__init__`

```python
def __init__() -> None
```








---


### <kbd>method</kbd> `attach_environment`

```python
async def attach_environment(
    config: FabricConfig,
    reference: EnvironmentReference,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> EnvironmentHandle
```

Verify and attach to an existing caller-owned environment.

Attachment does not create the provider resource and does not grant Fabric deletion authority. The returned handle can be passed to ``start_runtime_in()`` and later to ``release_environment()`` to detach.



**Args:**

 - <b>`config`</b>:  Complete typed ``FabricConfig`` with caller-owned environment settings.
 - <b>`reference`</b>:  Provider-specific identity of the existing resource.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.



**Returns:**
 A verified, immutable ``EnvironmentHandle``.



**Raises:**

 - <b>`FabricConfigError`</b>:  If config, reference, or returned handle is invalid.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not installed.
 - <b>`FabricRuntimeError`</b>:  If environment verification or attachment fails.

---


### <kbd>method</kbd> `attach_service`

```python
async def attach_service(
    config: FabricConfig,
    reference: ServiceReference,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> Service
```

Validate and attach to a caller-owned long-lived service.

The adapter validates the reference against the normalized configuration and writes any resolved credentials only to private, process-local connection material. Releasing the returned service detaches NeMo Fabric without stopping the caller-owned service.



**Args:**

 - <b>`config`</b>:  Complete typed ``FabricConfig``.
 - <b>`reference`</b>:  Adapter-specific endpoint and credential references.  Do not include credential values.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.



**Returns:**
 An active caller-owned ``Service``. Use it as an asynchronous context manager to guarantee detach.



**Raises:**

 - <b>`FabricConfigError`</b>:  If the reference, plan, or adapter configuration  is invalid.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.
 - <b>`FabricRuntimeError`</b>:  If attachment or validation fails.

---


### <kbd>method</kbd> `doctor`

```python
async def doctor(
    config: FabricConfig,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> DoctorReport
```

Diagnose a planned agent without starting its runtime.

Doctor checks the resolved adapter, capability mappings, and declared environment requirements using the native NeMo Fabric core.



**Args:**

 - <b>`config`</b>:  Complete typed ``FabricConfig``.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.



**Returns:**
 A ``DoctorReport`` with aggregate status and ordered checks.



**Raises:**

 - <b>`FabricConfigError`</b>:  If inputs or native diagnostic output are  invalid.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.

---


### <kbd>method</kbd> `plan`

```python
def plan(
    config: FabricConfig,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> RunPlan
```

Resolve a complete typed configuration into an immutable execution plan.

Planning resolves the selected adapter and reports optional runtime capabilities such as streaming, updates, and cancellation. Planning does not start the runtime.



**Args:**

 - <b>`config`</b>:  Complete typed ``FabricConfig``. Raw mappings are not  accepted.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.



**Returns:**
 A ``RunPlan`` containing the canonical config, path context, adapter, and declared runtime capabilities.



**Raises:**

 - <b>`FabricConfigError`</b>:  If the config or adapter resolution is invalid.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.

---


### <kbd>method</kbd> `prepare_environment`

```python
async def prepare_environment(
    config: FabricConfig,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> EnvironmentHandle
```

Prepare an execution environment.

The returned handle is independent of any runtime session. The caller owns the lifecycle decision and must eventually pass it to ``release_environment()``.



**Args:**

 - <b>`config`</b>:  Complete typed ``FabricConfig`` describing the environment.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.



**Returns:**
 A typed, immutable ``EnvironmentHandle``.



**Raises:**

 - <b>`FabricConfigError`</b>:  If config resolution or the returned handle is invalid.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not installed.
 - <b>`FabricRuntimeError`</b>:  If environment preparation fails.

---


### <kbd>method</kbd> `prepare_service`

```python
async def prepare_service(
    config: FabricConfig,
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> Service
```

Create and supervise a Fabric-owned long-lived service.

The selected adapter maps the normalized configuration into its service configuration. The returned handle is process-local and can be shared by multiple runtimes created from the same resolved plan.



**Args:**

 - <b>`config`</b>:  Complete typed ``FabricConfig``.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.



**Returns:**
 An active Fabric-owned ``Service``. Use it as an asynchronous context manager to guarantee shutdown.



**Raises:**

 - <b>`FabricConfigError`</b>:  If planning or adapter configuration is invalid.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.
 - <b>`FabricRuntimeError`</b>:  If service startup fails.

---


### <kbd>method</kbd> `release_environment`

```python
async def release_environment(environment: EnvironmentHandle) -> None
```

Release or detach a prepared environment through its provider.

Local and externally owned environments detach without deletion. Provider-managed, Fabric-owned environments may be deleted according to their normalized ownership contract.



**Args:**

 - <b>`environment`</b>:  Handle returned by ``prepare_environment()`` or  ``attach_environment()``.



**Raises:**

 - <b>`FabricConfigError`</b>:  If ``environment`` is not a typed handle.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not installed.
 - <b>`FabricRuntimeError`</b>:  If release or detach fails.

---


### <kbd>method</kbd> `run`

```python
async def run(
    config: FabricConfig,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    input: Any = None,
    request: RunRequest | None = None,
    expected_descriptor_sha256: str | None = None,
) -> RunResult
```

Execute one complete start, invoke, and stop lifecycle.

``input`` and ``request`` are mutually exclusive. Omitting both produces an empty text input. Use ``RunRequest`` when the invocation needs a caller-owned request ID, context, or overrides. NeMo Fabric attempts to stop a started runtime even when invocation fails.



**Args:**

 - <b>`config`</b>:  Complete typed ``FabricConfig``.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.
 - <b>`input`</b>:  JSON-compatible invocation input.
 - <b>`request`</b>:  Complete validated ``RunRequest``.
 - <b>`expected_descriptor_sha256`</b>:  Optional host-inspection fingerprint.  Reject descriptor drift before starting the runtime.



**Returns:**
 The normalized ``RunResult``, including output, artifacts, telemetry references, lifecycle events, and structured error data.



**Raises:**

 - <b>`FabricConfigError`</b>:  If input and request are combined, request data is not  JSON-compatible, or config resolution fails.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.
 - <b>`FabricRuntimeError`</b>:  If the native runtime lifecycle fails before a  normalized result can be returned.

---


### <kbd>method</kbd> `start_runtime`

```python
async def start_runtime(
    config: FabricConfig,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
    streaming: bool = False,
    launch_collector: bool | None = None,
    completion_wait_timeout: float = 1.0,
    service: Service | None = None,
    expected_descriptor_sha256: str | None = None,
) -> Runtime
```

Start a stateful runtime for one or more ordered invocations.

Each call starts a new logical runtime. Runtime-scoped overrides are recursively merged below invocation-scoped overrides. With NVIDIA NeMo Relay enabled, ``streaming=True`` uses collector-backed streaming. By default, streaming starts an embedded collector. Set ``launch_collector=False`` to use an externally managed collector. Pi requires the embedded collector because its ATOF records do not carry NeMo Fabric request IDs.



**Args:**

 - <b>`config`</b>:  Complete typed ``FabricConfig``.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.
 - <b>`overrides`</b>:  JSON-compatible overrides applied to every invocation  in the runtime unless superseded by invocation overrides.
 - <b>`streaming`</b>:  Whether to enable collector-backed NeMo Relay ATOF  streaming for ``Runtime.invoke_stream()``.
 - <b>`launch_collector`</b>:  Whether to launch an embedded collector. ``None``  defaults to ``True`` when streaming is enabled. ``False`` uses  an externally managed collector. Pi does not support ``False``.  This argument cannot be set unless ``streaming=True``.
 - <b>`completion_wait_timeout`</b>:  Maximum seconds the embedded collector  waits for a Pi ``agent_settled`` marker after invocation. Increase  this value when Relay delivery can be delayed. This value is  ignored unless Pi streaming uses the embedded collector.
 - <b>`service`</b>:  Optional prepared or attached service. When supplied, the  runtime connects to that service instead of creating its own.
 - <b>`expected_descriptor_sha256`</b>:  Optional descriptor fingerprint from host  inspection. Matching metadata does not qualify an attached service's  deployment-owned skills or MCP configuration.



**Returns:**
 An active ``Runtime``. Use it as an asynchronous context manager to guarantee runtime shutdown.



**Raises:**

 - <b>`FabricConfigError`</b>:  If inputs or overrides are invalid, streaming is  requested without NeMo Relay enabled, ``launch_collector`` is  set without streaming, Pi is configured with an external  collector, or an external collector has no sink.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.
 - <b>`FabricRuntimeError`</b>:  If runtime startup fails.

---


### <kbd>method</kbd> `start_runtime_in`

```python
async def start_runtime_in(
    config: FabricConfig,
    environment: EnvironmentHandle,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
    streaming: bool = False,
) -> Runtime
```

Start one stateful runtime in an explicitly prepared or attached environment.

Starting or stopping the runtime does not release ``environment``. One runtime session can be active in an environment at a time. After it stops, the consumer can start another session or release the environment.



**Args:**

 - <b>`config`</b>:  Complete typed ``FabricConfig`` matching the environment.
 - <b>`environment`</b>:  Handle returned by ``prepare_environment()`` or  ``attach_environment()``.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.
 - <b>`overrides`</b>:  JSON-compatible runtime-scoped invocation overrides.
 - <b>`streaming`</b>:  Whether to provision NeMo Relay ATOF streaming.



**Returns:**
 An active ``Runtime`` bound to ``environment``.



**Raises:**

 - <b>`FabricConfigError`</b>:  If inputs are invalid or the handle does not match the plan.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not installed.
 - <b>`FabricRuntimeError`</b>:  If runtime startup fails.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

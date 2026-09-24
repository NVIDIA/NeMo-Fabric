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

Every lifecycle method accepts a complete, typed ``FabricConfig`` plus an optional ``base_dir`` used to resolve relative paths. Compose variants in Python before calling the SDK. The ``doctor()``, ``plan()``, and ``run()`` results are typed, read-only mapping models. ``start_runtime()`` returns an active ``Runtime`` handle.

``Fabric`` uses the native Rust extension. SDK calls raise ``FabricNativeUnavailableError`` when the native extension is not installed.

See the Getting Started overview for runnable single-invocation, typed-config, and multi-turn examples.


### <kbd>method</kbd> `__init__`

```python
def __init__() -> None
```








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


### <kbd>method</kbd> `discover`

```python
def discover(
    *,
    discovery: DiscoveryConfig | None = None,
    base_dir: str | os.PathLike[str] | None = None,
) -> DescriptorCatalog
```

Return the adapter and target descriptors that planning can select.

The catalog uses the same registry as ``plan()``: bundled descriptors, descriptors installed in the selected Python environment, and any ``discovery.local_paths``. Entries are sorted by exact identifier and keep every source of an identical descriptor. Malformed or ambiguous metadata raises ``FabricConfigError``. Discovery does not import adapter runners, start runtimes, or check whether declared requirements are installed.



**Args:**

 - <b>`discovery`</b>:  Optional typed explicit local descriptor paths.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.

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


### <kbd>method</kbd> `run`

```python
async def run(
    config: FabricConfig,
    *,
    base_dir: str | os.PathLike[str] | None = None,
    input: Any = None,
    request: RunRequest | None = None,
) -> RunResult
```

Execute one complete start, invoke, and stop lifecycle.

``input`` and ``request`` are mutually exclusive. Omitting both produces an empty text input. Use ``RunRequest`` when the invocation needs a caller-owned request ID, context, or overrides. NeMo Fabric attempts to stop a started runtime even when invocation fails.



**Args:**

 - <b>`config`</b>:  Complete typed ``FabricConfig``.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.
 - <b>`input`</b>:  JSON-compatible invocation input.
 - <b>`request`</b>:  Complete validated ``RunRequest``.



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



**Returns:**
 An active ``Runtime``. Use it as an asynchronous context manager to guarantee runtime shutdown.



**Raises:**

 - <b>`FabricConfigError`</b>:  If inputs or overrides are invalid, streaming is  requested without NeMo Relay enabled, ``launch_collector`` is  set without streaming, Pi is configured with an external  collector, or an external collector has no sink.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.
 - <b>`FabricRuntimeError`</b>:  If runtime startup fails.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

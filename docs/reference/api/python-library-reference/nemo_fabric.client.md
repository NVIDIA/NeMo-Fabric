---
title: "Client"
slug: "/reference/api/python-library-reference/client"
description: "Resolve, plan, diagnose, and run agents with Fabric."
---
{/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 */}

# <kbd>module</kbd> `nemo_fabric.client`
Native Python client for resolving and running NeMo Fabric agents.



---


## <kbd>class</kbd> `Fabric`
Primary Python entrypoint for NeMo Fabric.

Every lifecycle method accepts a complete, typed ``FabricConfig`` plus an optional ``base_dir`` used to resolve relative paths. Compose variants in Python before calling the SDK. All inspection and execution APIs return typed, read-only mapping models.

``Fabric`` is native-only. The ``nemo-fabric`` CLI is backed by this public SDK; SDK calls raise ``FabricNativeUnavailableError`` when the native extension is not installed.

See the Getting Started overview for runnable one-shot, typed-config, and multi-turn examples.




---


### <kbd>method</kbd> `doctor`

```python
doctor(
    config: 'FabricConfig',
    base_dir: 'str | PathLike[str] | None' = None
) → DoctorReport
```

Diagnose a planned agent without starting its runtime.

Doctor checks the resolved adapter, capability mappings, and declared environment requirements using the native Fabric core.



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
plan(
    config: 'FabricConfig',
    base_dir: 'str | PathLike[str] | None' = None
) → RunPlan
```

Resolve a complete typed configuration into an immutable execution plan.

Planning resolves the selected adapter and reports optional runtime capabilities such as streaming, updates, and cancellation. Planning does not start the runtime.



**Args:**

 - <b>`config`</b>:  Complete typed ``FabricConfig``. Raw mappings are not  accepted.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.



**Returns:**
 A ``RunPlan`` containing the effective config, adapter, and declared runtime capabilities.



**Raises:**

 - <b>`FabricConfigError`</b>:  If the config or adapter resolution is invalid.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.

---


### <kbd>method</kbd> `resolve`

```python
resolve(
    config: 'FabricConfig',
    base_dir: 'str | PathLike[str] | None' = None
) → EffectiveConfig
```

Resolve a complete typed configuration.

Resolution validates and normalizes configuration but does not resolve an adapter or compute runtime capabilities. Use ``plan()`` when those execution details are required.



**Args:**

 - <b>`config`</b>:  Complete typed ``FabricConfig``. Raw mappings are not  accepted; convert them with  ``FabricConfig.from_mapping()``.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.



**Returns:**
 The normalized ``EffectiveConfig`` snapshot.



**Raises:**

 - <b>`FabricConfigError`</b>:  If the config is invalid.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.

---


### <kbd>method</kbd> `run`

```python
run(
    config: 'FabricConfig',
    base_dir: 'str | PathLike[str] | None' = None,
    input: 'Any' = None,
    request: 'RunRequest | None' = None
) → RunResult
```

Execute one complete start, invoke, and stop lifecycle.

``input`` and ``request`` are mutually exclusive. Omitting both produces an empty text input. Use ``RunRequest`` when the invocation needs a caller-owned request ID, context, or overrides. Fabric attempts to stop a started runtime even when invocation fails.



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
start_runtime(
    config: 'FabricConfig',
    base_dir: 'str | PathLike[str] | None' = None,
    overrides: 'Mapping[str, Any] | None' = None
) → Runtime
```

Start a stateful runtime for one or more ordered invocations.

Each call starts a new logical runtime. Runtime-scoped overrides are recursively merged below invocation-scoped overrides.



**Args:**

 - <b>`config`</b>:  Complete typed ``FabricConfig``.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths.
 - <b>`overrides`</b>:  JSON-compatible overrides applied to every invocation  in the runtime unless superseded by invocation overrides.



**Returns:**
 An active ``Runtime``. Use it as an asynchronous context manager to guarantee runtime shutdown.



**Raises:**

 - <b>`FabricConfigError`</b>:  If inputs or overrides are invalid.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.
 - <b>`FabricRuntimeError`</b>:  If runtime startup fails.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

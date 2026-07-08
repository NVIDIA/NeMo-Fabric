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

The client accepts either a path-backed agent package or a typed ``FabricConfig``. Path-backed sources select profiles by name; typed sources accept ordered ``FabricProfileConfig`` values and may use ``base_dir`` to resolve relative paths. All inspection and execution APIs return typed, read-only mapping models.

``Fabric`` is native-only. The ``fabric`` CLI is a separate public surface over the same Rust core; SDK calls raise ``FabricNativeUnavailableError`` when the native extension is not installed.

See the Getting Started overview for runnable one-shot, typed-config, and multi-turn examples.




---


### <kbd>method</kbd> `doctor`

```python
doctor(
    agent: 'AgentSource',
    profiles: 'PathProfiles | TypedProfiles | None' = None,
    base_dir: 'PathSource | None' = None
) → DoctorReport
```

Diagnose a planned agent without starting its runtime.

Doctor checks the resolved adapter, capability mappings, and declared environment requirements using the native Fabric core.



**Args:**

 - <b>`agent`</b>:  Agent-package directory or config-file path, or a typed  ``FabricConfig``.
 - <b>`profiles`</b>:  One profile name or an ordered sequence of names for a  path-backed source. For a typed source, an ordered sequence of  ``FabricProfileConfig`` values.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths in a typed  config. Valid only when ``agent`` is a typed config source.



**Returns:**
 A ``DoctorReport`` with aggregate status and ordered checks.



**Raises:**

 - <b>`FabricConfigError`</b>:  If inputs or native diagnostic output are  invalid.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.

---


### <kbd>method</kbd> `plan`

```python
plan(
    agent: 'AgentSource',
    profiles: 'PathProfiles | TypedProfiles | None' = None,
    base_dir: 'PathSource | None' = None
) → RunPlan
```

Resolve an agent source into an immutable execution plan.

Planning applies profiles, resolves the selected adapter, and reports optional runtime capabilities such as streaming, updates, and cancellation. It does not start the runtime.



**Args:**

 - <b>`agent`</b>:  Agent-package directory or config-file path, or a typed  ``FabricConfig``. Raw  mappings are not accepted.
 - <b>`profiles`</b>:  One profile name or an ordered sequence of names for a  path-backed source. For a typed source, an ordered sequence of  ``FabricProfileConfig`` values.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths in a typed  config. Valid only when ``agent`` is a typed config source.



**Returns:**
 A ``RunPlan`` containing the effective config, adapter, and declared runtime capabilities.



**Raises:**

 - <b>`FabricConfigError`</b>:  If the source, profile stack, config, or adapter  resolution is invalid.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.

---


### <kbd>method</kbd> `resolve`

```python
resolve(
    agent: 'AgentSource',
    profiles: 'PathProfiles | TypedProfiles | None' = None,
    base_dir: 'PathSource | None' = None
) → EffectiveConfig
```

Resolve an agent source and its ordered profile overlays.

Resolution validates and normalizes configuration but does not resolve an adapter or compute runtime capabilities. Use ``plan()`` when those execution details are required.



**Args:**

 - <b>`agent`</b>:  Agent-package directory or config-file path, or a typed  ``FabricConfig``. Raw  mappings are not accepted; convert them with  ``FabricConfig.from_mapping()``.
 - <b>`profiles`</b>:  One profile name or an ordered sequence of names for a  path-backed source. For a typed source, an ordered sequence of  ``FabricProfileConfig`` values.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths in a typed  config. Valid only when ``agent`` is a typed config source.



**Returns:**
 The normalized ``EffectiveConfig`` snapshot.



**Raises:**

 - <b>`FabricConfigError`</b>:  If the source, profile stack, or resolved config  is invalid.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.

---


### <kbd>method</kbd> `run`

```python
run(
    agent: 'AgentSource',
    profiles: 'PathProfiles | TypedProfiles | None' = None,
    base_dir: 'PathSource | None' = None,
    input: 'Any' = None,
    input_file: 'str | Path | None' = None,
    request: 'RunRequest | None' = None,
    request_file: 'str | Path | None' = None,
    request_id: 'str | None' = None,
    context: 'Mapping[str, Any] | None' = None,
    overrides: 'Mapping[str, Any] | None' = None
) → RunResult
```

Execute one complete start, invoke, and stop lifecycle.

Exactly zero or one of ``input``, ``input_file``, ``request``, and ``request_file`` may be supplied. Omitting all four produces an empty text input. A complete ``request`` or ``request_file`` cannot be mixed with separate ``request_id``, ``context``, or ``overrides`` fields. Fabric attempts to stop a started runtime even when invocation fails.



**Args:**

 - <b>`agent`</b>:  Agent-package directory or config-file path, or a typed  ``FabricConfig``.
 - <b>`profiles`</b>:  One profile name or an ordered sequence of names for a  path-backed source. For a typed source, an ordered sequence of  ``FabricProfileConfig`` values.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths in a typed  config. Valid only when ``agent`` is a typed config source.
 - <b>`input`</b>:  JSON-compatible invocation input.
 - <b>`input_file`</b>:  UTF-8 file whose contents become the invocation input.
 - <b>`request`</b>:  Complete validated ``RunRequest``.
 - <b>`request_file`</b>:  UTF-8 JSON file containing a complete request.
 - <b>`request_id`</b>:  Caller-owned request identifier. Fabric generates one  when omitted.
 - <b>`context`</b>:  Caller-owned, JSON-compatible request metadata.
 - <b>`overrides`</b>:  JSON-compatible invocation-scoped config overrides.



**Returns:**
 The normalized ``RunResult``, including output, artifacts, telemetry references, lifecycle events, and structured error data.



**Raises:**

 - <b>`FabricConfigError`</b>:  If sources are combined, request data is not  JSON-compatible, or config resolution fails.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.
 - <b>`FabricRuntimeError`</b>:  If the native runtime lifecycle fails before a  normalized result can be returned.

---


### <kbd>method</kbd> `start_runtime`

```python
start_runtime(
    agent: 'AgentSource',
    profiles: 'PathProfiles | TypedProfiles | None' = None,
    base_dir: 'PathSource | None' = None,
    overrides: 'Mapping[str, Any] | None' = None
) → Runtime
```

Start a stateful runtime for one or more ordered invocations.

Each call starts a new logical runtime. Runtime-scoped overrides are recursively merged below invocation-scoped overrides.



**Args:**

 - <b>`agent`</b>:  Agent-package directory or config-file path, or a typed  ``FabricConfig``.
 - <b>`profiles`</b>:  One profile name or an ordered sequence of names for a  path-backed source. For a typed source, an ordered sequence of  ``FabricProfileConfig`` values.
 - <b>`base_dir`</b>:  Base directory for resolving relative paths in a typed  config. Valid only when ``agent`` is a typed config source.
 - <b>`overrides`</b>:  JSON-compatible overrides applied to every invocation  in the runtime unless superseded by invocation overrides.



**Returns:**
 An active ``Runtime``. Use it as an asynchronous context manager to guarantee runtime shutdown.



**Raises:**

 - <b>`FabricConfigError`</b>:  If inputs or overrides are invalid.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is not  installed.
 - <b>`FabricRuntimeError`</b>:  If runtime startup fails.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

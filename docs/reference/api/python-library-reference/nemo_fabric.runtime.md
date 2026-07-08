---
title: "Runtime"
slug: "/reference/api/python-library-reference/runtime"
description: "Drive stateful multi-turn execution through the Runtime API."
---
{/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 */}

# <kbd>module</kbd> `nemo_fabric.runtime`
Runtime lifecycle support for the Fabric Python SDK.



---


## <kbd>class</kbd> `RuntimeStatus`
Lifecycle state of a runtime.

``ACTIVE`` accepts invocations, ``STOPPED`` has released its runtime, and ``FAILED`` records a lifecycle failure that prevents further use.





---


## <kbd>class</kbd> `Runtime`
One logical, stateful harness execution.

Create runtimes with ``Fabric.start_runtime()`` rather than calling the constructor. A runtime serializes invocations and preserves adapter-owned harness state across turns. Use it as an asynchronous context manager to stop the runtime on exit.

Runtime-scoped overrides are recursively merged with invocation overrides; invocation values win.


---

#### <kbd>property</kbd> handle

Return a detached snapshot of the runtime handle.

---

#### <kbd>property</kbd> invocations

Return copied request, runtime, and invocation IDs for completed turns.

---

#### <kbd>property</kbd> messages

Return a deep copy of the latest harness-provided message history.

---

#### <kbd>property</kbd> runtime_id

Return the unique identifier for this started runtime lifecycle.

---

#### <kbd>property</kbd> status

Return the current ``ACTIVE``, ``STOPPED``, or ``FAILED`` state.



---


### <kbd>method</kbd> `invoke`

```python
invoke(
    input: 'Any' = None,
    request: 'RunRequest | RunRequestModel | Mapping[str, Any] | None' = None,
    request_id: 'str | None' = None,
    context: 'Mapping[str, Any] | None' = None,
    overrides: 'Mapping[str, Any] | None' = None
) → RunResult
```

Run one turn on this runtime.

A complete ``request`` cannot be combined with separate ``request_id``, ``context``, or ``overrides`` fields. Runtime overrides are merged below invocation overrides. Concurrent turns on the same runtime are rejected.



**Args:**

 - <b>`input`</b>:  JSON-compatible turn input.
 - <b>`request`</b>:  Complete ``RunRequest`` or compatible mapping.
 - <b>`request_id`</b>:  Caller-owned request identifier; generated when omitted.
 - <b>`context`</b>:  Caller-owned, JSON-compatible request metadata.
 - <b>`overrides`</b>:  JSON-compatible invocation-scoped config overrides.



**Returns:**
 The normalized ``RunResult`` for this turn.



**Raises:**

 - <b>`FabricConfigError`</b>:  If request fields conflict or are not  JSON-compatible.
 - <b>`FabricStateError`</b>:  If the runtime is not active, is stopping, or is  already running a turn.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is missing.
 - <b>`FabricRuntimeError`</b>:  If native invocation fails before returning a  normalized result.

---


### <kbd>method</kbd> `stop`

```python
stop() → None
```

Destroy an idle runtime exactly once.

Repeated calls after a successful stop are no-ops. A failed runtime or an in-flight invocation must reach a terminal state before cleanup can proceed.



**Raises:**

 - <b>`FabricStateError`</b>:  If the runtime failed, is already stopping, or  has an invocation in flight.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is missing.
 - <b>`FabricRuntimeError`</b>:  If native runtime shutdown fails.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

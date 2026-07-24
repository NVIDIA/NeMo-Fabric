---
title: "Runtime"
slug: "/reference/api/python-library-reference/runtime"
description: "Drive stateful multi-turn execution through the Runtime API."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.runtime`
Runtime lifecycle support for the Fabric Python SDK.



---


## <kbd>class</kbd> `RuntimeStatus`
Lifecycle state of a runtime.

``ACTIVE`` accepts invocations, ``STOPPED`` has released its runtime, and ``FAILED`` records a lifecycle failure that prevents further invocations but still permits cleanup.





---


## <kbd>class</kbd> `Runtime`
One logical, stateful harness execution.

Create runtimes with ``Fabric.start_runtime()`` rather than calling the constructor. A runtime serializes invocations and preserves adapter-owned harness state across turns. Use it as an asynchronous context manager to stop the runtime on exit.

Runtime-scoped overrides are recursively merged with invocation overrides; invocation values win.


---

### <kbd>property</kbd> handle

Return a detached snapshot of the runtime handle.

---

### <kbd>property</kbd> invocations

Return copied request, runtime, and invocation IDs for completed turns.

---

### <kbd>property</kbd> messages

Return a deep copy of the latest harness-provided message history.

---

### <kbd>property</kbd> runtime_id

Return the unique identifier for this started runtime lifecycle.

---

### <kbd>property</kbd> status

Return the current ``ACTIVE``, ``STOPPED``, or ``FAILED`` state.

---

### <kbd>property</kbd> supports_streaming

Return whether Relay-backed ATOF streaming is enabled.



---


### <kbd>method</kbd> `invoke`

```python
invoke(input: 'Any' = None, request: 'RunRequest | None' = None) → RunResult
```

Run one turn on this runtime.

``input`` and ``request`` are mutually exclusive. Runtime overrides are merged below invocation overrides from ``RunRequest``. Concurrent turns on the same runtime are rejected.



**Args:**

 - <b>`input`</b>:  JSON-compatible turn input.
 - <b>`request`</b>:  Complete validated ``RunRequest``.



**Returns:**
 The normalized ``RunResult`` for this turn.



**Raises:**

 - <b>`FabricConfigError`</b>:  If request fields conflict or are not  JSON-compatible.
 - <b>`FabricStateError`</b>:  If the runtime is not active, is stopping, or is  already running a turn.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is missing.
 - <b>`FabricRuntimeError`</b>:  If native invocation fails before returning a  normalized result.

---


### <kbd>method</kbd> `invoke_stream`

```python
invoke_stream(
    input: 'Any' = None,
    request: 'RunRequest | None' = None
) → InvokeStream
```

Start one turn and stream raw Relay ATOF records as they arrive.

``input`` and ``request`` are mutually exclusive. The returned :class:`InvokeStream` yields raw ATOF dictionaries. Await ``stream.result()`` for the terminal normalized :class:`RunResult`.



**Raises:**

 - <b>`FabricCapabilityError`</b>:  If the runtime was not started with Relay  enabled and ``streaming=True``.
 - <b>`FabricConfigError`</b>:  If request fields conflict or are not  JSON-compatible.
 - <b>`FabricStateError`</b>:  If another turn or stream is active.

---


### <kbd>method</kbd> `stop`

```python
stop() → None
```

Destroy an idle runtime exactly once.

Repeated calls after a successful stop are no-ops. A failed runtime may still be stopped so its resources are released.



**Raises:**

 - <b>`FabricStateError`</b>:  If the runtime is already stopping or has an  invocation in flight.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is missing.
 - <b>`FabricRuntimeError`</b>:  If native runtime shutdown fails.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

---
title: "Sessions"
slug: "/reference/api/python-library-reference/sessions"
description: "Drive stateful multi-turn runtimes through the Session API."
---
{/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 */}

# <kbd>module</kbd> `nemo_fabric.session`
Session lifecycle support for the Fabric Python SDK.



---


## <kbd>class</kbd> `SessionStatus`
Lifecycle state of a session runtime.

``ACTIVE`` accepts invocations, ``STOPPED`` has released its runtime, and ``FAILED`` records a lifecycle failure that prevents further use.





---


## <kbd>class</kbd> `Session`
One ordered multi-turn conversation over a Fabric runtime.

Create sessions with ``Fabric.start_session()`` rather than calling the constructor. A session owns one started runtime, serializes invocations, and preserves harness state across turns. Use it as an asynchronous context manager to stop the runtime on exit.

Session-scoped overrides are recursively merged with invocation overrides; invocation values win. Runtime identity and conversation identity are distinct: ``runtime_id`` identifies this lifecycle, while ``session_id`` is the stable caller-owned resume key.


---

#### <kbd>property</kbd> handle

Return a detached snapshot of the public session handle.

---

#### <kbd>property</kbd> info

Return a typed snapshot of session identity, status, and capabilities.

---

#### <kbd>property</kbd> invocations

Return copied request, runtime, and invocation IDs for completed turns.

---

#### <kbd>property</kbd> messages

Return a deep copy of the latest harness-provided message history.

---

#### <kbd>property</kbd> runtime

Return a detached snapshot of the underlying runtime handle.

---

#### <kbd>property</kbd> runtime_id

Return the unique identifier for this started runtime lifecycle.

---

#### <kbd>property</kbd> session_id

Return the stable session ID.

---

#### <kbd>property</kbd> status

Return the current ``ACTIVE``, ``STOPPED``, or ``FAILED`` state.



---


### <kbd>method</kbd> `cancel`

```python
cancel() → None
```

Report whether runtime cancellation is available.



**Raises:**

 - <b>`FabricCapabilityError`</b>:  If cancellation is unsupported or the  cancellation transport is not yet implemented.

---


### <kbd>method</kbd> `invoke`

```python
invoke(
    input: 'Any' = None,
    request: 'RunRequest | Mapping[str, Any] | None' = None,
    request_id: 'str | None' = None,
    context: 'Mapping[str, Any] | None' = None,
    overrides: 'Mapping[str, Any] | None' = None
) → RunResult
```

Run one turn on the session's existing runtime.

A complete ``request`` cannot be combined with separate ``request_id``, ``context``, or ``overrides`` fields. The session identifier is injected into request context, and session overrides are merged below invocation overrides. Concurrent turns on the same handle are rejected.



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
 - <b>`FabricStateError`</b>:  If the session is not active, is stopping, or is  already running a turn.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is missing.
 - <b>`FabricRuntimeError`</b>:  If native invocation fails before returning a  normalized result.

---


### <kbd>method</kbd> `stop`

```python
stop() → None
```

Destroy an idle runtime exactly once.

Repeated calls after a successful stop are no-ops. A failed session or an in-flight invocation must reach a terminal state before cleanup can proceed.



**Raises:**

 - <b>`FabricStateError`</b>:  If the session failed, is already stopping, or  has an invocation in flight.
 - <b>`FabricNativeUnavailableError`</b>:  If the native extension is missing.
 - <b>`FabricRuntimeError`</b>:  If native runtime shutdown fails.

---


### <kbd>method</kbd> `stream`

```python
stream(
    input: 'Any' = None,
    request: 'RunRequest | Mapping[str, Any] | None' = None,
    request_id: 'str | None' = None,
    context: 'Mapping[str, Any] | None' = None,
    overrides: 'Mapping[str, Any] | None' = None
) → AsyncIterator[FabricEvent | RunResult]
```

Yield buffered events followed by one terminal result.

Some adapters may buffer internally; this API does not promise that events arrive in real time. Request validation and failure behavior are identical to ``invoke()``.



**Args:**

 - <b>`input`</b>:  JSON-compatible turn input.
 - <b>`request`</b>:  Complete ``RunRequest`` or compatible mapping.
 - <b>`request_id`</b>:  Caller-owned request identifier; generated when omitted.
 - <b>`context`</b>:  Caller-owned, JSON-compatible request metadata.
 - <b>`overrides`</b>:  JSON-compatible invocation-scoped config overrides.



**Yields:**
 Each normalized ``FabricEvent``, then the terminal ``RunResult``.

---


### <kbd>method</kbd> `update`

```python
update(update: 'RuntimeUpdate') → RuntimeUpdateResult
```

Validate a runtime update and report transport availability.



**Args:**

 - <b>`update`</b>:  Typed update containing overrides and caller metadata.



**Raises:**

 - <b>`FabricConfigError`</b>:  If ``update`` is not a ``RuntimeUpdate``.
 - <b>`FabricCapabilityError`</b>:  If updates are unsupported or the update  transport is not yet implemented.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

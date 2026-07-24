---
title: "Errors"
slug: "/reference/api/python-library-reference/errors"
description: "Structured exception hierarchy for config, capability, state, and runtime failures."
---
{/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 */}

# <kbd>module</kbd> `nemo_fabric.errors`
Public exception hierarchy for the NeMo Fabric Python SDK.



---


## <kbd>class</kbd> `FabricError`

Base class for structured SDK-level Fabric errors.

Catch this type to handle any SDK failure while preserving machine-readable stage, code, retryability, and detail fields.



**Attributes:**

 - <b>`stage`</b>:  Lifecycle stage that failed, when known.
 - <b>`code`</b>:  Stable machine-readable error code, when available.
 - <b>`retryable`</b>:  Whether retrying may succeed without changing the request.
 - <b>`details`</b>:  Detached structured error details.


### <kbd>method</kbd> `__init__`

```python
__init__(
    message: 'str',
    stage: 'str | None' = None,
    code: 'str | None' = None,
    retryable: 'bool' = False,
    details: 'Mapping[str, Any] | None' = None
) → None
```

Initialize a structured Fabric exception.



**Args:**

 - <b>`message`</b>:  Human-readable failure description.
 - <b>`stage`</b>:  Optional lifecycle stage that failed.
 - <b>`code`</b>:  Optional stable machine-readable error code.
 - <b>`retryable`</b>:  Whether callers may safely retry unchanged input.
 - <b>`details`</b>:  Optional structured diagnostic context. The exception  stores a deep copy.





---


## <kbd>class</kbd> `FabricConfigError`

Invalid SDK input, request shape, factory, or resolved config.


### <kbd>method</kbd> `__init__`

```python
__init__(
    message: 'str',
    stage: 'str | None' = None,
    code: 'str | None' = None,
    retryable: 'bool' = False,
    details: 'Mapping[str, Any] | None' = None
) → None
```

Initialize a structured Fabric exception.



**Args:**

 - <b>`message`</b>:  Human-readable failure description.
 - <b>`stage`</b>:  Optional lifecycle stage that failed.
 - <b>`code`</b>:  Optional stable machine-readable error code.
 - <b>`retryable`</b>:  Whether callers may safely retry unchanged input.
 - <b>`details`</b>:  Optional structured diagnostic context. The exception  stores a deep copy.





---


## <kbd>class</kbd> `FabricRuntimeError`

Failure while starting, invoking, stopping, or otherwise driving a runtime.


### <kbd>method</kbd> `__init__`

```python
__init__(
    message: 'str',
    stage: 'str | None' = None,
    code: 'str | None' = None,
    retryable: 'bool' = False,
    details: 'Mapping[str, Any] | None' = None
) → None
```

Initialize a structured Fabric exception.



**Args:**

 - <b>`message`</b>:  Human-readable failure description.
 - <b>`stage`</b>:  Optional lifecycle stage that failed.
 - <b>`code`</b>:  Optional stable machine-readable error code.
 - <b>`retryable`</b>:  Whether callers may safely retry unchanged input.
 - <b>`details`</b>:  Optional structured diagnostic context. The exception  stores a deep copy.





---


## <kbd>class</kbd> `FabricStateError`

Operation rejected because a local runtime is in the wrong state.


### <kbd>method</kbd> `__init__`

```python
__init__(
    message: 'str',
    stage: 'str | None' = None,
    code: 'str | None' = None,
    retryable: 'bool' = False,
    details: 'Mapping[str, Any] | None' = None
) → None
```

Initialize a structured Fabric exception.



**Args:**

 - <b>`message`</b>:  Human-readable failure description.
 - <b>`stage`</b>:  Optional lifecycle stage that failed.
 - <b>`code`</b>:  Optional stable machine-readable error code.
 - <b>`retryable`</b>:  Whether callers may safely retry unchanged input.
 - <b>`details`</b>:  Optional structured diagnostic context. The exception  stores a deep copy.





---


## <kbd>class</kbd> `FabricCapabilityError`

Operation rejected by resolved runtime capabilities or implementation status.


### <kbd>method</kbd> `__init__`

```python
__init__(
    message: 'str',
    stage: 'str | None' = None,
    code: 'str | None' = None,
    retryable: 'bool' = False,
    details: 'Mapping[str, Any] | None' = None
) → None
```

Initialize a structured Fabric exception.



**Args:**

 - <b>`message`</b>:  Human-readable failure description.
 - <b>`stage`</b>:  Optional lifecycle stage that failed.
 - <b>`code`</b>:  Optional stable machine-readable error code.
 - <b>`retryable`</b>:  Whether callers may safely retry unchanged input.
 - <b>`details`</b>:  Optional structured diagnostic context. The exception  stores a deep copy.





---


## <kbd>class</kbd> `FabricNativeUnavailableError`

SDK call requires the PyO3 extension, but it is not installed or importable.


### <kbd>method</kbd> `__init__`

```python
__init__(
    message: 'str',
    stage: 'str | None' = None,
    code: 'str | None' = None,
    retryable: 'bool' = False,
    details: 'Mapping[str, Any] | None' = None
) → None
```

Initialize a structured Fabric exception.



**Args:**

 - <b>`message`</b>:  Human-readable failure description.
 - <b>`stage`</b>:  Optional lifecycle stage that failed.
 - <b>`code`</b>:  Optional stable machine-readable error code.
 - <b>`retryable`</b>:  Whether callers may safely retry unchanged input.
 - <b>`details`</b>:  Optional structured diagnostic context. The exception  stores a deep copy.







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

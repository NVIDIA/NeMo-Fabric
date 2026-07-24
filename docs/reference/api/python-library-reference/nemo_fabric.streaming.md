---
title: "Streaming"
slug: "/reference/api/python-library-reference/streaming"
description: "Consume Relay-backed raw ATOF records and terminal invocation results."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.streaming`
Relay-backed streaming support for the NVIDIA NeMo Fabric Python SDK.



---


## <kbd>class</kbd> `InvokeStream`
Async iterator of raw ATOF records for one runtime invocation.

Consume the final normalized result separately with :meth:`result`. If iteration stops early, call :meth:`aclose` before starting another turn.




---


### <kbd>method</kbd> `aclose`

```python
async def aclose() → None
```

Stop iteration and drain this turn without cancelling the invocation.

---


### <kbd>method</kbd> `result`

```python
async def result() → RunResult
```

Return the terminal normalized result without adding it to the stream.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

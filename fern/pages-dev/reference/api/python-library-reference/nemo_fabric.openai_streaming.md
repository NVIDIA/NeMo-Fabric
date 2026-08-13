---
title: "OpenAI Streaming"
slug: "/reference/api/python-library-reference/openai-streaming"
description: "Consume adapter-native OpenAI Chat Completions chunks and terminal invocation results."
---
{/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 */}

# <kbd>module</kbd> `nemo_fabric.openai_streaming`

Adapter-native OpenAI streaming for the NVIDIA NeMo Fabric Python SDK.



---


## <kbd>class</kbd> `OpenAIInvokeStream`

Async iterator of OpenAI chat-completion chunks for one invocation.

Await ``result()`` for the separate normalized terminal result. Call ``aclose()`` when iteration stops early; it drains the stream without cancelling the invocation.




---


### <kbd>method</kbd> `aclose`

```python
async def aclose() -> None
```

Discard unread chunks and wait without cancelling the invocation.

---


### <kbd>method</kbd> `result`

```python
async def result() -> RunResult
```

Drain the stream and return its separate normalized terminal result.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

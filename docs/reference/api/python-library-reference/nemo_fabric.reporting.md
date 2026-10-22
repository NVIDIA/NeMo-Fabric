---
title: "Run Evidence Reporting"
slug: "/reference/api/python-library-reference/reporting"
description: "Inspect saved invocation results, accounting coverage, and diagnostic references."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.reporting`

Inspect saved invocation evidence without a benchmark manifest or trace import.

**Global Variables**
---------------
- **MAX_RESULT_BYTES**
- **RESOURCE_SCOPE**

---

## <kbd>function</kbd> `summarize_result`

```python
def summarize_result(result: Mapping[str, Any]) -> dict[str, Any]
```

Describe one returned invocation, preserving independent usable evidence.

This is not a task verdict or proof of a complete scheduled run. Normalized accounting uses the shared reducer. Legacy usage remains labeled diagnostics; logs and Relay/ATIF totals never fill missing normalized dimensions.


---

## <kbd>function</kbd> `read_result`

```python
def read_result(path: Path) -> dict[str, Any]
```

Read one saved full result without requiring task labels or a manifest.


---

## <kbd>function</kbd> `read_evidence`

```python
def read_evidence(path: Path) -> dict[str, Any]
```

Read a full result or invocation record; never follow references.


---

## <kbd>function</kbd> `read_invocation_record`

```python
def read_invocation_record(path: Path) -> dict[str, Any]
```

Read bounded start/end evidence; a truncated tail never creates a result.

This reports the core invocation window before one-shot shutdown. Only the typed model_usage projection supplies normalized accounting; legacy counters never gain a completeness guarantee.


---

## <kbd>function</kbd> `render_invocation_record_summary`

```python
def render_invocation_record_summary(
    summary: Mapping[str, Any],
    *,
    details: bool = False,
) -> str
```

Describe the invocation interval without assuming whole-runtime success.


---

## <kbd>function</kbd> `render_result_summary`

```python
def render_result_summary(summary: Mapping[str, Any], *, details: bool = False) -> str
```

Render a concise overview; details add resource evidence.

Token and cost completeness remain separate. JSON quantities and accounting semantics are unchanged by this presentation.


---

## <kbd>function</kbd> `main`

```python
def main() -> None
```

Inspect a saved invocation result as a concise summary or JSON.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

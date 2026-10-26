---
title: "Run Evidence Reporting"
slug: "/reference/api/python-library-reference/reporting"
description: "Inspect saved invocation results, accounting coverage, and diagnostic references."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.reporting`

Inspect saved run journals and legacy results without importing traces.

**Global Variables**
---------------
- **MAX_RESULT_BYTES**
- **MAX_HISTORY_EVENTS**
- **MAX_HISTORY_BYTES**
- **MAX_HISTORY_MILESTONE_RESERVE_BYTES**
- **MAX_HISTORY_FIELD_CHARS**
- **MAX_EVENT_BYTES**
- **MAX_HISTORY_MESSAGE_CHARS**
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

Read a run journal or legacy result without opening referenced artifacts.


---


## <kbd>function</kbd> `read_invocation_record`

```python
def read_invocation_record(path: Path) -> dict[str, Any]
```

Read legacy start/end evidence; a truncated tail never creates a result.

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

Render legacy invocation evidence without claiming one-shot completion.


---


## <kbd>function</kbd> `render_result_summary`

```python
def render_result_summary(summary: Mapping[str, Any], *, details: bool = False) -> str
```

Render concise measurement values; details add retained history and sources.


---


## <kbd>function</kbd> `main`

```python
def main() -> None
```

Inspect a saved run journal or legacy result as readable text or JSON.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

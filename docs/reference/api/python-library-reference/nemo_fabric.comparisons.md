---
title: "Task Comparisons"
slug: "/reference/api/python-library-reference/comparisons"
description: "Compare caller-defined task attempts with explicit evidence coverage."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.comparisons`

Compare caller-defined task attempts using saved NeMo Fabric invocation evidence.

**Global Variables**
---------------
- **MAX_RESULT_BYTES**

---

## <kbd>function</kbd> `summarize_attempts`

```python
def summarize_attempts(
    attempts: Iterable[Mapping[str, Any]],
    *,
    root: Path,
    inventory_complete: bool,
) -> dict[str, Any]
```

Summarize an explicit inventory of scheduled custom-task attempts.

Each attempt names its task/revision/configuration and expected requests. ``inventory_complete`` asserts all scheduled attempts, including retries, are listed; ``requests_complete`` makes the corresponding per-attempt claim. Unknown inventory never establishes a complete cohort denominator. Evaluator verdicts and caller elapsed boundaries remain independent of execution. Resources retain per-invocation scopes. Logs and traces are references only. ``pricing_alignment`` compares contributing shared-reducer bases, not cohort eligibility or accounting completeness; per-configuration totals are retained.


---

## <kbd>function</kbd> `read_comparison_manifest`

```python
def read_comparison_manifest(path: Path, *, root: Path | None = None) -> dict[str, Any]
```

Read a bounded manifest; evidence paths are relative to its explicit root.


---

## <kbd>function</kbd> `render_comparison_summary`

```python
def render_comparison_summary(
    report: Mapping[str, Any],
    *,
    details: bool = False,
) -> str
```

Render per-metric coverage; details retain identities and pricing evidence.

Timing describes observed caller attempts with one common scope, never a sum of invocation durations. Neither partial spend nor task quality is ranked.


---

## <kbd>function</kbd> `main`

```python
def main() -> None
```

Print an explicit comparison manifest as a table or JSON without running agents.




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

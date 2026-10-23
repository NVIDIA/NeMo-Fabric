---
title: "Model Usage Accounting"
slug: "/reference/api/python-library-reference/accounting"
description: "Summarize complete and observed model usage with explicit coverage."
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# <kbd>module</kbd> `nemo_fabric.accounting`

Typed model accounting and invocation summaries with explicit coverage.

This module never backfills from ATIF or interprets absent values as zero. Observed values are deduplicated evidence, not guaranteed mathematical bounds.


---


## <kbd>function</kbd> `summarize_usage`

```python
def summarize_usage(
    results: Iterable[Mapping[str, Any]],
    *,
    expected: Iterable[tuple[str, str]],
) -> UsageSummary
```

Summarize invocations against an explicit expected identity manifest.

Include failed and cancelled results. Identical duplicates count once; conflicts and unexpected results are errors. A complete total exists only when every expected invocation has a complete value. Observations remain visible alongside coverage. No observations is null, never zero.


---


## <kbd>class</kbd> `DimensionSummary`

An aggregate together with its explicit invocation coverage.






### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `expected_count` | `int` |
| `complete_count` | `int` |
| `incomplete_count` | `int` |
| `missing_result_count` | `int` |
| `complete_total` | `int \| None` |
| `observed_sum` | `int \| None` |
| `reasons` | `dict[str, int]` |
| `aggregation_error` | `NotRequired[str]` |

---


## <kbd>class</kbd> `UsageSummary`

Coverage of an explicit invocation manifest, including missing results.








### Fields

The mapping exposes the following typed fields:

| Field | Type |
| --- | --- |
| `unit` | `str` |
| `dimensions` | `dict[str, nemo_fabric.accounting.DimensionSummary]` |
| `pricing` | `NotRequired[dict[str, Any]]` |
| `pricing_groups` | `NotRequired[list[dict[str, Any]]]` |

---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

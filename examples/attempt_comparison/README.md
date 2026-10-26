<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Compare Your Own Task Attempts

Compare saved NVIDIA NeMo Fabric runs across harnesses or model configurations.
Your application lists the tasks and requests it scheduled. NeMo Fabric reads
their recorded results and shows how much evidence is available for each metric.
Harbor, a benchmark runner and NeMo Relay are optional.

## Try the Example

Run these commands from the repository root with the development environment
installed:

```bash
python -m examples.attempt_comparison /tmp/fabric-comparison-demo
python -m nemo_fabric.comparisons /tmp/fabric-comparison-demo/manifest.json
python -m nemo_fabric.comparisons /tmp/fabric-comparison-demo/manifest.json --details
python -m nemo_fabric.comparisons /tmp/fabric-comparison-demo/manifest.json --json
```

Use a new or empty directory. The example creates one `run.jsonl` in each request
directory and a `manifest.json` listing all four requests:

| Directory | Evidence |
| --- | --- |
| `normal/` | Returned result with complete token totals. |
| `partial/` | Returned result with 80 input tokens recorded; the input total is unknown. |
| `interrupted/` | Settled invocation accounting without a final request summary. |
| `missing/` | Request registration without invocation or completion evidence. |

All timestamps and counts are fictional. The example makes no model calls and
does not supply benchmark results, caller timing measurements, evaluator verdicts
or prices. The candidate has three listed attempts and the baseline has one, so
the report warns that their workloads differ. It does not select a subset or
rank incomplete spend.

## Capture a Real Task

List every scheduled attempt, including retries, before dispatch. Give each
expected NeMo Fabric request a distinct `request_id`. A task can include several
requests running in sequence or in parallel.

Enable `runtime.record.enabled` and set `runtime.artifacts` in your Fabric
configuration. Each request saves a `run.jsonl` containing lifecycle events,
available diagnostics, optional resource samples and its terminal summary.
The returned `run_log` artifact gives the path. A failed one-shot SDK request
exposes the exact saved path as `error.details["run_file"]` when recording started.

```python
from time import monotonic_ns
from nemo_fabric import Fabric, RunRequest

started = monotonic_ns()
result = await Fabric().run(
    config, base_dir=workspace,
    request=RunRequest(request_id="review-1", input="Review my change"),
)
elapsed_ms = (monotonic_ns() - started) // 1_000_000
run_file = next(
    artifact.path
    for artifact in result.artifacts.artifacts
    if artifact.kind == "run_log"
)
```

Add the path relative to your evidence root to a caller-owned manifest. Keep
scheduled requests in the manifest even when no result or file is available.

```json
{
  "schema": "nemo.fabric.comparison_manifest",
  "version": "v1alpha1",
  "inventory_complete": true,
  "attempts": [{
    "attempt_id": "review-attempt-1",
    "task_id": "my-change-review",
    "task_revision": "source-revision-1",
    "configuration_id": "harness-and-model-settings-1",
    "requests_complete": true,
    "requests": [{"request_id": "review-1", "evidence": "review-1/run.jsonl"}],
    "elapsed_ms": 4200,
    "elapsed_scope": "caller_one_shot"
  }]
}
```

Replace the example path and elapsed value with your captured values.
**elapsed_ms** is the stored caller duration; the human report displays seconds.
Set **inventory_complete** only when every scheduled attempt is listed, and
**requests_complete** only when that attempt lists every expected request. Both
default to false. Empty attempts remain visible and do not imply zero spend.
Request IDs and attempt IDs must be unique in the manifest.

For Python, call `summarize_attempts(attempts, root=Path(...),
inventory_complete=True)`. The CLI uses the manifest directory as the evidence
root; `--root` selects another root. Paths cannot escape that root, including
through symlinks. The reader does not open referenced traces or arbitrary artifacts.

## Read the Comparison

- **Task outcomes:** Optional evaluator verdicts come from your application.
  Add an evaluation object with status passed or failed, evaluator name and
  version. Otherwise the attempt is `not_evaluated`. Successful execution does
  not establish task correctness. Different evaluator identities or versions
  do not produce a combined pass fraction.
- **Token and cost totals:** Each metric shows its own complete/listed-request
  count. "80 recorded; total unknown" preserves useful evidence without claiming
  a complete total. Missing cost does not erase known tokens or task outcomes.
  `--details` includes total tokens and cache reads and writes.
- **Pricing:** Catalog estimates keep their pricing basis. The report warns
  when contributing bases differ or a configuration has no priced evidence.
  Incompatible pricing groups remain separate in JSON and are not added.
  Recorded amounts with unknown totals cannot establish which configuration is cheaper.
- **Timing:** The report shows the median of available caller attempt times,
  their coverage count and measurement scope. It does not add parallel
  invocation durations to calculate task latency. Human durations use seconds.
- **Resources and provenance:** Individual requests retain available resource
  scopes, models, artifacts and trace references. Matching task labels do not
  prove equivalent served models, tools or environments.

A terminal returned result is authoritative, including unknown or invalid
accounting. Earlier events cannot fill its missing totals. If the final request
summary is missing or reports no returned result, a settled invocation event can
still supply typed accounting. It counts once and leaves the outer result missing.
A registration-only file supplies no invocation accounting.

JSON retains request identities, accounting sources, metric gaps and pricing
groups. **execution.missing_result_count** counts requests without returned results.
**missing_accounting_evidence_count** counts listed requests without an authoritative
accounting source. Those counts can differ when an interrupted request retained
settled invocation accounting.

## Read Older Saved Results

The reader also accepts full result JSON and older invocation records. An
explicit optional record path can recover typed invocation accounting when
the primary result file is absent or unreadable. A valid primary result remains
authoritative. Request identities must match, and any known runtime and
invocation identities must agree. Legacy counters alone do not establish
complete accounting. A result and its record never count twice.

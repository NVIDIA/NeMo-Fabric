<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Compare Your Own Task Attempts

Use saved Fabric evidence to compare your tasks across harnesses or model
configurations. Task IDs, revisions and configuration labels are caller-defined.
A task can be a code review, document review, investigation or benchmark problem.
An ordinary Fabric run does not require these labels.

## Try the Example

From a matching development checkout:

```bash
python -m examples.attempt_comparison /tmp/fabric-comparison-demo
python -m nemo_fabric.comparisons /tmp/fabric-comparison-demo/manifest.json
python -m nemo_fabric.comparisons /tmp/fabric-comparison-demo/manifest.json --details
python -m nemo_fabric.comparisons /tmp/fabric-comparison-demo/manifest.json --json
```

Use a new or empty directory. This creates synthetic normal, partial and missing
results. It makes no model calls and invents no latency, evaluator verdict or
price. The candidate's extra listed attempt stays visible, with a workload warning.
The report does not choose a matched subset or rank partial spend.

## Capture a Real Task

List every scheduled attempt, including retries, before dispatch. Give each
expected Fabric request a distinct `request_id`. Recording saves full results
and request evidence under the configured artifact directory. Keep missing
requests in the inventory when startup, timeout or recording prevents a result.

The caller can capture a request and its own elapsed boundary:

```python
from time import monotonic_ns
from nemo_fabric import Fabric, RunRequest

started = monotonic_ns()
result = await Fabric().run(
    config, base_dir=workspace,
    request=RunRequest(request_id="review-1", input="Review my change"),
)
elapsed_ms = (monotonic_ns() - started) // 1_000_000
```

Enable `runtime.record.enabled` and set `runtime.artifacts` in `config`. The
returned `run_result` artifact identifies the full saved result. Add its path,
relative to your evidence root, to this caller-owned manifest:

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
    "requests": [{"request_id": "review-1", "evidence": "review-result.json"}],
    "elapsed_ms": 4200,
    "elapsed_scope": "caller_one_shot"
  }]
}
```

Replace the illustrative path and elapsed value with the captured values. Set
`inventory_complete` only when every scheduled attempt is listed, and
`requests_complete` only when the attempt lists every expected request. Both
default to false. An empty scheduled attempt remains visible and never becomes
measured zero spend. Request IDs and attempt IDs must be unique in the inventory.

Use `from nemo_fabric.comparisons import summarize_attempts` with the `attempts`
list, `root=Path(...)` and explicit `inventory_complete` for the Python API.
The CLI defaults to the manifest's directory as the evidence root; `--root`
selects another root. Paths cannot escape that root, including through symlinks.

## Interpret the Report

- Execution status describes Fabric execution. It does not establish that your
  task was solved. Add an optional caller verdict as
  `"evaluation": {"status": "passed", "evaluator": "my-tests", "version": "v1"}`
  or use `failed`. Otherwise the task remains `not_evaluated`. Mixed evaluator
  identities have no combined pass fraction.
- Complete token totals and observed subtotals retain independent coverage.
  Missing cost does not erase known tokens, execution or quality. The same
  shared accounting reducer handles invocations and comparisons.
- Catalog estimates retain their pricing basis. The overview shows a compact basis label and warns when bases differ or a configuration has no priced evidence. `--details` retains full bases and all token dimensions. Every displayed usage metric shows its own complete/expected-request coverage. Caller timing shows the median of available attempts only when measured scopes agree, with the observation count; invocation intervals are not summed. JSON exposes `pricing_alignment`: `aligned` means every configuration has priced evidence under one shared basis, `different_bases` means the contributing bases differ, `unverified` means only some configurations have priced evidence, and `unavailable` means none do. These labels describe pricing evidence, not complete accounting or equivalent workloads. Individual valid totals remain available. Incompatible bases within one configuration remain separate groups in JSON; their amounts are not added together. Observed spend is incomplete and cannot establish which configuration is cheaper.
- Caller elapsed time names its measurement boundary. Parallel invocation
  durations are never added to create attempt latency. Runtime CPU/RSS snapshots
  retain each invocation's host scope; the report does not sum incompatible trees.
- Task IDs/revisions and unmatched workloads remain visible. Matching labels do
  not prove that served models, tools or environments were equivalent. Inspect
  available configuration and artifact evidence when checking conditions.

Logs, telemetry and Relay references remain links to supporting evidence. Their
payloads are not read or used to fill missing Fabric accounting. Harbor,
OpenShell, a benchmark runner and a telemetry backend are not required.

## Use a Bounded Invocation Record When the Full Result Is Missing

A request can name an explicit fallback:

```json
{"request_id": "review-1", "evidence": "review-result.json", "record": "run-record.jsonl"}
```

A valid full result is authoritative, including unknown or malformed accounting.
The reader uses the explicit matching record only when the full result itself
is missing, unreadable, malformed or exceeds the bounded reader's 16 MiB limit. Both paths
use the canonical evidence reader. Request IDs must match; known runtime and
invocation IDs must agree. No referenced log, trace or arbitrary artifact is opened.

A record path can also be the primary `evidence`. Typed normalized accounting in
a complete end record can remain useful even when the full result was not saved.
The report labels `accounting_source: "record"` and
`accounting_boundary: "core_invocation"`. Its pre-shutdown status never becomes
a returned one-shot result or a task verdict. Legacy-only record counters stay
unverified. A result and its record never count twice.

`execution.missing_result_count` counts absent full results. Accounting coverage
counts authoritative returned results or usable typed record projections, so a
record can provide accounting while execution remains without a full result.
Top-level `missing_accounting_evidence_count` includes every listed request without an authoritative accounting source. Each pricing group covers only invocations assigned to that known basis; missing accounting cannot be assigned to a pricing group. A returned result with missing or invalid normalized accounting is incomplete, not missing. Inspect
`accounting_sources`, `accounting_evidence_count`, `normalized_accounting_count`,
and each request's
source boundary when these counts differ.

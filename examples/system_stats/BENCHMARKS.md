<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Two-run benchmark qualification

**Identical configs can produce different amounts of agent work and encounter different system conditions. Stats should help separate those two.**

Start with a small, fixed selection of real Harbor Terminal-Bench tasks through
the NVIDIA NeMo Fabric agent. Resolve the latest published dataset once and retain its
immutable version and selected task identities. Cover different resource
patterns where the dataset permits it: CPU-heavy work, temporary memory demand,
and a longer tool sequence. Validate evidence and inspect representative
trajectories before scaling to the full suite. Oracle runs can check task
environments, but do not qualify NeMo Fabric's agent invocation collection.

Run the complete selected benchmark twice with native system stats enabled.
Use the same NeMo Fabric commit, task set, model weights, serving configuration,
concurrency, resource limits, sampling settings, and supported random seeds.
Record the warmup and cache policy before either pass; preserve it in both.
Keep failures and timeouts in the report.

Before spending GPU time, install the NeMo Fabric build under test in the actual task-runner
environment and run the credential-free resource probe there. Require resolved
stats artifacts, matching invocation IDs, a retained transient peak, and explicit
metric availability. A host-side package version alone does not establish which
NeMo Fabric build or collector ran inside a task container.

## Evidence to retain

```text
qualification/
  manifest.json                # commits, commands, configs, tasks, environment
  pass-1/
    benchmark-results/         # original harness outputs and logs
    invocations/               # original RunResults and referenced stats files
    server-telemetry/          # separately sourced inference/GPU observations
  pass-2/
    benchmark-results/
    invocations/
    server-telemetry/
  comparison.md
```

The manifest must record the actual host and task-runner environments separately,
GPU allocation and driver details, model/server identity, collection settings,
and the association between benchmark task IDs and NeMo Fabric invocation IDs.
Do not put credentials or complete environment dumps in it. Preserve raw files
before deriving tables; a summary without its source series is insufficient.

## Questions the comparison must answer

| Question | Evidence |
| --- | --- |
| Did one pass do more agent work? | Task outcomes, available tokens/model/tool calls, wall time, process-tree CPU estimate. |
| Did endpoint memory miss temporary demand? | Before/final RSS versus invocation sampled peak and time-weighted mean; include a representative trajectory. |
| Did system conditions differ? | Actual cgroup CPU/memory limits, throttled-time delta, pressure counters, accounted memory, OOM events. |
| When did demand occur? | Retained sample offsets joined to available trajectory events; disclose clock conversion and overlapping scopes. |
| Did evidence survive task execution? | Task-to-invocation association, result/artifact round-trip, and preserved failure artifacts where available. |
| Is the series trustworthy? | Sample count, actual gaps, coverage, dropped readings, truncation, availability, and collection time. |

Report per-task comparisons plus suite totals and p50/p95 timings. Choose the
coverage acceptance threshold before running and disclose all exclusions.
Validate that all retained rows belong to the expected invocation, offsets are
monotonic and within its window, artifact references resolve, reductions agree
with the raw series, and unavailable metrics remain unknown. Compare valid
counter deltas only within the same target and counter epoch.

## Scope and claims

The native collector currently measures the adapter host's process tree and,
on supported local Linux execution, its cgroup. A task-side Harbor runner must
produce these observations inside the task environment. The cgroup can include
other processes; do not add its totals to overlapping process-tree measurements.

An inference server outside that scope needs separate telemetry. GPU utilization,
GPU memory, and server request/token measurements must identify their actual
collector, server, device, and timestamps. They contextualize NeMo Fabric results;
they are not native GPU metrics produced by this implementation. Shared GPU
observations do not establish exclusive attribution to one invocation.

Two instrumented passes demonstrate practical collection and comparison. They
do not establish statistical performance significance or causal sampling overhead.
The collector's reported read time is not an end-to-end overhead measurement;
that claim requires additional matched runs with sampling enabled and disabled.

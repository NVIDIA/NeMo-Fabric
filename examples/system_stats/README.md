<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Native System Stats

This credential-free example starts a real worker, touches 128 MiB of memory, burns CPU, and exits before the final snapshot. Native sampling retains its transient RSS peak; endpoint memory can return near baseline. No model endpoint, Relay collector, or invocation recorder is required.

The small process adapter follows the public lifecycle contract and implements only this controlled resource probe. It does not represent an LLM benchmark.

From a built source checkout, run:

```bash
just build-python
.venv/bin/python examples/system_stats/run.py --output /tmp/fabric-stats-example
```

The example enables all metric groups with `SystemStatsConfig(sampling=SystemStatsSamplingConfig(interval_ms=100, max_samples=1000))`. Omit `sampling` for endpoints only, set `collect=["process"]` for process-tree measurements only, or set `config.system_stats=None` to disable invocation stats. The separate final `resource_snapshot` is preserved.

Inspect `run-result.json`, its `system_stats.sampled` reductions, and the JSONL/summary paths in `artifacts`. Every sample retains runtime/invocation/request IDs. `sampled_peak` is a sampled maximum, not a true high-water mark. Retained offsets allow benchmark tooling to correlate samples with trajectory events.

For real Linux cgroup constraints, build the development wheel from this checkout and run a bounded container:

```bash
docker build -f examples/system_stats/Dockerfile -t fabric-system-stats:dev .
mkdir -p /tmp/fabric-stats-container
docker run --rm --user "$(id -u):$(id -g)" --cpus=0.5 --memory=512m \
  -v /tmp/fabric-stats-container:/output \
  fabric-system-stats:dev --output /output
```

The cgroup observation names the adapter host's cgroup. Visible CPU and memory limits should reflect these constraints. Throttled time and CPU use are cumulative counters with invocation deltas; accounted memory and provider-lifetime peak remain separate from process RSS and invocation sampled peak. Do not add overlapping scopes. Hidden ancestors, unsupported controllers, and unavailable reads remain explicit.

A qualification run with these limits retained 14 readings at a 100 ms target cadence:

| Observation | Before | Sampled peak | After |
| --- | ---: | ---: | ---: |
| Process-tree RSS | 10.3 MiB | 146.3 MiB | 10.4 MiB |
| Live processes | 1 | 2 | 1 |

The same result reported a 0.5-core visible limit, a 512 MiB memory limit, an estimated 0.39 seconds of process-tree CPU, and a cgroup throttled-time delta of 0.62 seconds. Values vary by machine and execution conditions; these are measured example values, not expected outputs or performance guarantees.

Add `--harbor-runner` to either command to exercise NeMo Fabric's existing task-side Harbor spec/result transport. The serialized result and referenced stats artifacts should retain the same invocation identity. This checks the runner hop; a full Harbor reward trial is separate qualification.

The Dockerfile builds current source into a development wheel. This is not registry-only release qualification; a released consumer must use a NeMo Fabric version containing these APIs.

For a complete workload on a remote GPU machine, follow the [two-run benchmark qualification plan](BENCHMARKS.md). Retain both full passes and separate inference-server observations from NeMo Fabric's measured scopes.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Response-limit test checklist

Use the current SDK, native core and rebuilt OpenShell provider/task images.
Keep logs, reports, endpoint addresses and run evidence outside tracked files.

- [ ] Validate positive integer types, defaults, omitted fields and unknown keys.
- [ ] Check below, at and above each of the four independent budgets.
- [ ] Verify compact UTF-8 response counting, including multibyte characters.
- [ ] Verify explicit failed results with limit, configured/observed values and asset.
- [ ] Confirm a diagnostic can be returned even with a one-byte response budget.
- [ ] Verify duplicate artifact references count once and SDK diagnostics are excluded.
- [ ] Verify no partial artifact manifest is returned after a limit failure.
- [ ] Preserve artifact-reader traversal, symlink, FIFO and file-change protections.
- [ ] Run `just test-rust`, `just test-python` and `just test-typescript`.
- [ ] Build docs and native bindings; check schemas and generated references.
- [ ] Run `response-limits-workload.sh` through `examples/openshell/run-demo.sh`.
      Check all 15 controlled cases on both host and OpenShell, including a 5 MiB
      response, raised file count, binary files above 1 MiB and aggregate transfer
      above the former provider transport cap. Verify complete file hashes.
- [ ] Run `code-review-response-limits-workload.sh` through the same launcher with
      the local inference endpoint configured through the code-review environment
      variables, including `CODE_REVIEW_RESPONSE_EVIDENCE` set to a persistent local
      report path outside tracked files. Check Hermes and Deep Agents on host and OpenShell, with a tiny
      rejecting budget and a raised accepting budget: eight cells, cleanup required.
- [ ] Review only the child branch diff against PR #601; preserve the GitHub stack.

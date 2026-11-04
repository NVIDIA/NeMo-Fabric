<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Invocation Deadline Qualification — November 4, 2026

All 16 real-inference cases passed using the code-review example, Hermes and Deep Agents, and the same self-hosted Nemotron model on the host and in OpenShell. The [sanitized results](invocation-deadlines-2026-11-04.json) retain image and harness identities, durations, termination reasons, and cleanup outcomes. Raw normalized results and logs remain in the local `.tmp/req12-inference/` directory; credentials and internal endpoint addresses are excluded from this report.

Runtime behavior is from commit `4cbb46b1`. The review follow-up adds the qualification harness and documentation; it does not change runtime implementation.

| Harness | Location | Smoke | Finite (900 s) | Unlimited | Timeout (15 s) | Cleanup |
| --- | --- | --- | --- | --- | --- | --- |
| hermes | host | 3.12 s | 334.03 s | 333.79 s | `timeout/invoke`, 17.03 s | Passed |
| hermes | openshell | 4.40 s | 335.22 s | 336.12 s | `timeout/invoke`, 17.39 s | Passed |
| deepagents | host | 3.37 s | 334.16 s | 333.85 s | `timeout/invoke`, 17.02 s | Passed |
| deepagents | openshell | 6.34 s | 337.19 s | 339.45 s | `timeout/invoke`, 17.39 s | Passed |

The long invocations performed two sequential 165-second source-reader tool calls between real model requests, then produced final responses containing the unique tool receipt and the actual source value `41`. Final-response checks were verified for all retained records. Each 15-second deadline returned after approximately 17 seconds, including native process teardown. No source-reader processes or OpenShell containers remained after qualification.

## Completed Checks

- [x] Host and OpenShell inference with two real harnesses.
- [x] Finite and unlimited invocations exceed 320 seconds and complete normally.
- [x] Finite deadlines preserve `timeout` at `invoke`.
- [x] Final model responses contain the tool receipt and reviewed source value.
- [x] Runtime stop, sandbox release, and post-run process/container checks.
- [x] Existing model-free OpenShell qualification: two tests passed, including 326-second finite/unlimited runs.
- [x] Harbor qualification test file: 118 passed in a clean checkout, preserving the existing staged vendor tree in the main workspace.

## Validation

- [x] Python suite: 3,129 passed and 148 optional skips in the main checkout, plus all 118 Harbor qualification tests passed in a clean detached checkout. The split preserves the pre-existing staged vendor tree inside the main Harbor sample task. Ambient Claude credentials were removed from the unit-test process so its mock-model health test remains isolated.
- [x] Updated the SDK-health provider fixture to preserve a null invocation timeout; affected health tests passed, including Pi through the command bridge.
- [x] Rust workspace: 324 passed at the unchanged runtime implementation commit; TypeScript suite also passed. No Rust, schema, or TypeScript sources changed in this review follow-up.
- [x] Changed-file Ruff, pre-commit, copyright, shell syntax, and release/checklist relative links passed.
- [x] Documentation generation and Fern validation passed with 0 errors and 12 existing generated-reference warnings.
- [x] All eight short inference cases passed again after requiring receipts in final responses and explicitly probing server readiness before lifecycle startup.
- [x] Both the qualification image and the existing Hermes image with its unchanged default pin build successfully.

## Review Focus

- The inference matrix and controlled source reader in `tests/e2e/`: actual model responses are checked, and the long delay occurs inside tool execution.
- Release gating in `.agents/skills/create-release-tag/SKILL.md`, `RELEASING.md`, and the reusable checklist: subsets and skipped live cases do not qualify.
- The Hermes code-review image now installs from its frozen upstream lockfile, preserving the already installed runtime. This avoids the incompatible Pillow requirements encountered by the prior unlocked install at the repository's Hermes pin.
- The documentation changes describe normalized runtime behavior on host and sandbox, remove implementation history from user guidance, and remove the two requested main README bullets.

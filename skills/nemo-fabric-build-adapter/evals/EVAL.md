<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Adapter Evaluation Scope

Stage `files/project/` identically for both arms and copy it to `/workspace/repo` before agent work. The deterministic Acme target SDK is a fixture, not a replacement Fabric SDK. The real published adapter contract and common host helpers are installed by setup. Use `/workspace/.fabric-eval-venv/bin/python`; no external service, harness, model call, or credentials are required.

Implementation cases must identify the starter files and request bounded code and tests. Contract-review cases request concrete lifecycle code or a design/test matrix and must not be scored as if they had required a live deployment, published wheel, or production persistence engine. Do not claim durable sessions for the in-memory Acme fixture; the durable-session case is an explicit design exercise.

Keep activation in `expected_skill` and `should_trigger`, not positive `expected_behavior`, because the baseline shares the behavior assertions. Negative cases should identify the appropriate workflow without invoking an unavailable alternate skill or implementing that workflow.

The sandbox needs Python 3.11+ and system pip with `--python` support. Setup creates an isolated venv without `ensurepip`, then uses system pip to install into it. Package downloads use a 60-second read timeout and bounded connection retries. A setup failure or gateway interruption is infrastructure evidence, not a skill failure; retain the trial details and resolve it before starting another full matrix.

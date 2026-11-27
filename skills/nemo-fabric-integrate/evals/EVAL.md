<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Consumer Integration Evaluation Scope

Stage the same fixtures and dependencies in both evaluation arms. `files/project/` contains the existing application interfaces, copied to `/workspace/repo` before agent work. `files/results/` contains the saved results used by the analysis case. Use `/workspace/.fabric-eval-venv/bin/python` for executable exercises; setup installs the pinned published nightly without asking the agent to install dependencies.

Implementation cases must name the supplied interfaces and request concrete edits and focused tests. Do not require a live model call, real harness startup, or real credentials. Use spec'd mocks at public SDK lifecycle boundaries for consumer tests. Configuration and planning can be checked against the installed SDK; mock declared-environment checks when a test needs a provisioned environment.

The response-limits case reviews a development contract newer than the pinned nightly. It requests a code proposal and error-handling explanation, not execution against a package that lacks that API. Keep that distinction explicit until a published package contains the feature.

Positive `expected_behavior` assertions must describe task outcomes, not skill activation: the baseline receives the same assertions without the skill. Record activation separately with `expected_skill` and `should_trigger`.

Negative cases test workflow selection or the supplied result-analysis task. They must not require an unavailable alternate skill, unrelated repository, native harness installation, or implementation of the out-of-scope task. Keep `expected_skill: null`, `should_trigger: false`, an observable non-activation assertion, and an answerable user goal.

The sandbox needs Python 3.11+ and system pip with `--python` support. Setup creates an isolated venv without `ensurepip`, then uses system pip to install into it. Package downloads use a 60-second read timeout and bounded connection retries. A setup failure or gateway interruption is infrastructure evidence, not a skill failure; retain the trial details and resolve it before starting another full matrix.

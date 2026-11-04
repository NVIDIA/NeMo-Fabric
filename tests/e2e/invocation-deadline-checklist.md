<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Invocation Deadline Qualification

Use this checklist on the exact release candidate or PR commit. Record the full source commit SHA with the model ID, endpoint, harness versions, task image digest, commands, normalized results, elapsed times, and cleanup outcomes. For release qualification, the SHA must match `RELEASE_SHA`, and the host SDK, provider, and task image must be built from that commit. Keep run evidence in local output directories or external review artifacts; do not commit it. Keep credentials out of evidence. A skipped live test is incomplete qualification.

- [ ] Unit and regression suites: configuration validation, optional deadline propagation, finite enforcement, termination reasons, and unchanged lifecycle/health budgets.
- [ ] Model-free OpenShell tests: finite and unlimited invocations exceed 320 seconds, short finite deadline retains `timeout` at `invoke`, and sandboxes are deleted.
- [ ] Hermes on host: actual code review through the example, finite and unlimited invocations exceed 320 seconds, finite deadline is enforced, and cleanup succeeds.
- [ ] Deep Agents on host: same cases.
- [ ] Hermes in OpenShell: same cases using the rebuilt task SDK and provider.
- [ ] Deep Agents in OpenShell: same cases using the rebuilt task SDK and provider.
- [ ] Confirm each long invocation includes real model calls before and after controlled tool execution; do not count a sleep outside invocation as deadline coverage.
- [ ] Run changed-example tests, formatting, documentation generation, and required language suites; record every failure or unavailable check explicitly.

The model-free transport regression uses the standard OpenShell launcher:

```bash
UV_PROJECT_ENVIRONMENT="$PWD/.venv" OPENSHELL_EXAMPLE_MODE=fabric_owned \
  bash examples/openshell/run-demo.sh tests/e2e/openshell-timeouts-workload.sh
```

The inference matrix uses the code-review example configurations and a controlled MCP source reader. It requires an OpenAI-compatible model endpoint and installed Hermes and Deep Agents environments. The task image must contain the current SDK, both adapters, both harnesses, and the code-review workload.

Prepare the host harnesses with the repository installation recipes, including `just install-hermes-agent`, and an environment with the Deep Agents harness extra installed. Then configure the inference run:

```bash
export CODE_REVIEW_MODEL_URL="<OpenAI-compatible URL>"
export CODE_REVIEW_MODEL="<served model ID>"
export OPENAI_API_KEY="<model credential, or an unused value for a no-auth endpoint>"
export CODE_REVIEW_HERMES_PYTHON="$PWD/.venv-hermes/bin/python"
export CODE_REVIEW_DEEPAGENTS_PYTHON="$PWD/.venv/bin/python"
export CODE_REVIEW_POLICY="<absolute path to endpoint-specific sandbox policy>"
export CODE_REVIEW_EVIDENCE="$PWD/.tmp/deadline-qualification/results.json"
UV_PROJECT_ENVIRONMENT="$PWD/.venv" OPENSHELL_EXAMPLE_MODE=fabric_owned \
  bash examples/openshell/run-demo.sh tests/e2e/code-review-deadlines-workload.sh
```

Restrict the policy to the model endpoint and client Python binary. OpenShell native TCP rules require a DNS hostname, not an IP literal; for a private model, verify that hostname resolves to the intended address and allow only that address with `allowed_ips`. Set `CODE_REVIEW_SANDBOX_MODEL_URL` when the sandbox must use that hostname to reach the same endpoint. Host requests continue to use `CODE_REVIEW_MODEL_URL`.

The full inference matrix covers smoke completion, a 900-second finite deadline, no total deadline, and a 15-second enforced deadline for each harness and location. Two sequential 165-second tool calls make each long invocation exceed 320 seconds while staying below independent native per-tool budgets. The model must read the example's actual `calculator.py` and return the tool's unique receipt and the value returned by `answer()`. Each case stops its runtime and releases any owned sandbox.

`CODE_REVIEW_DEADLINE_CASES` and `CODE_REVIEW_DEADLINE_LOCATIONS` select developer subsets. A subset run does not satisfy the release gate; qualification requires all 16 cases with successful cleanup.

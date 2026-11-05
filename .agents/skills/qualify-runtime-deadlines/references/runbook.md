<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deadline Qualification Runbook

Run from a clean disposable checkout on Linux with Git, Python 3.12+, `uv`, Cargo
and Docker. Provide existing supported Hermes and Deep Agents dependency
interpreters with pytest available; provide a clean Hermes source checkout at
`bf0203a794a07aac3ff764972ab9087baa4e4f4e`, matching the code-review task image.
The agent owns preparing these prerequisites, using the repository installation
recipes without overwriting a user's dirty harness checkout.

Keep output directories outside version control. Prepare the plan after the
setup conversation, substituting the user's already-approved inputs:

```bash
GATE=.agents/skills/qualify-runtime-deadlines/scripts/qualify.py
python3 "$GATE" plan \
  --revision "$(git rev-parse HEAD)" \
  --model-url "$CODE_REVIEW_MODEL_URL" --model "$CODE_REVIEW_MODEL" \
  --sandbox-model-url "$CODE_REVIEW_SANDBOX_MODEL_URL" \
  --api-key-env OPENAI_API_KEY \
  --hermes-python "$CODE_REVIEW_HERMES_PYTHON" \
  --hermes-source "$HERMES_SOURCE" \
  --deepagents-python "$CODE_REVIEW_DEEPAGENTS_PYTHON" \
  --policy "$CODE_REVIEW_POLICY" \
  --output /absolute/local/deadline-plan.json

python3 "$GATE" run --plan /absolute/local/deadline-plan.json \
  --output /absolute/local/deadline-run --release
python3 "$GATE" verify --report /absolute/local/deadline-run/report.json \
  --revision "$(git rev-parse HEAD)" --release
```

`run` refuses a nonempty destination, installs SDK wheels in fresh host
environments, verifies SDK imports, builds the task/provider through the standard
launcher, and retains all exit codes. It runs transport and inference workloads
serially, with separate temporary gateways. Once execution starts it writes a report even on failure;
there are no automatic retries. Reproduce failures in a new output directory.
Build caches may be reused, but active SDK imports and wheel hashes must identify
the selected source. No existing model service is stopped or reconfigured.

`plan --cases smoke --locations host` selects a developer subset. Its report is
explicitly incomplete and `verify --release` rejects it. It still runs the
model-free regression, which proves the transport outlasts the former cap.

Inspect `report.json`, `installation.json`, `inference.json`, both JUnit files
and launcher/build logs. A full pass requires 16 unique inference cells with
cleanup and two passing model-free tests. The model-free long test exercises
finite and unlimited runs concurrently. `verify` checks long elapsed times,
short timeout reasons/timing, real-source receipts, revision identity and
retained file hashes; it rejects missing, skipped or failed evidence. Keep
credentials out of logs and never commit these files.

Use the same verifier from the stable-release skill with `--revision "$RELEASE_SHA"`.
A successful gate validates the selected source; creating or publishing a tag is
a separate release action.

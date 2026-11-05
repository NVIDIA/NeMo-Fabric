---
name: qualify-runtime-deadlines
description: Prepare a saved inference plan and qualify normalized invocation deadlines with Hermes and Deep Agents on host and OpenShell; verify exact-revision evidence for a release gate.
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# Qualify NVIDIA NeMo Fabric Invocation Deadlines

Use the runnable gate in this directory for SDK → harness → inference → tool →
result → cleanup checks. The scope is Hermes and Deep Agents, each on the host
and in OpenShell. It supplements the unit suites; it does not qualify the entire
adapter catalog or publish a release.

## Set Up The Run

Reuse the user's existing model, credential and execution choices. Ask only for
missing inputs. Start with an existing OpenAI-compatible endpoint and its served
model ID; do not install or manage inference servers. Read [setup](references/setup.md)
for credential handling, endpoint policy and the coverage preview. The agent
constructs the saved plan; the user need not edit JSON or discover harness flags.

Preview both harnesses, both execution locations and all four cases before a
release run. Missing prerequisites stay visible as nonpasses. Developer subsets
are useful for triage but cannot pass the release gate.

## Run The Saved Plan

Follow the [runbook](references/runbook.md). Run from a clean disposable checkout
of the selected source revision. The gate builds noneditable SDK/adapter wheels
from that revision and installs them in fresh host environments, while reusing
explicitly selected harness dependency environments. It verifies actual SDK
import locations and wheel identities; ambient editable SDK installs do not
qualify. Hermes may use a clean pinned upstream source checkout, independently
of its Python interpreter version.

The gate runs the existing model-free transport regression and real code-review
inference matrix, rebuilding the provider and task images from the selected
checkout. Keep all plans, installation records, logs, reports and credentials
outside tracked files. Never commit run evidence.

## Verify The Release Gate

Run `verify --release` against the saved report and exact release SHA before
creating a tag. Require all 16 inference cells, both model-free tests, successful
cleanup, finite/unlimited calls longer than 320 seconds, canonical invocation
timeouts, and model completion receipts from the actual example source. Require
matching source revision and unchanged artifact hashes. A skipped case, missing
record, failed cleanup, subset, changed revision or failed command is a nonpass.

The release-tag workflow calls this verifier after its ordinary local validation
and before tagging. Share the sanitized summary and local evidence locations
with the reviewer. Do not infer all-harness or all-capability coverage from this
focused qualification.

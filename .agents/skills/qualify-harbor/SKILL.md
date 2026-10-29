---
name: qualify-harbor
description: Guide interactive model and credential setup, then qualify NeMo Fabric across its adapter catalog or a selected subset through Harbor and real harnesses. Use for reproducible integration checks and all-harness failure triage.
license: Apache-2.0
---

# Qualify NVIDIA NeMo Fabric With Harbor

Use the runnable gate in this directory to test Harbor → NeMo Fabric → harness → inference → result → verifier. Harbor owns containers, task lifecycle, and scoring. The gate runs exactly one shared Terminal-Bench task across the catalog, builds isolated package installs on that task environment, and retains a complete adapter denominator. It does not install or manage inference servers.

## Start With A Short Setup Conversation

Use the user's existing choices and authorization. When starting a new run without a configured plan, ask how they want to supply models:

- **One shared BYOM endpoint** for every selected adapter that supports its API and model.
- **Individual provider credentials**, grouping adapters that can share the same provider/key reference.

Offer a mix when the user wants coverage beyond the shared endpoint’s compatible subset; otherwise record those exceptions as skipped. If the user already chose a mode or supplied a plan, continue from it; do not restart the questionnaire. Ask only for missing information, in small steps, using the available question tool or a concise chat question.

Read [interactive setup](references/setup.md) for the conversation, credential handling, compatibility decisions, and coverage preview. The agent owns plan construction and adapter-specific configuration; users should not have to edit JSON rows or discover provider flags themselves.

Before builds or model calls, show which adapters are ready, need setup, or will be skipped, with model/provider and credential **variable names only**. Resolve outstanding scope or cost decisions on that concrete preview, reusing any authorization already given. A shared BYOM choice never implies universal compatibility. Kiro requires its native provider and `KIRO_API_KEY`; BYOM is unsupported and missing credentials mean skipped execution, retained in the catalog denominator.

## Execute The Saved Plan

Read [the runbook](references/runbook.md) for reproducible commands, preparation profiles, and failure triage. The conversational setup produces the same noninteractive plan used by these commands.

1. Identify the exact package version or source commit and compatible Harbor package version. Use a disposable checkout of that version with a fresh catalog (`just check-adapter-catalog`). Do not use a developer's ambient editable environment as reproducible installation evidence.
2. Establish inference prerequisites before running. A local checkpoint must already be served through a compatible HTTP API. Verify the served model ID, container reachability, protocol, authentication, and available resources. Use the same served model for compatible harnesses; record provider restrictions rather than assuming universal compatibility. Do not stop existing services, take allocated GPUs, mount credential directories, or publish new network listeners.
3. Select one task directory, not a benchmark dataset. The runbook pins TB2 `kv-store-grpc` as a comprehensive baseline. Generate and inspect the plan; its task hash covers the instruction, environment, and unchanged upstream verifier. It includes every catalog adapter, exact package/descriptor metadata, settings, estimated provider cost, and capability coverage. Resolve prerequisites by configuring individual rows and dedicated task images. Registered workflows require explicit `workflow_target_id` and optional `workflow_settings` with the SDK bridge; validate catalog binding, both descriptor fingerprints, settings transport, and installed target provenance. Keep blocked, unsupported, and excluded rows in the plan with reasons.
4. Prepare isolated host and task installs, then run the selected matrix. Use `--only` for a developer subset; its report cannot establish a full-catalog pass. The default is one trial at a time, one attempt, bounded runtime/build deadlines, and no automatic retries. Preserve each failure before reproducing it in a new output directory.
5. Use compatible host/task Python versions, selecting `task_python` per row when needed. Keep NeMo Fabric packages noneditable; allow upstream-supported pinned Hermes source installs with verified clean Git provenance. Do not require the task base image’s Python version for every harness.
6. Inspect `report.json`, `summary.md`, installation logs, and trial artifacts. The pass criterion is Harbor → NeMo Fabric → harness → inference → result integration: lifecycle, selection/fingerprint, package identity, result/accounting transport, requested telemetry, and cleanup. A completed verifier's task reward is informational; reward zero can pass integration. Harness usage availability is also informational: retain unknown counters and scope limitations without inventing totals. Accounting that changes in transport remains a failure. Reproduce findings and classify them before attributing a regression. Fix gate defects narrowly; report unrelated product defects separately.
7. Share the sanitized report and exact evidence locations with the reviewer. A full pass requires every catalog entry to pass integration. Missing credentials, incompatible providers, install failures, missing or invalid verifier evidence, timeout, cancellation, and skipped coverage remain nonpasses. Model task quality and documented harness capabilities do not fail an otherwise successful round trip.

One shared TB2 task is a scored integration check, not a full benchmark or proof of all-feature equivalence. MCP, skills, ATIF/export, and subagent paths require their own evidence where supported. The report labels untested paths. Use the existing Harbor integration suites for deterministic lifecycle, error, credential, MCP, skills, and telemetry coverage; record any live capability tests separately. Never infer a capability pass from one successful task.

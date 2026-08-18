{/*
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}

# Stage 6: Verify the Adapter

Verify the minimum lifecycle and every claim in the selected Adapter
Descriptor before publishing an adapter. NVIDIA NeMo Fabric 0.2 runs
adapter-contract checks through `just test-python` and
`just test-typescript`; the installed adapter lifecycle and optional
capability checks below remain manual.
Completing this checklist does not imply NVIDIA review, trust, certification,
or verification.

## Verify the Minimum Profile

Run these checks against an installed adapter package:

1. Discover the packaged `*.fabric-adapter.json` without importing adapter
   code. Also verify an explicit `discovery.local_paths` record for the
   development workflow.
2. Run `Fabric().plan(...)` with the smallest valid configuration and inspect
   the selected descriptor and projected `AgentConfig`.
3. Confirm that planning rejects one unsupported normalized field and invalid
   data for every published settings, model, tool-definition, target, and
   extension schema.
4. Run `doctor(...)` with both satisfied and missing runtime requirements.
5. Exercise typed request projection, at least two ordered successful
   invocations, target failure, and stop.
6. Exercise partial-start failure, invocation transport failure, malformed or
   untyped `AgentRunResult`, and end-of-file cleanup without exposing secrets.
7. Run two independent Fabric runtimes and confirm that they do not share
   mutable target state.

Preserve runtime, invocation, and request IDs as opaque values. Check logs and
persisted diagnostics for credential values, authorization headers, complete
environment mappings, and arbitrary user input.

## Verify Declared Capabilities

Test each descriptor claim independently:

| Claim | Evidence |
| --- | --- |
| Normalized config field | One accepted case and one unsupported or invalid case. |
| Adapter-owned schema | Valid and invalid examples exercised through planning. |
| Runtime requirement | `doctor(...)` reports both satisfied and missing states. |
| Telemetry output | The output is produced and correlated to the intended invocation. |
| Relay-backed stream | Ordinary `invoke` completes while correlated Agent Trajectory Observability Format (ATOF) records reach `Runtime.invoke_stream()`. |
| Native OpenAI stream | Empty and multi-chunk streams, invalid records, early close, a separate terminal value, and exactly one target invocation. |

Do not claim reserved cancellation, update, or service capabilities until the
installed NeMo Fabric runtime binding exposes and tests the corresponding
adapter operation.

## Record the Result

Record the exact adapter package version, `contract_version`, NeMo Fabric
version, minimum-profile result, test environment, and every optional
capability as supported or unsupported. Link the evidence to the adapter
release so a later release does not inherit the claim automatically.

Use [Examples and References](examples.md) to compare the finished adapter with
the closest maintained implementation.

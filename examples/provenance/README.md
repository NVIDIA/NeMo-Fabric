<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Software Evidence

`RunResult.execution` records selected software versions inline. The default-on `FabricConfig.provenance` control and typed owner reports work through the existing SDK, adapter and task dispatch paths. Set `ProvenanceConfig(enabled=False)` before initialization to skip host and guest collectors. See the [SDK evidence contract](../../docs/sdk/python.mdx#selected-software-evidence).

## Reproduce The Six Paths

From the repository root, prepare the normal development environment with `just build-python` and the test dependencies from `just test-python`. The focused acceptance run below writes actual RunResults only when `FABRIC_PROVENANCE_ARTIFACTS` is set:

```bash
FABRIC_PROVENANCE_ARTIFACTS="$PWD/outputs/provenance-validation/run-results" \
  uv run --no-sync pytest \
  tests/python/test_provenance_paths.py \
  tests/python/test_provenance.py \
  tests/python/test_openshell_remote_runtime.py \
  tests/e2e/test_openclaw_checks.py \
  -k 'provenance or software or independent_remote or external_platform or agent_server or shared_service'
```

| Path | Exercised implementation | External boundary |
| --- | --- | --- |
| Direct SDK | SDK, native core, persistent Python adapter host; repeated invocations | Repository model-free adapter fixture |
| Separate venvs | Caller SDK plus an independent adapter venv with normally installed contract/common packages | No model or remote service |
| External platform dispatch | Harbor agent initialization report and typed payload; runner in a separate real Python process | Local file transport replaces a Harbor environment/container |
| Agent-server embedding | FastAPI handler calling the public SDK and forwarding an explicit application startup report | Gym's embedding pattern; no live Gym deployment |
| Native sandbox dispatch | Production OpenShell bridge, separate guest SDK process, persistent runtime and artifact transfer | Model-free stdio provider replaces a live OpenShell gateway; existing tests cover create and attach |
| Independently deployed remote agent | Production remote-agent adapter and real HTTP client traffic | Controlled API server; no reported remote software is invented |
| Shared service | Production OpenClaw adapter, native service handle, multiple runtimes and SDK instances | Controlled Gateway HTTP endpoints; no live OpenClaw binary/deployment |

Provider and container-engine fixture values are explicit owner reports, not measured live infrastructure versions. Shared-service capture times remain unchanged across runtimes. The test-only comparison projection excludes IDs, capture times and artifact roots, retains unknowns/coverage, and identifies one changed component. It is not a public SDK comparison API or a claim of equivalent environments.

## Validate An Installed SDK

This model-free run checks normal wheel ownership of the SDK/contract metadata. Keep `ADAPTER_PYTHON` pointed at the development interpreter that has the conformance adapter's shared host package:

```bash
uv sync --project sdk/python/nemo-fabric-runtime --frozen --no-default-groups --no-editable --reinstall
ADAPTER_PYTHON="$PWD/.venv/bin/python" \
  sdk/python/nemo-fabric-runtime/.venv/bin/python \
  examples/provenance/installed_sdk.py \
  --output outputs/provenance-validation/run-results/installed-sdk.json
```

The SDK and adapter installations remain separate even when their version strings match. Editable checkouts can report an unavailable package version when installed metadata does not own the loaded source file; the evidence retains that reason instead of asserting loaded-code identity.

The initial scope captures selected software and existing provider/engine reports. Safe configuration/input capture is the P0 follow-up. No dependency scan, arbitrary executable/service probe, journal, new storage service or provenance version field is introduced.

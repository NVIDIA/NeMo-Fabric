# Harbor Multi-Harness Demo

This demo runs one Harbor task through one `FabricAgent` import path while
Fabric selects the execution harness from an ordered profile stack.

## Variants

| Command | Fabric profiles | Purpose |
| --- | --- | --- |
| `./run.sh smoke` | `smoke` | Credential-free Harbor, Fabric SDK, workspace, and verifier check |
| `./run.sh hermes` | `hermes` | Real Hermes CLI coding-agent run |
| `./run.sh hermes-relay` | `hermes`, `telemetry` | Same Hermes run with Relay ATOF/ATIF enabled |
| `./run.sh codex` | `codex` | Real Codex CLI coding-agent run |

Harbor owns the task container and verifier. `FabricAgent` launches
`nemo_fabric.integrations.harbor_runner` inside that container; the runner loads
the YAML files into typed Fabric config objects and invokes `FabricClient`.

## Run

Requirements: Python 3.12+, `uv`, Docker, and this checkout. The first run builds
the demo image and can take several minutes.

```bash
chmod +x integrations/harbor/demo/run.sh
FABRIC_DEMO_FORCE_BUILD=1 integrations/harbor/demo/run.sh smoke
```

For the real harnesses:

```bash
NVIDIA_API_KEY=... integrations/harbor/demo/run.sh hermes
NVIDIA_API_KEY=... integrations/harbor/demo/run.sh hermes-relay
OPENAI_API_KEY=... integrations/harbor/demo/run.sh codex
```

Set `FABRIC_CODEX_MODEL` only when you want to override the Codex CLI default.

## Recording Flow

1. Show the single Harbor agent import path in `run.sh`.
2. Run `hermes`, then `codex`; only the Fabric profile and credential change.
3. Run `hermes-relay` to show telemetry as an additive profile.
4. Open all results with:

```bash
uv run --extra harbor harbor view integrations/harbor/demo/runs
```

Each trial keeps `fabric-result.json` plus Fabric artifacts under the Harbor
agent logs.

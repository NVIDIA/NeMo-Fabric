# SWE-Bench Run Plan

This note captures the path from the current Fabric POC to a real SWE-Bench run
and a Harbor integration.

## Goal

Run a SWE-Bench task through Fabric-managed Hermes and produce the artifacts
needed for evaluation:

- model/harness run logs;
- workspace patch;
- Fabric run metadata;
- Relay ATOF/ATIF when telemetry is enabled;
- verifier result from Harbor or SWE-Bench.

Fabric should act as the agent harness runner. It should not become the
SWE-Bench dataset adapter or verifier.

## Ownership Split

Fabric owns:

- harness selection and profile resolution;
- Hermes invocation;
- normalized request/result handling;
- patch, log, and metadata artifacts;
- Relay telemetry config/pass-through.

Harbor or SWE-Bench owns:

- dataset/task materialization;
- repository checkout and task environment setup;
- verifier execution;
- reward/pass-fail calculation.

This keeps Fabric focused on the harness boundary while letting existing
evaluation systems keep their dataset and verifier semantics.

## Full SWE-Bench Run

The fastest path is to reuse Harbor's SWE-Bench adapter output first. Harbor
already knows how to materialize SWE-Bench tasks into task directories,
instructions, Docker/env setup, and verifier scripts.

The first real target should be `django__django-13741`, since it has been used
across prior POCs.

Required work:

1. Generate or locate a Harbor SWE-Bench task directory for
   `django__django-13741`. Done for the local POC via Harbor's generated
   `datasets/swebench-opencode-smoke/django__django-13741` task.
2. Add a Fabric profile that attaches to the prepared task workspace instead of
   creating a new environment. A test-only shim profile exists under
   `tests/fixtures/hermes-shim-agent`; the MVP path should use the real Hermes
   SDK adapter.
3. Pass the SWE-Bench problem statement, workspace path, model config, instance
   metadata, and profile into Hermes. Done through structured `RunRequest`
   JSON.
4. Capture `workspace.patch`, logs, run metadata, and optional ATOF/ATIF. Patch,
   logs, and metadata are captured; ATOF/ATIF is already available on the
   Hermes Relay profile but has not been combined with this Harbor SWE-Bench
   smoke yet.
5. Improve patch capture so new/untracked files are represented, not only
   tracked diffs.
6. Run the Harbor or SWE-Bench verifier against the resulting workspace/patch.
   The optional verifier path is wired in the smoke, but the current generated
   verifier script expects `uv` inside the SWE-Bench image, so the reliable
   green path remains patch capture rather than reward verification.

Milestone:

```text
Harbor SWE-Bench task -> Fabric -> Hermes -> patch/artifacts -> Harbor verifier
```

## Harbor Integration

The first Harbor integration should be a Harbor `BaseAgent` wrapper around
Fabric.

Harbor continues to own:

- task generation;
- environment/container lifecycle;
- verifier and reward;
- Harbor task/artifact layout.

The Harbor Fabric agent owns:

- loading the Fabric agent config/profile;
- mapping Harbor task input into a Fabric run request;
- invoking Fabric SDK or CLI;
- copying Fabric artifacts into Harbor logs/artifacts;
- exposing patch and telemetry artifacts back to Harbor.

Recommended shape:

```text
Harbor task/env -> FabricAgent -> Fabric SDK/CLI -> selected harness -> artifacts -> Harbor verifier
```

For the MVP, Fabric should run in the Harbor-prepared environment or treat that
environment as a local workspace from inside the container. Fabric does not need
to provision Docker, OpenSandbox, or Kubernetes for this path.

## Config Mapping

Initial Harbor-to-Fabric mapping:

| Harbor input | Fabric field |
| --- | --- |
| `instruction` | `RunRequest.input` |
| task id / instance id | `RunRequest.context` |
| repository workspace | `environment.workspace` or request context |
| `model_name` | `models.default` override |
| `mcp_servers` | `mcp` profile field |
| `skills_dir` | `skills` profile field |
| Harbor logs/artifacts dir | Fabric artifact output root |

## Implementation Sequence

1. Run one Harbor-generated SWE-Bench task through Fabric and Hermes outside
   Harbor's agent interface. Done for Docker-backed patch capture.
2. Fix patch capture for new/untracked files.
3. Add `FabricAgent(BaseAgent)` for Harbor. Done in the Harbor POC branch.
4. Run a simple Harbor task through `FabricAgent`.
5. Run `django__django-13741` through Harbor + Fabric + Hermes.
6. Enable Relay/ATIF capture on the same path.
7. Switch harness/profile without changing Harbor integration code.

## Non-Goals For The First Pass

- Reimplement SWE-Bench dataset conversion in Fabric.
- Move verification or reward calculation into Fabric.
- Make Fabric own Docker/OpenSandbox/Kubernetes provisioning for Harbor runs.
- Build a persistent Fabric service runtime before the one-task path works.

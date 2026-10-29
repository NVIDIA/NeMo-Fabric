<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor Qualification Runbook

When invoked as an agent skill, start with [interactive setup](setup.md): choose a shared BYOM endpoint or individual provider credentials, resolve compatible routes, and preview coverage. The commands below are the replayable execution reference. The agent generates and validates the plan; a user does not need to hand-edit the per-adapter examples.

## Prerequisites And Scope

Run from the repository root on Linux x86_64 with Git, Python 3.12 or newer, `uv`, Docker, and Docker Compose. The image builder preserves the selected task Docker environment and adds Node.js 24.16.0, Rust 1.94.0, and packaged NVIDIA NeMo Fabric and harness dependencies. The baseline task image uses Python 3.13 on Debian Bookworm; that is not a Fabric or Harbor version pin. Select any compatible installed interpreter for each component. Custom tasks must provide a compatible Debian environment with pip and apt-get. It can download an isolated host Python interpreter through `uv`. Use a dedicated output directory on storage visible to the Docker daemon; Snap Docker needs a `TMPDIR` visible outside its private `/tmp`, as described in the [Harbor examples](../../../../examples/harbor/README.md#docker-installed-with-snap). Remote Docker daemons need the same absolute paths as the client. No host credential directories are mounted.

Choose the exact NVIDIA NeMo Fabric package version and matching checkout/catalog, plus an exact Harbor version compatible with that NeMo Fabric line. The native Harbor `fabric` agent is the default; `--bridge sdk` explicitly tests `nemo_fabric.integrations.harbor:FabricAgent` instead. Record the distinction when comparing runs. Native Harbor and the SDK bridge are independently versioned implementations.

`--mode registry` installs the exact published version. `--mode source` builds noneditable wheels from the recorded committed source, installs the same runtime wheel on host and task, and builds TypeScript tarballs from the checked-in lockfiles. Source mode qualifies source integration, not registry publication. Its local tarballs can satisfy package dependencies that are not published yet. Run registry mode against the selected packages before treating registry installability as proven. Build logs and identities retain actual resolved harness versions; unpinned upstream ranges can resolve differently on another date.

For a published prerelease, pass `--catalog-json` with the catalog resource extracted from that exact catalog wheel. Published prerelease packages may use stamped metadata while the committed catalog can retain the stable base version. Keep the reference checkout at the matching source revision: the gate still checks its complete adapter denominator and descriptor bytes against the selected packages, then verifies the installed catalog's full hash. It preserves Python and npm versions independently (for example, `0.6.0rc1` and `0.6.0-rc.1`). To obtain the resource without importing any harness, run:

```bash
uv tool run --from pip pip download --no-deps --only-binary=:all: \
  --dest "$WORK/catalog" "nemo-fabric-adapter-catalog==$PACKAGE_VERSION"
python3 - "$WORK/catalog" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import sys
directory = Path(sys.argv[1])
with ZipFile(next(directory.glob('*.whl'))) as wheel:
    (directory / 'catalog.json').write_bytes(
        wheel.read('nemo_fabric_adapter_catalog/catalog.json'))
PY
```

Use `--mode registry --version "$PACKAGE_VERSION" --catalog-json "$WORK/catalog/catalog.json"` with the other plan arguments below. Use a fresh catalog directory per package version.

The denominator is every descriptor in the checked-in metadata catalog, including both NOOA variants and remote-agent. Workflow-only adapters remain unsupported until an explicit registered target is selected with the SDK bridge; their catalog target IDs remain listed in the report. The separate BenchAgent entry exercises standalone NOOA. Future catalog entries appear automatically and need a reviewed installation recipe/provider profile. They are not silently removed when no harness or credential is available. The builder derives Python extras and TypeScript optional peers from package manifests; a few harnesses deliberately installed outside adapter dependencies have explicit tested recipes in `prepare.py`. Review those recipes when the supported harness pins change.

The OpenHands recipe uses the adapter's documented compatible stack:
`openhands-sdk==1.50.0`, `openhands-tools==1.50.0`, `litellm==1.100.1`, and
`mcp==1.28.1`. Newer incompatible LiteLLM/MCP combinations can surface as an
SDK service-unavailable error before any provider request. Inspect retained
stderr and installed versions before attributing that error to inference.
The image runs `pip check`. Task dependencies such as gRPC/protobuf resolve
normally; their observed versions remain in the evidence without becoming
extra pins or integration pass criteria.

## Serve A Local Checkpoint Or Select A Provider

A filesystem checkpoint path is not an inference API. Start with an existing operator-managed NIM/vLLM or other compatible server for that checkpoint, using an allocated GPU and the server's documented serving command. This gate deliberately does not load checkpoints, install serving infrastructure, allocate GPUs, or modify existing model services. Validate these prerequisites:

- The endpoint advertises the exact served model ID and supports tool calling. Responses, Chat Completions, and Anthropic Messages are different protocols. Check the API description and exercise the selected harness; an advertised route alone is not a passing integration test.
- The task container can reach the endpoint. Its `127.0.0.1` is the container, not the host. Keep a local server bound to loopback by default and arrange an operator-approved container route/proxy if needed. Do not expose a model service publicly to make the gate work.
- HTTPS is the default for remote endpoints. For an explicitly trusted private HTTP endpoint, name the exact host with `--trusted-http-host`; it is also forwarded to harnesses that advertise `insecure_http_model_hosts`. This transmits prompts and any credential without TLS to that host.
- Export the selected credential variable through your usual secret source. Never put the value in the plan, command arguments, Docker build arguments, settings, URLs, or committed files. For a confirmed unauthenticated endpoint, a nonsecret `MODEL_API_KEY=unused` satisfies clients that require a key. It is not an authentication test.

Claude needs Anthropic Messages, Codex normally uses Responses, and most other local coding harnesses use Chat Completions. `--endpoint-kind openai-and-anthropic` permits the same endpoint to be attempted through both families. Kiro requires its native provider, authentication and advertised model. Droid uses provider `factory`; its supported native BYOK profile can route to a custom endpoint without setting normalized model `base_url`.

For Kiro, keep the row prerequisite-blocked (skipped execution) unless the
native CLI, `KIRO_API_KEY`, and a Kiro-advertised model are available. Kiro does
not support BYOM through this adapter: do not use the local Nemotron endpoint,
a dummy `MODEL_API_KEY`, or an experimental protocol bridge. To enable a native
trial, select `model_provider=kiro`, the advertised model ID, `base_url=null`,
and `api_key_env=KIRO_API_KEY`, then mark the prepared row runnable. The runner
checks credential presence before task installation or launch; only the secret
variable name belongs in the plan. Missing credentials retain a blocked row,
not an integration failure or a full-catalog pass.

For Remote Agent HTTP integration, explicitly set its row to `action=run`, clear
normalized `base_url`, and configure `harness_settings` with `base_url` and
`api_type=openai-completions` for a compatible endpoint. Keep the actual served
model ID. This exercises the declared HTTP adapter contract; autonomous remote
workspace/tool execution remains untested. Task reward zero does not negate a
successful round trip.

For Droid BYOK, the prepared image includes CLI `droid@0.233.0` and the adapter's
pinned SDK. Extend its isolated task image with native
`/root/.factory/settings.json`, following the [Droid BYOK guide](../../../../docs/integrations/harness/droid.mdx#optional-use-a-custom-byok-model).
Set `cloudSessionSync=false` and define the real served model in `customModels`
using provider `generic-chat-completion-api`, the endpoint `baseUrl`, and an
`apiKey` environment reference. Select `custom:<served-model-id>` in the plan,
`model_provider=factory`, `base_url=null`, and `api_key_env=FACTORY_API_KEY`.
For an explicitly unauthenticated local endpoint only, the reference can also
be `${FACTORY_API_KEY}` with dummy value `EMPTY`; this is not authentication
coverage. Keep that profile confined to the disposable task image, record its
Dockerfile/settings and final image identity, then mark the row runnable. Do
not mount ambient Factory configuration or credentials.

Hermes uses its upstream-supported Python 3.14 runtime and source/assets layout;
its broader package metadata also supports older updaters, not older runtimes.
Use a prepared task image with an isolated 3.14 environment, its frozen core +
`mcp` dependency graph, and the exact supported source installation. Do not
bypass its wheel/sdist build guard. Set the row's `task_python` to that
environment's executable. Harbor may use a different compatible host Python;
the runner and identity/descriptor probes use `task_python`, while the upstream
task verifier keeps its original interpreter and instructions. No exact match
between host and task Python versions is required.

NeMo Fabric runtime/catalog/adapter packages must remain noneditable. A pinned
Hermes source installation is supported even when its metadata says editable:
record `{"hermes-agent":"<exact-commit>"}` in
`/opt/qualification-harness-sources.json`. The identity collector records its
actual Git revision, source URL and dirty state; the gate requires a matching
clean source revision before and after the trial. The complete dependency and
interpreter identities remain evidence. Read-only image permissions are useful
but are not a sandbox against the task's root user.

The Codex profile uses `danger-full-access` and `deny_all` **inside the gate-created Harbor task container**. Harbor supplies the isolation boundary; nested `bwrap` sandboxing requires user namespaces that many Docker hosts prohibit. Do not reuse this profile on a host workspace or attach host credentials, Docker sockets, or unrelated files to the task. The gate never enables privileged Docker mode or changes host namespace policy.

Estimate paid-provider cost before running, using the selected model's current prices, expected prompt/output tokens per trial, and the number of runnable entries. Supply the estimate and assumptions explicitly. The estimate is recorded evidence, not an enforced spending cap. The deadline bounds runtime, not token billing. Never retry silently or infer a zero cost from unavailable accounting.

## Plan, Prepare, And Run

Run **one shared task across the harnesses**, never the full TB2 dataset. The baseline is [`kv-store-grpc`](https://github.com/harbor-framework/terminal-bench-2/tree/2fd12b88aafdd04a52c298e3940bcb189f9766d6/kv-store-grpc): dependency installation, protocol definition, generated code, server implementation, and a background service checked with real RPCs. Fetch the pinned task outside the NeMo Fabric checkout:

```bash
WORK=/path/to/private/qualification
mkdir -m 700 "$WORK"
git clone --no-checkout https://github.com/harbor-framework/terminal-bench-2.git "$WORK/tb2"
git -C "$WORK/tb2" sparse-checkout set kv-store-grpc
git -C "$WORK/tb2" checkout 2fd12b88aafdd04a52c298e3940bcb189f9766d6
```

The gate requires `--task` pointing to that single directory. It records a hash of the task inputs and verifier, rejects a dataset root, and rejects changed files after planning. Preparation builds the task's original Dockerfile before adding NeMo Fabric and each harness. Harbor builds from the prepared image rather than pulling the task-only `docker_image`. The instruction, task assets, resources, and upstream verifier stay intact; the oracle solution is excluded from staging. A wrapper collects the agent's final package and selection evidence before running the upstream verifier, whose own dependency installation is outside the agent's execution. NeMo Fabric package/wheel identity, installed Python harness distributions, npm harness packages, interpreter/source identity, catalog and descriptor provenance remain protected. Other Python package changes are recorded, allowing task setup to install or replace its own dependencies. If such changes break execution, lifecycle/result checks still fail. The original base and final image IDs are retained. This integration overlay is not an unmodified leaderboard submission.

Set the version, compatible Harbor pin, served model ID, and container-reachable endpoint for your environment. The following example uses an already running local NIM endpoint and a nonsecret placeholder key:

```bash
QUALIFY=.agents/skills/qualify-harbor/scripts
export MODEL_API_KEY=unused

python3 "$QUALIFY/qualify.py" plan \
  --output "$WORK/plan.json" \
  --task "$WORK/tb2/kv-store-grpc" \
  --mode source --version 0.6.0 \
  --harbor-version 0.24.1.dev202610250408 \
  --model nvidia/nemotron-3-ultra-550b-a55b \
  --base-url http://model-gateway:8000/v1 \
  --trusted-http-host model-gateway \
  --endpoint-kind openai-and-anthropic \
  --backend 'Operator-managed NIM; record the exact image/version' \
  --estimated-cost-usd 0 \
  --cost-basis 'Existing local inference; no metered provider charge; hardware excluded' \
  --timeout 900

# Inspect and configure rows before preparing; retain the full denominator.
python3 "$QUALIFY/prepare.py" --plan "$WORK/plan.json" --work "$WORK/environment"

"$WORK/environment/host/bin/python" "$QUALIFY/qualify.py" run \
  --plan "$WORK/plan.json" --environment "$WORK/environment/environment.json" \
  --output "$WORK/full-run"
```

Preparation and execution each require new directories and never overwrite previous evidence. Pass `--host-python /absolute/path/to/python` to `prepare.py` to use an existing Python 3.12+ interpreter rather than downloading Python 3.12. A base installation failure writes `environment.json` and logs before exiting; run the gate with `python3` in that case to produce a fully blocked report. `prepare.py` continues past individual harness installation failures, which become failed matrix rows. Retain the environment directory while inspecting or reproducing a run. Images are tagged with the source revision; record their immutable IDs in the report. No image is pushed.

For a keyed HTTPS provider, export `MODEL_API_KEY` securely, use its model slug and `--base-url https://...`, supply a nonzero cost estimate with token/pricing assumptions, and set the appropriate endpoint kind. Each plan row can override `model`, `model_provider`, `base_url`, `api_key_env`, `harness_settings`, `telemetry`, and `image`. A custom task image must contain the matching runtime/catalog/adapter, its harness, `/qualification/identity.py`, and `/opt/qualification-identity.json` produced by that script. NeMo Fabric packages must be noneditable; supported pinned Hermes source installs follow the provenance rules above. Set a prerequisite row's `action` to `run` only after providing those dependencies. Do not pass a generic local model to a proprietary provider under a fabricated model alias.

For an OpenAI-compatible endpoint with a vendor-prefixed served ID, set that row’s `model_provider` to `openai` without changing `model`. This requires the SDK bridge containing [PR #534](https://github.com/NVIDIA/NeMo-Fabric/pull/534); generate the plan with `--bridge sdk`. Native Harbor’s equivalent option is deferred to a separate upstream change. The selected provider is retained in both admission input and the report. Older SDKs remain usable when the override is omitted.

For adapter-declared model settings, use the SDK bridge and set that plan row's
`model_settings` object. It projects to `fabric_model_settings` and remains
subject to the adapter model schema. For example, an explicitly unpriced
Mini-SWE model uses `{"cost_tracking":"disabled"}` with a positive `max_turns`
on the same row. This disables monetary budgeting and keeps cost unavailable;
it does not report a zero cost. `max_turns` projects to `fabric_max_turns`.
The bridge and runner must both contain the model-settings transport from #534.

For a registered workflow, generate the plan with `--bridge sdk` and configure
that adapter's row with `workflow_target_id` and optional `workflow_settings`.
For example, the generic NOOA adapter can select its registered coding target:

```json
{
  "workflow_target_id": "nvidia.nooa.coding-agent",
  "workflow_settings": {},
  "action": "run",
  "reason": ""
}
```

Keep the row's existing adapter ID, catalog record, model, credentials reference,
and all other catalog rows. Do not substitute the separate BenchAgent adapter.
The target must exist in the exact catalog snapshot, bind to the selected adapter,
and have an unchanged descriptor. Settings must be JSON without inline
credentials; the SDK validates them against the target's schema before setup.
This requires matching host/task SDKs with registered-workflow Harbor support.
Native Harbor workflow transport is not qualified by this path.

The command and installed-image preflight transport the same target/settings.
Task evidence retains the target's factory and installed descriptor provenance;
the admission fingerprint covers both normalized adapter and target descriptors.
A factory, settings schema, selection or origin mismatch fails integration.
Workflow selection does not claim complete usage, MCP, skills or subagent
coverage. Unknown accounting remains unknown and is checked for exact transport.

A developer subset retains every other row as excluded and sets `scope=developer_subset`:

```bash
"$WORK/environment/host/bin/python" "$QUALIFY/qualify.py" run \
  --plan "$WORK/plan.json" --environment "$WORK/environment/environment.json" \
  --output "$WORK/pi-reproduction" --only nvidia.fabric.pi
```

Exit zero means the requested subset passed, or every entry passed in a full run. A subset never sets `full_catalog_passed=true`. Nonzero means failure, blocked coverage, cancellation, or invalid prerequisites.

The gate tests integration, not model task quality. A valid reward of zero does
not fail a completed Harbor/NeMo Fabric/harness round trip. The report keeps task
reward and `checks.task_success` as informational evidence. Missing, malformed,
or nonfinite reward evidence still fails verifier completion; Harbor exceptions,
failed invocations, timeouts, and cleanup failures remain failures.

Harness accounting availability is a separate capability observation. Missing
complete counters remain unknown, with a recorded limitation; they do not fail
integration when NeMo Fabric transports the same normalized unknown values into
Harbor. `checks.usage_available` is informational. A mismatch in result identity
or accounting transport still fails, regardless of reward or usage availability.
Reports record these pass criteria explicitly and show reward alongside the
integration outcome. A subset cannot establish full-catalog coverage.

## Evidence And Triage

`report.json` is the sanitized machine summary; `summary.md` is the human matrix. Both are rewritten atomically after each row, with pending rows retained if interrupted. `plan.json` retains the full descriptor snapshot. Each attempted row includes a reproducible command, timestamps, image identity, host/task package and harness versions, runtime wheel hash, observed runtime/invocation/adapter identity, selected workflow target/settings when present, reward, usage, capability limitations, and cleanup status. Catalog hashes cover raw metadata; Harbor admission hashes cover the core-normalized adapter descriptor and selected target descriptor. The gate records both and compares the uploaded admission hash against SDK metadata inspection, never directly against the raw catalog hash. Raw task artifacts stay in the private run directory. The gate redacts known credential values from text output, but binary artifacts and unknown secrets are not shareable by assumption; inspect before exporting them.

Source-built task images remove the build tree after packaging. The native core can discover its compile-time source directory when that directory still exists; identical descriptor bytes do not guarantee the intended execution origin. A metadata-only task-container preflight verifies installed Python package data or the explicitly selected npm package before any model call. The verifier records the actual task's descriptor provenance again. A bundled/source-tree origin or drift between those observations fails admission even when versions and fingerprints match.

Use the failure categories to choose the next investigation:

| Category | Start With |
| --- | --- |
| Prerequisite/install | `environment.json`, `task-build.log`, `base-build.log`, `host-install.log`, per-image `install.log`; package availability, Python/Node compatibility, harness pins |
| Auth/provider/model | Endpoint protocol, exact served ID, credential environment reference, harness error code, and task's model config |
| Adapter admission/fingerprint | Selected catalog SHA, uploaded `fabric-run-*.json`, installed descriptor, package versions, task identity |
| Transport/runner | Harbor trial exception, `agent/fabric-result-*.json` or runner-error envelope, runner stdout/stderr |
| Harbor lifecycle/verifier | Trial `result.json`, `exception_info`, verifier output, reward file; valid reward zero is an informational task outcome, missing/invalid reward is a failure |
| Usage | Invocation usage and Harbor accounting; missing complete counters are a harness limitation, transport mismatches fail; unknown is not zero |
| Telemetry/export | Relay validation, canonical ATIF schema, promoted `trajectory.json`, artifacts; baseline runs do not test telemetry |
| Timeout/cancellation/cleanup | Recorded deadline/exit code, partial logs, owned Compose resources and `cleanup.log` |
| Infrastructure | Connectivity, image fetch/build, storage exhaustion, server/rate-limit errors; reproduce before declaring a product regression |

Harbor normally deletes task containers. After completion or interruption, the gate removes only Compose containers/networks whose project label starts with this run's randomly allocated task prefix, plus metadata-probe containers carrying its exact qualification label. It reads resource labels without collecting container environments. It never prunes Docker globally or stops a model service. SIGINT is forwarded to its own process group, followed by bounded TERM/KILL escalation. Abrupt host loss or SIGKILL cannot guarantee cleanup; retain the run's task prefix and inspect its exact labels before manual cleanup. Images and build evidence are deliberately retained for reproduction. The gate does not retry trials; harness-native retries can still occur within the recorded deadline and remain visible in the logs.

For deterministic gate and integration checks, use the repository environment:

```bash
uv run --no-sync pytest tests/scripts/test_harbor_qualification.py \
  tests/integrations/test_harbor_runner.py tests/integrations/test_harbor_mcp.py \
  tests/integrations/test_harbor_skills.py tests/integrations/test_harbor_telemetry.py \
  tests/python/test_harbor_accounting.py tests/python/test_harbor_credentials.py \
  tests/python/test_adapter_capabilities.py
```

The baseline leaves MCP, skills, subagents, and telemetry untested where declared, or unsupported/not declared where metadata does not advertise them. For optional Relay qualification, configure a supported row with `telemetry=relay` and a compatible image containing Relay and any required harness extension; the gate validates export and canonical ATIF. Use the [SWE-Bench examples](../../../../examples/harbor/swebench/README.md) for live MCP/skills/telemetry scenarios and attach their exact commands and artifacts separately. Subagent functionality needs harness-specific evidence; neither one task nor ATIF schema validity establishes it. Historical flaky CI tests are context, not present-run exemptions.

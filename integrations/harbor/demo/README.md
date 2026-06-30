# Harbor Multi-Harness Demo

This demo keeps one Harbor external-agent surface while Fabric selects the
execution harness from an ordered profile stack. Harbor owns the task,
container, verifier, reward, and run layout. `FabricAgent` invokes the Fabric
Python SDK inside the task container; it does not invoke the Fabric CLI.

## Requirements

- Python 3.12+
- `uv`
- Docker
- this repository checkout, with the changes to test committed
- a host `codex login` for the real Codex variant

The first image build can take several minutes.

## Prepare the Build Context

Harbor builds `task/environment/Dockerfile` with the environment directory as
its Docker context. Export committed `HEAD` there so the image installs the
exact Fabric revision under test:

```bash
DEMO_DIR="$PWD/integrations/harbor/demo"
TASK_DIR="$DEMO_DIR/task"
RUNS_DIR="$DEMO_DIR/runs"
VENDOR_DIR="$TASK_DIR/environment/vendor/nemo-fabric"

rm -rf "$TASK_DIR/environment/vendor"
mkdir -p "$VENDOR_DIR"
git archive HEAD | tar -x -C "$VENDOR_DIR"
```

Keep this shell open for the commands below. Use a new `--job-name`, or remove
the matching generated directory under `$RUNS_DIR`, before repeating a run.

## Harbor Arguments

| Argument | Meaning |
| --- | --- |
| `--path` | Harbor task directory containing `task.toml`, environment, and verifier |
| `--agent` | Harbor external agent class imported from the local Fabric package |
| `--ak` | Constructor argument passed by Harbor to `FabricAgent` |
| `fabric_config_path` | Base Fabric YAML path inside the task container |
| `fabric_profile_paths` | Ordered Fabric profile YAML paths inside the task container |
| `--model` | Harbor model selection; Fabric applies it after the file-backed profiles |
| `--ae` | Environment variable passed to the Harbor agent inside the container |
| `--mounts` | Host-to-container mounts managed by Harbor |
| `--extra-docker-compose` | Compose overlay applied to the Harbor task environment |
| `--job-name` | Stable Harbor output directory name for this variant |
| `--force-build` | Rebuild the Harbor task image from the prepared context |

The JSON array passed to `fabric_profile_paths` is one Harbor `--ak` value. It
is not a Fabric CLI argument.

## 1. Credential-Free Smoke

This proves Harbor task setup, the external agent import, sandbox-local SDK
execution, Fabric profile resolution, workspace mutation, and verification:

```bash
uv run --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_config_path=/opt/fabric-demo/agent.yaml \
  --ak 'fabric_profile_paths=["/opt/fabric-demo/profiles/smoke.yaml"]' \
  --job-name fabric-smoke \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

Expected Harbor summary: one trial, zero exceptions, and mean reward `1.000`.

## 2. Hermes CLI

The Harbor command is unchanged except for model selection, the credential,
and the Fabric profile path:

```bash
export NVIDIA_API_KEY=...

uv run --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_config_path=/opt/fabric-demo/agent.yaml \
  --ak 'fabric_profile_paths=["/opt/fabric-demo/profiles/hermes.yaml"]' \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ae "NVIDIA_API_KEY=$NVIDIA_API_KEY" \
  --job-name fabric-hermes \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

## 3. Hermes with Relay Telemetry

This composes two ordered Fabric profiles. The first selects Hermes; the second
adds Relay OpenInference traces, ATOF events, and an ATIF trajectory without
changing the Harbor agent. Start Phoenix on the host before the Harbor run:

```bash
docker rm -f fabric-phoenix 2>/dev/null || true
docker run --rm --detach \
  --name fabric-phoenix \
  --publish 6006:6006 \
  arizephoenix/phoenix:latest

until curl --fail --silent http://localhost:6006 >/dev/null; do sleep 1; done
```

Visit `http://localhost:6006` in a browser.

The telemetry profile sends OTLP/HTTP traces from the Harbor task container to
Phoenix at `host.docker.internal`. The checked-in Compose overlay maps that name
to Docker's host gateway, including on Linux. Then run:

```bash
uv run --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_config_path=/opt/fabric-demo/agent.yaml \
  --ak 'fabric_profile_paths=["/opt/fabric-demo/profiles/hermes.yaml","/opt/fabric-demo/profiles/telemetry.yaml"]' \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --ae "NVIDIA_API_KEY=$NVIDIA_API_KEY" \
  --extra-docker-compose "$DEMO_DIR/host-gateway.compose.yaml" \
  --job-name fabric-hermes-relay \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

Keep Phoenix open. The completed run appears as an OpenInference trace in its
Traces view. Relay also writes the portable ATOF and ATIF records into Harbor's
collected agent logs:

```bash
find "$RUNS_DIR/fabric-hermes-relay" \
  -path '*/agent/fabric-artifacts/hermes-relay/relay/events.atof.jsonl' \
  -print -exec sed -n '1,5p' {} \;

find "$RUNS_DIR/fabric-hermes-relay" \
  -path '*/agent/fabric-artifacts/hermes-relay/relay/*.atif.json' \
  -print -exec python -m json.tool {} \;
```

## 4. Codex CLI

Codex uses the same Harbor agent and task. For this local Docker demo, Harbor
mounts the host Codex login as a read-only secret. The setup command copies it
into a writable container-local `CODEX_HOME`; Fabric only inherits that
environment and never reads the credential.

```bash
codex login status

CODEX_HOME_DIR="${CODEX_HOME:-$HOME/.codex}"
test -f "$CODEX_HOME_DIR/auth.json"
CODEX_AUTH_MOUNT="[{\"type\":\"bind\",\"source\":\"$CODEX_HOME_DIR/auth.json\",\"target\":\"/run/secrets/codex-auth.json\",\"read_only\":true}]"

uv run --extra harbor harbor run \
  --path "$TASK_DIR" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak fabric_config_path=/opt/fabric-demo/agent.yaml \
  --ak 'fabric_profile_paths=["/opt/fabric-demo/profiles/codex.yaml"]' \
  --ak 'fabric_install_command=mkdir -p "$CODEX_HOME" && cp /run/secrets/codex-auth.json "$CODEX_HOME/auth.json"' \
  --model openai/gpt-5.4 \
  --ae CODEX_HOME=/tmp/fabric-codex-home \
  --mounts "$CODEX_AUTH_MOUNT" \
  --job-name fabric-codex \
  --jobs-dir "$RUNS_DIR" \
  --n-concurrent 1 \
  --n-attempts 1 \
  --force-build
```

The image pins Codex CLI `0.142.4`. Harbor passes the selected model to the
Fabric SDK as the final typed profile, and the Codex profile pins a compatible
reasoning effort. The profile uses Codex `danger-full-access` because Harbor's
task container is the outer sandbox and nested Linux namespace creation is not
available there. The auth mount grants that trusted container access to your
Codex account for this run.

## Inspect the Result

Harbor records Fabric's normalized result in the trial's agent logs. For the
smoke variant:

```bash
find "$RUNS_DIR/fabric-smoke" -path '*/agent/fabric-result.json' -print -exec cat {} \;
cat "$RUNS_DIR/fabric-smoke/result.json"
uv run --extra harbor harbor view "$RUNS_DIR"
```

Check `status`, `profiles`, `harness`, `adapter_id`, runtime and invocation IDs,
artifacts, telemetry, Harbor exceptions, and reward. A successful smoke run has
Fabric status `succeeded` and Harbor mean reward `1.0`.

After the demo, remove the generated build-context copy:

```bash
rm -rf "$TASK_DIR/environment/vendor"
```

## Recording Flow

1. Show the common `--agent` and `fabric_config_path` values.
2. Run the credential-free smoke and inspect `fabric-result.json`.
3. Run Hermes, then Codex, changing only profile, model, and credential
   provisioning.
4. Start Phoenix, run Hermes plus telemetry, and open the resulting
   OpenInference trace.
5. Show the same run's ATOF events and ATIF trajectory from Harbor's logs.
6. Open all four jobs with `harbor view`.

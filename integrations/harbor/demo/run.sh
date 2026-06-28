#!/usr/bin/env bash
set -euo pipefail

variant="${1:-smoke}"
demo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_dir="$demo_dir/task"
runs_dir="$demo_dir/runs"

model_name=""
agent_env=""
case "$variant" in
  smoke)
    profile_paths='["/opt/fabric-demo/profiles/smoke.yaml"]'
    ;;
  hermes)
    : "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY for the Hermes demo}"
    profile_paths='["/opt/fabric-demo/profiles/hermes.yaml"]'
    model_name="nvidia/nemotron-3-nano-30b-a3b"
    agent_env="NVIDIA_API_KEY=$NVIDIA_API_KEY"
    ;;
  hermes-relay)
    : "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY for the Hermes telemetry demo}"
    profile_paths='["/opt/fabric-demo/profiles/hermes.yaml","/opt/fabric-demo/profiles/telemetry.yaml"]'
    model_name="nvidia/nemotron-3-nano-30b-a3b"
    agent_env="NVIDIA_API_KEY=$NVIDIA_API_KEY"
    ;;
  codex)
    : "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the Codex demo}"
    profile_paths='["/opt/fabric-demo/profiles/codex.yaml"]'
    agent_env="OPENAI_API_KEY=$OPENAI_API_KEY"
    if [[ -n "${FABRIC_CODEX_MODEL:-}" ]]; then
      model_name="openai/$FABRIC_CODEX_MODEL"
    fi
    ;;
  *)
    echo "usage: $0 {smoke|hermes|hermes-relay|codex}" >&2
    exit 2
    ;;
esac

repo_root="$(git -C "$demo_dir" rev-parse --show-toplevel)"
vendor_dir="$task_dir/environment/vendor"
rm -rf "$vendor_dir"
mkdir -p "$vendor_dir/nemo-fabric"
git -C "$repo_root" archive HEAD | tar -x -C "$vendor_dir/nemo-fabric"
trap 'rm -rf "$vendor_dir"' EXIT

harbor_args=(
  run
  --path "$task_dir"
  --agent nemo_fabric.integrations.harbor:FabricAgent
  --ak "fabric_config_path=/opt/fabric-demo/agent.yaml"
  --ak "fabric_profile_paths=$profile_paths"
  --job-name "fabric-$variant"
  --jobs-dir "$runs_dir"
  --n-concurrent 1
  --n-attempts 1
)
if [[ -n "$model_name" ]]; then
  harbor_args+=(--model "$model_name")
fi
if [[ -n "$agent_env" ]]; then
  harbor_args+=(--ae "$agent_env")
fi
if [[ "${FABRIC_DEMO_FORCE_BUILD:-0}" == "1" ]]; then
  harbor_args+=(--force-build)
fi

uv run --extra harbor harbor "${harbor_args[@]}"

python3 - "$runs_dir/fabric-$variant/result.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
stats = json.loads(path.read_text())["stats"]
means = [
    metric["mean"]
    for evaluation in stats["evals"].values()
    for metric in evaluation["metrics"]
    if "mean" in metric
]
if stats["n_errored_trials"] or means != [1.0]:
    raise SystemExit(f"Harbor demo failed; inspect {path}")
PY

echo
echo "Inspect all variants with: uv run --extra harbor harbor view $runs_dir"

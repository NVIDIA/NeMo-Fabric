#!/usr/bin/env bash
set -euo pipefail

variant="${1:-smoke}"
demo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_dir="$demo_dir/task"
runs_dir="$demo_dir/runs"

model_args=()
env_args=()
case "$variant" in
  smoke)
    profile_paths='["/opt/fabric-demo/profiles/smoke.yaml"]'
    ;;
  hermes)
    : "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY for the Hermes demo}"
    profile_paths='["/opt/fabric-demo/profiles/hermes.yaml"]'
    model_args=(--model "nvidia/nemotron-3-nano-30b-a3b")
    env_args=(--ae "NVIDIA_API_KEY=$NVIDIA_API_KEY")
    ;;
  hermes-relay)
    : "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY for the Hermes telemetry demo}"
    profile_paths='["/opt/fabric-demo/profiles/hermes.yaml","/opt/fabric-demo/profiles/telemetry.yaml"]'
    model_args=(--model "nvidia/nemotron-3-nano-30b-a3b")
    env_args=(--ae "NVIDIA_API_KEY=$NVIDIA_API_KEY")
    ;;
  codex)
    : "${OPENAI_API_KEY:?Set OPENAI_API_KEY for the Codex demo}"
    profile_paths='["/opt/fabric-demo/profiles/codex.yaml"]'
    env_args=(--ae "OPENAI_API_KEY=$OPENAI_API_KEY")
    if [[ -n "${FABRIC_CODEX_MODEL:-}" ]]; then
      model_args=(--model "openai/$FABRIC_CODEX_MODEL")
    fi
    ;;
  *)
    echo "usage: $0 {smoke|hermes|hermes-relay|codex}" >&2
    exit 2
    ;;
esac

force_build=()
if [[ "${FABRIC_DEMO_FORCE_BUILD:-0}" == "1" ]]; then
  force_build=(--force-build)
fi

uv run --extra harbor harbor run \
  --path "$task_dir" \
  --agent nemo_fabric.integrations.harbor:FabricAgent \
  --ak "fabric_config_path=/opt/fabric-demo/agent.yaml" \
  --ak "fabric_profile_paths=$profile_paths" \
  --job-name "fabric-$variant" \
  --jobs-dir "$runs_dir" \
  --n-concurrent 1 \
  --n-attempts 1 \
  "${model_args[@]}" \
  "${env_args[@]}" \
  "${force_build[@]}"

echo
echo "Inspect all variants with: uv run --extra harbor harbor view $runs_dir"

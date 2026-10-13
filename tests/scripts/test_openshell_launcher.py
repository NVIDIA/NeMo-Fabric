# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the shared OpenShell demo launcher's local orchestration.

The tests check command construction and ownership cleanup without starting
Docker, the OpenShell gateway, or a model endpoint.
"""

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).parents[2]

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="The OpenShell example launcher is a Bash/Linux script.",
)

MOCK_COMMAND = r"""
import json, os, pathlib, socket, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
root = pathlib.Path(os.environ["MOCK_ROOT"])
with (root / "operations.jsonl").open("a") as stream:
    stream.write(json.dumps({"command": name, "args": args,
                             "provider": os.environ.get("NEMO_FABRIC_OPEN_SHELL_PROVIDER"),
                             "openai_key_present": "OPENAI_API_KEY" in os.environ,
                             "model_key_matches_expected":
                                 os.environ.get("OPENAI_API_KEY") ==
                                 os.environ.get("MOCK_EXPECTED_MODEL_API_KEY")
                                 if "MOCK_EXPECTED_MODEL_API_KEY" in os.environ else None}) + "\n")
if name == "uname":
    print(os.environ.get("MOCK_OS", "Linux") if args == ["-s"] else "x86_64")
elif name == "cargo":
    binary = root / "target/debug/fabric-environment-openshell"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("mock provider")
    binary.chmod(0o755)
elif name == "docker":
    if args[:2] == ["image", "inspect"]:
        print("sha256:" + "a" * 64)
elif name == "openssl":
    if "-out" in args:
        output = pathlib.Path(args[args.index("-out") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("mock key")
    elif args[:2] == ["rand", "-hex"]:
        print("a" * (int(args[2]) * 2))
elif name == "openshell-gateway":
    port = int(args[args.index("--port") + 1])
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen()
        while True:
            connection, _ = server.accept()
            connection.close()
elif name == "openshell":
    if "sandbox" not in args:
        if not any(token in args for token in ("settings", "provider")):
            raise RuntimeError("unexpected OpenShell setup command")
        sys.exit(0)
    state = root / "sandbox-state.json"
    operation = args[args.index("sandbox") + 1]
    sandbox_name = args[args.index(operation) + 1] if operation != "create" else args[args.index("--name") + 1]
    if operation == "create":
        if len(sandbox_name) > 19:
            raise RuntimeError("OpenShell sandbox name exceeds 19 characters")
        state.write_text(json.dumps({"name": sandbox_name,
                                     "id": os.environ.get("MOCK_SANDBOX_ID", "mock-resource-id")}))
    elif operation == "delete":
        if state.exists() and json.loads(state.read_text())["name"] == sandbox_name:
            state.unlink()
    elif operation == "get":
        if not state.exists() or json.loads(state.read_text())["name"] != sandbox_name:
            sys.exit(1)
        print(json.dumps({"id": os.environ.get("MOCK_SANDBOX_ID", "mock-resource-id")}))
elif name == "curl":
    raise RuntimeError("tests must use cached release assets")
elif name == "uv":
    if args[0] == "run":
        sys.exit(int(os.environ.get("MOCK_CONSUMER_EXIT", "0")))
"""


@pytest.fixture(name="launcher")
def launcher_fixture(tmp_path):
    shared = tmp_path / "examples/openshell"
    shared.mkdir(parents=True)
    for filename in ["run-demo.sh", "Dockerfile.sdk", "Dockerfile.sdk.dockerignore"]:
        shutil.copyfile(ROOT / "examples/openshell" / filename, shared / filename)

    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    command = f"#!{sys.executable}\n" + MOCK_COMMAND
    for name in ["cargo", "curl", "docker", "openssl", "uname", "uv"]:
        executable = mock_bin / name
        executable.write_text(command)
        executable.chmod(0o755)

    cache = tmp_path / ".tmp/openshell-releases/v0.1.2"
    cache.mkdir(parents=True)
    assets = [
        ("openshell", "openshell", "musl", "openshell-checksums-sha256.txt"),
        (
            "openshell-gateway",
            "openshell-gateway",
            "gnu",
            "openshell-gateway-checksums-sha256.txt",
        ),
        (
            "openshell-sandbox",
            "openshell-sandbox",
            "musl",
            "openshell-sandbox-checksums-sha256.txt",
        ),
    ]
    for component, binary, target, sums_name in assets:
        asset = cache / f"{component}-x86_64-unknown-linux-{target}.tar.gz"
        content = command.encode()
        with tarfile.open(asset, "w:gz") as archive:
            info = tarfile.TarInfo(binary)
            info.size, info.mode = len(content), 0o755
            archive.addfile(info, io.BytesIO(content))
        cache.joinpath(sums_name).write_text(
            hashlib.sha256(asset.read_bytes()).hexdigest() + "  " + asset.name + "\n"
        )

    host_python = tmp_path / "host-env/bin/python"
    host_python.parent.mkdir(parents=True)
    host_python.write_text("mock python")
    host_python.chmod(0o755)
    port = _free_port()
    env = {
        **os.environ,
        "MOCK_ROOT": str(tmp_path),
        "PATH": str(mock_bin) + os.pathsep + os.environ["PATH"],
        "UV_PROJECT_ENVIRONMENT": str(host_python.parents[1]),
        "OPENSHELL_EXAMPLE_PORT": str(port),
        "OPENSHELL_EXAMPLE_GATEWAY_START_TIMEOUT": "10",
        "OPENSHELL_EXAMPLE_RESULTS_DIR": str(tmp_path / "results"),
    }
    return shared / "run-demo.sh", tmp_path, env


def _free_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


def _operations(root: Path) -> list[dict]:
    path = root / "operations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _run(launcher, workload, env):
    script, _, base_env = launcher
    return subprocess.run(
        ["bash", str(script), str(workload), "--input", "words with spaces"],
        env={**base_env, **env},
        text=True,
        capture_output=True,
        timeout=20,
    )


def test_generic_workload_gets_ownership_and_consumer_args(launcher):
    _, root, _ = launcher
    workload = root / "third-workload"
    workload.mkdir()
    hook = workload / "demo.sh"
    hook.write_text(
        """WORKLOAD_NAME=third-workload
SUPPORTED_MODES=(caller_owned fabric_owned)
IMAGE_DOCKERFILE="${WORKLOAD_ROOT}/runtime-image.Dockerfile"
POLICY_FILE="${WORKLOAD_ROOT}/policy.yaml"
stage_workload_image() { :; }
run_workload_consumer() { local mode="$1"; shift; uv run --no-sync consumer "$mode" "$@"; }
"""
    )

    completed = _run(launcher, hook, {"OPENSHELL_EXAMPLE_MODE": "both"})

    assert completed.returncode == 0, completed.stderr
    calls = [operation for operation in _operations(root) if operation["command"] == "uv"]
    assert [call["args"][3] for call in calls] == ["caller_owned", "fabric_owned"]
    assert all(call["args"][-2:] == ["--input", "words with spaces"] for call in calls)
    assert all(call["provider"].endswith("/fabric-environment-openshell") for call in calls)
    assert not (root / "sandbox-state.json").exists()


@pytest.mark.parametrize("mode", ["caller_owned", "fabric_owned"])
def test_sdk_example_hook_passes_normal_sdk_lifecycle_options(launcher, mode):
    _, root, _ = launcher
    completed = _run(
        launcher,
        ROOT / "examples/openshell_echo/demo.sh",
        {"OPENSHELL_EXAMPLE_MODE": mode},
    )

    assert completed.returncode == 0, completed.stderr
    call = next(operation for operation in _operations(root) if operation["command"] == "uv")
    assert call["args"][:5] == [
        "run",
        "--no-sync",
        "python",
        "-m",
        "examples.openshell_echo.consumer",
    ]
    assert "--ownership" in call["args"]
    assert call["args"][call["args"].index("--ownership") + 1] == mode
    assert "--sandbox-name" in call["args"]
    if mode == "caller_owned":
        assert "--sandbox-id" in call["args"]
    else:
        assert "--sandbox-id" not in call["args"]


def test_consumer_failure_cleans_up_caller_owned_sandbox(launcher):
    _, root, _ = launcher
    completed = _run(
        launcher,
        ROOT / "examples/openshell_echo/demo.sh",
        {"OPENSHELL_EXAMPLE_MODE": "caller_owned", "MOCK_CONSUMER_EXIT": "9"},
    )

    assert completed.returncode == 9
    assert not (root / "sandbox-state.json").exists()
    assert any(operation["command"] == "openshell" and "delete" in operation["args"] for operation in _operations(root))


def test_occupied_gateway_port_stops_before_building(launcher):
    _, root, env = launcher
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", int(env["OPENSHELL_EXAMPLE_PORT"])))
        occupied.listen()
        completed = _run(launcher, ROOT / "examples/openshell_echo/demo.sh", {})

    assert completed.returncode == 2
    assert "already in use; set OPENSHELL_EXAMPLE_PORT" in completed.stderr
    assert not any(operation["command"] in {"cargo", "docker", "openshell", "uv"} for operation in _operations(root))


@pytest.mark.parametrize("current_id,deleted", [("owned-id", True), ("replacement-id", False)])
def test_fabric_owned_failure_cleanup_requires_the_recorded_sandbox_id(
    launcher, current_id, deleted
):
    _, root, _ = launcher
    workload = root / "fabric-owned-workload.sh"
    workload.write_text(
        """WORKLOAD_NAME=fabric-owned-workload
SUPPORTED_MODES=(fabric_owned)
stage_workload_image() { :; }
run_workload_consumer() {
  local mode="$1"
  shift
  local sandbox_name="$2"
  printf '{"name":"%s","id":"%s"}\\n' "$sandbox_name" "owned-id" > "$OPENSHELL_FABRIC_SANDBOX_STATE_FILE"
  printf '{"name":"%s","id":"%s"}\\n' "$sandbox_name" "$MOCK_SANDBOX_ID" > "$MOCK_ROOT/sandbox-state.json"
  uv run --no-sync consumer "$mode" "$@"
}
"""
    )

    completed = _run(
        launcher,
        workload,
        {
            "OPENSHELL_EXAMPLE_MODE": "fabric_owned",
            "MOCK_CONSUMER_EXIT": "9",
            "MOCK_SANDBOX_ID": current_id,
        },
    )

    assert completed.returncode == 9
    assert (root / "sandbox-state.json").exists() is (not deleted)
    deleted_calls = [
        operation
        for operation in _operations(root)
        if operation["command"] == "openshell" and "delete" in operation["args"]
    ]
    assert bool(deleted_calls) is deleted


def test_code_review_workload_configures_hosted_provider_before_sandbox(launcher):
    _, root, _ = launcher
    completed = _run(
        launcher,
        ROOT / "examples/code_review_agent/demo-openshell-openclaw-sdk.sh",
        {
            "OPENSHELL_EXAMPLE_MODE": "caller_owned",
            "NVIDIA_API_KEY": "test-key-must-not-appear-in-command-arguments",
        },
    )

    assert completed.returncode == 0, completed.stderr
    operations = _operations(root)
    provider_calls = [
        operation
        for operation in operations
        if operation["command"] == "openshell"
        and "provider" in operation["args"]
        and operation["args"][operation["args"].index("provider") + 1]
        in ("profile", "create")
    ]
    profile, provider = provider_calls
    profile_args = profile["args"]
    assert profile_args[profile_args.index("provider") + 1 :][:2] == ["profile", "import"]
    assert Path(profile_args[profile_args.index("--file") + 1]).name == "openshell-provider.yaml"
    provider_args = provider["args"]
    assert provider_args[provider_args.index("provider") + 1] == "create"
    assert provider_args[provider_args.index("--name") + 1] == "hosted"
    assert provider_args[provider_args.index("--type") + 1] == "nvidia-code-review"
    assert provider_args[provider_args.index("--credential") + 1] == "OPENAI_API_KEY"
    assert provider["openai_key_present"]
    create = next(
        operation
        for operation in operations
        if operation["command"] == "openshell" and "sandbox" in operation["args"]
    )
    assert "hosted" in create["args"]
    assert "test-key-must-not-appear-in-command-arguments" not in json.dumps(operations)


@pytest.mark.parametrize("relay_enabled", [False, True])
def test_hermes_code_review_workload_selects_harness_and_optional_relay(
    launcher, relay_enabled
):
    _, root, _ = launcher
    profile = root / "local-model-profile.yaml"
    profile.write_text("id: local-code-review\n")
    completed = _run(
        launcher,
        ROOT / "examples/code_review_agent/demo-openshell-hermes-sdk.sh",
        {
            "OPENSHELL_EXAMPLE_MODE": "caller_owned",
            "OPENSHELL_CODE_REVIEW_RELAY": str(relay_enabled).lower(),
            "OPENAI_API_KEY": "test-key-must-not-appear-in-command-arguments",
            "MOCK_EXPECTED_MODEL_API_KEY": "test-key-must-not-appear-in-command-arguments",
            "OPENSHELL_MODEL_PROVIDER_PROFILE": str(profile),
            "OPENSHELL_CODE_REVIEW_BASE_URL": "https://inference.example.test/v1",
            "OPENSHELL_CODE_REVIEW_MODEL": "nvidia/nemotron-3-ultra-550b-a55b",
        },
    )

    assert completed.returncode == 0, completed.stderr
    operations = _operations(root)
    call = next(operation for operation in operations if operation["command"] == "uv")
    assert "examples.code_review_agent.openshell_sdk_consumer" in call["args"]
    assert call["args"][call["args"].index("--harness") + 1] == "hermes"
    assert ("--relay" in call["args"]) is relay_enabled
    assert call["args"][call["args"].index("--base-url") + 1] == (
        "https://inference.example.test/v1"
    )
    assert call["args"][call["args"].index("--model") + 1] == (
        "nvidia/nemotron-3-ultra-550b-a55b"
    )
    profile_call = next(
        operation
        for operation in operations
        if operation["command"] == "openshell" and "profile" in operation["args"]
    )
    assert profile_call["args"][profile_call["args"].index("--file") + 1] == (
        str(profile)
    )
    provider_call = next(
        operation
        for operation in operations
        if operation["command"] == "openshell" and "create" in operation["args"]
        and "provider" in operation["args"]
    )
    assert provider_call["model_key_matches_expected"]
    assert provider_call["args"][provider_call["args"].index("--type") + 1] == (
        "local-code-review"
    )
    assert "test-key-must-not-appear-in-command-arguments" not in json.dumps(operations)


def test_hermes_default_profile_uses_nvidia_key_when_both_keys_exist(launcher):
    _, root, _ = launcher
    completed = _run(
        launcher,
        ROOT / "examples/code_review_agent/demo-openshell-hermes-sdk.sh",
        {
            "OPENSHELL_EXAMPLE_MODE": "caller_owned",
            "NVIDIA_API_KEY": "expected-nvidia-key",
            "OPENAI_API_KEY": "different-openai-key",
            "MOCK_EXPECTED_MODEL_API_KEY": "expected-nvidia-key",
        },
    )

    assert completed.returncode == 0, completed.stderr
    provider_call = next(
        operation
        for operation in _operations(root)
        if operation["command"] == "openshell" and "create" in operation["args"]
        and "provider" in operation["args"]
    )
    assert provider_call["model_key_matches_expected"]


def test_hermes_missing_credential_stops_before_provider_creation(launcher):
    _, root, _ = launcher
    completed = _run(
        launcher,
        ROOT / "examples/code_review_agent/demo-openshell-hermes-sdk.sh",
        {
            "OPENSHELL_EXAMPLE_MODE": "caller_owned",
            "NVIDIA_API_KEY": "",
            "OPENAI_API_KEY": "",
        },
    )

    assert completed.returncode == 2
    assert "NVIDIA_API_KEY is required" in completed.stderr
    assert not any(
        operation["command"] == "openshell"
        and ("provider" in operation["args"] or "sandbox" in operation["args"])
        for operation in _operations(root)
    )


@pytest.mark.parametrize("relay,mcp", [(False, False), (True, True)])
def test_langgraph_openshell_workload_uses_sdk_consumer_and_host_provider(
    launcher, relay, mcp
):
    _, root, _ = launcher
    secret = "test-model-key-must-not-appear-in-command-arguments"
    completed = _run(
        launcher,
        ROOT / "examples/langgraph_custom_agent/demo-openshell-sdk.sh",
        {
            "OPENSHELL_EXAMPLE_MODE": "caller_owned",
            "OPENSHELL_LANGGRAPH_RELAY": str(relay).lower(),
            "OPENSHELL_LANGGRAPH_MCP": str(mcp).lower(),
            "OPENAI_API_KEY": secret,
            "OPENSHELL_MODEL_PROVIDER_PROFILE": "/tmp/langgraph-model-profile.yaml",
            "OPENSHELL_LANGGRAPH_BASE_URL": "https://models.example.test/v1",
        },
    )

    assert completed.returncode == 0, completed.stderr
    operations = _operations(root)
    call = next(operation for operation in operations if operation["command"] == "uv")
    assert "examples.langgraph_custom_agent.consumer.openshell" in call["args"]
    assert ("--relay" in call["args"]) is relay
    assert ("--mcp" in call["args"]) is mcp
    assert call["args"][call["args"].index("--base-url") + 1] == (
        "https://models.example.test/v1"
    )
    profile_call = next(
        operation
        for operation in operations
        if operation["command"] == "openshell" and "profile" in operation["args"]
    )
    assert profile_call["args"][profile_call["args"].index("--file") + 1] == (
        "/tmp/langgraph-model-profile.yaml"
    )
    provider_call = next(
        operation
        for operation in operations
        if operation["command"] == "openshell"
        and "provider" in operation["args"]
        and operation["args"][operation["args"].index("provider") + 1] == "create"
    )
    assert "nvidia-langgraph" in provider_call["args"]
    assert secret not in json.dumps(operations)


@pytest.mark.parametrize(
    "environment,message",
    [({"MOCK_OS": "Darwin"}, "requires Linux"),
     ({"OPENSHELL_EXAMPLE_MODE": "unknown"}, "must be caller_owned")],
)
def test_invalid_platform_or_mode_stops_before_building(launcher, environment, message):
    script, root, base_env = launcher
    workload = root / "workload.sh"
    workload.write_text(
        "WORKLOAD_NAME=workload\nSUPPORTED_MODES=(caller_owned)\n"
        "stage_workload_image() { :; }\nrun_workload_consumer() { :; }\n"
    )
    completed = subprocess.run(
        ["bash", str(script), str(workload)],
        env={**base_env, **environment},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert message in completed.stderr
    assert not any(operation["command"] in {"cargo", "uv", "openshell"} for operation in _operations(root))

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the shared OpenShell launcher without Docker or network access."""

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

MOCK_COMMAND = r"""
import json, os, pathlib, socket, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
root = pathlib.Path(os.environ["MOCK_ROOT"])
with (root / "operations.jsonl").open("a") as stream:
    stream.write(json.dumps({"command": name, "args": args,
                             "provider": os.environ.get("NEMO_FABRIC_OPEN_SHELL_PROVIDER")}) + "\n")
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
    state = root / "sandbox-state.json"
    operation = args[args.index("sandbox") + 1]
    sandbox_name = args[args.index(operation) + 1] if operation != "create" else args[args.index("--name") + 1]
    if operation == "create":
        if len(sandbox_name) > 19:
            raise RuntimeError("OpenShell sandbox name exceeds 19 characters")
        state.write_text(json.dumps({"name": sandbox_name, "id": "mock-resource-id"}))
    elif operation == "delete":
        if state.exists() and json.loads(state.read_text())["name"] == sandbox_name:
            state.unlink()
    elif operation == "get":
        if not state.exists() or json.loads(state.read_text())["name"] != sandbox_name:
            sys.exit(1)
        print('{"id":"mock-resource-id"}')
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
        ROOT / "examples/openshell_sdk/demo.sh",
        {"OPENSHELL_EXAMPLE_MODE": mode},
    )

    assert completed.returncode == 0, completed.stderr
    call = next(operation for operation in _operations(root) if operation["command"] == "uv")
    assert call["args"][:5] == [
        "run",
        "--no-sync",
        "python",
        "-m",
        "examples.openshell_sdk.consumer",
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
        ROOT / "examples/openshell_sdk/demo.sh",
        {"OPENSHELL_EXAMPLE_MODE": "caller_owned", "MOCK_CONSUMER_EXIT": "9"},
    )

    assert completed.returncode == 9
    assert not (root / "sandbox-state.json").exists()
    assert any(operation["command"] == "openshell" and "delete" in operation["args"] for operation in _operations(root))


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

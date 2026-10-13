# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit environment lifecycle remains separate from runtime shutdown."""

from pathlib import Path, WindowsPath
from unittest.mock import MagicMock

import pytest
from nemo_fabric import (
    EnvironmentHandle,
    EnvironmentReference,
    Fabric,
    FabricConfigError,
)
from _utils.configs import hermes_shim_config


async def test_prepared_local_environment_supports_multiple_runtimes(
    hermes_shim_agent_dir,
):
    fabric = Fabric()
    config = hermes_shim_config()
    environment = await fabric.prepare_environment(
        config, base_dir=hermes_shim_agent_dir
    )
    assert isinstance(environment, EnvironmentHandle)
    assert environment.workspace == hermes_shim_agent_dir / "repos" / "my-service"
    runtimes = []
    try:
        for _ in range(2):
            runtime = await fabric.start_runtime_in(
                config, environment, base_dir=hermes_shim_agent_dir
            )
            runtimes.append(runtime)
            assert (
                runtime.handle.environment["environment_id"]
                == environment.environment_id
            )
        results = [
            await runtime.invoke(input={"turn": i})
            for i, runtime in enumerate(runtimes)
        ]
        assert all(result.status == "succeeded" for result in results)
        assert runtimes[0].runtime_id != runtimes[1].runtime_id
    finally:
        for runtime in runtimes:
            await runtime.stop()
        await fabric.release_environment(environment)
    assert hermes_shim_agent_dir.is_dir()


@pytest.mark.parametrize("method", ["start", "release", "attach"])
async def test_environment_methods_require_typed_handles_before_native_dispatch(
    method, monkeypatch
):
    fabric = Fabric()
    native = MagicMock()
    monkeypatch.setattr(fabric, "_require_native_module", native)
    with pytest.raises(FabricConfigError):
        if method == "start":
            await fabric.start_runtime_in(hermes_shim_config(), {})
        elif method == "release":
            await fabric.release_environment({})
        else:
            await fabric.attach_environment(hermes_shim_config(), {})
    native.assert_not_called()


@pytest.fixture(name="remote_handle")
def remote_handle_fixture():
    return {
        "environment_id": "environment-1",
        "provider": "openshell",
        "control_location": "in_env_control",
        "ownership": "caller_owned",
        "workspace": "/sandbox/workspace",
        "artifacts": "/sandbox/artifacts",
        "connection": {"sandbox_id": "immutable-1"},
    }


def test_remote_paths_do_not_use_host_path_conversion(monkeypatch, remote_handle):
    import nemo_fabric.types as types

    monkeypatch.setattr(types, "Path", WindowsPath)
    handle = EnvironmentHandle.from_mapping(remote_handle)
    assert handle.workspace == "/sandbox/workspace"
    assert handle.artifacts == "/sandbox/artifacts"
    assert handle.to_mapping()["connection"]["sandbox_id"] == "immutable-1"


@pytest.mark.parametrize(
    "changes",
    [
        {"ownership": "unknown"},
        {"control_location": "host"},
        {"environment_id": ""},
        {"workspace": []},
        {"workspace": "relative"},
        {"artifacts": "/sandbox/../host"},
        {"extra": True},
        {"env": {"TOKEN": 3}},
    ],
)
def test_environment_handle_rejects_invalid_binding(changes, remote_handle):
    with pytest.raises(FabricConfigError):
        EnvironmentHandle.from_mapping({**remote_handle, **changes})


def test_environment_reference_is_an_immutable_snapshot():
    original = {"provider": "openshell", "resource": {"sandbox_name": "review"}}
    reference = EnvironmentReference.from_mapping(original)
    original["resource"]["sandbox_name"] = "changed"
    assert reference.resource["sandbox_name"] == "review"
    with pytest.raises(FabricConfigError):
        EnvironmentReference.from_mapping({**original, "unexpected": True})


@pytest.mark.parametrize(
    "path",
    [
        "relative",
        "C:/workspace",
        "/sandbox/../host",
        "/sandbox\\workspace",
        "/sandbox\0workspace",
        Path("/sandbox/workspace"),
    ],
)
def test_openshell_config_rejects_host_or_traversal_paths(path):
    from pydantic import ValidationError
    from nemo_fabric import EnvironmentConfig

    with pytest.raises(ValidationError, match="absolute POSIX task path"):
        EnvironmentConfig(provider="openshell", workspace=path)

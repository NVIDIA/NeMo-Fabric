# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import HTTPError, URLError

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "bootstrap", ROOT / "scripts/ci/check_npm_bootstrap.py"
)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def manifest(root, path, **values):
    directory = root / path
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "package.json").write_text(json.dumps(values))
    return directory


def test_discovery_future_scoped_package_private_and_dependency_order(tmp_path):
    manifest(tmp_path, "adapter-contract/typescript", name="contract")
    manifest(tmp_path, "adapters/typescript", private=True, workspaces=["*"])
    future = manifest(
        tmp_path,
        "adapters/typescript/a-future",
        name="@fabric/future",
        dependencies={"common": "1"},
    )
    common = manifest(
        tmp_path,
        "adapters/typescript/z-common",
        name="common",
        dependencies={"contract": "1"},
    )
    manifest(tmp_path, "adapters/typescript/internal", name="internal", private=True)
    assert bootstrap.package_directories(tmp_path) == [
        tmp_path / "adapter-contract/typescript",
        common,
        future,
    ]


def test_discovery_cycle_fails(tmp_path):
    manifest(
        tmp_path,
        "adapter-contract/typescript",
        name="contract",
        dependencies={"adapter": "1"},
    )
    manifest(tmp_path, "adapters/typescript", workspaces=["adapter"])
    manifest(
        tmp_path,
        "adapters/typescript/adapter",
        name="adapter",
        dependencies={"contract": "1"},
    )
    with pytest.raises(ValueError, match="cycle"):
        bootstrap.package_directories(tmp_path)


@pytest.mark.parametrize(
    "dependency_field", ["peerDependencies", "optionalDependencies"]
)
def test_discovery_orders_internal_peer_and_optional_dependencies(
    tmp_path, dependency_field
):
    manifest(tmp_path, "adapter-contract/typescript", name="contract")
    manifest(tmp_path, "adapters/typescript", workspaces=["*"])
    consumer = manifest(
        tmp_path,
        "adapters/typescript/a-consumer",
        name="consumer",
        **{dependency_field: {"provider": "1", "external": "1"}},
    )
    provider = manifest(tmp_path, "adapters/typescript/z-provider", name="provider")
    assert bootstrap.package_directories(tmp_path) == [
        tmp_path / "adapter-contract/typescript",
        provider,
        consumer,
    ]


def test_discovery_unmatched_workspace_fails(tmp_path):
    manifest(tmp_path, "adapter-contract/typescript", name="contract")
    manifest(tmp_path, "adapters/typescript", workspaces=["missing-*"])
    with pytest.raises(ValueError, match="Workspace pattern has no matches: missing-"):
        bootstrap.package_directories(tmp_path)


def test_discovery_duplicate_public_name_fails(tmp_path):
    manifest(tmp_path, "adapter-contract/typescript", name="contract")
    manifest(tmp_path, "adapters/typescript", workspaces=["*"])
    manifest(tmp_path, "adapters/typescript/first", name="duplicate")
    manifest(tmp_path, "adapters/typescript/second", name="duplicate")
    with pytest.raises(ValueError, match="Duplicate public package name: duplicate"):
        bootstrap.package_directories(tmp_path)


@pytest.mark.parametrize("code", [401, 403, 429, 500, 503])
def test_http_failure_is_not_missing_package(tmp_path, code):
    directory = manifest(tmp_path, "pkg", name="pkg")
    with pytest.raises(
        ValueError, match=f"HTTP {code}; existence could not be verified"
    ) as caught:
        bootstrap.check_packages(
            [directory],
            open_url=MagicMock(side_effect=HTTPError("url", code, "error", {}, None)),
        )
    assert "bootstrap this package" not in str(caught.value)


@pytest.mark.parametrize("error", [URLError("offline"), TimeoutError("timeout")])
def test_transport_failure_is_not_missing_package(tmp_path, error):
    directory = manifest(tmp_path, "pkg", name="pkg")
    with pytest.raises(ValueError, match="existence could not be verified"):
        bootstrap.check_packages([directory], open_url=MagicMock(side_effect=error))


def test_missing_package_checks_every_package_and_uses_bounded_package_url(tmp_path):
    first = manifest(tmp_path, "first", name="@fabric/new")
    second = manifest(tmp_path, "second", name="existing", version="99.0.0")
    response = MagicMock()
    response.__enter__.return_value.status = 200
    opener = MagicMock(
        side_effect=[HTTPError("url", 404, "missing", {}, None), response]
    )
    with pytest.raises(ValueError, match="@fabric/new: package metadata HTTP 404"):
        bootstrap.check_packages([first, second], open_url=opener)
    assert opener.call_count == 2
    assert (
        opener.call_args_list[0].args[0].full_url
        == "https://registry.npmjs.org/%40fabric%2Fnew"
    )
    assert (
        opener.call_args_list[1].args[0].full_url
        == "https://registry.npmjs.org/existing"
    )
    assert all(call.kwargs == {"timeout": 15} for call in opener.call_args_list)


def test_current_repository_discovery():
    directories = bootstrap.package_directories(ROOT)
    public = [ROOT / "adapter-contract/typescript"] + [
        p.parent
        for p in (ROOT / "adapters/typescript").glob("*/package.json")
        if not json.loads(p.read_text()).get("private")
    ]
    assert set(directories) == set(public)
    assert directories[0] == ROOT / "adapter-contract/typescript"

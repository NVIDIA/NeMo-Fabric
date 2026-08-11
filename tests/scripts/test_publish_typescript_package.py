# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest


CI_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ci"
sys.path.insert(0, str(CI_SCRIPTS))

import publish_typescript_package  # noqa: E402


PACKAGE = "@nvidia/nemo-fabric-adapter-contract"
VERSION = "0.2.0"
INTEGRITY = "sha512-expected"
TARBALL = "nvidia-nemo-fabric-adapter-contract-0.2.0.tgz"


def _result(
    stdout: str = "",
    *,
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["npm"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _pack_result() -> subprocess.CompletedProcess[str]:
    return _result(
        json.dumps(
            [
                {
                    "name": PACKAGE,
                    "version": VERSION,
                    "integrity": INTEGRITY,
                    "filename": TARBALL,
                }
            ]
        )
    )


class NpmRunner:
    def __init__(
        self,
        responses: Sequence[
            tuple[Sequence[str], subprocess.CompletedProcess[str]]
        ],
    ) -> None:
        self.responses = list(responses)

    def __call__(
        self,
        arguments: Sequence[str],
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd.is_dir()
        assert self.responses, f"Unexpected npm call: {arguments}"
        expected, result = self.responses.pop(0)
        assert list(arguments) == list(expected)
        return result

    def assert_finished(self) -> None:
        assert self.responses == []


def _package_directory(tmp_path: Path) -> Path:
    package_directory = tmp_path / "package"
    package_directory.mkdir()
    (package_directory / TARBALL).write_bytes(b"package")
    return package_directory


def _exact_view_responses(
    *,
    integrity: str = INTEGRITY,
    dist_tag_version: str = VERSION,
) -> list[tuple[list[str], subprocess.CompletedProcess[str]]]:
    package_version = f"{PACKAGE}@{VERSION}"
    return [
        (["view", package_version, "version"], _result(VERSION)),
        (["view", package_version, "dist.integrity"], _result(integrity)),
        (["view", PACKAGE, "dist-tags.latest"], _result(dist_tag_version)),
    ]


def test_existing_exact_package_is_an_idempotent_success(tmp_path: Path) -> None:
    package_directory = _package_directory(tmp_path)
    runner = NpmRunner(
        [(["pack", "--json"], _pack_result()), *_exact_view_responses()]
    )

    publish_typescript_package.publish_package(
        package_directory,
        VERSION,
        "latest",
        run_npm=runner,
        sleep=lambda _: None,
    )

    runner.assert_finished()


@pytest.mark.parametrize(
    ("integrity", "dist_tag_version"),
    [("sha512-wrong", VERSION), (INTEGRITY, "0.1.0")],
)
def test_existing_conflicting_package_fails(
    tmp_path: Path,
    integrity: str,
    dist_tag_version: str,
) -> None:
    package_directory = _package_directory(tmp_path)
    runner = NpmRunner(
        [
            (["pack", "--json"], _pack_result()),
            *_exact_view_responses(
                integrity=integrity,
                dist_tag_version=dist_tag_version,
            ),
        ]
    )

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match="artifact or dist-tag does not match",
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "latest",
            run_npm=runner,
            sleep=lambda _: None,
        )

    runner.assert_finished()


def test_absent_package_publishes_and_verifies(tmp_path: Path) -> None:
    package_directory = _package_directory(tmp_path)
    missing = _result(returncode=1, stderr="npm error code E404")
    runner = NpmRunner(
        [
            (["pack", "--json"], _pack_result()),
            (["view", f"{PACKAGE}@{VERSION}", "version"], missing),
            (["view", PACKAGE, "dist-tags.latest"], _result("0.1.0")),
            (
                ["publish", f"./{TARBALL}", "--access", "public", "--tag", "latest"],
                _result("published"),
            ),
            *_exact_view_responses(),
        ]
    )

    publish_typescript_package.publish_package(
        package_directory,
        VERSION,
        "latest",
        run_npm=runner,
        sleep=lambda _: None,
    )

    runner.assert_finished()


def test_non_404_lookup_failure_fails_closed(tmp_path: Path) -> None:
    package_directory = _package_directory(tmp_path)
    runner = NpmRunner(
        [
            (["pack", "--json"], _pack_result()),
            (
                ["view", f"{PACKAGE}@{VERSION}", "version"],
                _result(returncode=1, stderr="npm error code E503"),
            ),
        ]
    )

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match="E503",
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "latest",
            run_npm=runner,
            sleep=lambda _: None,
        )

    runner.assert_finished()


def test_dist_tag_cannot_move_backward(tmp_path: Path) -> None:
    package_directory = _package_directory(tmp_path)
    missing = _result(returncode=1, stderr="npm error code E404")
    runner = NpmRunner(
        [
            (["pack", "--json"], _pack_result()),
            (["view", f"{PACKAGE}@{VERSION}", "version"], missing),
            (["view", PACKAGE, "dist-tags.latest"], _result("0.3.0")),
        ]
    )

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match="Refusing to move latest backward",
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "latest",
            run_npm=runner,
            sleep=lambda _: None,
        )

    runner.assert_finished()


def test_ambiguous_publish_failure_reconciles_registry_state(tmp_path: Path) -> None:
    package_directory = _package_directory(tmp_path)
    missing = _result(returncode=1, stderr="npm error code E404")
    runner = NpmRunner(
        [
            (["pack", "--json"], _pack_result()),
            (["view", f"{PACKAGE}@{VERSION}", "version"], missing),
            (["view", PACKAGE, "dist-tags.latest"], _result("0.1.0")),
            (
                ["publish", f"./{TARBALL}", "--access", "public", "--tag", "latest"],
                _result(returncode=1, stderr="network connection closed"),
            ),
            *_exact_view_responses(),
        ]
    )

    publish_typescript_package.publish_package(
        package_directory,
        VERSION,
        "latest",
        run_npm=runner,
        sleep=lambda _: None,
    )

    runner.assert_finished()


@pytest.mark.parametrize(
    ("candidate", "current", "is_newer"),
    [
        ("0.2.1", "0.2.0", True),
        ("0.2.0", "0.2.0-rc.1", True),
        ("0.2.0-rc.2", "0.2.0-rc.1", True),
        ("0.2.0-beta.2", "0.2.0-rc.1", False),
        ("0.2.0-rc", "0.2.0-rc.0", False),
        ("0.2.1", "0.3.0", False),
    ],
)
def test_version_order(candidate: str, current: str, is_newer: bool) -> None:
    assert (
        publish_typescript_package._version_key(candidate)
        > publish_typescript_package._version_key(current)
    ) is is_newer

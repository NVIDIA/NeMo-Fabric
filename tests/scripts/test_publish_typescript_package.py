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


PACKAGE = "nemo-fabric-adapter-contract"
VERSION = "0.2.0"
INTEGRITY = "sha512-expected"
TARBALL = "nvidia-nemo-fabric-adapter-contract-0.2.0.tgz"
README = "# NVIDIA NeMo Fabric Adapter Contract for TypeScript"


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


def _pack_result(
    *,
    version: str = VERSION,
    filename: str = TARBALL,
    files: Sequence[str] = ("README.md",),
) -> subprocess.CompletedProcess[str]:
    return _result(
        json.dumps(
            [
                {
                    "name": PACKAGE,
                    "version": version,
                    "integrity": INTEGRITY,
                    "filename": filename,
                    "files": [{"path": path} for path in files],
                }
            ]
        )
    )


class NpmRunner:
    def __init__(
        self,
        responses: Sequence[tuple[Sequence[str], subprocess.CompletedProcess[str]]],
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


@pytest.fixture(name="package_directory")
def package_directory_fixture(tmp_path: Path) -> Path:
    package_directory = tmp_path / "package"
    package_directory.mkdir()
    (package_directory / TARBALL).write_bytes(b"package")
    (package_directory / "README.md").write_text(f"{README}\n", encoding="utf-8")
    (package_directory / "package.json").write_text(
        json.dumps(
            {
                "name": PACKAGE,
                "version": VERSION,
                "dependencies": {},
            }
        ),
        encoding="utf-8",
    )
    return package_directory


def test_runtime_dependencies_are_preflighted_against_npm(
    package_directory: Path,
):
    dependency = "nemo-fabric-adapters-common"
    dependency_version = "0.3.0"
    manifest = json.loads(
        (package_directory / "package.json").read_text(encoding="utf-8")
    )
    manifest["dependencies"] = {dependency: dependency_version}
    (package_directory / "package.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    runner = NpmRunner(
        [
            (["pack", "--json", "--ignore-scripts"], _pack_result()),
            (
                ["view", f"{dependency}@{dependency_version}", "version"],
                _result(dependency_version),
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


def test_unpublished_runtime_dependency_blocks_publication(
    package_directory: Path,
):
    dependency = "nemo-fabric-adapters-common"
    dependency_version = "0.3.0"
    manifest = json.loads(
        (package_directory / "package.json").read_text(encoding="utf-8")
    )
    manifest["dependencies"] = {dependency: dependency_version}
    (package_directory / "package.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    missing = _result(returncode=1, stderr="npm error code E404")
    runner = NpmRunner(
        [
            (["pack", "--json", "--ignore-scripts"], _pack_result()),
            (["view", f"{dependency}@{dependency_version}", "version"], missing),
        ]
    )

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match=(
            "Required runtime dependency "
            f"{dependency}@{dependency_version} is not published"
        ),
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "latest",
            run_npm=runner,
            sleep=lambda _: None,
        )

    runner.assert_finished()


def test_runtime_dependency_versions_must_be_exact(package_directory: Path):
    manifest = json.loads(
        (package_directory / "package.json").read_text(encoding="utf-8")
    )
    manifest["dependencies"] = {"nemo-fabric-adapters-common": "^0.3.0"}
    (package_directory / "package.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    runner = NpmRunner([(["pack", "--json", "--ignore-scripts"], _pack_result())])

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match=r"must use an exact npm version, got \^0\.3\.0",
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "latest",
            run_npm=runner,
            sleep=lambda _: None,
        )

    runner.assert_finished()


def _exact_view_responses(
    *,
    integrity: str = INTEGRITY,
    dist_tag: str = "latest",
    dist_tag_version: str = VERSION,
) -> list[tuple[list[str], subprocess.CompletedProcess[str]]]:
    package_version = f"{PACKAGE}@{VERSION}"
    return [
        (["view", package_version, "version"], _result(VERSION)),
        (["view", package_version, "dist.integrity"], _result(integrity)),
        (["view", PACKAGE, f"dist-tags.{dist_tag}"], _result(dist_tag_version)),
    ]


def test_invalid_readme_encoding_fails_closed(package_directory: Path):
    (package_directory / "README.md").write_bytes(b"invalid utf-8: \xff")
    runner = NpmRunner([])

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match=r"Package README\.md could not be read",
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "latest",
            run_npm=runner,
            sleep=lambda _: None,
        )

    runner.assert_finished()


@pytest.mark.parametrize("dist_tag", ["alpha", "latest", "next"])
def test_existing_exact_package_without_registry_readme_is_idempotent_success(
    package_directory: Path,
    dist_tag: str,
):
    runner = NpmRunner(
        [
            (["pack", "--json", "--ignore-scripts"], _pack_result()),
            *_exact_view_responses(dist_tag=dist_tag),
        ]
    )

    publish_typescript_package.publish_package(
        package_directory,
        VERSION,
        dist_tag,
        run_npm=runner,
        sleep=lambda _: None,
    )

    runner.assert_finished()


@pytest.mark.parametrize(
    ("dist_tag", "integrity", "dist_tag_version", "error"),
    [
        (dist_tag, integrity, dist_tag_version, error)
        for dist_tag in ("alpha", "latest", "next")
        for integrity, dist_tag_version, error in (
            ("", VERSION, "Published integrity is missing"),
            ("sha512-wrong", VERSION, "Expected integrity"),
            (INTEGRITY, "0.1.0", f"Published {dist_tag} dist-tag"),
        )
    ],
)
def test_existing_conflicting_package_fails(
    package_directory: Path,
    dist_tag: str,
    integrity: str,
    dist_tag_version: str,
    error: str,
):
    runner = NpmRunner(
        [
            (["pack", "--json", "--ignore-scripts"], _pack_result()),
            *_exact_view_responses(
                integrity=integrity,
                dist_tag=dist_tag,
                dist_tag_version=dist_tag_version,
            ),
        ]
    )

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match=error,
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            dist_tag,
            run_npm=runner,
            sleep=lambda _: None,
        )

    runner.assert_finished()


def test_packed_artifact_must_include_readme(package_directory: Path):
    runner = NpmRunner(
        [
            (
                ["pack", "--json", "--ignore-scripts"],
                _pack_result(files=()),
            ),
        ]
    )

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match=r"Packed artifact is missing README\.md",
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "latest",
            run_npm=runner,
            sleep=lambda _: None,
        )

    runner.assert_finished()


@pytest.mark.parametrize("dist_tag", ["alpha", "latest", "next"])
def test_absent_package_publishes_directory_and_verifies(
    package_directory: Path,
    dist_tag: str,
):
    missing = _result(returncode=1, stderr="npm error code E404")
    runner = NpmRunner(
        [
            (["pack", "--json", "--ignore-scripts"], _pack_result()),
            (["view", f"{PACKAGE}@{VERSION}", "version"], missing),
            (["view", PACKAGE, f"dist-tags.{dist_tag}"], _result("0.1.0")),
            (
                [
                    "publish",
                    ".",
                    "--ignore-scripts",
                    "--access",
                    "public",
                    "--tag",
                    dist_tag,
                ],
                _result("published"),
            ),
            *_exact_view_responses(dist_tag=dist_tag),
        ]
    )

    publish_typescript_package.publish_package(
        package_directory,
        VERSION,
        dist_tag,
        run_npm=runner,
        sleep=lambda _: None,
    )

    runner.assert_finished()


def test_non_404_lookup_failure_fails_closed(package_directory: Path):
    runner = NpmRunner(
        [
            (["pack", "--json", "--ignore-scripts"], _pack_result()),
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


@pytest.mark.parametrize("dist_tag", ["alpha", "latest", "next"])
def test_dist_tag_cannot_move_backward(
    package_directory: Path,
    dist_tag: str,
):
    missing = _result(returncode=1, stderr="npm error code E404")
    runner = NpmRunner(
        [
            (["pack", "--json", "--ignore-scripts"], _pack_result()),
            (["view", f"{PACKAGE}@{VERSION}", "version"], missing),
            (["view", PACKAGE, f"dist-tags.{dist_tag}"], _result("0.3.0")),
        ]
    )

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match=f"Refusing to move {dist_tag} backward",
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            dist_tag,
            run_npm=runner,
            sleep=lambda _: None,
        )

    runner.assert_finished()


def test_ambiguous_publish_failure_reconciles_registry_state(package_directory: Path):
    missing = _result(returncode=1, stderr="npm error code E404")
    runner = NpmRunner(
        [
            (["pack", "--json", "--ignore-scripts"], _pack_result()),
            (["view", f"{PACKAGE}@{VERSION}", "version"], missing),
            (["view", PACKAGE, "dist-tags.latest"], _result("0.1.0")),
            (
                [
                    "publish",
                    ".",
                    "--ignore-scripts",
                    "--access",
                    "public",
                    "--tag",
                    "latest",
                ],
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


def test_invalid_dist_tag_fails_before_packing(package_directory: Path):
    runner = NpmRunner([])

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match="Unsupported npm dist-tag: beta",
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "beta",
            run_npm=runner,
        )

    runner.assert_finished()


@pytest.mark.parametrize("verification_attempts", [0, -1])
def test_invalid_verification_attempts_fail_before_packing(
    package_directory: Path,
    verification_attempts: int,
):
    runner = NpmRunner([])

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match="At least one registry verification attempt is required",
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "latest",
            run_npm=runner,
            verification_attempts=verification_attempts,
        )

    runner.assert_finished()


def test_packed_version_must_match_release(package_directory: Path):
    runner = NpmRunner(
        [(["pack", "--json", "--ignore-scripts"], _pack_result(version="0.2.1"))]
    )

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match=r"Packed version 0\.2\.1 does not match release 0\.2\.0",
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "latest",
            run_npm=runner,
        )

    runner.assert_finished()


def test_packed_filename_must_be_safe(package_directory: Path):
    runner = NpmRunner(
        [
            (
                ["pack", "--json", "--ignore-scripts"],
                _pack_result(filename="../package.tgz"),
            )
        ]
    )

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match=r"npm pack returned an unsafe filename: ../package\.tgz",
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "latest",
            run_npm=runner,
        )

    runner.assert_finished()


def test_packed_tarball_must_exist(package_directory: Path):
    runner = NpmRunner(
        [(["pack", "--json", "--ignore-scripts"], _pack_result(filename="missing.tgz"))]
    )

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match=r"npm pack did not create missing\.tgz",
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "latest",
            run_npm=runner,
        )

    runner.assert_finished()


def test_visible_post_publish_conflict_fails_without_retry(package_directory: Path):
    missing = _result(returncode=1, stderr="npm error code E404")
    runner = NpmRunner(
        [
            (["pack", "--json", "--ignore-scripts"], _pack_result()),
            (["view", f"{PACKAGE}@{VERSION}", "version"], missing),
            (["view", PACKAGE, "dist-tags.latest"], _result("0.1.0")),
            (
                [
                    "publish",
                    ".",
                    "--ignore-scripts",
                    "--access",
                    "public",
                    "--tag",
                    "latest",
                ],
                _result(returncode=1, stderr="network connection closed"),
            ),
            *_exact_view_responses(integrity="sha512-conflict"),
        ]
    )
    delays: list[float] = []

    with pytest.raises(
        publish_typescript_package.PublicationError,
        match="Expected integrity",
    ):
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "latest",
            run_npm=runner,
            sleep=delays.append,
        )

    assert delays == []
    runner.assert_finished()


def test_failed_publish_exhaustion_reports_both_failures(package_directory: Path):
    missing = _result(returncode=1, stderr="npm error code E404")
    runner = NpmRunner(
        [
            (["pack", "--json", "--ignore-scripts"], _pack_result()),
            (["view", f"{PACKAGE}@{VERSION}", "version"], missing),
            (["view", PACKAGE, "dist-tags.latest"], _result("0.1.0")),
            (
                [
                    "publish",
                    ".",
                    "--ignore-scripts",
                    "--access",
                    "public",
                    "--tag",
                    "latest",
                ],
                _result(returncode=1, stderr="network connection closed"),
            ),
            (["view", f"{PACKAGE}@{VERSION}", "version"], missing),
            (["view", f"{PACKAGE}@{VERSION}", "version"], missing),
            (["view", f"{PACKAGE}@{VERSION}", "version"], missing),
        ]
    )
    delays: list[float] = []

    with pytest.raises(
        publish_typescript_package.PublicationError,
    ) as error:
        publish_typescript_package.publish_package(
            package_directory,
            VERSION,
            "latest",
            run_npm=runner,
            sleep=delays.append,
            verification_attempts=3,
        )

    message = str(error.value)
    assert "npm publish failed: network connection closed" in message
    assert "registry verification also failed" in message
    assert "package version is not visible in npm" in message
    assert delays == [5, 10]
    runner.assert_finished()


@pytest.mark.parametrize(
    "case",
    [
        ("0.2.1", "0.2.0", True),
        ("0.2.0", "0.2.0-rc.1", True),
        ("0.2.0-rc.2", "0.2.0-rc.1", True),
        ("0.2.0-beta.2", "0.2.0-rc.1", False),
        ("0.2.0-rc", "0.2.0-rc.0", False),
        ("0.2.1", "0.3.0", False),
    ],
)
def test_version_order(case: tuple[str, str, bool]):
    candidate, current, is_newer = case
    assert (
        publish_typescript_package._version_key(candidate)
        > publish_typescript_package._version_key(current)
    ) is is_newer

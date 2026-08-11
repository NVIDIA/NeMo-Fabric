# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


VERSION_PATTERN = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:-(?P<label>alpha|beta|rc)(?:\.(?P<number>\d+))?)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
PRERELEASE_ORDER = {"alpha": 0, "beta": 1, "rc": 2}
NPM_NOT_FOUND_MARKERS = ("E404", "404 Not Found")


class PublicationError(RuntimeError):
    """The npm registry state is unsafe or publication failed."""


@dataclass(frozen=True)
class PackageArtifact:
    name: str
    version: str
    integrity: str
    filename: str


@dataclass(frozen=True)
class PublishedState:
    version: str
    integrity: str
    dist_tag_version: str


RunNpm = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]
Sleep = Callable[[float], None]


def _run_npm(arguments: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["npm", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _command_error(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )


def _is_not_found(result: subprocess.CompletedProcess[str]) -> bool:
    output = _command_error(result)
    return any(marker in output for marker in NPM_NOT_FOUND_MARKERS)


def _require_output(
    result: subprocess.CompletedProcess[str],
    *,
    action: str,
) -> str:
    if result.returncode != 0:
        detail = _command_error(result)
        raise PublicationError(f"{action} failed{f': {detail}' if detail else ''}")
    return result.stdout.strip()


def _version_key(version: str) -> tuple[int, int, int, int, int, int]:
    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise PublicationError(f"Unsupported published npm version: {version}")

    release = tuple(int(match.group(name)) for name in ("major", "minor", "patch"))
    label = match.group("label")
    if label is None:
        prerelease = (len(PRERELEASE_ORDER), 0, 0)
    else:
        number = match.group("number")
        prerelease = (
            PRERELEASE_ORDER[label],
            0 if number is None else 1,
            int(number or 0),
        )
    return (*release, *prerelease)


def _pack_package(
    package_directory: Path,
    expected_version: str,
    run_npm: RunNpm,
) -> PackageArtifact:
    result = run_npm(["pack", "--json"], package_directory)
    output = _require_output(result, action="npm pack")
    try:
        values = json.loads(output)
        value = values[0] if len(values) == 1 else None
        artifact = PackageArtifact(
            name=value["name"],
            version=value["version"],
            integrity=value["integrity"],
            filename=value["filename"],
        )
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise PublicationError("npm pack returned an unexpected result") from error

    if not all((artifact.name, artifact.version, artifact.integrity, artifact.filename)):
        raise PublicationError("npm pack returned empty package metadata")
    if artifact.version != expected_version:
        raise PublicationError(
            f"Packed version {artifact.version} does not match release {expected_version}"
        )
    if Path(artifact.filename).name != artifact.filename:
        raise PublicationError(f"npm pack returned an unsafe filename: {artifact.filename}")
    if not (package_directory / artifact.filename).is_file():
        raise PublicationError(f"npm pack did not create {artifact.filename}")
    return artifact


def _view(
    package_directory: Path,
    run_npm: RunNpm,
    *arguments: str,
) -> str | None:
    result = run_npm(["view", *arguments], package_directory)
    if result.returncode == 0:
        return result.stdout.strip()
    if _is_not_found(result):
        return None
    detail = _command_error(result)
    raise PublicationError(
        f"npm view {' '.join(arguments)} failed{f': {detail}' if detail else ''}"
    )


def _published_state(
    package_directory: Path,
    artifact: PackageArtifact,
    dist_tag: str,
    run_npm: RunNpm,
) -> PublishedState | None:
    package_version = f"{artifact.name}@{artifact.version}"
    version = _view(package_directory, run_npm, package_version, "version")
    if version is None:
        return None
    integrity = _view(package_directory, run_npm, package_version, "dist.integrity")
    dist_tag_version = _view(
        package_directory,
        run_npm,
        artifact.name,
        f"dist-tags.{dist_tag}",
    )
    return PublishedState(
        version=version,
        integrity=integrity or "",
        dist_tag_version=dist_tag_version or "",
    )


def _state_matches(
    state: PublishedState,
    artifact: PackageArtifact,
) -> bool:
    return (
        state.version == artifact.version
        and state.integrity == artifact.integrity
        and state.dist_tag_version == artifact.version
    )


def _describe_conflict(
    state: PublishedState,
    artifact: PackageArtifact,
    dist_tag: str,
) -> str:
    return "\n".join(
        (
            f"{artifact.name}@{artifact.version} already exists, but its artifact or dist-tag does not match",
            f"Expected integrity: {artifact.integrity}",
            f"Published integrity: {state.integrity or '<unset>'}",
            f"Expected {dist_tag} dist-tag: {artifact.version}",
            f"Published {dist_tag} dist-tag: {state.dist_tag_version or '<unset>'}",
        )
    )


def publish_package(
    package_directory: Path,
    version: str,
    dist_tag: str,
    *,
    run_npm: RunNpm = _run_npm,
    sleep: Sleep = time.sleep,
    verification_attempts: int = 6,
) -> None:
    if dist_tag not in {"latest", "next"}:
        raise PublicationError(f"Unsupported npm dist-tag: {dist_tag}")
    if verification_attempts < 1:
        raise PublicationError("At least one registry verification attempt is required")

    artifact = _pack_package(package_directory, version, run_npm)
    state = _published_state(package_directory, artifact, dist_tag, run_npm)
    if state is not None:
        if _state_matches(state, artifact):
            print(
                f"{artifact.name}@{artifact.version} already has the expected "
                f"artifact and {dist_tag} dist-tag; skipping"
            )
            return
        raise PublicationError(_describe_conflict(state, artifact, dist_tag))

    current_dist_tag = _view(
        package_directory,
        run_npm,
        artifact.name,
        f"dist-tags.{dist_tag}",
    )
    if current_dist_tag and _version_key(version) <= _version_key(current_dist_tag):
        raise PublicationError(
            f"Refusing to move {dist_tag} backward from {current_dist_tag} to {version}"
        )

    publish_result = run_npm(
        ["publish", f"./{artifact.filename}", "--access", "public", "--tag", dist_tag],
        package_directory,
    )
    if publish_result.stdout:
        print(publish_result.stdout.rstrip())
    if publish_result.stderr:
        print(publish_result.stderr.rstrip(), file=sys.stderr)

    last_error: PublicationError | None = None
    for attempt in range(verification_attempts):
        try:
            state = _published_state(package_directory, artifact, dist_tag, run_npm)
            if state is not None and _state_matches(state, artifact):
                print(
                    f"Verified {artifact.name}@{artifact.version} with "
                    f"{dist_tag} dist-tag"
                )
                return
            last_error = (
                PublicationError(_describe_conflict(state, artifact, dist_tag))
                if state is not None
                else PublicationError("The package version is not visible in npm")
            )
        except PublicationError as error:
            last_error = error
        if attempt + 1 < verification_attempts:
            sleep(5)

    publish_detail = _command_error(publish_result)
    if publish_result.returncode != 0:
        raise PublicationError(
            f"npm publish failed{f': {publish_detail}' if publish_detail else ''}; "
            f"registry verification also failed: {last_error}"
        )
    raise PublicationError(f"npm publish completed, but verification failed: {last_error}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish and reconcile the TypeScript adapter-contract package"
    )
    parser.add_argument("--package-directory", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist-tag", choices=("latest", "next"), required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    publish_package(
        arguments.package_directory,
        arguments.version,
        arguments.dist_tag,
    )


if __name__ == "__main__":
    try:
        main()
    except PublicationError as error:
        raise SystemExit(str(error)) from error

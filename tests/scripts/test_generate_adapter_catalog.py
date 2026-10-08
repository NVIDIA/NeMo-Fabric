# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ci"))
from generate_adapter_catalog import BUNDLE, CATALOG, catalog_data, generate_catalog  # noqa: E402


@pytest.fixture(name="catalog_repo")
def catalog_repo_fixture(tmp_path: Path) -> Path:
    project = tmp_path / CATALOG / "pyproject.toml"
    project.parent.mkdir(parents=True)
    project.write_text('[project]\nversion = "0.5.0"\n', encoding="utf-8")
    return tmp_path


@pytest.fixture(name="adapter_source")
def adapter_source_fixture(catalog_repo: Path) -> Path:
    directory = catalog_repo / "adapters/python/example"
    directory.mkdir(parents=True)
    (directory / "pyproject.toml").write_text(
        '[project]\nname = "example-adapter"\nversion = "1.2.3"\n', encoding="utf-8"
    )
    source = directory / "example.fabric-adapter.json"
    source.write_text(
        json.dumps(
            {"adapter_id": "example.adapter", "runner": {"module": "not_installed"}}
        ),
        encoding="utf-8",
    )
    return source


def test_bundle_preserves_descriptors_and_separates_source_metadata(
    catalog_repo: Path, adapter_source: Path
):
    targets = adapter_source.parent / "targets"
    targets.mkdir()
    target = targets / "example.fabric-target.json"
    target.write_text(
        json.dumps({"id": "example.target", "adapter_id": "example.adapter"}),
        encoding="utf-8",
    )
    data = catalog_data(catalog_repo)
    assert data["catalog_version"] == "0.5.0"
    assert len(data["adapters"]) == len(data["targets"]) == 1
    for kind in ("adapters", "targets"):
        for record in data[kind].values():
            original = json.loads((catalog_repo / record["source"]).read_bytes())
            assert record["descriptor"] == original
            canonical = json.dumps(
                original, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
            assert record["sha256"] == hashlib.sha256(canonical).hexdigest()
            assert record["package"] == {"name": "example-adapter", "version": "1.2.3"}


def test_check_accepts_windows_checkout_line_endings(
    catalog_repo: Path, adapter_source: Path
):
    generate_catalog(catalog_repo)
    output = catalog_repo / BUNDLE
    output.write_bytes(output.read_bytes().replace(b"\n", b"\r\n"))
    generate_catalog(catalog_repo, check=True)


def test_typescript_records_are_discovered_without_sdk_or_python_manifest(
    catalog_repo: Path, adapter_source: Path
):
    directory = catalog_repo / "adapters/typescript/example"
    directory.mkdir(parents=True)
    (directory / "package.json").write_text(
        json.dumps({"name": "example-ts", "version": "1.2.3-rc.1"}), encoding="utf-8"
    )
    source = directory / "example.fabric-adapter.json"
    source.write_text(
        json.dumps(
            {
                "adapter_id": "example.typescript",
                "runner": {"command": "node", "script": "dist/cli.js"},
            }
        ),
        encoding="utf-8",
    )
    ignored = directory / "tests/fixture.fabric-adapter.json"
    ignored.parent.mkdir()
    ignored.write_text("{}", encoding="utf-8")
    data = catalog_data(catalog_repo)
    assert len(data["adapters"]) == 2
    record = data["adapters"]["example.typescript"]
    assert record["ecosystem"] == "typescript"
    assert record["package"]["version"] == "1.2.3-rc.1"
    assert record["descriptor"] == json.loads(source.read_bytes())


def test_refresh_is_reproducible_and_drops_removed_records(
    catalog_repo: Path, adapter_source: Path
):
    generate_catalog(catalog_repo)
    original = (catalog_repo / BUNDLE).read_bytes()
    generate_catalog(catalog_repo)
    assert (catalog_repo / BUNDLE).read_bytes() == original
    generate_catalog(catalog_repo, check=True)
    second = adapter_source.parent / "second.fabric-adapter.json"
    second.write_text(json.dumps({"adapter_id": "example.second"}), encoding="utf-8")
    generate_catalog(catalog_repo)
    second.unlink()
    with pytest.raises(ValueError, match="catalog is stale"):
        generate_catalog(catalog_repo, check=True)
    generate_catalog(catalog_repo)
    assert (catalog_repo / BUNDLE).read_bytes() == original


@pytest.mark.parametrize(
    "change", ["descriptor", "package_version", "catalog_version", "missing_bundle"]
)
def test_check_rejects_stale_bundle(
    catalog_repo: Path, adapter_source: Path, change: str
):
    generate_catalog(catalog_repo)
    if change == "missing_bundle":
        (catalog_repo / BUNDLE).unlink()
    else:
        source = {
            "descriptor": adapter_source,
            "package_version": adapter_source.parent / "pyproject.toml",
            "catalog_version": catalog_repo / CATALOG / "pyproject.toml",
        }[change]
        original = source.read_text(encoding="utf-8")
        updated = (
            original.replace("not_installed", "changed_runner")
            if change == "descriptor"
            else original.replace('"1.2.3"', '"2.0.0"').replace('"0.5.0"', '"2.0.0"')
        )
        source.write_text(updated, encoding="utf-8")
    with pytest.raises(ValueError, match="catalog is stale"):
        generate_catalog(catalog_repo, check=True)


@pytest.mark.parametrize("record_id", ["example.adapter", "", None])
def test_duplicate_or_invalid_ids_fail_before_writing(
    catalog_repo: Path, adapter_source: Path, record_id: str | None
):
    (adapter_source.parent / "second.fabric-adapter.json").write_text(
        json.dumps({"adapter_id": record_id}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate|Invalid"):
        generate_catalog(catalog_repo)
    assert not (catalog_repo / BUNDLE).exists()


def test_target_requires_a_bundled_adapter(catalog_repo: Path, adapter_source: Path):
    targets = adapter_source.parent / "targets"
    targets.mkdir()
    (targets / "missing.fabric-target.json").write_text(
        json.dumps({"id": "example.target", "adapter_id": "missing.adapter"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unbundled adapter"):
        catalog_data(catalog_repo)


def test_committed_bundle_matches_all_bundled_sources():
    generate_catalog(Path(__file__).resolve().parents[2], check=True)

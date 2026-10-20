# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path
from unittest.mock import Mock

import pytest
from jsonschema import validate

import nemo_fabric_adapter_catalog as catalog


@pytest.fixture(name="bundle", scope="session")
def bundle_fixture():
    return json.loads(
        files(catalog).joinpath("catalog.json").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "kind,getter",
    [
        ("adapters", catalog.get_adapter_descriptor),
        ("targets", catalog.get_target_descriptor),
    ],
)
def test_lookup_preserves_canonical_descriptors(bundle, kind, getter, repo_root: Path):
    schema_name = (
        "adapter-descriptor" if kind == "adapters" else "adapter-target-descriptor"
    )
    schema = json.loads(
        (
            repo_root / "schemas/adapter-contract" / f"{schema_name}.schema.json"
        ).read_bytes()
    )
    for record_id, record in bundle[kind].items():
        original = json.loads((repo_root / record["source"]).read_bytes())
        assert getter(record_id) == original
        validate(getter(record_id), schema)


@pytest.mark.parametrize(
    "getter", [catalog.get_adapter_descriptor, catalog.get_target_descriptor]
)
@pytest.mark.parametrize(
    "record_id", ["", "not.a.bundled.id", "../../escape", "NVIDIA.FABRIC.CODEX"]
)
def test_unknown_ids_do_not_fall_back(getter, record_id):
    with pytest.raises(KeyError):
        getter(record_id)


def test_lookup_returns_independent_nested_objects(bundle):
    adapter_id = next(iter(bundle["adapters"]))
    first = catalog.get_adapter_descriptor(adapter_id)
    first["runner"]["changed_by_caller"] = True
    second = catalog.get_adapter_descriptor(adapter_id)
    assert "changed_by_caller" not in second["runner"]


def test_lookup_is_independent_of_task_interpreter(bundle):
    os.environ["ADAPTER_PYTHON"] = "/does/not/exist/python"
    adapter_id = next(iter(bundle["adapters"]))
    assert (
        catalog.get_adapter_descriptor(adapter_id)
        == bundle["adapters"][adapter_id]["descriptor"]
    )


@pytest.mark.parametrize(
    "content,error",
    [("{", json.JSONDecodeError), ('{"format_version": 2}', ValueError)],
)
def test_invalid_resources_fail_clearly(
    monkeypatch: pytest.MonkeyPatch, content: str, error
):
    resource = Mock()
    resource.joinpath.return_value.read_text.return_value = content
    monkeypatch.setattr(catalog, "files", Mock(return_value=resource))
    with pytest.raises(error):
        catalog.get_adapter_descriptor("not.a.bundled.id")


def test_every_bundled_descriptor_declares_the_versions_report(bundle):
    adapters = bundle["adapters"]

    assert adapters
    for adapter_id, entry in adapters.items():
        run_result = entry["descriptor"]["extension_schemas"]["run_result"]
        assert run_result["properties"]["nemo_fabric.versions.v1"] == {
            "type": "object"
        }, adapter_id
        assert "nemo_fabric.versions.v1" not in run_result.get("required", []), (
            adapter_id
        )


SKIPPED_DIRECTORIES = {"node_modules", "target", "build", "dist"}


def _checked_in_descriptors(repo_root: Path):
    for directory, subdirectories, filenames in os.walk(repo_root):
        subdirectories[:] = [
            name
            for name in subdirectories
            if not name.startswith(".") and name not in SKIPPED_DIRECTORIES
        ]
        for filename in filenames:
            if filename.endswith(".fabric-adapter.json"):
                yield Path(directory) / filename


def test_every_copy_of_a_bundled_descriptor_declares_the_versions_report(
    bundle, repo_root: Path
):
    # Bundled adapters always attach the versions report, and core fails a run
    # whose descriptor does not declare it, so copies used by examples and the
    # CLI presets must declare it too.
    bundled_ids = set(bundle["adapters"])
    bundled_modules = {
        entry["descriptor"]["runner"]["module"]
        for entry in bundle["adapters"].values()
        if "module" in entry["descriptor"]["runner"]
    }
    sources = {
        (repo_root / entry["source"]).resolve() for entry in bundle["adapters"].values()
    }

    copies = []
    for path in _checked_in_descriptors(repo_root):
        descriptor = json.loads(path.read_bytes())
        if (
            descriptor.get("adapter_id") in bundled_ids
            or descriptor.get("runner", {}).get("module") in bundled_modules
        ):
            copies.append(path)
            run_result = descriptor.get("extension_schemas", {}).get("run_result", {})
            assert run_result.get("properties", {}).get("nemo_fabric.versions.v1") == {
                "type": "object"
            }, path

    assert {path.resolve() for path in copies} - sources

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
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


def _descriptors_on_disk(repo_root: Path) -> list[Path]:
    found = []
    for directory, subdirectories, filenames in os.walk(repo_root):
        # Installed dependencies are not part of the source tree.
        subdirectories[:] = [
            name for name in subdirectories if name not in {".git", "node_modules"}
        ]
        found.extend(
            Path(directory) / filename
            for filename in filenames
            if filename.endswith(".fabric-adapter.json")
        )
    return found


def _checked_in_descriptors(repo_root: Path) -> list[Path]:
    """Every tracked descriptor, or every descriptor on disk outside a git checkout."""

    try:
        listed = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z", "*.fabric-adapter.json"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return _descriptors_on_disk(repo_root)
    return [
        repo_root / name
        for name in listed.decode("utf-8").split("\0")
        if name and (repo_root / name).is_file()
    ]


def test_every_checked_in_copy_of_a_bundled_descriptor_matches_its_source(
    bundle, repo_root: Path
):
    # Bundled adapters always attach the versions report, and core fails a run
    # whose result carries an undeclared key, so a copy used by an example or a
    # CLI preset must stay byte-identical to the descriptor it copies, as the
    # CLI's packaged-descriptor test already requires for its own copies.
    sources_by_id: dict[str, Path] = {}
    sources_by_module: dict[str, Path] = {}
    sources_by_script: dict[str, list[Path]] = {}
    for adapter_id, entry in bundle["adapters"].items():
        source = repo_root / entry["source"]
        runner = entry["descriptor"]["runner"]
        sources_by_id[adapter_id] = source
        if "module" in runner:
            sources_by_module[runner["module"]] = source
        if "script" in runner:
            sources_by_script.setdefault(runner["script"], []).append(source)

    copies = []
    for path in _checked_in_descriptors(repo_root):
        descriptor = json.loads(path.read_bytes())
        runner = descriptor.get("runner", {})
        if descriptor.get("adapter_id") in sources_by_id:
            candidates = [sources_by_id[descriptor["adapter_id"]]]
        elif runner.get("module") in sources_by_module:
            candidates = [sources_by_module[runner["module"]]]
        elif runner.get("script") in sources_by_script:
            # Several TypeScript adapters share one script path.
            candidates = sources_by_script[runner["script"]]
        else:
            continue
        copies.append(path)
        assert any(path.read_bytes() == source.read_bytes() for source in candidates), (
            f"{path} drifted from {[str(source) for source in candidates]}"
        )

    sources = {path.resolve() for path in sources_by_id.values()}
    assert {path.resolve() for path in copies} - sources

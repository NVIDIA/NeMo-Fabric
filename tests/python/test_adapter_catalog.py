# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fabric.discover() returns the descriptors that planning selects."""

import json
import sysconfig
from pathlib import Path

import pytest
from nemo_fabric import DiscoveryConfig, Fabric, FabricConfig, FabricConfigError

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/discovery/discoverable.fabric-adapter.json"
)
ADAPTER_ID = "org.fabric.fixture.discoverable"


@pytest.fixture(autouse=True, name="installed")
def installed_fixture(monkeypatch, tmp_path):
    """Isolate installed descriptors in a temporary data root and return it."""

    monkeypatch.delenv("ADAPTER_PYTHON", raising=False)
    data = tmp_path / "data"
    original = sysconfig.get_path
    monkeypatch.setattr(
        sysconfig,
        "get_path",
        lambda name, *args: str(data) if name == "data" else original(name, *args),
    )
    return data / "share/nemo-fabric/adapters"


def _discover(*paths, **options):
    return Fabric().discover(
        discovery=DiscoveryConfig(local_paths=list(paths)), **options
    )


def _record(catalog):
    return next(r for r in catalog.adapters if r.descriptor["adapter_id"] == ADAPTER_ID)


def _config(**fields):
    return FabricConfig.from_mapping({"metadata": {"name": "catalog"}, **fields})


def test_planning_selects_the_discovered_local_descriptor(tmp_path):
    record = _record(_discover(FIXTURE, base_dir=tmp_path))
    assert record.provenance[0]["source"] == "explicit_local"
    config = _config(
        discovery={"local_paths": [str(FIXTURE)]},
        harness={
            "adapter_id": ADAPTER_ID,
            "settings": {"mode": "advanced", "budget": 3},
        },
    )
    assert (
        Fabric().plan(config, base_dir=tmp_path)["adapter_descriptor"]
        == record.to_mapping()
    )
    config.harness.settings.pop("budget")
    with pytest.raises(FabricConfigError, match="budget"):
        Fabric().plan(config, base_dir=tmp_path)


def test_planning_selects_the_discovered_installed_descriptor(installed):
    (installed / "any-package").mkdir(parents=True)
    (installed / "any-package/any-name.fabric-adapter.json").write_bytes(
        FIXTURE.read_bytes()
    )
    record = _record(Fabric().discover())
    assert record.provenance[0]["source"] == "installed_package"
    config = _config(harness={"adapter_id": ADAPTER_ID, "settings": {"mode": "simple"}})
    assert Fabric().plan(config)["adapter_descriptor"] == record.to_mapping()


def test_planning_selects_the_discovered_workflow_target():
    catalog = Fabric().discover()
    target = next(
        t for t in catalog.targets if t.descriptor["id"] == "nvidia.nooa.coding-agent"
    )
    config = _config(workflow={"target_id": "nvidia.nooa.coding-agent"})
    assert Fabric().plan(config)["adapter_target_descriptor"] == target.to_mapping()


def test_identical_descriptors_merge_and_any_other_conflict_fails(tmp_path):
    copy = tmp_path / "copy.fabric-adapter.json"
    copy.write_bytes(FIXTURE.read_bytes())
    catalog = _discover(FIXTURE, copy)
    assert len(_record(catalog).provenance) == 2
    ids = [r.descriptor["adapter_id"] for r in catalog.adapters]
    assert ids == sorted(ids)

    different = {**json.loads(FIXTURE.read_text()), "runner": {"module": "other"}}
    copy.write_text(json.dumps(different))
    with pytest.raises(FabricConfigError, match="ambiguous adapter"):
        _discover(FIXTURE, copy)
    for name, text in [
        ("bad.fabric-adapter.json", '{"adapter_id": "bad"}'),
        ("bad.fabric-target.json", '{"id": "bad"}'),
    ]:
        (tmp_path / name).write_text(text)
        with pytest.raises(FabricConfigError):
            _discover(tmp_path / name)
    with pytest.raises(FabricConfigError, match="does not exist"):
        _discover(tmp_path / "missing")


def test_catalog_is_read_only_and_round_trips():
    catalog = _discover(FIXTURE)
    record = _record(catalog)
    record.descriptor["settings_schema"]["required"].append("budget")
    record.provenance[0]["path"] = "mutated"
    assert record.descriptor["settings_schema"]["required"] == ["mode"]
    assert record.provenance[0]["path"] != "mutated"

    mapping = catalog.to_mapping()
    assert type(catalog).from_mapping(json.loads(json.dumps(mapping))) == catalog
    with pytest.raises(FabricConfigError, match="adapter provenance"):
        type(catalog).from_mapping(
            {"adapters": [{**mapping["adapters"][0], "provenance": []}]}
        )
    with pytest.raises(FabricConfigError, match="DiscoveryConfig"):
        Fabric().discover(discovery={"local_paths": []})

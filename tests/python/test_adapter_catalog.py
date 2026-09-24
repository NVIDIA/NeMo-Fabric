# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public discovery exposes the same canonical metadata used by planning."""

import json
import sys
from pathlib import Path

import pytest
from nemo_fabric import DiscoveryConfig, Fabric, FabricConfig, FabricConfigError

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/discovery/future.fabric-adapter.json"
)
ADAPTER_ID = "org.fabric.fixture.discoverable"


@pytest.fixture(autouse=True)
def isolated_installed_descriptor_root(monkeypatch, tmp_path):
    import sysconfig

    original = sysconfig.get_path
    monkeypatch.setattr(
        sysconfig,
        "get_path",
        lambda name, *args, **kwargs: (
            str(tmp_path / "packages")
            if name == "data"
            else original(name, *args, **kwargs)
        ),
    )


def _adapter(catalog):
    return next(
        item for item in catalog.adapters if item.descriptor["adapter_id"] == ADAPTER_ID
    )


def test_discovery_and_plan_share_settings_and_model_contracts(tmp_path, monkeypatch):
    monkeypatch.delenv("ADAPTER_PYTHON", raising=False)
    discovery = DiscoveryConfig(local_paths=[FIXTURE])
    record = _adapter(Fabric().discover(discovery=discovery, base_dir=tmp_path))
    assert "fabric_discovery_fixture" not in sys.modules
    assert (
        record.descriptor["settings_schema"]
        == json.loads(FIXTURE.read_text())["settings_schema"]
    )
    assert record.provenance[0]["source"] == "explicit_local"
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "discovery-fixture"},
            "discovery": {"local_paths": [str(FIXTURE)]},
            "harness": {
                "adapter_id": ADAPTER_ID,
                "settings": {"mode": "advanced", "budget": 3},
            },
            "models": {"default": {"provider": "openai", "model": "fixture-model"}},
        }
    )
    plan = Fabric().plan(config, base_dir=tmp_path)
    assert plan["adapter_descriptor"] == record.to_mapping()
    config.harness.settings.pop("budget")
    with pytest.raises(FabricConfigError, match="budget"):
        Fabric().plan(config, base_dir=tmp_path)
    config.harness.settings = {"mode": "simple"}
    config.models["default"].model = "unaccepted-model"
    with pytest.raises(FabricConfigError, match="models.default.model"):
        Fabric().plan(config, base_dir=tmp_path)
    assert "fabric_discovery_fixture" not in sys.modules


def test_discovery_rejects_ambiguous_and_malformed_metadata(tmp_path):
    raw = json.loads(FIXTURE.read_text())
    raw["runner"]["module"] = "fixture.different_module"
    extra = tmp_path / "different.fabric-adapter.json"
    extra.write_text(json.dumps(raw))
    with pytest.raises(FabricConfigError, match="ambiguous adapter"):
        Fabric().discover(discovery=DiscoveryConfig(local_paths=[FIXTURE, extra]))
    extra.write_text('{"adapter_id":"invalid"}')
    with pytest.raises(FabricConfigError):
        Fabric().discover(discovery=DiscoveryConfig(local_paths=[extra]))


def test_discovery_deduplicates_identical_records_and_keeps_all_sources(tmp_path):
    duplicate = tmp_path / "renamed.fabric-adapter.json"
    duplicate.write_bytes(FIXTURE.read_bytes())
    catalog = Fabric().discover(
        discovery=DiscoveryConfig(local_paths=[FIXTURE, duplicate])
    )
    record = _adapter(catalog)
    assert len(record.provenance) == 2
    assert [item.descriptor["adapter_id"] for item in catalog.adapters] == sorted(
        item.descriptor["adapter_id"] for item in catalog.adapters
    )
    value = record.descriptor
    value["adapter_id"] = "mutated"
    assert record.descriptor["adapter_id"] == ADAPTER_ID
    value["settings_schema"]["properties"]["mode"]["enum"].append("mutated")
    assert record.descriptor["settings_schema"]["properties"]["mode"]["enum"] == [
        "simple",
        "advanced",
    ]
    provenance = record.provenance
    provenance[0]["path"] = "mutated"
    assert record.provenance[0]["path"] != "mutated"


def test_discovery_uses_the_same_installed_root_as_planning(tmp_path, monkeypatch):
    import sysconfig

    package_root = tmp_path / "share/nemo-fabric/adapters/unrelated-package-name"
    package_root.mkdir(parents=True)
    (package_root / "unrelated-filename.fabric-adapter.json").write_bytes(
        FIXTURE.read_bytes()
    )
    original_get_path = sysconfig.get_path
    monkeypatch.delenv("ADAPTER_PYTHON", raising=False)
    monkeypatch.setattr(
        sysconfig,
        "get_path",
        lambda name, *args, **kwargs: (
            str(tmp_path)
            if name == "data"
            else original_get_path(name, *args, **kwargs)
        ),
    )
    record = _adapter(Fabric().discover())
    assert record.provenance[0]["source"] == "installed_package"
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "installed-discovery-fixture"},
            "harness": {"adapter_id": ADAPTER_ID, "settings": {"mode": "simple"}},
        }
    )
    assert Fabric().plan(config)["adapter_descriptor"] == record.to_mapping()


def test_discovery_rejects_missing_paths_and_untyped_configuration(tmp_path):
    with pytest.raises(FabricConfigError, match="does not exist"):
        Fabric().discover(discovery=DiscoveryConfig(local_paths=[tmp_path / "missing"]))
    with pytest.raises(FabricConfigError, match="DiscoveryConfig"):
        Fabric().discover(discovery={"local_paths": []})


async def test_fabric_only_adapter_runs_with_authored_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(FIXTURE.parent))
    monkeypatch.setenv("ADAPTER_PYTHON", sys.executable)
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "fabric-only"},
            "discovery": {"local_paths": [str(FIXTURE)]},
            "harness": {
                "adapter_id": ADAPTER_ID,
                "settings": {"mode": "advanced", "budget": 4},
            },
            "models": {"default": {"provider": "openai", "model": "fixture-model"}},
        }
    )
    result = await Fabric().run(config, base_dir=tmp_path, input={"message": "hello"})
    assert result.status == "succeeded"
    assert result.output == {
        "settings": {"mode": "advanced", "budget": 4},
        "input": {"message": "hello"},
    }


def test_discovered_workflow_target_is_plannable_without_source_reconstruction(
    monkeypatch,
):
    monkeypatch.delenv("ADAPTER_PYTHON", raising=False)
    catalog = Fabric().discover()
    target = next(
        item
        for item in catalog.targets
        if item.descriptor["id"] == "nvidia.nooa.coding-agent"
    )
    assert target.provenance
    assert target.descriptor["adapter_id"] == "nvidia.fabric.nooa"
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "discovered-workflow"},
            "workflow": {"target_id": "nvidia.nooa.coding-agent"},
        }
    )
    assert Fabric().plan(config)["adapter_target_descriptor"] == target.to_mapping()


def test_catalog_mapping_round_trips_for_snapshot_planning(tmp_path, monkeypatch):
    monkeypatch.delenv("ADAPTER_PYTHON", raising=False)
    catalog = Fabric().discover(
        discovery=DiscoveryConfig(local_paths=[FIXTURE]), base_dir=tmp_path
    )
    mapping = catalog.to_mapping()
    assert set(mapping) == {"adapters", "targets"}
    assert type(catalog).from_mapping(json.loads(json.dumps(mapping))) == catalog
    with pytest.raises(FabricConfigError, match="adapter provenance"):
        type(catalog).from_mapping(
            {"adapters": [{**mapping["adapters"][0], "provenance": []}]}
        )

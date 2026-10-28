# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Defensive selected metadata behavior adapted from PR #458's source cases."""

import importlib.metadata
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from nemo_fabric_adapter_contract import metadata


def test_loaded_module_ownership_and_full_version_are_preserved(monkeypatch, tmp_path):
    loaded = tmp_path / "package/__init__.py"
    module = SimpleNamespace(__file__=str(loaded))
    monkeypatch.setitem(sys.modules, "provenance_test_package", module)
    info = tmp_path / "example-1.2.3.dist-info"
    info.mkdir()
    version = "1.2.3-rc.7+" + "build" * 100
    (info / "METADATA").write_text(
        f"Name: example\nVersion: {version}\n\n" + "body" * 100000
    )
    (info / "RECORD").write_text("package/__init__.py,,\n")
    installed = importlib.metadata.PathDistribution(info)
    monkeypatch.setattr(
        metadata.importlib.metadata, "distribution", lambda name: installed
    )
    observation = metadata.package_version(
        "example", "provenance_test_package", "test:metadata", "2026-10-27T12:00:00Z"
    )
    assert observation.value == version
    assert observation.observed_at == "2026-10-27T12:00:00Z"
    module.__file__ = str(tmp_path / "shadow/__init__.py")
    assert (
        metadata.package_version(
            "example",
            "provenance_test_package",
            "test:metadata",
            observation.observed_at,
        ).reason
        == "loaded_module_not_attributed"
    )


def test_missing_and_unreadable_metadata_remain_expected_unknowns(monkeypatch):
    for error, reason in [
        (importlib.metadata.PackageNotFoundError(), "distribution_not_found"),
        (OSError(), "metadata_read_failed"),
    ]:
        mock_distribution = MagicMock(side_effect=error)
        monkeypatch.setattr(
            metadata.importlib.metadata, "distribution", mock_distribution
        )
        report = metadata.collect_selected("adapter", [("harness", "missing", None)])
        env = report.software_environments[0]
        assert env.complete
        assert env.entries[1].version.reason == reason


def test_hung_lookup_is_bounded_and_does_not_accumulate_workers(monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(metadata, "COLLECTION_SECONDS", 0.02)

    def blocked():
        entered.set()
        release.wait(5)
        finished.set()
        return metadata.ExecutionEvidence.gap("adapter", "test")

    try:
        report = metadata.bounded_collect("adapter", blocked)
        assert entered.is_set()
        assert report.software_environments[0].reason == "collection_time_budget"
        assert (
            metadata.bounded_collect("adapter", blocked).software_environments[0].reason
            == "collector_busy"
        )
    finally:
        release.set()
        assert finished.wait(1)
        with metadata._worker:
            pass


def test_target_bounds_are_explicit_without_scanning_all_packages():
    report = metadata.collect_selected(
        "adapter", (("harness", f"missing-{i}", None) for i in range(1000))
    )
    env = report.software_environments[0]
    assert not env.complete
    assert env.reason
    assert len(env.entries) <= metadata.MAX_TARGETS + 1


def test_work_budget_preserves_partial_inventory_before_the_hard_wait(monkeypatch):
    monkeypatch.setattr(
        metadata.time, "monotonic", MagicMock(side_effect=[0.0, 0.0, 0.21])
    )
    mock_version = MagicMock(
        return_value=metadata.observation(
            "1.0+full", "test:metadata", metadata.observed_at()
        )
    )
    monkeypatch.setattr(metadata, "package_version", mock_version)
    report = metadata.collect_selected(
        "adapter", [("harness", "first", None), ("framework", "second", None)]
    )
    env = report.software_environments[0]
    assert not env.complete
    assert env.reason == "collection_truncated_or_work_budget"
    assert [entry.name for entry in env.entries] == ["python", "first"]
    assert env.entries[1].version.value == "1.0+full"
    mock_version.assert_called_once()


def test_source_metadata_bytes_are_bounded_before_parsing(monkeypatch, tmp_path):
    info = tmp_path / "example-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Version: 1.0\n\n")
    (info / "RECORD").write_text("x" * (metadata.MAX_RECORD_BYTES + 1))
    monkeypatch.setitem(
        sys.modules,
        "bounded_metadata_test",
        SimpleNamespace(__file__=str(tmp_path / "example.py")),
    )
    monkeypatch.setattr(
        metadata.importlib.metadata,
        "distribution",
        lambda name: importlib.metadata.PathDistribution(info),
    )
    report = metadata.collect_selected(
        "adapter", [("harness", "example", "bounded_metadata_test")]
    )
    assert not report.software_environments[0].complete
    assert (
        report.software_environments[0].entries[1].version.reason
        == "metadata_byte_limit"
    )
    (info / "METADATA").write_text("Version: " + "a" * metadata.MAX_HEADER_BYTES)
    value = metadata.package_version(
        "example", None, "test:metadata", "2026-10-27T12:00:00Z"
    )
    assert value.reason == "metadata_byte_limit"

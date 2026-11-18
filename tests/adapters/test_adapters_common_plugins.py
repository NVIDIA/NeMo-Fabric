# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared plugin start checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from nemo_fabric_adapters.common import plugins
from nemo_fabric_adapters.common.lifecycle import LifecycleError

LISTING = {"loaded": "", "broken": "ImportError: boom"}


def _check(*requested: str) -> None:
    plugins.check_requested(
        [
            (f"harness.settings.plugins.enabled[{i}]", p)
            for i, p in enumerate(requested)
        ],
        LISTING.get,
        harness="example",
        label="Example",
    )


def test_check_requested_accepts_loaded_plugins():
    _check("loaded")
    _check()


def test_check_requested_reports_a_missing_plugin():
    with pytest.raises(LifecycleError) as error:
        _check("loaded", "typo")

    assert error.value.code == "example_plugin_not_found"
    assert error.value.metadata == {
        "field": "harness.settings.plugins.enabled[1]",
        "plugin": "typo",
    }
    assert "'typo'" in error.value.message


def test_check_requested_reports_a_plugin_that_did_not_load():
    with pytest.raises(LifecycleError) as error:
        _check("broken")

    assert error.value.code == "example_plugin_unavailable"
    assert error.value.metadata == {
        "field": "harness.settings.plugins.enabled[0]",
        "plugin": "broken",
        "reason": "ImportError: boom",
    }
    assert "ImportError: boom" in error.value.message


def test_check_requested_keeps_the_end_of_a_long_reason():
    reason = "x" * 3000 + "tail"

    with pytest.raises(LifecycleError) as error:
        plugins.check_requested(
            [("f", "p")], lambda _: reason, harness="example", label="Example"
        )

    assert len(error.value.metadata["reason"]) == plugins.MAX_REASON
    assert error.value.metadata["reason"].endswith("tail")


def test_resolve_plugin_path_resolves_relative_paths(tmp_path: Path):
    (tmp_path / "plugins" / "one").mkdir(parents=True)

    resolved = plugins.resolve_plugin_path(
        "plugins/one", tmp_path, harness="example", label="Example", field="f"
    )

    assert resolved == (tmp_path / "plugins" / "one").resolve()


def test_resolve_plugin_path_reports_a_missing_path(tmp_path: Path):
    with pytest.raises(LifecycleError) as error:
        plugins.resolve_plugin_path(
            "missing",
            tmp_path,
            harness="example",
            label="Example",
            field="harness.settings.plugins.external.paths[0]",
        )

    assert error.value.code == "example_plugin_not_found"
    assert error.value.metadata == {
        "field": "harness.settings.plugins.external.paths[0]",
        "path": str(tmp_path / "missing"),
    }

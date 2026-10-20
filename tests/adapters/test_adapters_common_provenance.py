# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import types
from pathlib import Path

import deepagents
import minisweagent
import pytest
from nemo_fabric_adapters.common import provenance


def known(value: str, method: str = "package_metadata") -> dict[str, str]:
    return {
        "value": value,
        "source": "reported",
        "collected_by": "adapter",
        "method": method,
    }


def missing(state: str, reason: str, method: str = "package_metadata") -> dict[str, str]:
    return {
        "state": state,
        "reason": reason,
        "source": "reported",
        "collected_by": "adapter",
        "method": method,
    }


# The same vectors as core's string length test in `what_ran.rs`.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("é" * 256, "é" * 256),
        ("m" * 300 + "a", "m" * 240 + "~96858eedeb3c8e6"),
        ("é" * 300, "é" * 240 + "~7250b66610f8b7d"),
    ],
)
def test_bounded_text_matches_core_string_length_rule(text, expected):
    assert provenance.bounded_text(text) == expected


def test_known_fact_replaces_a_lone_surrogate_so_the_result_stays_valid_json():
    fact = provenance.known_fact("claude-\ud800", "harness_api")

    assert fact["value"] == "claude-?"
    json.dumps(fact).encode("utf-8")


def test_package_version_fact_reads_installed_metadata():
    assert provenance.package_version_fact("nemo-fabric-adapters-common") == known(
        "0.6.0"
    )
    assert provenance.package_version_fact("nemo-fabric-no-such-package") == missing(
        "unavailable", "not_collected"
    )


@pytest.mark.parametrize(
    ("distribution", "module", "expected"),
    [
        ("deepagents", deepagents.__name__, known("0.7.9")),
        ("mini-swe-agent", minisweagent.__name__, known("2.4.6")),
        # The distribution exists but did not install the loaded module.
        ("mini-swe-agent", deepagents.__name__, missing("unavailable", "not_attributed")),
        ("deepagents", "nemo_fabric_not_loaded", missing("unavailable", "not_collected")),
    ],
)
def test_loaded_module_version_fact_accepts_only_the_module_the_package_installed(
    distribution, module, expected
):
    assert provenance.loaded_module_version_fact(distribution, module) == expected


def test_loaded_module_version_fact_rejects_a_shadowing_source_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    shadow = types.ModuleType("deepagents")
    shadow.__file__ = str(tmp_path / "deepagents" / "__init__.py")
    monkeypatch.setitem(sys.modules, "deepagents", shadow)

    assert provenance.loaded_module_version_fact("deepagents", "deepagents") == missing(
        "unavailable", "not_attributed"
    )


def test_versions_report_keeps_16_roles_in_sorted_order_and_bounds_values():
    served = {f"role-{index:02}": provenance.no_source_fact() for index in range(17)}
    served["r" * 129] = provenance.no_source_fact()

    report = provenance.versions_report(
        package_version=provenance.known_fact("0.6.0", "package_metadata"),
        harness_name=provenance.known_fact("harness", "harness_api"),
        harness_version=provenance.known_fact("v" * 300, "harness_api"),
        served_models=served,
    )

    assert list(report["models"]) == [f"role-{index:02}" for index in range(16)]
    assert report["harness"]["version"]["value"] == provenance.bounded_text("v" * 300)
    assert report["adapter"]["package_version"] == known("0.6.0")


def probe_threads() -> int:
    return sum(
        thread.name == "nemo-fabric-version-probe" for thread in threading.enumerate()
    )


async def test_blocking_probe_bounds_a_function_that_never_returns():
    release = threading.Event()
    calls: list[str] = []

    def blocking() -> str:
        calls.append("called")
        release.wait()
        return "1.0.0"

    probe = provenance.BlockingProbe()
    ticks = 0

    async def count_ticks() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(count_ticks())
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            await probe.call(blocking, timeout=0.2)
        elapsed = time.monotonic() - started
        # A second call while the first still runs starts no new thread.
        with pytest.raises(TimeoutError):
            await probe.call(blocking, timeout=5)
        assert time.monotonic() - started < 1
        assert calls == ["called"]
        assert probe_threads() == 1
    finally:
        ticker.cancel()
        release.set()

    assert 0.2 <= elapsed < 1
    # The event loop kept running while the function blocked its thread.
    assert ticks >= 5
    while probe_threads():
        await asyncio.sleep(0.01)
    assert await probe.call(lambda: "2.0.0", timeout=1) == "2.0.0"


async def test_blocking_probe_returns_the_result_or_the_raised_error():
    probe = provenance.BlockingProbe()

    def fail() -> str:
        raise RuntimeError("no version")

    assert await probe.call(lambda: "1.2.3") == "1.2.3"
    with pytest.raises(RuntimeError, match="no version"):
        await probe.call(fail)


def test_package_version_report_marks_harness_and_served_models_no_source():
    report = provenance.package_version_report(
        "nemo-fabric-adapters-common", ["judge", "default"]
    )

    no_source = missing("unsupported", "no_source", "adapter_report")
    assert report == {
        "adapter": {"package_version": known("0.6.0")},
        "harness": {"name": no_source, "version": no_source},
        "models": {"default": {"served": no_source}, "judge": {"served": no_source}},
    }

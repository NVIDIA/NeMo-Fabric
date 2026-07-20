# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the committed Python SDK reference."""

from __future__ import annotations

import re
from collections import defaultdict
from inspect import getdoc, getmembers, isclass, isfunction, ismethod, getattr_static
from pathlib import Path

import nemo_fabric
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "docs" / "reference" / "api" / "python-library-reference"
LANDING_PAGE = ROOT / "docs" / "about-nemo-fabric" / "overview.mdx"
QUICK_START_PAGE = ROOT / "docs" / "getting-started" / "quickstart.mdx"
BEGINNER_TUTORIAL_PAGE = ROOT / "docs" / "getting-started" / "beginner-tutorial.mdx"
NAVIGATION = ROOT / "docs" / "index.yml"
MODULE_SLUGS = {
    "nemo_fabric.client": "/reference/api/python-library-reference/client",
    "nemo_fabric.runtime": "/reference/api/python-library-reference/runtime",
    "nemo_fabric.models": "/reference/api/python-library-reference/models",
    "nemo_fabric.types": "/reference/api/python-library-reference/types",
    "nemo_fabric.errors": "/reference/api/python-library-reference/errors",
}


def _exported_classes_by_module() -> dict[str, set[str]]:
    exports: dict[str, set[str]] = defaultdict(set)
    for name in nemo_fabric.__all__:
        exported = getattr(nemo_fabric, name)
        exports[exported.__module__].add(name)
    return dict(exports)


def _documented_classes(page: Path) -> set[str]:
    return set(
        re.findall(
            r"^## <kbd>class</kbd> `([^`]+)`$",
            page.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )


def test_python_reference_exactly_covers_exported_sdk_classes() -> None:
    exports = _exported_classes_by_module()
    expected_pages = {f"{module}.md" for module in exports}
    actual_pages = {
        page.name
        for page in REFERENCE_DIR.glob("nemo_fabric.*.md")
    }

    assert actual_pages == expected_pages
    for module, names in exports.items():
        page = REFERENCE_DIR / f"{module}.md"
        assert _documented_classes(page) == names
        assert f'slug: "{MODULE_SLUGS[module]}"' in page.read_text(encoding="utf-8")


def test_exported_sdk_classes_and_public_members_have_docstrings() -> None:
    for export_name in nemo_fabric.__all__:
        exported = getattr(nemo_fabric, export_name)
        assert getdoc(exported), export_name
        if not isclass(exported):
            continue

        for member_name, member in getmembers(exported):
            if member_name.startswith("_"):
                continue
            raw_member = getattr_static(exported, member_name)
            if not (
                isfunction(member)
                or ismethod(member)
                or isinstance(raw_member, property)
            ):
                continue
            if issubclass(exported, BaseModel) and hasattr(BaseModel, member_name):
                continue
            assert getdoc(member), f"{export_name}.{member_name}"


def test_generated_reference_uses_valid_heading_order() -> None:
    for page in REFERENCE_DIR.glob("*.md"):
        text = page.read_text(encoding="utf-8")
        assert "#### <kbd>property</kbd>" not in text, page


def test_landing_page_routes_new_users_through_the_product() -> None:
    landing = LANDING_PAGE.read_text(encoding="utf-8")
    quick_start = QUICK_START_PAGE.read_text(encoding="utf-8")
    beginner_tutorial = BEGINNER_TUTORIAL_PAGE.read_text(encoding="utf-8")
    navigation = NAVIGATION.read_text(encoding="utf-8")

    assert "      - section: API\n" in navigation
    assert "      - section: APIs\n" not in navigation

    for heading in (
        "## Benefits",
        "## Use Cases",
        "## Choose Your Interface",
        "## Core Workflow",
        "## Learn More",
    ):
        assert heading in landing

    for destination in (
        "/getting-started/install",
        "/getting-started/quick-start",
        "/getting-started/quickstart",
        "/nemo/fabric/sdk/python-sdk",
        "/reference/api/python-library-reference/client",
        "/reference/api/python-library-reference/runtime",
    ):
        assert destination in landing

    assert "client.plan(" not in quick_start
    assert "client.doctor(" not in quick_start
    assert "## Quickstart Steps" in quick_start
    assert "## Concepts Overview" in beginner_tutorial
    for tutorial_api in (
        "FabricConfig",
        "RunResult",
        "Fabric().run(",
        "Fabric().start_runtime(",
        "fabric.plan(",
        "fabric.doctor(",
        "01_quickstart.ipynb",
    ):
        assert tutorial_api in beginner_tutorial
    assert "## Summary" in beginner_tutorial
    assert (
        "      - page: Quickstart\n"
        "        path: ./getting-started/quickstart.mdx\n"
        "        slug: quick-start\n"
    ) in navigation
    assert (
        "      - page: Beginner Tutorial\n"
        "        path: ./getting-started/beginner-tutorial.mdx\n"
        "        slug: quickstart\n"
    ) in navigation
    assert "/nemo/fabric/sdk/python-sdk" in quick_start

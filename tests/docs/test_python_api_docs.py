# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the committed Python SDK reference."""

from __future__ import annotations

import importlib
import re
from collections import defaultdict
from enum import Enum
from inspect import (
    Parameter,
    formatannotation,
    get_annotations,
    getdoc,
    getmembers,
    getattr_static,
    isclass,
    iscoroutinefunction,
    isfunction,
    ismethod,
    signature,
)
from pathlib import Path

import nemo_fabric
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "docs" / "reference" / "api" / "python-library-reference"
LANDING_PAGE = ROOT / "docs" / "about-nemo-fabric" / "overview.mdx"
NAVIGATION = ROOT / "docs" / "index.yml"
MODULE_SLUGS = {
    "nemo_fabric.client": "/reference/api/python-library-reference/client",
    "nemo_fabric.runtime": "/reference/api/python-library-reference/runtime",
    "nemo_fabric.models": "/reference/api/python-library-reference/models",
    "nemo_fabric.types": "/reference/api/python-library-reference/types",
    "nemo_fabric.errors": "/reference/api/python-library-reference/errors",
}
MEMBER_SIGNATURE = re.compile(
    r"^### <kbd>(?:method|classmethod)</kbd> `([^`]+)`\n\n"
    r"```text\n(.*?)\n```",
    flags=re.MULTILINE | re.DOTALL,
)
MODEL_FIELD_ROW = re.compile(
    r"^\| `(?P<name>[^`]+)` \| `(?P<annotation>[^`]*)` \| "
    r"(?P<required>Yes|No) \| (?P<default>—|`[^`]*`) \| "
    r"(?P<constraints>—|`[^`]*`) \|",
    flags=re.MULTILINE,
)
TYPED_FIELD_ROW = re.compile(
    r"^\| `(?P<name>[^`]+)` \| `(?P<annotation>[^`]*)` \|$",
    flags=re.MULTILINE,
)


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


def _class_section(page: Path, class_name: str) -> str:
    marker = f"## <kbd>class</kbd> `{class_name}`"
    section = page.read_text(encoding="utf-8").split(marker, maxsplit=1)[1]
    return section.split("\n## <kbd>class</kbd>", maxsplit=1)[0]


def _annotation_text(annotation: object) -> str:
    if isinstance(annotation, str):
        if (
            len(annotation) >= 2
            and annotation[0] == annotation[-1]
            and annotation[0] in {'"', "'"}
        ):
            annotation = annotation[1:-1]
        return annotation.replace("|", r"\|")
    return formatannotation(annotation).replace("|", r"\|")


def _default_factory_text(factory: object) -> str:
    name = getattr(factory, "__name__", type(factory).__name__)
    if name == "<lambda>":
        return "<generated>"
    return f"{name}()"


def test_python_reference_exactly_covers_exported_sdk_classes() -> None:
    exports = _exported_classes_by_module()
    expected_pages = {f"{module}.md" for module in exports}
    actual_pages = {page.name for page in REFERENCE_DIR.glob("nemo_fabric.*.md")}

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


def test_generated_signatures_preserve_python_callable_semantics():
    signature_count = 0
    async_count = 0
    for module_name, class_names in _exported_classes_by_module().items():
        module = importlib.import_module(module_name)
        page = REFERENCE_DIR / f"{module_name}.md"

        for class_name in class_names:
            exported_class = getattr(module, class_name)
            section = _class_section(page, class_name)
            documented_members = re.findall(
                r"^### <kbd>(?:method|classmethod)</kbd> `([^`]+)`$",
                section,
                flags=re.MULTILINE,
            )
            rendered_signatures = list(MEMBER_SIGNATURE.finditer(section))
            assert [
                match.group(1) for match in rendered_signatures
            ] == documented_members

            for match in rendered_signatures:
                member_name, rendered = match.groups()
                member = getattr(exported_class, member_name)
                member_signature = signature(member)
                signature_count += 1
                assert "→" not in rendered

                is_async = iscoroutinefunction(member)
                async_count += is_async
                assert rendered.startswith("async ") is is_async

                has_keyword_only = any(
                    parameter.kind is Parameter.KEYWORD_ONLY
                    for parameter in member_signature.parameters.values()
                )
                has_var_positional = any(
                    parameter.kind is Parameter.VAR_POSITIONAL
                    for parameter in member_signature.parameters.values()
                )
                if has_keyword_only and not has_var_positional:
                    assert re.search(r"(?:\(\*,|, \*,|\n    \*,)", rendered)

    assert signature_count > 0
    assert async_count == 5


def test_generated_reference_documents_pydantic_field_contracts():
    for export_name in nemo_fabric.__all__:
        exported = getattr(nemo_fabric, export_name)
        if not isclass(exported) or not issubclass(exported, BaseModel):
            continue
        if not exported.model_fields:
            continue

        page = REFERENCE_DIR / f"{exported.__module__}.md"
        section = _class_section(page, export_name)
        rows = {
            match.group("name"): match.groupdict()
            for match in MODEL_FIELD_ROW.finditer(section)
        }
        assert set(rows) == set(exported.model_fields)
        annotations = get_annotations(exported, eval_str=False)

        for field_name, field in exported.model_fields.items():
            row = rows[field_name]
            assert row["annotation"] == _annotation_text(
                annotations.get(field_name, field.annotation)
            )
            assert row["required"] == ("Yes" if field.is_required() else "No")
            if field.is_required():
                assert row["default"] == "—"
            elif field.default_factory is not None:
                assert row["default"] == (
                    f"`{_default_factory_text(field.default_factory)}`"
                )
            else:
                assert row["default"] == f"`{field.default!r}`"
            if field.metadata:
                constraints = ", ".join(repr(item) for item in field.metadata)
                assert row["constraints"] == f"`{constraints}`"
            else:
                assert row["constraints"] == "—"


def test_generated_reference_documents_typed_mapping_fields():
    for export_name in nemo_fabric.__all__:
        exported = getattr(nemo_fabric, export_name)
        if not isclass(exported) or exported.__module__ != "nemo_fabric.types":
            continue

        annotations = get_annotations(exported, eval_str=False)
        page = REFERENCE_DIR / f"{exported.__module__}.md"
        section = _class_section(page, export_name)
        rows = {
            match.group("name"): match.group("annotation")
            for match in TYPED_FIELD_ROW.finditer(section)
        }
        assert set(rows) == set(annotations)
        for field_name, annotation in annotations.items():
            assert rows[field_name] == _annotation_text(annotation)
        assert "from_mapping(mapping: Mapping[str, Any]) -> Self" in section


def test_generated_reference_documents_error_bases_and_enum_values():
    for export_name in nemo_fabric.__all__:
        exported = getattr(nemo_fabric, export_name)
        if not isclass(exported):
            continue

        page = REFERENCE_DIR / f"{exported.__module__}.md"
        section = _class_section(page, export_name)
        if exported.__module__ == "nemo_fabric.errors":
            for base in exported.__bases__:
                assert f"`{base.__name__}`" in section
        if issubclass(exported, Enum):
            for member in exported:
                assert f"| `{member.name}` | `{member.value}` |" in section


def test_landing_page_routes_new_users_through_the_product() -> None:
    landing = LANDING_PAGE.read_text(encoding="utf-8")
    navigation = NAVIGATION.read_text(encoding="utf-8")

    assert "      - section: API\n" in navigation
    assert "      - section: APIs\n" not in navigation

    for heading in (
        "## What NeMo Fabric Gives You",
        "## How NeMo Fabric Fits",
        "## Quick Start",
        "## Choose Your Interface",
        "## Core Workflow",
        "## Next Steps",
    ):
        assert heading in landing

    for destination in (
        "/reference/api/python-library-reference/client",
        "/reference/api/python-library-reference/runtime",
        "/reference/api/python-library-reference/types",
        "/reference/api/python-library-reference/errors",
    ):
        assert destination in landing

    quick_start = landing.split("## Quick Start", maxsplit=1)[1].split(
        "## Choose Your Interface", maxsplit=1
    )[0]
    assert "client.plan(" not in quick_start
    assert "client.doctor(" not in quick_start
    assert "/sdk/python" in quick_start

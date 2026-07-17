# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generated ``docs-website`` branch layout."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "docs" / "sync_fern_docs_branch.py"
SPEC = importlib.util.spec_from_file_location("sync_fern_docs_branch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC)


def _docs_config(target: Path) -> dict:
    return yaml.safe_load((target / "fern" / "docs.yml").read_text(encoding="utf-8"))


def test_sync_main_bootstraps_publishing_layout(tmp_path: Path) -> None:
    SYNC.sync_main(ROOT, tmp_path)

    product = _docs_config(tmp_path)["products"][0]
    assert product["path"] == "./versions/soon.yml"
    assert [entry["slug"] for entry in product["versions"]] == ["soon", "main"]
    assert product["versions"][1]["path"] == "./versions/main.yml"

    navigation = (tmp_path / "fern" / "versions" / "main.yml").read_text(
        encoding="utf-8"
    )
    assert "../pages-main/getting-started/overview.mdx" in navigation
    assert (tmp_path / "fern" / "pages-soon" / "index.mdx").is_file()
    assert not (tmp_path / "fern" / "pages-main" / "package.json").exists()
    assert (tmp_path / ".github" / "workflows" / "publish-fern-docs.yml").is_file()


def test_release_snapshot_replaces_cover_only_for_stable_tag(tmp_path: Path) -> None:
    SYNC.sync_main(ROOT, tmp_path)
    SYNC.release_version(ROOT, tmp_path, "0.1.0-rc.1")

    prerelease_product = _docs_config(tmp_path)["products"][0]
    assert prerelease_product["path"] == "./versions/soon.yml"
    assert [entry["slug"] for entry in prerelease_product["versions"]] == [
        "soon",
        "main",
        "v0.1.0",
    ]

    released_page = tmp_path / "fern" / "pages-v0.1.0" / "sdk" / "python.mdx"
    released_text = released_page.read_text(encoding="utf-8")
    assert "](/v0.1.0/getting-started/overview)" in released_text
    assert "github.com/NVIDIA/NeMo-Fabric/tree/0.1.0-rc.1/" in released_text

    SYNC.release_version(ROOT, tmp_path, "0.1.0")
    stable_product = _docs_config(tmp_path)["products"][0]
    assert stable_product["path"] == "./versions/v0.1.0.yml"
    assert [entry["slug"] for entry in stable_product["versions"]] == [
        "latest",
        "main",
        "v0.1.0",
    ]

    # A later main sync updates bleeding-edge pages without dropping releases.
    SYNC.sync_main(ROOT, tmp_path)
    assert _docs_config(tmp_path)["products"][0] == stable_product


def test_alpha_docs_tags_are_not_publishable() -> None:
    with pytest.raises(ValueError, match="alpha docs tags are not published"):
        SYNC.parse_release_tag("0.1.0-alpha.20260717")

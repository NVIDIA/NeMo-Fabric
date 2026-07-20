# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

DOCS_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "docs"
sys.path.insert(0, str(DOCS_SCRIPTS))

import sync_fern_docs_branch  # noqa: E402


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _source_tree(root: Path) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "guide.mdx").write_text(
        "See https://github.com/NVIDIA/NeMo-Fabric/blob/main/README.md\n",
        encoding="utf-8",
    )
    (docs / "_source").mkdir()
    (docs / "_source" / "ignored.md").write_text("ignored\n", encoding="utf-8")
    _write_yaml(
        docs / "index.yml",
        {
            "navigation": [
                {"page": "Guide", "path": "guide.mdx"},
                {"page": "External", "path": "https://example.com"},
            ]
        },
    )

    fern = root / "fern"
    fern.mkdir()
    (fern / "fern.config.json").write_text('{"version": "5.37.10"}\n', encoding="utf-8")
    _write_yaml(
        fern / "docs.yml",
        {
            "title": "NVIDIA NeMo Fabric",
            "products": [
                {
                    "display-name": "NeMo Fabric",
                    "slug": "/",
                    "path": "../docs/index.yml",
                }
            ],
        },
    )


def test_sync_dev_rewrites_navigation_and_preserves_versions(tmp_path: Path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _source_tree(source_root)

    target_fern = target_root / "fern"
    (target_fern / "pages-dev").mkdir(parents=True)
    (target_fern / "pages-dev" / "stale.mdx").write_text("stale\n", encoding="utf-8")
    preserved_product = {
        "display-name": "NeMo Fabric",
        "slug": "/",
        "path": "./versions/v0.1.0.yml",
        "versions": [
            {
                "display-name": "Latest (v0.1.0)",
                "path": "./versions/v0.1.0.yml",
                "slug": "latest",
                "availability": "stable",
            }
        ],
    }
    _write_yaml(
        target_fern / "docs.yml",
        {"title": "Old title", "products": [preserved_product]},
    )

    sync_fern_docs_branch.sync_dev(source_root, target_root)

    assert (target_fern / "pages-dev" / "guide.mdx").is_file()
    assert not (target_fern / "pages-dev" / "stale.mdx").exists()
    assert not (target_fern / "pages-dev" / "index.yml").exists()
    assert not (target_fern / "pages-dev" / "_source").exists()
    assert (target_fern / "fern.config.json").is_file()

    navigation = sync_fern_docs_branch.read_yaml(target_fern / "versions" / "dev.yml")
    assert navigation["navigation"][0]["path"] == "../pages-dev/guide.mdx"
    assert navigation["navigation"][1]["path"] == "https://example.com"

    docs_yml = sync_fern_docs_branch.read_yaml(target_fern / "docs.yml")
    assert docs_yml["title"] == "NVIDIA NeMo Fabric"
    assert docs_yml["products"][0] == preserved_product


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("1.2.3", ("v1.2.3", "stable", True)),
        ("1.2.3-beta.4", ("v1.2.3", "beta", False)),
        ("1.2.3-rc.4", ("v1.2.3", "beta", False)),
    ],
)
def test_parse_release_tag(tag: str, expected: tuple[str, str, bool]):
    assert sync_fern_docs_branch.parse_release_tag(tag) == expected


@pytest.mark.parametrize("tag", ["v1.2.3", "1.2", "1.2.3-alpha.4", "1.2.3-beta"])
def test_parse_release_tag_rejects_unpublished_tags(tag: str):
    with pytest.raises(ValueError):
        sync_fern_docs_branch.parse_release_tag(tag)


def test_release_version_promotes_stable_snapshot(tmp_path: Path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _source_tree(source_root)

    target_fern = target_root / "fern"
    _write_yaml(target_fern / "versions" / "dev.yml", {"navigation": []})
    _write_yaml(
        target_fern / "docs.yml",
        {
            "products": [
                {
                    "display-name": "NeMo Fabric",
                    "slug": "/",
                    "path": "./versions/v0.1.0.yml",
                    "versions": [
                        {
                            "display-name": "Latest (v0.1.0)",
                            "path": "./versions/v0.1.0.yml",
                            "slug": "latest",
                            "availability": "stable",
                        },
                        {
                            "display-name": "dev",
                            "path": "./versions/dev.yml",
                            "slug": "dev",
                            "availability": "beta",
                        },
                        {
                            "display-name": "v0.1.0",
                            "path": "./versions/v0.1.0.yml",
                            "slug": "v0.1.0",
                            "availability": "stable",
                        },
                    ],
                }
            ]
        },
    )

    sync_fern_docs_branch.release_version(target_root, "0.2.0-rc.1", source_root)
    beta_docs_yml = sync_fern_docs_branch.read_yaml(target_fern / "docs.yml")
    beta_product = beta_docs_yml["products"][0]
    assert beta_product["path"] == "./versions/v0.1.0.yml"
    assert [entry["slug"] for entry in beta_product["versions"]] == [
        "latest",
        "dev",
        "v0.2.0",
        "v0.1.0",
    ]
    assert beta_product["versions"][2]["availability"] == "beta"

    sync_fern_docs_branch.release_version(target_root, "0.2.0", source_root)
    stable_docs_yml = sync_fern_docs_branch.read_yaml(target_fern / "docs.yml")
    stable_product = stable_docs_yml["products"][0]
    assert stable_product["path"] == "./versions/v0.2.0.yml"
    assert stable_product["versions"][0]["display-name"] == "Latest (v0.2.0)"
    assert stable_product["versions"][2]["availability"] == "stable"
    assert "blob/0.2.0/README.md" in (
        target_fern / "pages-v0.2.0" / "guide.mdx"
    ).read_text(encoding="utf-8")

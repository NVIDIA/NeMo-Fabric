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
    root.mkdir()
    (root / "README.md").write_text("# NeMo Fabric\n", encoding="utf-8")
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "guide.mdx").write_text(
        "\n".join(
            (
                "See https://github.com/NVIDIA/NeMo-Fabric/blob/main/README.md.",
                "Browse https://github.com/NVIDIA/NeMo-Fabric/tree/main/docs#guides.",
                "Read https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/My%20Guide.md.",
                "Missing https://github.com/NVIDIA/NeMo-Fabric/blob/main/notebooks/quickstart.ipynb.",
                "",
            )
        ),
        encoding="utf-8",
    )
    (docs / "My Guide.md").write_text("# My Guide\n", encoding="utf-8")
    (docs / "_source").mkdir()
    (docs / "_source" / "ignored.md").write_text("ignored\n", encoding="utf-8")
    python_reference = docs / "reference" / "api" / "python-library-reference"
    python_reference.mkdir(parents=True)
    (python_reference / "client.md").write_text(
        f"{sync_fern_docs_branch.MARKDOWN_SPDX_HEADER}\n\n# Client\n",
        encoding="utf-8",
    )
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


@pytest.mark.parametrize(
    "title",
    ["", ' "Repository README"', " 'Repository README'", " (Repository README)"],
)
def test_repository_links_are_resolved_only_in_published_pages(tmp_path, title):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _source_tree(source_root)
    schemas = source_root / "schemas"
    schemas.mkdir()
    (schemas / "result.json").write_text("{}")
    page = source_root / "docs" / "guide.mdx"
    text = (
        "[Schema](../schemas/result.json#definitions)\n"
        f"[Repository](../README.md{title})\n"
        "[Page](./My%20Guide.md)\n"
        "[Missing](../missing.json)\n"
        "![Image](../README.md)\n"
    )
    page.write_text(text)
    sync_fern_docs_branch.sync_dev(source_root, target_root)
    published = target_root / "fern" / "pages-dev" / "guide.mdx"
    assert page.read_text() == text
    assert published.read_text() == (
        "[Schema](https://github.com/NVIDIA/NeMo-Fabric/blob/main/schemas/result.json#definitions)\n"
        f"[Repository](https://github.com/NVIDIA/NeMo-Fabric/blob/main/README.md{title})\n"
        "[Page](./My%20Guide.md)\n"
        "[Missing](../missing.json)\n"
        "![Image](../README.md)\n"
    )
    sync_fern_docs_branch.release_version(target_root, "0.6.0", source_root)
    released = target_root / "fern" / "pages-v0.6.0" / "guide.mdx"
    assert "blob/v0.6.0/schemas/result.json#definitions" in released.read_text()
    assert f"blob/v0.6.0/README.md{title})" in released.read_text()


def test_card_source_links_resolve_only_in_published_pages(tmp_path: Path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _source_tree(source_root)
    guide = source_root / "docs" / "guide.mdx"
    original = (
        "---\nslug: /guides/overview\n---\n"
        '<Card title="Client" href="./My%20Guide.md?view=full#request">Client</Card>\n'
        '<Card\n title="Client again"\n href="./My%20Guide.md">Client</Card>\n'
        '<Card href="#local">Local</Card>\n'
        '<Card href="https://example.com/guide.md">External</Card>\n'
        '```mdx\n<Card href="./missing.mdx">Sample source</Card>\n```\n'
    )
    guide.write_text(original)
    (source_root / "docs" / "My Guide.md").write_text(
        "---\nslug: /sdk/client\n---\n# Client\n"
    )

    sync_fern_docs_branch.sync_dev(source_root, target_root)

    published = target_root / "fern" / "pages-dev" / "guide.mdx"
    assert guide.read_text() == original
    assert published.read_text() == original.replace(
        "./My%20Guide.md?view=full#request", "../sdk/client?view=full#request"
    ).replace('href="./My%20Guide.md"', 'href="../sdk/client"')
    sync_fern_docs_branch.release_version(target_root, "0.6.0", source_root)
    released = target_root / "fern" / "pages-v0.6.0" / "guide.mdx"
    assert 'href="../sdk/client?view=full#request"' in released.read_text()


@pytest.mark.parametrize(
    ("opening", "inside", "closing"),
    [
        ("~~~mdx", "~~~not-a-closing-fence", "~~~"),
        ("```mdx", "```not-a-closing-fence", "```"),
        ("~~~~mdx", "~~~", "~~~~"),
        ("````mdx", "```", "````"),
        ("~~~mdx", "```", "~~~~ \t"),
        ("```mdx", "~~~", "```` \t"),
        ("~~~mdx", "    ~~~", "   ~~~"),
        ("```mdx", "    ```", "   ```"),
        ("```mdx", "> ```", "```"),
        ("~~~mdx", "> ~~~", "~~~"),
    ],
)
@pytest.mark.parametrize("quote_prefix", ["", "> ", "> > ", "   > "])
def test_card_source_links_preserve_fenced_examples(
    tmp_path: Path, opening: str, inside: str, closing: str, quote_prefix: str
):
    page = tmp_path / "page.mdx"
    literal = '<Card href="./example-only.mdx">Literal example</Card>'
    original = (
        "---\nslug: /guides/overview\n---\n"
        f"{quote_prefix}{opening}\n{quote_prefix}{inside}\n"
        f"{quote_prefix}{literal}\n{quote_prefix}{closing}\n"
        '<Card href="./destination.mdx">Real link</Card>\n'
    )
    page.write_text(original)
    (tmp_path / "destination.mdx").write_text("---\nslug: /sdk/client\n---\n")

    sync_fern_docs_branch.rewrite_internal_card_links(tmp_path)

    assert page.read_text() == original.replace(
        'href="./destination.mdx"', 'href="../sdk/client"'
    )


def test_card_source_links_resume_after_unclosed_quoted_fence(tmp_path: Path):
    page = tmp_path / "page.mdx"
    original = (
        "---\nslug: /guides/overview\n---\n"
        '> ```mdx\n> <Card href="./example-only.mdx">Example</Card>\n\n'
        '<Card href="./destination.mdx">Real link</Card>\n'
    )
    page.write_text(original)
    (tmp_path / "destination.mdx").write_text("---\nslug: /sdk/client\n---\n")

    sync_fern_docs_branch.rewrite_internal_card_links(tmp_path)

    assert page.read_text() == original.replace(
        'href="./destination.mdx"', 'href="../sdk/client"'
    )


@pytest.mark.parametrize(
    "example",
    [
        '`<Card href="./example-only.mdx">Example</Card>`',
        '`` `<Card href="./example-only.mdx">Example</Card>` ``',
        '`<Card\n href="./example-only.mdx">Example</Card>`',
        '`unmatched before ``<Card href="./example-only.mdx">Example</Card>``',
        r'\\`<Card href="./example-only.mdx">Example</Card>`',
        r'\\\\`<Card href="./example-only.mdx">Example</Card>`',
    ],
)
def test_card_source_links_preserve_inline_examples(tmp_path: Path, example: str):
    page = tmp_path / "page.mdx"
    original = (
        "---\nslug: /guides/overview\n---\n"
        f'Example: {example} and <Card href="./destination.mdx">Real link</Card>\n'
    )
    page.write_text(original)
    (tmp_path / "destination.mdx").write_text("---\nslug: /sdk/client\n---\n")

    sync_fern_docs_branch.rewrite_internal_card_links(tmp_path)

    assert page.read_text() == original.replace(
        'href="./destination.mdx"', 'href="../sdk/client"'
    )


@pytest.mark.parametrize(
    "prose",
    [
        r'Escaped \` <Card href="./destination.mdx">Real link</Card> \`',
        r'Escaped \\\` <Card href="./destination.mdx">Real link</Card> \\\`',
        'Unmatched ` here.\n\n<Card href="./destination.mdx">Real link</Card>'
        "\n\nAnother unmatched ` here.",
        'Unmatched ` here.\n \t\n<Card href="./destination.mdx">Real link</Card>'
        "\n \t\nAnother unmatched ` here.",
        r'`Literal backslash \` <Card href="./destination.mdx">Real link</Card>',
    ],
)
def test_card_source_links_rewrite_outside_inline_code(tmp_path: Path, prose: str):
    page = tmp_path / "page.mdx"
    original = "---\nslug: /guides/overview\n---\n" + prose + "\n"
    page.write_text(original)
    (tmp_path / "destination.mdx").write_text("---\nslug: /sdk/client\n---\n")

    sync_fern_docs_branch.rewrite_internal_card_links(tmp_path)

    assert page.read_text() == original.replace(
        'href="./destination.mdx"', 'href="../sdk/client"'
    )


@pytest.mark.parametrize("quote_prefix", ["", "> ", "> > "])
@pytest.mark.parametrize(
    ("opening", "continuation"),
    [
        ("- ~~~mdx", "  "),
        ("* ```mdx", "  "),
        ("+ ~~~mdx", "  "),
        ("1. ```mdx", "   "),
        ("12. ~~~mdx", "    "),
        ("- 1. ```mdx", "     "),
        ("- Item\n\n  ```mdx", "  "),
        ("- Outer\n  - Inner\n\n    ~~~mdx", "    "),
    ],
)
def test_card_source_links_preserve_list_fences(
    tmp_path: Path, quote_prefix: str, opening: str, continuation: str
):
    fence = "~~~" if "~~~" in opening else "```"
    listed = (
        opening
        + "\n"
        + continuation
        + "    "
        + fence
        + "\n"
        + continuation
        + '<Card href="./example-only.mdx">Example</Card>\n'
        + continuation
        + fence
        + "\n"
    )
    original = (
        "---\nslug: /guides/overview\n---\n"
        + "".join(quote_prefix + line for line in listed.splitlines(keepends=True))
        + '<Card href="./destination.mdx">Real link</Card>\n'
    )
    page = tmp_path / "page.mdx"
    page.write_text(original)
    (tmp_path / "destination.mdx").write_text("---\nslug: /sdk/client\n---\n")

    sync_fern_docs_branch.rewrite_internal_card_links(tmp_path)

    assert page.read_text() == original.replace(
        'href="./destination.mdx"', 'href="../sdk/client"'
    )


def test_card_source_links_resume_after_unclosed_list_fence(tmp_path: Path):
    page = tmp_path / "page.mdx"
    original = (
        "---\nslug: /guides/overview\n---\n"
        '- ```mdx\n  <Card href="./example-only.mdx">Example</Card>\n\n'
        '<Card href="./destination.mdx">Real link</Card>\n'
    )
    page.write_text(original)
    (tmp_path / "destination.mdx").write_text("---\nslug: /sdk/client\n---\n")
    sync_fern_docs_branch.rewrite_internal_card_links(tmp_path)
    assert page.read_text() == original.replace(
        'href="./destination.mdx"', 'href="../sdk/client"'
    )


def test_card_source_links_reject_missing_targets(tmp_path: Path):
    page = tmp_path / "page.mdx"
    page.write_text('---\nslug: /page\n---\n<Card href="missing.mdx">Missing</Card>\n')
    with pytest.raises(ValueError, match="Card destination does not exist"):
        sync_fern_docs_branch.rewrite_internal_card_links(tmp_path)


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
    assert (
        (
            target_fern
            / "pages-dev"
            / "reference"
            / "api"
            / "python-library-reference"
            / "client.md"
        )
        .read_text(encoding="utf-8")
        .startswith(sync_fern_docs_branch.MDX_SPDX_HEADER)
    )

    navigation = sync_fern_docs_branch.read_yaml(target_fern / "versions" / "dev.yml")
    assert navigation["navigation"][0]["path"] == "../pages-dev/guide.mdx"
    assert navigation["navigation"][1]["path"] == "https://example.com"

    docs_yml = sync_fern_docs_branch.read_yaml(target_fern / "docs.yml")
    assert docs_yml["title"] == "NVIDIA NeMo Fabric"
    assert docs_yml["products"][0] == preserved_product


@pytest.mark.parametrize("release", [False, True])
def test_repository_image_links_resolve_after_publication(
    tmp_path: Path, release: bool
):
    """Images outside docs remain reachable in dev and release layouts."""
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _source_tree(source_root)
    (source_root / "assets").mkdir()
    image_bytes = b"ecosystem image"
    (source_root / "assets" / "ecosystem.png").write_bytes(image_bytes)
    page = source_root / "docs" / "about-nemo-fabric" / "ecosystem.mdx"
    page.parent.mkdir()
    image_link = "../../assets/ecosystem.png"
    page.write_text(f"![Ecosystem]({image_link})\n", encoding="utf-8")

    if release:
        _write_yaml(
            target_root / "fern" / "docs.yml",
            sync_fern_docs_branch.read_yaml(source_root / "fern" / "docs.yml"),
        )
        sync_fern_docs_branch.release_version(target_root, "1.2.3", source_root)
        pages_name = "pages-v1.2.3"
    else:
        sync_fern_docs_branch.sync_dev(source_root, target_root)
        pages_name = "pages-dev"

    published_page = target_root / "fern" / pages_name / "about-nemo-fabric" / page.name
    assert image_link in published_page.read_text(encoding="utf-8")
    assert (published_page.parent / image_link).read_bytes() == image_bytes


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
    guide = (target_fern / "pages-v0.2.0" / "guide.mdx").read_text(encoding="utf-8")
    assert "blob/v0.2.0/README.md" in guide
    assert "tree/v0.2.0/docs#guides" in guide
    assert "blob/v0.2.0/docs/My%20Guide.md" in guide
    assert "blob/main/notebooks/quickstart.ipynb" in guide
    assert (
        (
            target_fern
            / "pages-v0.2.0"
            / "reference"
            / "api"
            / "python-library-reference"
            / "client.md"
        )
        .read_text(encoding="utf-8")
        .startswith(sync_fern_docs_branch.MDX_SPDX_HEADER)
    )

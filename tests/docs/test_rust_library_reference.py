# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the Rustdoc-to-MDX generator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from bs4.element import Tag


DOCS_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "docs"
sys.path.insert(0, str(DOCS_SCRIPTS))

import generate_rust_library_reference as rust_reference  # noqa: E402


@pytest.fixture(name="page")
def page_fixture(tmp_path: Path) -> rust_reference.Page:
    return rust_reference.Page(
        html_path=tmp_path / "enum.Example.html",
        output_path=tmp_path / "enum-example.mdx",
        url="/reference/api/rust-library-reference/example/enum-example",
        crate_name="example",
        crate_dir_name="example",
    )


def test_collapsed_declaration_omits_rustdoc_toggle_label(
    page: rust_reference.Page,
):
    soup = BeautifulSoup(
        """
        <main id="main-content">
          <pre>pub enum Example {
          <details class="toggle type-contents-toggle">
            <summary class="hideme"><span>Show 1 variant</span></summary>
            HTTPServer,
          </details>
          }</pre>
        </main>
        """,
        "html.parser",
    )
    content = soup.select_one("#main-content")
    assert isinstance(content, Tag)
    rust_reference._remove_noisy_sections(content)
    declaration = content.find("pre")
    assert isinstance(declaration, Tag)

    rendered = rust_reference._linked_signature_block(declaration, page, {})

    assert "Show 1 variant" not in rendered
    assert "HTTPServer" in rendered


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        (
            "Struct <span class='struct'>HTTP<wbr/>Server<wbr/>Config</span>",
            "Struct HTTPServerConfig",
        ),
        (
            "Constant <span class='constant'>ADAPTER_<wbr/>CONTRACT_<wbr/>VERSION</span>",
            "Constant ADAPTER_CONTRACT_VERSION",
        ),
    ],
)
def test_page_title_preserves_exact_api_identifier(
    page: rust_reference.Page,
    heading: str,
    expected: str,
):
    soup = BeautifulSoup(
        f"""
        <main id="main-content">
          <div class="main-heading">
            <h1>{heading}<button>Copy item path</button></h1>
          </div>
        </main>
        """,
        "html.parser",
    )

    assert rust_reference._page_title(soup, page) == expected


def test_variant_fields_are_nested_below_their_variant(
    page: rust_reference.Page,
):
    soup = BeautifulSoup(
        """
        <main id="main-content">
          <h2 id="variants">Variants</h2>
          <section id="variant.HTTPServer" class="variant">
            <h3 class="code-header">HTTPServer</h3>
          </section>
          <div class="sub-variant" id="variant.HTTPServer.fields">
            <h4>Fields</h4>
            <span
              id="variant.HTTPServer.field.api_url"
              class="section-header"
            ><code>api_url: String</code></span>
          </div>
        </main>
        """,
        "html.parser",
    )
    content = soup.select_one("#main-content")
    assert isinstance(content, Tag)

    rendered = rust_reference._block_markdown(content, page, {})

    variant = rendered.index("### `HTTPServer`")
    fields = rendered.index("#### Fields")
    field = rendered.index("##### `api_url: String`")
    assert variant < fields < field
    assert "\n### `api_url: String`\n" not in rendered


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        (
            "https://doc.rust-lang.org/1.96.1/std/path/struct.PathBuf.html",
            "https://doc.rust-lang.org/stable/std/path/struct.PathBuf.html",
        ),
        (
            "https://doc.rust-lang.org/stable/core/option/enum.Option.html",
            "https://doc.rust-lang.org/stable/core/option/enum.Option.html",
        ),
    ],
)
def test_stdlib_links_use_stable_urls(
    page: rust_reference.Page,
    href: str,
    expected: str,
):
    assert rust_reference._resolve_href(page, href, {}) == expected

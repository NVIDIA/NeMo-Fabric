# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Link checks for the committed generated API reference."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "docs" / "reference" / "api"
MARKDOWN_LINK = re.compile(r"\]\(([^)]+)\)")
HTML_LINK = re.compile(r'href=\\?"([^"\\]+)\\?"')


def test_generated_api_internal_links_are_relative_and_resolve():
    for page in REFERENCE_DIR.rglob("*"):
        if page.suffix not in {".md", ".mdx"}:
            continue
        text = page.read_text(encoding="utf-8")
        targets = MARKDOWN_LINK.findall(text) + HTML_LINK.findall(text)
        for target in targets:
            if target.startswith(("http://", "https://", "#")):
                continue
            path = target.partition("#")[0]
            assert not path.startswith("/"), f"{page}: {target}"
            assert Path(path).suffix in {".md", ".mdx"}, f"{page}: {target}"
            assert (page.parent / path).is_file(), f"{page}: {target}"

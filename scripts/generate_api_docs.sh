#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generate the Python SDK API reference (Markdown) from the SDK docstrings via
# lazydocs, for the Fern docs site. The docstrings in python/src/nemo_fabric are
# the source of truth; run this before publishing docs (and in CI) so the
# reference stays in sync with the code.
#
#   pip install -e ".[docs]"        # provides lazydocs
#   scripts/generate_api_docs.sh
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

out="docs/reference/api/python-library-reference"
rm -rf "$out"
mkdir -p "$out"

PYTHONPATH="python/src" lazydocs \
  --output-path "$out" \
  --overview-file "index.md" \
  "nemo_fabric.client"

# Make the lazydocs output MDX-safe for Fern (Fern parses .md as MDX):
#  - drop source badges (relative links don't resolve on the site)
#  - strip HTML comments (<!-- ... -->), which are invalid in MDX
#  - remove trailing whitespace emitted by lazydocs
perl -ni -e 'print unless m{img\.shields\.io/badge/-source}' "$out"/*.md
perl -0pi -e 's/<!--.*?-->//gs' "$out"/*.md
perl -pi -e 's/[ \t]+$//' "$out"/*.md

# Drop the mkdocs-specific .pages file lazydocs emits; Fern does not use it.
rm -f "$out"/.pages

echo "Generated API reference in $out/"

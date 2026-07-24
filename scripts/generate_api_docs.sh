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
  "nemo_fabric.client" \
  "nemo_fabric.runtime" \
  "nemo_fabric.streaming" \
  "nemo_fabric.models" \
  "nemo_fabric.types" \
  "nemo_fabric.errors"

# Normalize the lazydocs output for Fern:
#  - drop source badges (relative links don't resolve on the site)
#  - strip lazydocs HTML comments before adding the generated SPDX header
#  - remove trailing whitespace emitted by lazydocs
perl -ni -e 'print unless m{img\.shields\.io/badge/-source}' "$out"/*.md
perl -0pi -e 's/<!--.*?-->//gs' "$out"/*.md
perl -pi -e 's/<object object at 0x[0-9A-Fa-f]+>/.../g' "$out"/*.md
perl -pi -e 's/[ \t]+$//' "$out"/*.md
perl -0pi -e 's/\A\s+//' "$out"/*.md
# lazydocs nests properties at h4 directly under h2 class sections. Flatten
# those headings to h3 so generated pages satisfy markdown heading order.
perl -pi -e 's/^#### (<kbd>property<\/kbd>)/### $1/' "$out"/*.md
# lazydocs emits some class headings without a separating blank line.
perl -0pi -e 's/(^## <kbd>class<\/kbd> `(ToolsConfig|RelayAtofFileSinkConfig|RelayAtofStreamSinkConfig)`\n)(?!\n)/$1\n/gm' \
  "$out/nemo_fabric.models.md"
# lazydocs omits the async marker from generated method signatures.
perl -0pi -e 's/(### <kbd>method<\/kbd> `(aclose|result)`\n\n```python\n)\2\(/${1}async def ${2}(/g' \
  "$out/nemo_fabric.streaming.md"

add_frontmatter() {
  local file="$1"
  local title="$2"
  local description="$3"
  local slug="$4"
  local temporary="${file}.tmp"

  {
    printf -- '---\ntitle: "%s"\nslug: "%s"\ndescription: "%s"\n---\n' \
      "$title" "$slug" "$description"
    printf '%s\n' '<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.'
    printf '%s\n\n' 'SPDX-License-Identifier: Apache-2.0 -->'
    command cat "$file"
  } > "$temporary"
  mv "$temporary" "$file"
}

add_frontmatter \
  "$out/index.md" \
  "Python SDK Reference" \
  "Complete reference for the public NeMo Fabric Python SDK." \
  "/reference/api/python-library-reference"
add_frontmatter \
  "$out/nemo_fabric.client.md" \
  "Client" \
  "Resolve, plan, diagnose, and run agents with NeMo Fabric." \
  "/reference/api/python-library-reference/client"
add_frontmatter \
  "$out/nemo_fabric.runtime.md" \
  "Runtime" \
  "Drive stateful multi-turn execution through the Runtime API." \
  "/reference/api/python-library-reference/runtime"
add_frontmatter \
  "$out/nemo_fabric.streaming.md" \
  "Streaming" \
  "Consume Relay-backed raw ATOF records and terminal invocation results." \
  "/reference/api/python-library-reference/streaming"
add_frontmatter \
  "$out/nemo_fabric.models.md" \
  "Models" \
  "Pydantic authoring models for NeMo Fabric config and request inputs." \
  "/reference/api/python-library-reference/models"
add_frontmatter \
  "$out/nemo_fabric.types.md" \
  "Types" \
  "Typed config, request, plan, result, artifact, telemetry, and runtime contracts." \
  "/reference/api/python-library-reference/types"
add_frontmatter \
  "$out/nemo_fabric.errors.md" \
  "Errors" \
  "Structured exception hierarchy for config, capability, state, and runtime failures." \
  "/reference/api/python-library-reference/errors"

# Drop the mkdocs-specific .pages file lazydocs emits; Fern does not use it.
rm -f "$out"/.pages

echo "Generated API reference in $out/"

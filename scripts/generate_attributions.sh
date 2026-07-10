#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate an ATTRIBUTIONS-*.md file from the relevant lockfile.
# Usage: ./scripts/generate_attributions.sh <rust|python>

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LANG="${1:-}"
case "${LANG}" in
  rust)
    if ! command -v cargo-about >/dev/null 2>&1; then
      echo "error: cargo-about not found. Install with: cargo install cargo-about --locked" >&2
      exit 1
    fi
    uv run --no-sync python "${ROOT}/scripts/licensing/attributions_lockfile_md.py" rust
    ;;
  python)
    uv run --no-sync python "${ROOT}/scripts/licensing/attributions_lockfile_md.py" python
    ;;
  *)
    echo "Usage: $0 <rust|python>" >&2
    exit 1
    ;;
esac

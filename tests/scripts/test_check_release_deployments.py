# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = (
    REPO_ROOT
    / ".agents/skills/check-release-deployments/scripts/check_release_deployments.sh"
)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_nightly_tag_selects_its_nightly_workflow_run(tmp_path: Path):
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    (tmp_path / "crates").mkdir()
    gh_log = tmp_path / "gh-run-list.log"

    _write_executable(
        mock_bin / "git",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "rev-parse --show-toplevel" ]]; then
    printf '%s\n' "$MOCK_REPO_ROOT"
elif [[ "$*" == "ls-files" ]]; then
    exit 0
else
    printf 'Unexpected git arguments: %s\n' "$*" >&2
    exit 1
fi
""",
    )
    _write_executable(
        mock_bin / "just",
        """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
    normalize-release-tag) printf '0.4.0-alpha.20260915\n' ;;
    release-tag-to-py-version) printf '0.4.0a20260915\n' ;;
    python-package-paths) exit 0 ;;
    *) printf 'Unexpected just arguments: %s\n' "$*" >&2; exit 1 ;;
esac
""",
    )
    _write_executable(
        mock_bin / "gh",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "api" && "$2" == */git/ref/tags/* ]]; then
    exit 0
elif [[ "$1" == "api" && "$2" == */commits/* ]]; then
    printf 'nightly-sha\n'
elif [[ "$1 $2" == "run list" ]]; then
    printf '%s\n' "$*" > "$MOCK_GH_LOG"
    if [[ " $* " == *" --workflow nightly-alpha-tag.yml "* &&
          " $* " == *" --created 2026-09-15 "* ]]; then
        printf '%s\n' '[{"databaseId":42,"workflowName":"Create nightly alpha tag","status":"completed","conclusion":"success","url":"https://example.test/nightly"}]'
    else
        printf '%s\n' '[{"databaseId":99,"workflowName":"Publish Rust crates","status":"completed","conclusion":"failure","url":"https://example.test/unrelated"}]'
    fi
else
    printf 'Unexpected gh arguments: %s\n' "$*" >&2
    exit 1
fi
""",
    )
    _write_executable(mock_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")

    environment = dict(os.environ)
    environment.update(
        {
            "PATH": f"{mock_bin}:{environment['PATH']}",
            "MOCK_GH_LOG": str(gh_log),
            "MOCK_REPO_ROOT": str(tmp_path),
        }
    )
    result = subprocess.run(
        ["bash", str(CHECKER), "v0.4.0-alpha.20260915"],
        check=True,
        capture_output=True,
        encoding="utf-8",
        env=environment,
    )

    arguments = gh_log.read_text(encoding="utf-8")
    assert "--commit nightly-sha" in arguments
    assert "--workflow nightly-alpha-tag.yml" in arguments
    assert "--created 2026-09-15" in arguments
    assert "--event push" not in arguments
    assert "Create nightly alpha tag" in result.stdout
    assert "Publish Rust crates" not in result.stdout

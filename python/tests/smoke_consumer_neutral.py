# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: the SDK core stays consumer-neutral and dependency-free.

WS4 guardrail. The public SDK core -- the ``nemo_fabric`` package outside the
``integrations`` subpackage -- must not depend on any third-party package: no
harness (Hermes), no consumer (Harbor/Platform), no telemetry backend (Relay).
Adapters are loaded dynamically at runtime via importlib, and consumer glue
lives under ``nemo_fabric.integrations`` behind an optional extra; the core
never imports any of them.

Two checks:

1. Static -- every top-level import in the core resolves to the standard library
   or ``nemo_fabric`` itself. This also pins the SDK's zero-dependency contract
   (pyproject ``dependencies = []``).
2. Runtime -- a plain ``import nemo_fabric`` (what a consumer like Platform does)
   pulls in no consumer/harness package.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parents[2] / "python" / "src" / "nemo_fabric"
PYPROJECT = Path(__file__).resolve().parents[2] / "python" / "pyproject.toml"
ALLOWED = set(sys.stdlib_module_names) | {"nemo_fabric", "__future__"}
# Consumer/harness packages that must never leak into a plain ``import nemo_fabric``.
CONSUMER_SPECIFIC = [
    "harbor",
    "hermes",
    "relay",
    "nemo_relay",
    "nemo_fabric_adapters",
    "nemo_fabric_test_adapters",
]


def _top_level_imports(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            # node.level == 0 -> absolute import; relative imports stay in-package.
            roots.add(node.module.split(".")[0])
    return roots


def core_imports_only_stdlib_and_self() -> None:
    """Static: no third-party import anywhere in the core (non-integrations)."""

    core = sorted(
        path
        for path in SDK_ROOT.rglob("*.py")
        if "integrations" not in path.relative_to(SDK_ROOT).parts
    )
    assert core, f"no core SDK sources found under {SDK_ROOT}"

    offenders: dict[str, list[str]] = {}
    for path in core:
        bad = sorted(
            root
            for root in _top_level_imports(ast.parse(path.read_text()))
            if root not in ALLOWED
        )
        if bad:
            offenders[path.name] = bad

    assert not offenders, (
        "SDK core must import only the standard library and nemo_fabric "
        "(consumer glue belongs under nemo_fabric.integrations); "
        f"found third-party imports: {offenders}"
    )

    # Pin the declared contract too: a dependency added to pyproject would slip
    # past the import scan above if nothing in the core imports it yet. Assert the
    # structure explicitly so a missing [project]/dependencies fails loudly
    # instead of defaulting to an empty list and hiding a metadata regression.
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert "project" in pyproject, f"{PYPROJECT} is missing the [project] table"
    project = pyproject["project"]
    assert "dependencies" in project, f"{PYPROJECT} [project] is missing 'dependencies'"
    deps = project["dependencies"]
    assert deps == [], f"SDK must stay zero-dependency; found dependencies={deps!r}"


def importing_the_sdk_pulls_in_no_consumer_package() -> None:
    """Runtime: ``import nemo_fabric`` does not drag in a consumer/harness dep."""

    probe = (
        "import importlib, sys\n"
        "importlib.import_module('nemo_fabric')\n"
        f"forbidden = {CONSUMER_SPECIFIC!r}\n"
        "roots = {name.split('.')[0] for name in sys.modules}\n"
        "print(','.join(sorted(roots & set(forbidden))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    leaked = result.stdout.strip()
    assert not leaked, f"`import nemo_fabric` leaked consumer packages: {leaked}"


def main() -> None:
    core_imports_only_stdlib_and_self()
    importing_the_sdk_pulls_in_no_consumer_package()
    print("smoke_consumer_neutral ok")


if __name__ == "__main__":
    main()

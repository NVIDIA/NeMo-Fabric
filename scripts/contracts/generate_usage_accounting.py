# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embed the canonical usage-accounting schema in declaring adapter descriptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEY = "nemo_fabric.usage_accounting.v1"


def synchronize(*, check: bool) -> None:
    schema = json.loads(
        (ROOT / "schemas/extensions/usage-accounting.schema.json").read_text()
    )
    definitions = schema.pop("$defs", {})
    schema.pop("$schema", None)
    descriptors = [
        *ROOT.glob("adapters/python/*/*.fabric-adapter.json"),
        *ROOT.glob("adapters/typescript/*/*.fabric-adapter.json"),
        *ROOT.glob("crates/fabric-cli/assets/adapters/*/*.fabric-adapter.json"),
    ]
    for path in descriptors:
        original = path.read_text()
        document = json.loads(original)
        extensions = document.get("extension_schemas", {})
        usage = extensions.get("usage", {})
        if KEY not in usage.get("properties", {}):
            continue
        # This generator owns the accounting definitions at the usage-schema root.
        # Replace them so a removed canonical definition cannot survive in a descriptor.
        usage["$defs"] = definitions
        usage["properties"][KEY] = schema
        # Replace just this object; retain unrelated descriptor formatting.
        start = original.index('"extension_schemas":') + len('"extension_schemas":')
        start += len(original[start:]) - len(original[start:].lstrip())
        _, end = json.JSONDecoder().raw_decode(original[start:])
        replacement = json.dumps(extensions, indent=2).replace("\n", "\n  ")
        updated = original[:start] + replacement + original[start + end :]
        if original != updated:
            if check:
                raise SystemExit(f"accounting schema drift: {path.relative_to(ROOT)}")
            path.write_text(updated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    synchronize(check=parser.parse_args().check)

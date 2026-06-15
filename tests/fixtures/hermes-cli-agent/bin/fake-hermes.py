#!/usr/bin/env python3
"""Fake Hermes CLI used by Fabric smoke tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    args = sys.argv[1:]
    if "-z" not in args:
        print("fake hermes expected -z", file=sys.stderr)
        return 2
    prompt = args[args.index("-z") + 1]
    hermes_home = os.environ.get("HERMES_HOME", "")
    config_path = Path(hermes_home) / "config.yaml"
    if not config_path.is_file():
        print(f"missing config: {config_path}", file=sys.stderr)
        return 3
    result = {
        "fake_hermes": True,
        "prompt": prompt,
        "argv": args,
        "hermes_home": hermes_home,
        "has_config": True,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

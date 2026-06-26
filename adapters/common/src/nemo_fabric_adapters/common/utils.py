# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared adapter utility helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def write_relay_plugins_toml(plugin_config: dict[str, Any]) -> Path | None:
    try:
        import tomli_w

        config_path = os.environ.get("FABRIC_RELAY_CONFIG_PATH")
        if not config_path:
            raise RuntimeError("FABRIC_RELAY_CONFIG_PATH is required when Relay is enabled")

        path = Path(config_path).with_name("relay-plugins.toml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomli_w.dumps(plugin_config), encoding="utf-8")
        return path
    except ImportError:
        print("tomli_w is not installed, skipping writing relay plugins TOML", file=sys.stderr)
        return None

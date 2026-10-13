# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys


def test_windows_import_without_fcntl_and_deliberate_unsupported_command():
    code = """
import builtins, sys
from nemo_fabric import Fabric
sys.platform = "win32"
original = builtins.__import__
def import_module(name, *args, **kwargs):
    if name == "fcntl":
        raise ModuleNotFoundError("Windows does not provide fcntl")
    return original(name, *args, **kwargs)
builtins.__import__ = import_module
from nemo_fabric.runtime_server import cli
assert cli.main(["serve"]) == 1
raise SystemExit(cli.main(["stop", "--agent", "review", "--runtime-id", "runtime-1"]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 1
    assert len(result.stdout.splitlines()) == 1, result.stderr
    response = json.loads(result.stdout)
    assert response["status"] == "unsupported"
    assert response["error"]["code"] == "platform_unsupported"
    assert response["error"]["effects"] == "none"
    assert "requires a Unix platform" in result.stderr

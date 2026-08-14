# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_missing_harbor_dependency_reports_python_requirement():
    probe = textwrap.dedent(
        """
        import sys

        sys.modules["harbor"] = None
        from nemo_fabric.integrations.harbor import FabricAgent

        try:
            FabricAgent()
        except ModuleNotFoundError as error:
            print(error)
        else:
            raise SystemExit("FabricAgent did not report the missing Harbor dependency")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "requires Python 3.12 or later" in result.stdout
    assert "install nemo-fabric[harbor]" in result.stdout

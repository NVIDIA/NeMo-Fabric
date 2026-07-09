#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic adapter used by the credential-free Harbor example."""

import json
import sys
from pathlib import Path

payload = json.load(sys.stdin)
calculator = Path("/app/calculator.py")
calculator.write_text(
    "def add(a, b):\n"
    "    return a + b\n\n\n"
    "def multiply(a, b):\n"
    "    return a * b\n",
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "harness": "scripted",
            "response": "Fixed multiply(a, b) in /app/calculator.py",
            "request_id": payload["request"]["request_id"],
        }
    )
)

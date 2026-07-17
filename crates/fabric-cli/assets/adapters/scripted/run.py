#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic process adapter for credential-free CLI experiments."""

import json
import sys


payload = json.load(sys.stdin)
request = payload["request"]
print(
    json.dumps(
        {
            "response": request.get("input"),
            "request_id": request["request_id"],
        }
    )
)

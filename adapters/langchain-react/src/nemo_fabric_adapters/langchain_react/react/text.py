# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re


def remove_r1_think_tags(text: str) -> str:
    pattern = r"(<think>)?.*?</think>\s*(.*)"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(2)
    return text

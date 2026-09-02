# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic task usage for the BenchAgent subprocess fixture."""


def start_task_tokens() -> None:
    return None


def get_task_tokens() -> dict[str, int]:
    return {"n_input_tokens": 12, "n_output_tokens": 4}

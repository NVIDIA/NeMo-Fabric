# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

def version() -> str: ...
def plan_config(
    config_json: str,
    base_dir: str | None = None,
) -> str: ...
def doctor_config(
    config_json: str,
    base_dir: str | None = None,
) -> str: ...
def run_config(
    config_json: str,
    base_dir: str | None = None,
    input_text: str | None = None,
    input_file: str | None = None,
    request_json: str | None = None,
    request_file: str | None = None,
) -> str: ...
def start_runtime(plan_json: str) -> str: ...
def invoke_runtime(
    plan_json: str,
    runtime_json: str,
    request_json: str,
) -> str: ...
def stop_runtime(plan_json: str, runtime_json: str) -> str: ...

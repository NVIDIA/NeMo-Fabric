# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

path = Path("/app/calculator.py")
spec = importlib.util.spec_from_file_location("calculator", path)
assert spec and spec.loader
calculator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calculator)

assert calculator.add(2, 3) == 5
assert calculator.multiply(4, 3) == 12
assert calculator.multiply(-4, 3) == -12
assert calculator.multiply(0, 99) == 0

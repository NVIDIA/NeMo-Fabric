# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone collector for NVIDIA NeMo Fabric ATOF records."""

from nemo_fabric_collector.app import create_app
from nemo_fabric_collector.server import serve_collector

__all__ = ["create_app", "serve_collector"]

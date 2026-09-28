<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adapted Source

The Brave search MCP server in `brave_search.py` and its tests were adapted from
NVIDIA NemoClaw revision `9146224da4`: `image/fabric/brave_search.py` and
`test_brave_search.py`. The original NVIDIA Apache-2.0 notices are retained.

Modified on 2026-09-24: the server is packaged with this adapter, bounds each
search by a total deadline, and reports only expected search failures without
their details.

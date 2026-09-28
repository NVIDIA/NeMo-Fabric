<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adapted Source

The API server mode in `api_server.py`, the Tavily plugin in
`plugins/tavily/`, and their tests were adapted from NVIDIA NemoClaw revision
`9146224da4`: `image/fabric/hermes_adapter.py`, `hermes_tavily.py`,
`hermes-tavily.plugin.yaml`, and `test_hermes_tavily.py`. The original NVIDIA
Apache-2.0 notices are retained.

Modified on 2026-09-24: the code reads the NVIDIA NeMo Fabric `AgentConfig`
instead of NemoClaw's inference format, keeps native state in an explicit or
runtime-scoped directory, and leaves deployment lifecycle to the consumer.

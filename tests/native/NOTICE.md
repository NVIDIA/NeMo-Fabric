<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Source Notices

`hermes_security.py` derives from NVIDIA NemoClaw `test/hermes_native.py` at
revision `9146224da4`. On 2026-09-24 it moved to the NVIDIA NeMo Fabric native
qualification suite without changes to its Apache-2.0 notices.

`qualify.py` is original NeMo Fabric test code. Both scripts run only against
local processes that they own. The qualification script runs the installed
harness against a local simulated inference endpoint, so it does not establish
compatibility with an external provider.

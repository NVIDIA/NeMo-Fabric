<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adapted Source

The retained interface credential in `credentials.py` was adapted from NVIDIA
NemoClaw revision `9146224da4`: `image/fabric/interfaces.py`. The original
NVIDIA Apache-2.0 notices are retained.

Modified on 2026-09-24: adapters that expose native interfaces share the
credential, which is now published atomically and read without following
links or blocking on non-regular files.

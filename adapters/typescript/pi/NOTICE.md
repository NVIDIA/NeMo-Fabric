<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Adapted Source

The model-role loader in `src/pi-model.ts` and its tests in
`test/model-roles.test.mjs` were adapted from NVIDIA NemoClaw revision
`9146224da4`: `image/fabric/pi-model.ts` and `test_pi_model.mts`. The original
NVIDIA Apache-2.0 notices are retained.

Modified on 2026-09-24: the loader reads NVIDIA NeMo Fabric model roles
directly, keeps catalog models and the selected role on their Pi providers for
NeMo Relay, and switches roles through the adapter runtime.

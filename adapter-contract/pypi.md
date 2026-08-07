<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Adapter Contract

`nemo-fabric-adapter-contract` provides the typed Python configuration and
execution contract implemented by NeMo Fabric adapters. It does not include a
lifecycle host, harness integration, or NeMo Relay integration.

Python adapters can use this package for Pydantic validation. Adapters in other
languages can consume the JSON Schemas published by NeMo Fabric without a
Python package dependency.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# External Adapter References

This directory contains source-only reference adapters that exercise the public
NVIDIA NeMo Fabric adapter contract without becoming bundled NeMo Fabric
adapters. Each reference owns its descriptor, implementation, examples, and
documentation.

These adapters are not published as wheels and are not wired into bundled or
installed-adapter discovery. Packaging and discovery are separate concerns from
the adapter contract demonstrated here.

The following source-only reference adapter is available:

| Harness | Adapter ID | Reference |
| --- | --- | --- |
| NVIDIA NeMo Agent Toolkit | `nvidia.fabric.nat` | [NeMo Agent Toolkit adapter](nat/README.md) |

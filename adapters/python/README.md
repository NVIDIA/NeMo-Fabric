<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Python Adapter Packages

This directory contains the independently published Python adapter packages
and their shared runtime utilities. The `python` directory is a source-layout
boundary only: published package names, Python import paths, adapter IDs, and
installed descriptor paths do not include it.

Refer to the [adapter catalog](../README.md) for supported harnesses and their
capabilities. Each child package owns its `pyproject.toml`, `uv.lock`, package
README, adapter descriptor when applicable, and a license link to the
repository license. The `common` package provides shared first-party lifecycle
and Relay utilities; it is not an adapter descriptor.

## Build and Test

Run the following commands from the repository root:

```bash
just build-python
just test-python
just wheels
just check-wheel-licenses
```

Source packages live under `adapters/python/<name>`. Adapter wheels continue to
install descriptors under `share/nemo-fabric/adapters/<name>` so discovery is
stable across editable and packaged installations.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Environment Providers for NVIDIA NeMo Fabric

Environment providers connect Fabric's normalized environment and runtime
lifecycle to an execution backend. Providers own backend-specific connection,
authentication, resource verification, and execution transport behavior.

Fabric core remains provider-neutral. An environment provider does not replace
the Fabric adapter contract or make Fabric responsible for scheduling an
environment fleet.

Available providers:

- [OpenShell](openshell/README.md) provides experimental support for running
  Fabric adapters and custom agents in OpenShell sandboxes.

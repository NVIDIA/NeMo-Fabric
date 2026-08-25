<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric TypeScript Adapter Workspace

This private npm workspace coordinates the independently published common and
Pi adapter packages. It provides shared build, test, package-content, and
consumer-install checks without becoming a published package itself.

## Build and Test

Run the following commands from the repository root:

```bash
just build-typescript
just test-typescript-adapters
```

The adapter test recipe builds the local TypeScript contract before it compiles
the adapter packages.

To install only the Pi adapter workspace and its exact-pinned development
harness, run:

```bash
just install-typescript-pi
```

## Dependency Rationale

The workspace links `nemo-fabric-adapter-contract` from the checked-out source
tree so all adapter builds and tests use the contract being reviewed. Resolving
the package from the registry would test an older published contract, while
copying its schemas would create a second authority. This file dependency is
development-only; the published child manifests declare registry-safe contract
versions.

Harness SDKs are optional peers of their published adapters so consumers can
select a compatible harness version. Each adapter exact-pins its tested harness
under `devDependencies`; source installs and CI therefore remain reproducible
without making the harness a production dependency.

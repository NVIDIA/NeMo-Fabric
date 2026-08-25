#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Process runner referenced by the adapter descriptor and npm binary. It
// validates the Node.js host, then composes the shared lifecycle transport with
// the Pi runtime while deferring SDK loading until startup.

import { createRequire } from "node:module";

import { serve } from "nemo-fabric-adapters-common";

import { assertSupportedNodeVersion } from "./node-version.js";

const manifest = createRequire(import.meta.url)("../package.json") as {
  engines?: { node?: unknown };
};
assertSupportedNodeVersion(process.versions.node, manifest.engines?.node);

await serve(async () => {
  const [{ PiSdkSessionFactory }, { PiAdapterRuntime }] = await Promise.all([
    import("./pi-sdk.js"),
    import("./runtime.js"),
  ]);
  return new PiAdapterRuntime(new PiSdkSessionFactory());
});

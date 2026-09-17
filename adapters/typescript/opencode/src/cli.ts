#!/usr/bin/env bun
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Process runner referenced by the adapter descriptor and npm binary. SDK
// loading is deferred until start so descriptor discovery and protocol errors
// never require the OpenCode harness to be installed.

import { createRequire } from "node:module";

import { serve } from "nemo-fabric-adapters-common";

import { assertSupportedBunVersion } from "./bun-version.js";

const manifest = createRequire(import.meta.url)("../package.json") as {
  engines?: { bun?: unknown };
};

await serve(async () => {
  assertSupportedBunVersion(process.versions.bun, manifest.engines?.bun);
  const [{ OpenCodeSdkSessionFactory }, { OpenCodeAdapterRuntime }] = await Promise.all([
    import("./opencode-sdk.js"),
    import("./runtime.js"),
  ]);
  return new OpenCodeAdapterRuntime(new OpenCodeSdkSessionFactory());
});

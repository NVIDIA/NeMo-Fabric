#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { serve } from "nemo-fabric-adapters-common";

await serve(async () => {
  const [{ DroidSdkSessionFactory }, { DroidAdapterRuntime }] = await Promise.all([
    import("./droid-sdk.js"),
    import("./runtime.js"),
  ]);
  return new DroidAdapterRuntime(new DroidSdkSessionFactory());
}, { software: { packageJson: new URL("../package.json", import.meta.url), harness: "droid", packages: [{"role": "framework", "name": "@agentclientprotocol/sdk"}] } });

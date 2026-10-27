#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { serve } from "nemo-fabric-adapters-common";

await serve(async () => {
  const [{KiloSdkSessionFactory}, {KiloAdapterRuntime}] = await Promise.all([
    import("./kilo-sdk.js"), import("./runtime.js"),
  ]);
  return new KiloAdapterRuntime(new KiloSdkSessionFactory());
}, { software: { packageJson: new URL("../package.json", import.meta.url), harness: "kilo", packages: [{"role": "framework", "name": "@agentclientprotocol/sdk"}] } });

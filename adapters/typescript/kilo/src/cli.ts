#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { serve, withPackageVersion } from "nemo-fabric-adapters-common";

await serve(async () => {
  const [{KiloSdkSessionFactory}, {KiloAdapterRuntime}] = await Promise.all([
    import("./kilo-sdk.js"), import("./runtime.js"),
  ]);
  return withPackageVersion(new KiloAdapterRuntime(new KiloSdkSessionFactory()), new URL("../package.json", import.meta.url));
});

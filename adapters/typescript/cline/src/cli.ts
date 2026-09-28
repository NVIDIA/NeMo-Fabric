#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { serve } from "nemo-fabric-adapters-common";

await serve(async () => {
  const [{ ClineSdkSessionFactory }, { ClineAdapterRuntime }] = await Promise.all([
    import("./cline-sdk.js"),
    import("./runtime.js"),
  ]);
  return new ClineAdapterRuntime(new ClineSdkSessionFactory());
});

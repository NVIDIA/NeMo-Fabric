#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { serve } from "nemo-fabric-adapters-common";

const [major, minor] = process.versions.node.split(".").map(Number);
if (major === undefined || minor === undefined || major < 22 || (major === 22 && minor < 19)) {
  throw new Error("The Qwen adapter requires Node.js >=22.19.0");
}

await serve(async () => {
  const [{ QwenSdkSessionFactory }, { QwenAdapterRuntime }] = await Promise.all([
    import("./qwen-sdk.js"),
    import("./runtime.js"),
  ]);
  return new QwenAdapterRuntime(new QwenSdkSessionFactory());
});

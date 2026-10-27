#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { serve } from "nemo-fabric-adapters-common";

await serve(async () => {
  const [{ StrandsAcpSessionFactory }, { StrandsAdapterRuntime }] =
    await Promise.all([import("./strands-acp.js"), import("./runtime.js")]);
  return new StrandsAdapterRuntime(new StrandsAcpSessionFactory());
}, { software: { packageJson: new URL("../package.json", import.meta.url), harness: "strands", packages: [{"role": "framework", "name": "@agentclientprotocol/sdk"}] } });

#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { serve } from "nemo-fabric-adapters-common";

await serve(async () => {
  const [{ DeepAgentsCodeAcpSessionFactory }, { DeepAgentsCodeAdapterRuntime }] = await Promise.all([
    import("./deepagents-code-acp.js"),
    import("./runtime.js"),
  ]);
  return new DeepAgentsCodeAdapterRuntime(new DeepAgentsCodeAcpSessionFactory());
}, { software: { packageJson: new URL("../package.json", import.meta.url), harness: "deepagents-code", packages: [{"role": "framework", "name": "@agentclientprotocol/sdk"}] } });

#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { missingFact, serve, withPackageVersion } from "nemo-fabric-adapters-common";

await serve(async () => {
  const [{ ClineSdkSessionFactory, selectedModelRole }, { ClineAdapterRuntime }] = await Promise.all([
    import("./cline-sdk.js"),
    import("./runtime.js"),
  ]);
  // The SDK's result names a model, but nothing yet shows that it is the model
  // that served this invocation, so the selected role reads not_attributed.
  return withPackageVersion(
    new ClineAdapterRuntime(new ClineSdkSessionFactory()),
    new URL("../package.json", import.meta.url),
    (input) => {
      const role = selectedModelRole(input.config);
      return role === undefined
        ? undefined
        : { role, fact: missingFact("unavailable", "not_attributed", "harness_api") };
    },
  );
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { auditDependencies } from "../../../scripts/ci/audit_typescript_dependencies.mjs";

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  auditDependencies(
    dirname(dirname(fileURLToPath(import.meta.url))),
    "contract",
  );
}

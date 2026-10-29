// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ExecutionEvidence, SoftwareEntry, SoftwareEnvironment, VersionObservation } from "../src/index.js";
const observation: VersionObservation = {
  status: "available", value: "1.0.0-rc.1+full-build", source: "adapter:package.json",
  origin: "observed", observed_at: "2026-10-27T12:00:00Z",
};
// Named arrays must compose with inventories, as they do in real collectors.
const entries: SoftwareEntry[] = [{ role: "adapter", kind: "npm_package", name: "example", version: observation }];
const report: ExecutionEvidence = { software_environments: [{
  context_ref: "env-0", participants: ["adapter"], status: "available", complete: true,
  entries,
}] };
void report;

// @ts-expect-error Available observations require a value.
const missingValue: VersionObservation = { status: "available", source: "owner", origin: "reported", observed_at: "2026-10-27T12:00:00Z" };
// @ts-expect-error Missing observations require a reason.
const missingReason: VersionObservation = { status: "unavailable", source: "owner", origin: "reported", observed_at: "2026-10-27T12:00:00Z" };
// @ts-expect-error Incomplete inventories require a reason.
const partial: SoftwareEnvironment = { context_ref: "env-0", participants: ["adapter"], status: "available", complete: false, entries };
void [missingValue, missingReason, partial];

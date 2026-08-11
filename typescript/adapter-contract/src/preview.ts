// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentRunResult } from "./generated/agent-run-result.js";

export type * from "./generated/agent-run-request.js";
export type * from "./generated/agent-run-result.js";
export type { JsonObject, JsonValue } from "./json.js";

/** Completion status reported by an adapter target. */
export type AgentRunStatus = AgentRunResult["status"];

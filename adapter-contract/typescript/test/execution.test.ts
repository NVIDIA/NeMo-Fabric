// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  AgentRunError,
  AgentRunRequest,
  AgentRunResult,
  AgentRunStatus,
} from "../src/index.js";

const requests: AgentRunRequest[] = [
  { input: null },
  { input: true },
  { input: 42 },
  { input: "prompt" },
  { input: ["prompt", null] },
  { input: { messages: [{ role: "user", content: "hello" }] } },
];

const results: AgentRunResult[] = [
  { output: null, status: "succeeded" },
  {
    error: { code: "target_error", message: "target failed" },
    output: { partial: true },
    status: "failed",
  },
  { error: null, output: ["partial"], status: "cancelled" },
];
const status: AgentRunStatus = "succeeded";
const runError: AgentRunError = { code: "example", message: "example" };

void requests;
void results;
void status;
void runError;

// @ts-expect-error successful results cannot include an error
const invalidSuccess: AgentRunResult = {
  error: { code: "unexpected", message: "unexpected" },
  output: "done",
  status: "succeeded",
};
void invalidSuccess;

// @ts-expect-error failed results require a non-null error
const missingFailure: AgentRunResult = { output: null, status: "failed" };
void missingFailure;

// @ts-expect-error failed results cannot carry a null error
const nullFailure: AgentRunResult = {
  error: null,
  output: null,
  status: "failed",
};
void nullFailure;

// @ts-expect-error functions are not JSON values
const invalidRequest: AgentRunRequest = { input: () => "not JSON" };
void invalidRequest;

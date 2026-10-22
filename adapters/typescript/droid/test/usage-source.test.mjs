// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

const source = process.env.NEMO_FABRIC_TEST_DROID_SDK;

test("pinned Droid SDK: a mixed omission has the same positive aggregate as a full call", { skip: !source }, async () => {
  const sdk = await import(pathToFileURL(source).href);
  const complete = sdk.sumTokenUsage([{ inputTokens: 40, outputTokens: 8 }]);
  const incomplete = sdk.sumTokenUsage([{ inputTokens: 40 }, { outputTokens: 8 }]);
  assert.deepEqual(complete, incomplete);
  assert.equal(complete.inputTokens, 40);
  assert.equal(complete.outputTokens, 8);
  assert.equal(complete.cacheReadTokens, 0);
});

test("pinned Droid SDK: explicit terminal usage overrides a cumulative update, omission falls back", { skip: !source }, async () => {
  const sdk = await import(pathToFileURL(source).href);
  const tracker = new sdk.StreamStateTracker({ sessionId: "session", startedAt: 1 });
  tracker.processMessage({ type: "token_usage_update", inputTokens: 1000, outputTokens: 200,
    cacheReadTokens: 0, cacheCreationTokens: 0, thinkingTokens: 0 });
  const explicit = { inputTokens: 40, outputTokens: 8, cacheReadTokens: 0, cacheCreationTokens: 0, thinkingTokens: 0 };
  assert.equal(tracker.completeTurn({ reason: "completed", tokenUsage: explicit, durationMs: 1 }).tokenUsage, explicit);
  const fallback = tracker.completeTurn({ reason: "completed", durationMs: 1 });
  assert.equal(fallback.tokenUsage.inputTokens, 1000);
});

test("pinned Droid SDK: usage notifications expose no stable model-call identity", { skip: !source }, async () => {
  const sdk = await import(pathToFileURL(source).href);
  const notification = sdk.SessionTokenUsageChangedNotificationSchema.parse({
    type: "session_token_usage_changed", sessionId: "session",
    tokenUsage: { inputTokens: 40, outputTokens: 8, cacheReadTokens: 0, cacheCreationTokens: 0, thinkingTokens: 0 },
    lastCallTokenUsage: { inputTokens: 40, cacheReadTokens: 0, outputTokens: 8 },
  });
  assert.equal(notification.requestId, undefined);
  assert.equal(notification.messageId, undefined);
  const update = sdk.convertNotificationToStreamMessage(notification);
  assert.equal(update.inputTokens, 40);
  assert.equal(update.outputTokens, 8);
});

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { AgentSession } from "@earendil-works/pi-coding-agent";
import type { AgentUsage } from "nemo-fabric-adapter-contract";

type Entries = ReturnType<AgentSession["sessionManager"]["getEntries"]>;

// Pi 1.0.3 keeps retries and compacted/other-branch messages in its raw ledger.
// Its projected conversation and final assistant message omit billable calls.
export function invocationUsage(
  before: readonly string[],
  entries: Entries,
): AgentUsage | undefined {
  if (
    entries.length < before.length ||
    before.some((id, index) => entries[index]?.id !== id)
  )
    return undefined;
  let input = 0;
  let output = 0;
  let cacheRead = 0;
  let cacheWrite = 0;
  let reasoning = 0;
  let hasReasoning = true;
  let count = 0;
  for (const entry of entries.slice(before.length)) {
    let usage;
    if (
      entry.type === "usage" ||
      entry.type === "compaction" ||
      entry.type === "branch_summary"
    ) {
      usage = entry.usage;
      if (usage === undefined) return undefined;
    } else if (entry.type === "message" && entry.message.role === "assistant") {
      usage = entry.message.usage;
    } else if (
      entry.type === "message" &&
      entry.message.role === "toolResult"
    ) {
      // Ordinary tools need no tokens. Pi attaches nested model usage when present.
      usage = entry.message.usage;
      if (usage === undefined) continue;
    } else {
      continue;
    }
    if (usage === undefined) return undefined;
    const counters = [
      usage.input,
      usage.output,
      usage.cacheRead,
      usage.cacheWrite,
      usage.totalTokens,
    ];
    if (counters.some((value) => !Number.isSafeInteger(value) || value < 0))
      return undefined;
    const total =
      usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
    // Pi initializes missing provider usage to all zeros. It cannot distinguish
    // a genuine all-zero response from an unreported call; neither proves totals.
    if (total === 0 || total !== usage.totalTokens) return undefined;
    if (
      usage.reasoning !== undefined &&
      (!Number.isSafeInteger(usage.reasoning) ||
        usage.reasoning < 0 ||
        usage.reasoning > usage.output)
    )
      return undefined;
    input += usage.input;
    output += usage.output;
    cacheRead += usage.cacheRead;
    cacheWrite += usage.cacheWrite;
    if (usage.reasoning === undefined) hasReasoning = false;
    else reasoning += usage.reasoning;
    count += 1;
  }
  if (count === 0) return undefined;
  // Pi's input excludes reads AND writes. Its output already includes reasoning.
  input += cacheRead + cacheWrite;
  const total = input + output;
  if (
    ![input, output, cacheRead, cacheWrite, total].every(Number.isSafeInteger)
  )
    return undefined;
  return {
    input_tokens: input,
    output_tokens: output,
    cached_input_tokens: cacheRead,
    input_tokens_include_cache: true,
    total_tokens: total,
    // SDK cost is computed using configured/catalog rates, not provider billing.
    extensions: {
      "nemo_fabric.detailed_token_usage.v1": {
        input_tokens: input,
        output_tokens: output,
        cached_input_tokens: cacheRead,
        cache_write_input_tokens: cacheWrite,
        input_tokens_include_cache: true,
        output_tokens_include_reasoning: true,
        ...(hasReasoning ? { reasoning_tokens: reasoning } : {}),
        total_tokens: total,
      },
    },
  };
}

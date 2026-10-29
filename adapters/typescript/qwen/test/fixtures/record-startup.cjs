// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { appendFileSync } = require("node:fs");
const { getHeapStatistics } = require("node:v8");

appendFileSync(process.env.FABRIC_QWEN_STARTUP_LOG, `${JSON.stringify({
  pid: process.pid,
  heap_size_limit: getHeapStatistics().heap_size_limit,
})}\n`);

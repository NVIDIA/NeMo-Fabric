# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal NOOA middleware callbacks for the Relay subprocess fixture."""


async def nemo_relay_agent_call_middleware(ctx, nxt):
    return await nxt(ctx)


async def nemo_relay_llm_middleware(ctx, nxt):
    return await nxt(ctx)

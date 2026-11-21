# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter starter: the target SDK is in acme_sdk.py; Fabric mapping is absent."""


class AcmeRuntime:
    async def start(self, payload):
        raise NotImplementedError("Map validated AgentConfig into AcmeSession options")

    async def invoke(self, request, context):
        raise NotImplementedError("Return a normalized AgentRunResult")

    async def stop(self):
        raise NotImplementedError(
            "Close the target session after partial or full startup"
        )

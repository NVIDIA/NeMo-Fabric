# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixture Hermes plugin: one tool that returns a fixed marker."""

import json


def _handler(args, **_kwargs):
    return json.dumps({"result": f"HERMES-MOUNTED-OK:{args.get('word', '')}"})


def register(ctx):
    ctx.register_tool(
        name="hermes_mounted",
        toolset="hermes_mounted",
        schema={
            "name": "hermes_mounted",
            "description": "Return the hermes-mounted marker for a word",
            "parameters": {
                "type": "object",
                "properties": {"word": {"type": "string"}},
                "required": ["word"],
            },
        },
        handler=_handler,
    )

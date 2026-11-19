# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixture Hermes plugin: one tool that returns a fixed marker."""

import json


def _handler(args, **_kwargs):
    return json.dumps({"result": f"HERMES-INSTALLED-OK:{args.get('word', '')}"})


def register(ctx):
    ctx.register_tool(
        name="hermes_installed",
        toolset="hermes_installed",
        schema={
            "name": "hermes_installed",
            "description": "Return the hermes-installed marker for a word",
            "parameters": {
                "type": "object",
                "properties": {"word": {"type": "string"}},
                "required": ["word"],
            },
        },
        handler=_handler,
    )

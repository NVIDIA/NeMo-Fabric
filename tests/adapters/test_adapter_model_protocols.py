# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each adapter's model_schema admits exactly the protocols it speaks."""

import json

import jsonschema
import pytest


def _model(provider, api, model="m", **settings):
    return {"provider": provider, "model": model, "api": api} | (
        {"settings": settings} if settings else {}
    )


NOOA = ["nooa/nooa.fabric-adapter.json", "nooa/nooa-bench.fabric-adapter.json"]
CASES = [
    (
        "claude/claude.fabric-adapter.json",
        _model("anthropic", "anthropic-messages"),
        True,
    ),
    (
        "claude/claude.fabric-adapter.json",
        _model("anthropic", "openai-completions"),
        False,
    ),
    ("codex/codex.fabric-adapter.json", _model("openai", "openai-responses"), True),
    ("codex/codex.fabric-adapter.json", _model("openai", "openai-completions"), False),
    (
        "deepagents/deepagents.fabric-adapter.json",
        _model("nvidia", "openai-completions"),
        True,
    ),
    (
        "deepagents/deepagents.fabric-adapter.json",
        _model("anthropic", "anthropic-messages"),
        True,
    ),
    (
        "deepagents/deepagents.fabric-adapter.json",
        _model("anthropic", "openai-completions"),
        False,
    ),
    (
        "mini-swe-agent/mini-swe-agent.fabric-adapter.json",
        _model("openai", "openai-completions", "openai/gpt"),
        True,
    ),
    # LiteLLM routes a slash-qualified model to its prefix, not to the declared provider.
    (
        "mini-swe-agent/mini-swe-agent.fabric-adapter.json",
        _model("openai", "openai-completions", "anthropic/claude"),
        False,
    ),
    (
        "mini-swe-agent/mini-swe-agent.fabric-adapter.json",
        _model("anthropic", "openai-completions"),
        False,
    ),
    *[
        case
        for descriptor in NOOA
        for case in [
            (
                descriptor,
                _model("nvidia", "openai-completions", client_type="completion"),
                True,
            ),
            (
                descriptor,
                _model("openai", "openai-responses", client_type="responses"),
                True,
            ),
            (
                descriptor,
                _model("openai", "openai-completions", client_type="responses"),
                False,
            ),
            (descriptor, _model("nvidia", "openai-responses"), False),
            (descriptor, _model("anthropic", "openai-completions"), False),
        ]
    ],
]


@pytest.mark.parametrize(("descriptor", "model", "valid"), CASES)
def test_model_schema_admits_only_supported_protocols(
    repo_root, descriptor, model, valid
):
    path = repo_root / "adapters/python" / descriptor
    schema = json.loads(path.read_text(encoding="utf-8"))["model_schema"]
    assert jsonschema.Draft202012Validator(schema).is_valid(model) is valid

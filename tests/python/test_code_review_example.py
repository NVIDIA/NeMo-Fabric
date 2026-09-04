# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the code-review example."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from examples.code_review_agent import BASE_DIR
from examples.code_review_agent import __main__ as main_module
from examples.code_review_agent import base_config
from examples.code_review_agent import claude_config
from examples.code_review_agent import codex_config
from examples.code_review_agent import deepagents_config
from examples.code_review_agent import hermes_config
from examples.code_review_agent import nooa_config
from examples.code_review_agent import pi_config
from examples.code_review_agent import with_github_mcp
from examples.code_review_agent import with_native_otel
from examples.code_review_agent import with_opensandbox
from examples.code_review_agent import with_relay
from examples.code_review_agent import with_relay_openinference
from examples.code_review_agent import with_relay_otel
from examples.code_review_agent import with_skill_paths
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import RunOutput


def test_variant_builders_return_independent_complete_configs():
    base = base_config()
    hermes = hermes_config()
    codex = codex_config()
    claude = claude_config()
    deepagents = deepagents_config()
    nooa = nooa_config()
    pi = pi_config()

    for config in (base, hermes, codex, claude, deepagents, nooa, pi):
        assert isinstance(config, FabricConfig)
        assert config.metadata.name == "code-review-agent"
        assert config.environment is not None
        assert "default" in config.models

    assert hermes is not base
    assert hermes.harness is not base.harness
    assert codex.harness.adapter_id == "nvidia.fabric.codex"
    assert codex.mcp is None
    assert codex.skills is None
    assert claude is not base
    assert claude.harness is not base.harness
    assert claude.harness.adapter_id == "nvidia.fabric.claude"
    assert claude.models["default"].provider == "anthropic"
    assert claude.models["default"].model == "anthropic/claude-sonnet-4-5"
    assert claude.models["default"].api_key_env == "ANTHROPIC_API_KEY"
    assert claude.mcp is None
    assert claude.skills is None
    assert deepagents is not base
    assert deepagents.harness is not base.harness
    assert deepagents.harness.adapter_id == "nvidia.fabric.langchain.deepagents"
    assert deepagents.mcp is None
    assert deepagents.skills is not None
    assert deepagents.skills.paths == ["./skills/code-review"]
    assert pi is not base
    assert pi.harness is not base.harness
    assert pi.harness.adapter_id == "nvidia.fabric.pi"
    assert pi.models["default"].provider == "nvidia"
    assert pi.models["default"].api_key_env == "NVIDIA_API_KEY"
    assert pi.models["default"].temperature is None
    assert pi.skills is not None
    assert pi.skills.paths == ["./skills/code-review"]
    assert pi.tools is not None
    assert pi.tools.enabled == ["read"]
    assert deepagents.models == pi.models
    assert deepagents.instructions == pi.instructions
    assert deepagents.environment.workspace == pi.environment.workspace
    assert deepagents.tools is None
    assert nooa.harness is None
    assert nooa.workflow is not None
    assert nooa.workflow.target_id == "nvidia.nooa.coding-agent"
    assert nooa.skills is not None
    assert base.mcp is None
    assert base.skills is not None
    skill_path = BASE_DIR / base.skills.paths[0]
    assert (skill_path / "SKILL.md").is_file()


def test_capability_and_telemetry_variants_do_not_mutate_their_input():
    base = hermes_config()
    variants = (
        with_github_mcp(base),
        with_native_otel(codex_config()),
        with_opensandbox(base),
        with_relay(base),
        with_relay_openinference(base),
        with_relay_otel(base),
    )

    assert base.telemetry is not None
    assert base.telemetry.providers == {}
    assert base.environment is not None
    assert base.environment.provider == "local"
    assert base.mcp is None
    assert all(variant is not base for variant in variants)
    assert variants[0].mcp is not None
    assert variants[0].mcp.servers["github"].exposure == "harness_native"
    assert variants[1].telemetry is not None
    assert "native" in variants[1].telemetry.providers
    assert variants[2].environment is not None
    assert variants[2].environment.provider == "opensandbox"
    assert variants[3].telemetry is not None
    assert "relay" in variants[3].telemetry.providers


def test_skill_path_variants_replace_defaults_without_mutating_their_input():
    base = pi_config()

    without_skills = with_skill_paths(base)
    with_replacements = with_skill_paths(
        base,
        "./skills/code-review",
        "../../tests/fixtures/alternate",
    )

    assert base.skills is not None
    assert base.skills.paths == ["./skills/code-review"]
    assert without_skills.skills is None
    assert with_replacements.skills is not None
    assert with_replacements.skills.paths == [
        "./skills/code-review",
        "../../tests/fixtures/alternate",
    ]


@pytest.mark.usefixtures("nemo_relay")
def test_native_otel_variants_match_adapter_contracts():
    from nemo_relay import plugin

    codex = with_native_otel(codex_config())
    assert codex.telemetry is not None
    codex_config_payload = codex.telemetry.providers["native"].config
    codex_observability = codex_config_payload["components"][0]["config"]
    assert codex_observability["version"] == 1
    assert codex_observability["opentelemetry"]["endpoint"] == (
        "http://localhost:4318/v1/traces"
    )

    deepagents = with_native_otel(deepagents_config())
    assert deepagents.telemetry is not None
    deepagents_config_payload = deepagents.telemetry.providers["native"].config
    assert plugin.validate(deepagents_config_payload)["diagnostics"] == []

    with pytest.raises(ValueError, match="does not support native OpenTelemetry"):
        with_native_otel(hermes_config())


@pytest.mark.parametrize(
    ("variant", "endpoint_type", "endpoint"),
    [
        (
            with_relay_otel,
            "full",
            "http://localhost:4318/v1/traces",
        ),
        (
            with_relay_openinference,
            "openinference",
            "http://localhost:6006/v1/traces",
        ),
    ],
)
@pytest.mark.usefixtures("nemo_relay")
def test_relay_otel_variants_author_v3_endpoints(
    variant,
    endpoint_type: str,
    endpoint: str,
):
    from nemo_relay import plugin

    config = variant(hermes_config())

    observability = config.to_mapping()["relay"]["observability"]
    assert observability["version"] == 3
    assert "openinference" not in observability
    assert observability["opentelemetry"]["enabled"] is True
    assert observability["opentelemetry"]["endpoints"][0]["type"] == endpoint_type
    assert observability["opentelemetry"]["endpoints"][0]["endpoint"] == endpoint
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": observability,
            }
        ],
    }
    assert plugin.validate(plugin_config)["diagnostics"] == []


def test_variants_plan_from_complete_configs():
    client = Fabric()

    for config in (
        hermes_config(),
        codex_config(),
        claude_config(),
        deepagents_config(),
        pi_config(),
    ):
        plan = client.plan(config, base_dir=BASE_DIR)
        assert plan.base_dir == BASE_DIR
        assert plan.agent_name == "code-review-agent"
        assert plan.adapter.adapter_id == config.harness.adapter_id
        if config.harness.adapter_id == "nvidia.fabric.pi":
            continue
        github_plan = client.plan(with_github_mcp(config), base_dir=BASE_DIR)
        assert "github" in github_plan["capability_plan"]["native"]["mcp_servers"]
        assert "mcp_servers" not in github_plan["capability_plan"]["unsupported"]

    nooa_plan = client.plan(nooa_config(), base_dir=BASE_DIR)
    assert nooa_plan.base_dir == BASE_DIR
    assert nooa_plan.agent_name == "code-review-agent"
    assert nooa_plan.adapter.adapter_id == "nvidia.fabric.nooa"
    assert nooa_plan.config.workflow.target_id == "nvidia.nooa.coding-agent"
    nooa_github_plan = client.plan(with_github_mcp(nooa_config()), base_dir=BASE_DIR)
    assert "github" in nooa_github_plan["capability_plan"]["native"]["mcp_servers"]


def test_example_entrypoint_plans_without_starting_a_runtime():
    variants = (
        ("hermes", "nvidia.fabric.hermes"),
        ("codex", "nvidia.fabric.codex"),
        ("claude", "nvidia.fabric.claude"),
        ("deepagents", "nvidia.fabric.langchain.deepagents"),
        ("nooa", "nvidia.fabric.nooa"),
        ("pi", "nvidia.fabric.pi"),
    )
    cases = tuple(
        (
            ["--variant", variant, *(["--relay"] if relay_enabled else [])],
            adapter_id,
            relay_enabled,
        )
        for variant, adapter_id in variants
        for relay_enabled in (False, True)
    )

    for options, adapter_id, relay_enabled in cases:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "examples.code_review_agent",
                *options,
                "--plan",
            ],
            cwd=BASE_DIR.parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        plan = json.loads(completed.stdout)
        assert plan["agent_name"] == "code-review-agent"
        assert plan["adapter_descriptor"]["descriptor"]["adapter_id"] == adapter_id
        telemetry_plan = plan.get("telemetry_plan")
        if relay_enabled:
            assert telemetry_plan["relay_enabled"] is True
        else:
            assert telemetry_plan is None


@pytest.mark.parametrize(
    ("options", "expected_paths"),
    [
        (["--variant", "pi"], [BASE_DIR / "skills/code-review"]),
        (["--variant", "pi", "--no-skills"], []),
        (["--variant", "deepagents"], [BASE_DIR / "skills/code-review"]),
        (
            [
                "--variant",
                "pi",
                "--skill-path",
                "./skills/code-review",
                "--skill-path",
                "../../tests/fixtures/alternate",
            ],
            [
                BASE_DIR / "skills/code-review",
                BASE_DIR / "../../tests/fixtures/alternate",
            ],
        ),
    ],
)
def test_example_entrypoint_plans_skill_variants(options, expected_paths):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.code_review_agent",
            *options,
            "--plan",
        ],
        cwd=BASE_DIR.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    agent_config = json.loads(completed.stdout)["agent_config"]
    actual_paths = agent_config.get("skills", {}).get("paths", [])
    assert [Path(path).resolve() for path in actual_paths] == [
        path.resolve() for path in expected_paths
    ]


def test_example_entrypoint_rejects_conflicting_skill_options():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.code_review_agent",
            "--variant",
            "pi",
            "--skill-path",
            "./skills/code-review",
            "--no-skills",
            "--plan",
        ],
        cwd=BASE_DIR.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "not allowed with argument" in completed.stderr


def test_pi_variant_projects_explicit_skill_and_tool_policy():
    plan = Fabric().plan(pi_config(), base_dir=BASE_DIR)
    agent_config = plan.to_mapping()["agent_config"]

    assert agent_config["skills"] == {
        "paths": [str((BASE_DIR / "skills/code-review").resolve())]
    }
    assert agent_config["tools"] == {"enabled": ["read"]}
    assert plan.config.runtime.input_schema == "text"
    assert plan.config.runtime.output_schema == "message"


def test_pi_variant_requires_the_relay_extension_for_a_live_run():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.code_review_agent",
            "--variant",
            "pi",
            "--relay",
        ],
        cwd=BASE_DIR.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "Pi Relay runs require --pi-relay-extension-path" in completed.stderr


def test_pi_variant_rejects_relay_backed_streaming():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.code_review_agent",
            "--variant",
            "pi",
            "--relay",
            "--stream",
        ],
        cwd=BASE_DIR.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "Pi adapter does not support Relay-backed streaming yet" in completed.stderr


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--stream"], "--stream requires --relay"),
        (
            ["--relay", "--stream", "--plan"],
            "--stream cannot be combined with --plan",
        ),
    ],
)
def test_example_entrypoint_rejects_invalid_stream_options(options, message):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.code_review_agent",
            *options,
        ],
        cwd=BASE_DIR.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert message in completed.stderr


async def test_example_entrypoint_shows_response_after_normalized_output(
    monkeypatch,
    capsys,
):
    result = MagicMock()
    result.output = RunOutput.from_mapping({"response": "visible response"})
    result.to_mapping.return_value = {"output": result.output.to_mapping()}
    mock_fabric = MagicMock()
    mock_fabric.run = AsyncMock(return_value=result)
    monkeypatch.setattr(main_module, "Fabric", lambda: mock_fabric)
    monkeypatch.setattr(
        sys,
        "argv",
        ["code_review_agent", "--show-output", "--input", "review this"],
    )

    await main_module.main()

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert captured.err == ""
    assert lines[-1] == "visible response"
    assert json.loads("\n".join(lines[:-1])) == {
        "output": {"response": "visible response"}
    }


async def test_example_entrypoint_streams_relay_records_and_terminal_result(
    monkeypatch,
    capsys,
):
    result = MagicMock()
    result.output = RunOutput.from_mapping({"response": "streamed response"})
    result.error = None
    result.to_mapping.return_value = {
        "status": "succeeded",
        "output": result.output.to_mapping(),
    }

    stream = MagicMock(name="runtime_stream")
    stream.__aiter__.return_value = iter(
        [
            {"type": "scope_start", "request_id": "request-1"},
            {"type": "scope_end", "request_id": "request-1"},
        ]
    )
    stream.result = AsyncMock(return_value=result)
    runtime = MagicMock()
    runtime.invoke_stream.return_value = stream
    runtime_context = MagicMock(name="runtime_context")
    runtime_context.__aenter__ = AsyncMock(return_value=runtime)
    runtime_context.__aexit__ = AsyncMock(return_value=None)

    mock_fabric = MagicMock()
    mock_fabric.start_runtime = AsyncMock(return_value=runtime_context)
    monkeypatch.setattr(main_module, "Fabric", lambda: mock_fabric)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "code_review_agent",
            "--variant",
            "nooa",
            "--relay",
            "--stream",
            "--show-output",
            "--input",
            "review this",
        ],
    )

    await main_module.main()

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert captured.err == ""
    assert lines[-1] == "streamed response"
    payload = json.loads("\n".join(lines[:-1]))
    assert len(payload["atof_records"]) == 2
    assert payload["result"]["status"] == "succeeded"
    mock_fabric.start_runtime.assert_awaited_once()
    assert mock_fabric.start_runtime.call_args.kwargs["streaming"] is True
    runtime.invoke_stream.assert_called_once_with(input="review this")
    stream.result.assert_awaited_once_with()
    runtime_context.__aexit__.assert_awaited_once()

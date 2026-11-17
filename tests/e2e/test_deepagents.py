# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real Deep Agents smoke for single-invocation and multi-turn runtimes.

RUN_FABRIC_DEEPAGENTS_INTEGRATION=1 NVIDIA_API_KEY=... \
    pytest tests/e2e/test_deepagents.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import stat
import sys
import uuid
import warnings
from pathlib import Path

import pytest
import requests
from _utils.utils import assert_atof_model, assert_atof_skill_selection


@pytest.mark.usefixtures("mock_nvidia_api_key")
@pytest.mark.parametrize("effort", [None, "none"])
async def test_deepagents_persistent_host_with_mock_model(api_server, tmp_path, effort):
    pytest.importorskip("deepagents")
    from examples.code_review_agent import deepagents_config
    from nemo_fabric import EnvironmentConfig, Fabric, RuntimeConfig

    config = deepagents_config()
    if effort is not None:
        config.harness.settings["reasoning_effort"] = effort
    config.models["default"].base_url = f"{api_server}/v1"
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=tmp_path,
        artifacts=tmp_path / "artifacts",
    )
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts=tmp_path / "artifacts",
    )

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    assert first["metadata"]["adapter_runner"] == "persistent_local_host", results
    assert first["metadata"]["host_pid"] == second["metadata"]["host_pid"], results
    assert first["output"]["thread_id"] == second["output"]["thread_id"], results
    assert first["output"]["resumed"] is False, results
    assert second["output"]["resumed"] is True, results
    assert "user_count=2" in second["output"]["response"], results
    captured = requests.get(f"{api_server}/_requests", timeout=5)
    captured.raise_for_status()
    model_requests = captured.json()
    assert len(model_requests) == 2
    for request in model_requests:
        if effort is None:
            assert "reasoning_effort" not in request
        else:
            assert request["reasoning_effort"] == effort


@pytest.mark.usefixtures("mock_nvidia_api_key")
@pytest.mark.parametrize(
    "subagent_model",
    [None, "gpt-4o"],
    ids=["inherits-parent-model", "overrides-model"],
)
async def test_deepagents_delegated_requests_use_model_reasoning_effort(
    api_server, tmp_path, subagent_model
):
    pytest.importorskip("deepagents")
    from examples.code_review_agent import deepagents_config
    from nemo_fabric import EnvironmentConfig, Fabric, RuntimeConfig

    os.environ["OPENAI_API_KEY"] = "sk-test"
    os.environ["OPENAI_BASE_URL"] = f"{api_server}/v1"
    subagent = {
        "name": "worker",
        "description": "Handle the delegated test request.",
        "system_prompt": "Reply briefly to the delegated request.",
    }
    if subagent_model is not None:
        subagent["model"] = subagent_model
    config = deepagents_config()
    config.models["default"].model = "parent-model"
    config.models["default"].base_url = f"{api_server}/v1"
    config.harness.settings = {
        "reasoning_effort": "low",
        "deepagents": {"subagents": [subagent]},
    }
    config.environment = EnvironmentConfig(provider="local", workspace=tmp_path)
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts=tmp_path / "artifacts",
    )
    scenario = requests.post(
        f"{api_server}/_scenario",
        json={
            "tool_call": {
                "name": "task",
                "arguments": {
                    "subagent_type": "worker",
                    "description": "delegated reasoning check",
                },
            }
        },
        timeout=5,
    )
    scenario.raise_for_status()

    result = await Fabric().run(config, base_dir=tmp_path, input="Delegate to worker.")

    assert result["status"] == "succeeded", result.to_mapping()
    captured = requests.get(f"{api_server}/_requests", timeout=5)
    captured.raise_for_status()
    parent_first, delegated, parent_last = captured.json()
    for request in (parent_first, parent_last):
        assert request["model"] == "parent-model"
        assert request["reasoning_effort"] == "low"
    assert delegated["messages"][-1]["content"] == "delegated reasoning check"
    if subagent_model is None:
        assert delegated["model"] == "parent-model"
        assert delegated["reasoning_effort"] == "low"
    else:
        assert delegated["model"] == subagent_model
        assert "reasoning_effort" not in delegated


@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay")
@pytest.mark.parametrize(
    ("session_root", "uuid_request_ids"),
    [
        (None, True),
        ("018f47a4-3af7-7d94-8e61-9f0f89b5d314", True),
        ("018f47a4-3af7-7d94-8e61-9f0f89b5d314", False),
    ],
    ids=["per-request", "shared-root-uuid-requests", "shared-root-generated-requests"],
)
async def test_deepagents_persistent_host_with_relay_and_mock_model(
    api_server, tmp_path, session_root, uuid_request_ids
):
    pytest.importorskip("deepagents")
    from examples.code_review_agent import deepagents_config, with_relay
    from nemo_fabric import EnvironmentConfig, Fabric, RunRequest, RuntimeConfig

    config = with_relay(deepagents_config())
    config.models["default"].base_url = f"{api_server}/v1"
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=tmp_path,
        artifacts=tmp_path / "artifacts",
    )
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts=tmp_path / "artifacts",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        async with await Fabric().start_runtime(
            config,
            base_dir=tmp_path,
            streaming=True,
        ) as runtime:
            first_request = RunRequest(
                input="first",
                relay_session_root=session_root,
                context={"relay_session_root": "018f47a4-3af7-7d94-8e61-9f0f89b5d399"},
                **({"request_id": str(uuid.uuid4())} if uuid_request_ids else {}),
            )
            first_stream = runtime.invoke_stream(request=first_request)
            first_records = [record async for record in first_stream]
            first = await first_stream.result()
            first_atif_paths = {
                a["path"]
                for a in first["output"]["relay_artifacts"]
                if a["kind"] == "atif"
            }
            assert len(first_atif_paths) == 1
            first_atif_path = Path(next(iter(first_atif_paths)))
            first_atif = json.loads(first_atif_path.read_text())

            second_request = RunRequest(
                input="second",
                relay_session_root=session_root,
                context={"relay_session_root": "018f47a4-3af7-7d94-8e61-9f0f89b5d399"},
                **({"request_id": str(uuid.uuid4())} if uuid_request_ids else {}),
            )
            second_stream = runtime.invoke_stream(request=second_request)
            second_records = [record async for record in second_stream]
            second = await second_stream.result()
            second_atif = json.loads(
                Path(
                    next(
                        a["path"]
                        for a in second["output"]["relay_artifacts"]
                        if a["kind"] == "atif" and a["path"] not in first_atif_paths
                    )
                ).read_text()
            )

    results = (first.to_mapping(), second.to_mapping())
    assert first_records
    assert second_records
    assert first_request.request_id != second_request.request_id
    assert json.loads(first_atif_path.read_text()) == first_atif
    assert first_atif["session_id"] == (session_root or first_request.request_id)
    assert second_atif["session_id"] == (session_root or second_request.request_id)
    for request, records in (
        (first_request, first_records),
        (second_request, second_records),
    ):
        assert any(
            record.get("metadata", {}).get("nemo_fabric_request_id")
            == request.request_id
            for record in records
        ), records
    assert first["status"] == second["status"] == "succeeded", results
    assert first["metadata"]["host_pid"] == second["metadata"]["host_pid"], results
    assert first["output"]["thread_id"] == second["output"]["thread_id"], results
    assert first["output"]["resumed"] is False, results
    assert second["output"]["resumed"] is True, results
    assert "user_count=2" in second["output"]["response"], results
    for turn in (first, second):
        assert turn.telemetry[0].provider == "relay", turn.to_mapping()
        assert {artifact["kind"] for artifact in turn["output"]["relay_artifacts"]} >= {
            "atof",
            "atif",
        }, turn.to_mapping()


@pytest.mark.usefixtures("mock_nvidia_api_key")
async def test_env_secrets_in_headers(api_server, tmp_path):
    pytest.importorskip("deepagents")
    from examples.code_review_agent import deepagents_config
    from nemo_fabric import EnvironmentConfig, Fabric, RuntimeConfig

    os.environ["MY_KEY"] = "XYZ"
    scenario_response = requests.post(
        f"{api_server}/_scenario",
        json={
            "tool_call": {
                "name": "get_authorization_header",
                "arguments": {},
            }
        },
        timeout=5,
    )
    scenario_response.raise_for_status()

    config = deepagents_config()
    config.models["default"].base_url = f"{api_server}/v1"
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=tmp_path,
        artifacts=tmp_path / "artifacts",
    )
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts=tmp_path / "artifacts",
    )
    config.add_mcp_server(
        "headers",
        transport="streamable-http",
        url=f"{api_server}/mcp",
        authentication=None,
        custom_headers={"Authorization": "Bearer ${MY_KEY}"},
    )

    result = await Fabric().run(
        config,
        base_dir=tmp_path,
        input="Use the MCP tool to return its Authorization header.",
    )

    assert result["status"] == "succeeded", result.to_mapping()
    response = requests.get(f"{api_server}/_mcp_authorization_headers", timeout=5)
    response.raise_for_status()
    assert set(response.json()) == {"Bearer XYZ"}


@pytest.mark.usefixtures("mock_nvidia_api_key")
@pytest.mark.parametrize("relay", [False, True], ids=["native", "relay"])
async def test_unavailable_mcp_servers_do_not_block_startup(
    api_server, tmp_path, unused_tcp_port, relay, request
):
    """Start with the reachable MCP server and report the unavailable ones."""
    pytest.importorskip("deepagents")
    if relay:
        request.getfixturevalue("nemo_relay")
    from examples.code_review_agent import deepagents_config, with_relay
    from nemo_fabric import EnvironmentConfig, Fabric, RuntimeConfig

    os.environ["MY_KEY"] = "XYZ"
    scenario_response = requests.post(
        f"{api_server}/_scenario",
        json={"tool_call": {"name": "get_authorization_header", "arguments": {}}},
        timeout=5,
    )
    scenario_response.raise_for_status()

    config = deepagents_config()
    if relay:
        config = with_relay(config)
    config.models["default"].base_url = f"{api_server}/v1"
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=tmp_path,
        artifacts=tmp_path / "artifacts",
    )
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts=tmp_path / "artifacts",
    )
    config.add_mcp_server(
        "headers",
        transport="streamable-http",
        url=f"{api_server}/mcp",
        authentication=None,
        custom_headers={"Authorization": "Bearer ${MY_KEY}"},
    )
    # Nothing listens on this port, so the connection is refused.
    config.add_mcp_server(
        "offline",
        transport="streamable-http",
        url=f"http://127.0.0.1:{unused_tcp_port}/mcp",
    )
    config.add_mcp_server(
        "missing",
        transport="stdio",
        url="nemo-fabric-missing-mcp-server-command",
    )

    async with await Fabric().start_runtime(
        config, base_dir=tmp_path, streaming=relay
    ) as runtime:
        first = await runtime.invoke(
            input="Use the MCP tool to return its Authorization header."
        )
        second = await runtime.invoke(input="second")

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    response = requests.get(f"{api_server}/_mcp_authorization_headers", timeout=5)
    response.raise_for_status()
    assert set(response.json()) == {"Bearer XYZ"}
    for turn in (first, second):
        unavailable = turn["output"]["mcp"]["unavailable_servers"]
        assert {server["name"]: server["reason"] for server in unavailable} == {
            "offline": "failed",
            "missing": "failed",
        }, turn.to_mapping()
        assert "XYZ" not in json.dumps(unavailable)
    if relay:
        assert first.telemetry[0].provider == "relay", first.to_mapping()


@pytest.mark.usefixtures("mock_nvidia_api_key")
async def test_mcp_service_account_authentication(api_server, tmp_path):
    """Fetch, send and renew a client-credentials token through a real runtime."""
    pytest.importorskip("deepagents")
    from examples.code_review_agent import deepagents_config
    from nemo_fabric import (
        EnvironmentConfig,
        Fabric,
        McpAuthenticationConfig,
        RuntimeConfig,
    )

    secret = "sa-secret-do-not-report"
    os.environ["MCP_SA_SECRET"] = secret
    os.environ["MCP_SA_WRONG_SECRET"] = "not-the-secret"
    requests.post(
        f"{api_server}/_oauth2",
        json={"client_id": "fabric-agent", "client_secret": secret},
        timeout=5,
    ).raise_for_status()

    def call_mcp_tool_once() -> None:
        requests.post(
            f"{api_server}/_scenario",
            json={"tool_call": {"name": "get_authorization_header", "arguments": {}}},
            timeout=5,
        ).raise_for_status()

    def service_account(secret_env: str) -> McpAuthenticationConfig:
        return McpAuthenticationConfig(
            type="service_account",
            client_id="fabric-agent",
            client_secret_env=secret_env,
            token_url=f"{api_server}/oauth2/token",
            scopes=["tools.read"],
        )

    config = deepagents_config()
    config.models["default"].base_url = f"{api_server}/v1"
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=tmp_path,
        artifacts=tmp_path / "artifacts",
    )
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts=tmp_path / "artifacts",
    )
    config.add_mcp_server(
        "protected",
        transport="streamable-http",
        url=f"{api_server}/mcp-protected",
        authentication=service_account("MCP_SA_SECRET"),
    )
    config.add_mcp_server(
        "rejected",
        transport="streamable-http",
        url=f"{api_server}/mcp-protected",
        authentication=service_account("MCP_SA_WRONG_SECRET"),
    )

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        call_mcp_tool_once()
        first = await runtime.invoke(input="Return the MCP Authorization header.")
        # Invalidate the issued token so the MCP server answers 401.
        requests.post(f"{api_server}/_oauth2", json={"revoke": True}, timeout=5)
        call_mcp_tool_once()
        second = await runtime.invoke(input="Return it again.")

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    token_requests = requests.get(
        f"{api_server}/_oauth2_token_requests", timeout=5
    ).json()
    assert all(
        request == {"grant_type": "client_credentials", "scope": "tools.read"}
        for request in token_requests
    ), token_requests
    issued = {f"sa-token-{index}" for index in range(1, len(token_requests) + 1)}
    headers = requests.get(f"{api_server}/_mcp_authorization_headers", timeout=5).json()
    last_request = requests.get(f"{api_server}/_requests", timeout=5).json()[-1]
    tool_results = [
        "".join(part["text"] for part in message["content"])
        if isinstance(message["content"], list)
        else message["content"]
        for message in last_request["messages"]
        if message["role"] == "tool"
    ]
    # The first turn's tool call used the startup token; the second used a
    # token fetched after the MCP server rejected the revoked one.
    first_token, second_token = (
        result.removeprefix("Bearer ") for result in tool_results
    )
    assert first_token in issued and second_token in issued
    assert first_token != second_token
    assert f"Bearer {first_token}" in headers
    for turn in (first, second):
        assert turn["output"]["mcp"]["unavailable_servers"] == [
            {
                "name": "rejected",
                "transport": "streamable-http",
                "reason": "authentication_required",
                "error_type": "ServiceAccountTokenError",
            }
        ], turn.to_mapping()
    assert secret not in json.dumps(results)


@pytest.mark.usefixtures("mock_nvidia_api_key")
async def test_mcp_oauth2_browser_sign_in(api_server, tmp_path):
    """Sign in through a browser on the first invocation, then reuse the tokens."""
    pytest.importorskip("deepagents")
    from examples.code_review_agent import deepagents_config
    from nemo_fabric import (
        EnvironmentConfig,
        Fabric,
        McpAuthenticationConfig,
        RuntimeConfig,
    )
    from nemo_fabric_adapters.common import mcp_oauth

    # The browser: follow the authorization URL, whose server approves at once
    # and redirects to the adapter's loopback callback.
    browser = tmp_path / "browser.py"
    browser.write_text(
        "import sys, urllib.request\n"
        "urllib.request.urlopen(sys.argv[1], timeout=10).read()\n"
    )
    os.environ["BROWSER"] = shlex.join([sys.executable, str(browser), "%s"])

    def call_mcp_tool_once() -> None:
        requests.post(
            f"{api_server}/_scenario",
            json={"tool_call": {"name": "get_authorization_header", "arguments": {}}},
            timeout=5,
        ).raise_for_status()

    def last_tool_result() -> str:
        last_request = requests.get(f"{api_server}/_requests", timeout=5).json()[-1]
        (message,) = [m for m in last_request["messages"] if m["role"] == "tool"]
        content = message["content"]
        if isinstance(content, list):
            return "".join(part["text"] for part in content)
        return content

    config = deepagents_config()
    config.models["default"].base_url = f"{api_server}/v1"
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=tmp_path,
        artifacts=tmp_path / "artifacts",
    )
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts=tmp_path / "artifacts",
        keep_state=True,
    )
    config.add_mcp_server(
        "protected",
        transport="streamable-http",
        url=f"{api_server}/mcp-protected",
        authentication=McpAuthenticationConfig(type="oauth2", scopes=["tools.read"]),
    )

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        call_mcp_tool_once()
        first = await runtime.invoke(input="Return the MCP Authorization header.")
        first_header = last_tool_result()

    # A later runtime of the agent reuses the stored tokens without a browser.
    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        call_mcp_tool_once()
        second = await runtime.invoke(input="Return it again.")
        second_header = last_tool_result()

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    assert "mcp" not in first["output"] and "mcp" not in second["output"], results
    (authorization,) = requests.get(
        f"{api_server}/_oauth2_authorizations", timeout=5
    ).json()
    assert authorization["scope"] == "tools.read"
    assert authorization["code_challenge_method"] == "S256"
    assert authorization["client_id"] == "registered-client"
    assert first_header == second_header == "Bearer user-token-1"
    token_files = sorted(
        (tmp_path / "artifacts" / ".fabric" / "state").rglob("mcp-tokens/*/*")
    )
    assert [path.name for path in token_files] == [
        "client.json",
        "metadata.json",
        "tokens.json",
    ]
    assert {path.parent.name for path in token_files} == {
        mcp_oauth.storage_name("protected", f"{api_server}/mcp-protected")
    }
    if os.name == "posix":
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in token_files)


@pytest.mark.usefixtures("mock_nvidia_api_key")
@pytest.mark.parametrize("enabled", [True, False])
async def test_mcp_stdio_transport(api_server, tmp_path, enabled):
    pytest.importorskip("deepagents")
    from examples.code_review_agent import deepagents_config
    from nemo_fabric import EnvironmentConfig, Fabric, RuntimeConfig

    tool_name = "get_current_time"
    scenario_response = requests.post(
        f"{api_server}/_scenario",
        json={
            "tool_call": {
                "name": tool_name,
                "arguments": {"timezone": "America/Los_Angeles"},
            }
        },
        timeout=5,
    )
    scenario_response.raise_for_status()

    config = deepagents_config()
    config.models["default"].base_url = f"{api_server}/v1"
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=tmp_path,
        artifacts=tmp_path / "artifacts",
    )
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts=tmp_path / "artifacts",
    )
    config.add_mcp_server(
        "mcp_server_time",
        transport="stdio",
        url=sys.executable,
        args=["-m", "mcp_server_time"],
        env={"MCP_TIME_TEST": "enabled"},
    )

    if not enabled:
        config.remove_mcp_server("mcp_server_time")

    result = await Fabric().run(
        config,
        base_dir=tmp_path,
        input="Use the time MCP tool to get the current time in America/Los_Angeles.",
    )

    assert result["status"] == "succeeded", result.to_mapping()
    tool_calls = [
        call
        for message in result["output"]["messages"]
        for call in message.get("tool_calls", [])
        if call["name"] == tool_name
    ]
    tool_results = [
        message for message in result["output"]["messages"] if message["role"] == "tool"
    ]

    if not enabled:
        assert not tool_calls
        assert not tool_results
    else:
        assert tool_calls
        assert tool_results
        assert "America/Los_Angeles" in str(tool_results[0]["content"])


@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay")
@pytest.mark.parametrize("skill", ["default", "alternate", None])
async def test_skill_selection(
    api_server, tmp_path, skill, default_skill, alternate_skill
):
    pytest.importorskip("deepagents")
    from examples.code_review_agent import (
        deepagents_config,
        with_relay,
        with_skill_paths,
    )
    from nemo_fabric import EnvironmentConfig, Fabric, RuntimeConfig

    config = with_relay(with_skill_paths(deepagents_config()))
    config.models["default"].model = "fabric-echo"
    config.models["default"].base_url = f"{api_server}/v1"
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=default_skill.parents[2],
        artifacts=tmp_path / "artifacts",
    )
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts=tmp_path / "artifacts",
    )
    config.add_skill_path(default_skill)

    if skill == "alternate" or skill is None:
        config.remove_skill_path(default_skill)

    if skill == "alternate":
        config.add_skill_path(alternate_skill)

    if skill is not None:
        selected_skill = default_skill if skill == "default" else alternate_skill
        skill_file = selected_skill.relative_to(default_skill.parents[2]) / "SKILL.md"
        scenario_response = requests.post(
            f"{api_server}/_scenario",
            json={
                "tool_call": {
                    "name": "read_file",
                    "arguments": {"file_path": f"/{skill_file}"},
                }
            },
            timeout=5,
        )
        scenario_response.raise_for_status()

    result = await Fabric().run(
        config,
        base_dir=tmp_path,
        input=f"Use the {skill} skill." if skill else "Reply without using a skill.",
    )

    assert result["status"] == "succeeded", result.to_mapping()
    assert_atof_skill_selection(result["output"], skill)


@pytest.mark.usefixtures("mock_nvidia_api_key", "nemo_relay")
@pytest.mark.parametrize("model", ["m1", "m2"])
async def test_model_selection(api_server, tmp_path, model):
    pytest.importorskip("deepagents")
    from examples.code_review_agent import deepagents_config, with_relay
    from nemo_fabric import EnvironmentConfig, Fabric, RuntimeConfig

    config = with_relay(deepagents_config())
    config.models["default"].model = model
    config.models["default"].base_url = f"{api_server}/v1"
    config.environment = EnvironmentConfig(
        provider="local",
        workspace=tmp_path,
        artifacts=tmp_path / "artifacts",
    )
    config.runtime = RuntimeConfig(
        input_schema="chat",
        output_schema="message",
        artifacts=tmp_path / "artifacts",
    )

    result = await Fabric().run(config, base_dir=tmp_path, input="Reply with hello.")

    assert result["status"] == "succeeded", result.to_mapping()
    assert_atof_model(result["output"], model)


@pytest.fixture(name="_require_integration")
def _require_integration_fixture() -> None:
    if os.environ.get("RUN_FABRIC_DEEPAGENTS_INTEGRATION") != "1":
        pytest.skip("set RUN_FABRIC_DEEPAGENTS_INTEGRATION=1 to run")
    if importlib.util.find_spec("deepagents") is None:
        pytest.fail(
            "the deepagents package is required (pip install -e '.[deepagents]')"
        )
    if importlib.util.find_spec("nemo_fabric._native") is None:
        pytest.fail("the nemo_fabric native extension is required (pip install -e .)")
    if not os.environ.get("NVIDIA_API_KEY"):
        pytest.fail("NVIDIA_API_KEY is required")


@pytest.mark.usefixtures("_require_integration")
async def test_deepagents_single_invocation():
    from examples.code_review_agent import BASE_DIR, deepagents_config
    from nemo_fabric import Fabric

    client = Fabric()
    single = await client.run(
        deepagents_config(),
        base_dir=BASE_DIR,
        input="Reply with exactly: FABRIC_DEEPAGENTS_SINGLE_INVOCATION_OK",
    )
    assert single["status"] == "succeeded", single.to_mapping()
    assert single["output"]["response"], single.to_mapping()
    assert single["output"]["resumed"] is False, single.to_mapping()


@pytest.mark.usefixtures("_require_integration")
async def test_deepagents_multi_turn():
    from examples.code_review_agent import BASE_DIR, deepagents_config
    from nemo_fabric import Fabric

    client = Fabric()
    nonce = f"fabric-{uuid.uuid4().hex[:8]}"

    async with await client.start_runtime(
        deepagents_config(), base_dir=BASE_DIR
    ) as runtime:
        first = await runtime.invoke(input=f"Remember this value: {nonce}")
        second = await runtime.invoke(
            input="Reply with only the value I asked you to remember."
        )

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    # one started runtime keeps a stable LangGraph thread across turns
    assert first["output"]["thread_id"] == second["output"]["thread_id"], results
    assert first["metadata"]["adapter_runner"] == "persistent_local_host", results
    assert first["metadata"]["host_pid"] == second["metadata"]["host_pid"], results
    # The second turn continues the live runtime and recalls turn one.
    assert first["output"]["resumed"] is False, results
    assert second["output"]["resumed"] is True, results
    assert nonce in second["output"]["response"], second.to_mapping()


@pytest.mark.usefixtures("_require_integration", "nemo_relay")
async def test_deepagents_multi_turn_with_relay():
    from examples.code_review_agent import BASE_DIR, deepagents_config, with_relay
    from nemo_fabric import Fabric

    client = Fabric()
    nonce = f"fabric-relay-{uuid.uuid4().hex[:8]}"
    config = with_relay(deepagents_config())

    async with await client.start_runtime(config, base_dir=BASE_DIR) as runtime:
        first = await runtime.invoke(input=f"Remember this value: {nonce}")
        second = await runtime.invoke(
            input="Reply with only the value I asked you to remember."
        )

    results = (first.to_mapping(), second.to_mapping())
    assert first["status"] == second["status"] == "succeeded", results
    assert first["output"]["thread_id"] == second["output"]["thread_id"], results
    assert first["metadata"]["host_pid"] == second["metadata"]["host_pid"], results
    assert first["output"]["resumed"] is False, results
    assert second["output"]["resumed"] is True, results
    assert nonce in second["output"]["response"], second.to_mapping()
    for turn in (first, second):
        assert turn.telemetry[0].provider == "relay", turn.to_mapping()
        assert {artifact["kind"] for artifact in turn["output"]["relay_artifacts"]} >= {
            "atof",
            "atif",
        }, turn.to_mapping()


@pytest.mark.usefixtures("_require_integration")
async def test_deepagents_builtin_subagent_delegation():
    # Exercise the built-in delegated-subagent path end to end. The built-in
    # subagent inherits the parent model, tools, workspace, and tool policy, so
    # delegation cannot broaden capabilities.
    from examples.code_review_agent import BASE_DIR, deepagents_config
    from nemo_fabric import Fabric

    config = deepagents_config()

    client = Fabric()
    result = await client.run(
        config,
        base_dir=BASE_DIR,
        input=(
            "Delegate to the built-in general-purpose subagent via the task tool to "
            "echo the phrase FABRIC_DEEPAGENTS_SUBAGENT_OK, then reply with its result."
        ),
    )

    assert result["status"] == "succeeded", result.to_mapping()
    output = result["output"]
    assert "FABRIC_DEEPAGENTS_SUBAGENT_OK" in output["response"], result.to_mapping()
    task_calls = [
        call
        for message in output["messages"]
        for call in message.get("tool_calls", [])
        if call["name"] == "task"
    ]
    assert any(
        call["args"]["subagent_type"] == "general-purpose" for call in task_calls
    ), result.to_mapping()
    assert any("subgraph" in event for event in output["events"]), result.to_mapping()
    # delegated steps are folded into this turn's usage aggregation
    assert output["resumed"] is False, result.to_mapping()


@pytest.mark.usefixtures("_require_integration")
async def test_deepagents_doctor():
    from examples.code_review_agent import BASE_DIR, deepagents_config
    from nemo_fabric import Fabric

    config = deepagents_config()
    client = Fabric()
    report = await client.doctor(config, base_dir=BASE_DIR)

    # The adapter declares no static env requirement (auth is provider-specific and
    # validated by the runtime preflight), so doctor resolves without failures.
    assert report.status == "pass", report

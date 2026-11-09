# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in qualification against the pinned OpenClaw Gateway and real native clients."""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import pytest
from nemo_fabric import Fabric, FabricConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("FABRIC_OPENCLAW_TEST_COMMAND"),
    reason="Set FABRIC_OPENCLAW_TEST_COMMAND to pinned OpenClaw for native qualification",
)


@pytest.fixture(name="native_config")
def native_config_fixture(tmp_path, api_server, unused_tcp_port):
    os.environ["ADAPTER_PYTHON"] = sys.executable
    os.environ["FABRIC_TEST_MODEL_KEY"] = "native-parity-test-key"
    marker = "FABRIC_SHARED_PROMPT_72931"
    skill = tmp_path / "external-skills/native-parity"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: native-parity\ndescription: Check the Fabric native parity marker.\n---\nFABRIC_SKILL_98214\n"
    )
    config = FabricConfig.from_mapping(
        {
            "metadata": {"name": "openclaw-native-parity"},
            "harness": {
                "adapter_id": "nvidia.fabric.openclaw",
                "settings": {
                    "openclaw_command": os.environ["FABRIC_OPENCLAW_TEST_COMMAND"],
                    "agent_id": "reviewer",
                    "interfaces": {
                        "dashboard": {"port": unused_tcp_port},
                        "terminal": {},
                    },
                },
            },
            "models": {
                "default": {
                    "provider": "fabric-test",
                    "model": "fabric-echo",
                    "base_url": f"{api_server}/v1",
                    "api_key_env": "FABRIC_TEST_MODEL_KEY",
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_tokens": 128,
                }
            },
            "instructions": {"system": {"content": marker, "mode": "replace"}},
            "tools": {"blocked": ["exec", "process"]},
            "skills": {"paths": [str(skill)]},
            "mcp": {
                "servers": {
                    "fabricprobe": {
                        "transport": "streamable-http",
                        "url": f"{api_server}/mcp",
                        "custom_headers": {
                            "Authorization": "Bearer native-mcp-test-key"
                        },
                    }
                }
            },
            "environment": {
                "provider": "local",
                "workspace": str(tmp_path),
                "artifacts": str(tmp_path / "artifacts"),
            },
        }
    )
    return config


async def test_native_and_sdk_share_gateway_configuration(
    tmp_path, api_server, unused_tcp_port, native_config
):
    from nemo_fabric import CheckRequest, EnvironmentHandle
    from playwright.async_api import async_playwright
    import pexpect

    marker = native_config.instructions.system.content
    (tmp_path / "source.txt").write_text("FABRIC_SOURCE_MARKER_35291\n")
    fabric = Fabric()
    async with await fabric.start_runtime(native_config, base_dir=tmp_path) as runtime:
        directory = (
            tmp_path
            / "artifacts/.fabric/openclaw/runtimes"
            / runtime.runtime_id
            / "interfaces"
        )
        token = (directory / "dashboard.token").read_text()
        async with httpx.AsyncClient(trust_env=False, timeout=60) as client:

            async def scenario(path, turn):
                calls = (
                    [
                        {"name": "read", "arguments": {"path": "source.txt"}},
                        {
                            "name": "write",
                            "arguments": {
                                "path": f"{path}-result.txt",
                                "content": f"{path} configured workspace verified",
                            },
                        },
                    ]
                    if turn == 1
                    else [
                        {
                            "name": "tool_call",
                            "arguments": {
                                "id": "mcp:fabricprobe:fabricprobe__get_authorization_header",
                                "args": {},
                            },
                        },
                        {
                            "name": "read",
                            "arguments": {
                                "path": str(
                                    Path(native_config.skills.paths[0]) / "SKILL.md"
                                )
                            },
                        },
                    ]
                )
                response = await client.post(
                    f"{api_server}/_scenario",
                    json={
                        "tool_calls": calls,
                        "response_text": f"{path} turn {turn} complete",
                    },
                )
                response.raise_for_status()

            for turn in (1, 2):
                await scenario("sdk", turn)
                result = await runtime.invoke(input=f"SDK parity turn {turn}")
                assert result.status == "succeeded", result.to_mapping()
            assert (
                tmp_path / "sdk-result.txt"
            ).read_text() == "sdk configured workspace verified"

            for turn in (1, 2):
                await scenario("http", turn)
                response = await client.post(
                    f"http://127.0.0.1:{unused_tcp_port}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {token}"},
                    json={
                        "model": "openclaw/reviewer",
                        "user": "http-parity-client",
                        "messages": [
                            {"role": "user", "content": f"HTTP parity turn {turn}"}
                        ],
                    },
                )
                assert response.status_code == 200, response.text
            assert (
                tmp_path / "http-result.txt"
            ).read_text() == "http configured workspace verified"

            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    headless=True,
                    args=["--no-sandbox"],
                    executable_path=os.environ.get("FABRIC_TEST_CHROMIUM"),
                )
                try:
                    page = await browser.new_page()
                    await page.goto(
                        f"http://127.0.0.1:{unused_tcp_port}/#token={token}"
                    )
                    textarea = page.get_by_role(
                        "textbox", name="Chat composer", exact=True
                    )
                    await textarea.wait_for(timeout=20000)
                    for turn in (1, 2):
                        await scenario("browser", turn)
                        await textarea.fill(f"Browser parity turn {turn}")
                        await textarea.press("Enter")
                        await (
                            page.get_by_role("paragraph")
                            .filter(has_text=f"browser turn {turn} complete")
                            .wait_for(timeout=30000)
                        )
                        await page.wait_for_timeout(500)
                    await page.screenshot(path=str(tmp_path / "browser.png"))
                    (tmp_path / "browser-body.txt").write_text(
                        (await page.locator("body").inner_text()).replace(
                            token, "[redacted]"
                        )
                    )
                finally:
                    await page.screenshot(path=str(tmp_path / "browser.png"))
                    (tmp_path / "browser-body.txt").write_text(
                        (await page.locator("body").inner_text()).replace(
                            token, "[redacted]"
                        )
                    )
                    (tmp_path / "browser-model-requests.json").write_text(
                        json.dumps(
                            (await client.get(f"{api_server}/_requests")).json(),
                            indent=2,
                        )
                    )
                    await browser.close()
            assert (
                tmp_path / "browser-result.txt"
            ).read_text() == "browser configured workspace verified"

            def terminal_turns():
                child = pexpect.spawn(
                    str(directory / "terminal"),
                    ["--session", "parity-terminal"],
                    encoding="utf-8",
                    timeout=45,
                    dimensions=(32, 120),
                )
                try:
                    with (tmp_path / "terminal.log").open("w") as transcript:
                        child.logfile_read = transcript
                        child.expect(r"(?<!dis)connected", timeout=30)
                        for turn in (1, 2):
                            # The recorder is a separate server thread, so this sync request does not block Gateway work.
                            import requests

                            calls = (
                                [
                                    {
                                        "name": "read",
                                        "arguments": {"path": "source.txt"},
                                    },
                                    {
                                        "name": "write",
                                        "arguments": {
                                            "path": "terminal-result.txt",
                                            "content": "terminal configured workspace verified",
                                        },
                                    },
                                ]
                                if turn == 1
                                else [
                                    {
                                        "name": "tool_call",
                                        "arguments": {
                                            "id": "mcp:fabricprobe:fabricprobe__get_authorization_header",
                                            "args": {},
                                        },
                                    },
                                    {
                                        "name": "read",
                                        "arguments": {
                                            "path": str(
                                                Path(native_config.skills.paths[0])
                                                / "SKILL.md"
                                            )
                                        },
                                    },
                                ]
                            )
                            requests.post(
                                f"{api_server}/_scenario",
                                json={
                                    "tool_calls": calls,
                                    "response_text": f"terminal turn {turn} complete",
                                },
                                timeout=5,
                            ).raise_for_status()
                            child.send(f"Terminal parity turn {turn}\r")
                            child.expect(f"terminal turn {turn} complete", timeout=30)
                        child.send("/quit\r")
                        child.expect(pexpect.EOF, timeout=15)
                    child.close()
                    # The pinned client may use its own forced-exit fallback during disconnect.
                    assert child.exitstatus in {0, 130}
                finally:
                    if child.isalive():
                        child.terminate(force=True)

            await asyncio.to_thread(terminal_turns)
            assert (
                tmp_path / "terminal-result.txt"
            ).read_text() == "terminal configured workspace verified"
            assert len(runtime.invocations) == 2
            for level in ("live", "active", "ready"):
                report = await fabric.check(
                    environment=EnvironmentHandle.from_mapping(
                        runtime.handle.environment
                    ),
                    request=CheckRequest(
                        scope="runtime", runtime_id=runtime.runtime_id, level=level
                    ),
                )
                assert report.outcome == "passed", report.to_mapping()
                assert report.runtime["activity"] == "unknown"
            result = await runtime.invoke(
                input="Verify Gateway survives terminal detach"
            )
            assert result.status == "succeeded", result.to_mapping()
            assert len(runtime.invocations) == 3
            payloads = (await client.get(f"{api_server}/_requests")).json()
            (tmp_path / "captured-model-requests.json").write_text(
                json.dumps(payloads, indent=2)
            )
            assert len(payloads) == 25
            for payload in payloads:
                assert payload["model"] == "fabric-echo"
                assert [
                    message["content"]
                    for message in payload["messages"]
                    if message["role"] == "system"
                ] == [
                    marker
                    + "\n\nCurrent model identity: fabric-test/fabric-echo. If asked what model you are, answer with this value for the current run."
                ]
                assert payload["temperature"] == 0.2
                assert payload["top_p"] == 0.8
                assert (
                    payload.get("max_tokens", payload.get("max_completion_tokens"))
                    == 128
                )
                tools = {tool["function"]["name"] for tool in payload.get("tools", [])}
                assert {"read", "write"} <= tools
                assert not tools & {"exec", "process"}
            assert (
                await client.get(f"{api_server}/_chat_authorization_headers")
            ).json() == ["Bearer native-parity-test-key"] * len(payloads)
            assert (
                await client.get(f"{api_server}/_mcp_authorization_headers")
            ).json().count("Bearer native-mcp-test-key") >= 4
            messages = json.dumps(payloads)
            assert "FABRIC_SOURCE_MARKER_35291" in messages
            assert "FABRIC_SKILL_98214" in messages
    for name in (
        "terminal.running",
        "terminal",
        "gateway.json",
        "client.json",
        "dashboard.token",
    ):
        assert not (directory / name).exists()


async def test_native_mcp_and_skills_use_managed_configuration(
    tmp_path, api_server, native_config
):
    from nemo_fabric_adapters.openclaw import terminal

    fabric = Fabric()
    async with await fabric.start_runtime(native_config, base_dir=tmp_path) as runtime:
        async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
            await client.post(
                f"{api_server}/_scenario",
                json={
                    "tool_call": {
                        "name": "tool_search",
                        "arguments": {"query": "fabricprobe authorization header"},
                    }
                },
            )
            result = await runtime.invoke(
                input="Discover the configured MCP authorization tool."
            )
            assert result.status == "succeeded", result.to_mapping()
            payloads = (await client.get(f"{api_server}/_requests")).json()
            (tmp_path / "mcp-search-requests.json").write_text(
                json.dumps(payloads, indent=2)
            )
            tool_messages = [
                message
                for payload in payloads
                for message in payload["messages"]
                if message["role"] == "tool"
            ]
            assert any(
                "get_authorization_header" in str(message["content"])
                for message in tool_messages
            ), tool_messages
            await client.post(
                f"{api_server}/_scenario",
                json={
                    "tool_call": {
                        "name": "tool_call",
                        "arguments": {
                            "id": "mcp:fabricprobe:fabricprobe__get_authorization_header",
                            "args": {},
                        },
                    }
                },
            )
            result = await runtime.invoke(input="Call the discovered MCP tool.")
            assert result.status == "succeeded", result.to_mapping()
            assert (
                await client.get(f"{api_server}/_mcp_authorization_headers")
            ).json()[-1] == "Bearer native-mcp-test-key"
            forbidden = tmp_path / "forbidden-exec.txt"
            await client.post(
                f"{api_server}/_scenario",
                json={
                    "tool_call": {
                        "name": "tool_call",
                        "arguments": {
                            "id": "exec",
                            "args": {"command": f"touch {forbidden}"},
                        },
                    }
                },
            )
            result = await runtime.invoke(
                input="Check that blocked shell execution is rejected."
            )
            assert result.status == "succeeded", result.to_mapping()
            assert not forbidden.exists()
            payloads = (await client.get(f"{api_server}/_requests")).json()
            refusal = next(
                message["content"]
                for message in reversed(payloads[-1]["messages"])
                if message["role"] == "tool"
            )
            assert any(
                term in str(refusal).lower()
                for term in (
                    "unknown tool",
                    "not found",
                    "not allowed",
                    "denied",
                    "unavailable",
                )
            ), refusal
        connection = (
            tmp_path
            / "artifacts/.fabric/openclaw/runtimes"
            / runtime.runtime_id
            / "interfaces/gateway.json"
        )
        command, environment, _ = terminal.attachment(connection)
        process = await asyncio.create_subprocess_exec(
            command[0],
            "gateway",
            "call",
            "skills.status",
            "--params",
            '{"agentId":"reviewer"}',
            "--json",
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
        (tmp_path / "skills-status.txt").write_bytes(stdout)
        assert process.returncode == 0, stderr.decode()
        assert "native-parity" in stdout.decode()


async def test_four_clients_can_invoke_concurrently(
    tmp_path, api_server, unused_tcp_port, native_config
):
    import pexpect
    from playwright.async_api import async_playwright

    fabric = Fabric()
    async with await fabric.start_runtime(native_config, base_dir=tmp_path) as runtime:
        directory = (
            tmp_path
            / "artifacts/.fabric/openclaw/runtimes"
            / runtime.runtime_id
            / "interfaces"
        )
        token = (directory / "dashboard.token").read_text().strip()
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox"],
                executable_path=os.environ.get("FABRIC_TEST_CHROMIUM"),
            )
            child = None
            try:
                page = await browser.new_page()
                await page.goto(f"http://127.0.0.1:{unused_tcp_port}/#token={token}")
                composer = page.get_by_role("textbox", name="Chat composer", exact=True)
                await composer.wait_for(timeout=20000)
                child = pexpect.spawn(
                    str(directory / "terminal"),
                    ["--session", "concurrent-tui"],
                    encoding="utf-8",
                    timeout=30,
                    dimensions=(32, 120),
                )
                await asyncio.to_thread(child.expect, r"(?<!dis)connected")

                async def browser_turn():
                    await composer.fill("Browser concurrent request")
                    await composer.press("Enter")
                    await (
                        page.get_by_role("paragraph")
                        .filter(has_text="echo user_count=")
                        .last.wait_for(timeout=30000)
                    )

                async def terminal_turn():
                    child.send("Terminal concurrent request\r")
                    await asyncio.to_thread(child.expect, "echo user_count=")

                async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
                    sdk_result, http_result, _, _ = await asyncio.gather(
                        runtime.invoke(input="SDK concurrent request"),
                        client.post(
                            f"http://127.0.0.1:{unused_tcp_port}/v1/chat/completions",
                            headers={"Authorization": f"Bearer {token}"},
                            json={
                                "model": "openclaw/reviewer",
                                "user": "concurrent-http",
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "HTTP concurrent request",
                                    }
                                ],
                            },
                        ),
                        browser_turn(),
                        terminal_turn(),
                    )
                    assert sdk_result.status == "succeeded", sdk_result.to_mapping()
                    assert http_result.status_code == 200, http_result.text
                    assert len(runtime.invocations) == 1
                    payloads = (await client.get(f"{api_server}/_requests")).json()
                    assert len(payloads) == 4
                    for label in ("SDK", "HTTP", "Browser", "Terminal"):
                        assert (
                            sum(
                                label + " concurrent request"
                                in json.dumps(payload["messages"])
                                for payload in payloads
                            )
                            == 1
                        )
                    (tmp_path / "concurrent-model-requests.json").write_text(
                        json.dumps(payloads, indent=2)
                    )
                child.send("/quit\r")
                await asyncio.to_thread(child.expect, pexpect.EOF)
                child.close()
                assert child.exitstatus in {0, 130}
                result = await runtime.invoke(
                    input="SDK after concurrent native clients"
                )
                assert result.status == "succeeded", result.to_mapping()
            finally:
                if child is not None and child.isalive():
                    child.terminate(force=True)
                await browser.close()

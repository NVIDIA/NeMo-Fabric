# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in qualification of OpenClaw plugins through all four interfaces.

One runtime with the same ``harness.settings.plugins`` serves the SDK, the HTTP
API, the browser dashboard and the terminal TUI. A scripted model drives each
interface through ``tool_search`` and ``tool_call``, so the captured model
requests show whether each plugin's tool was offered, returned a fresh result,
and whether the external plugin's hook changed the prompt.

Set FABRIC_OPENCLAW_TEST_COMMAND to the pinned ``openclaw`` command. The browser
and terminal steps need Playwright and pexpect; set FABRIC_TEST_CHROMIUM to use
an installed Chromium or Chrome.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
from nemo_fabric import Fabric, FabricConfig

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("FABRIC_OPENCLAW_TEST_COMMAND"),
        reason="Set FABRIC_OPENCLAW_TEST_COMMAND to pinned OpenClaw for native qualification",
    ),
    pytest.mark.skipif(os.name != "posix", reason="The terminal client requires POSIX"),
]

BUNDLED = "memory-wiki"
EXTERNAL = "fabric-qual-probe"
# OpenClaw offers plugin tools through tool_search; tool_call runs them by id.
BUNDLED_TOOL = f"openclaw:{BUNDLED}:wiki_status"
EXTERNAL_TOOL = f"openclaw:{EXTERNAL}:fabric_qual_probe"
BUNDLED_RESULT = "Vault: ready"
DEPENDENCY_RESULT = "QUAL-DEP-OK"
HOOK_MARK = "QUAL-HOOK-OK"


@pytest.fixture(name="external_plugin")
def external_plugin_fixture(tmp_path: Path) -> Path:
    """A prepared plugin folder: built code plus its own node_modules.

    The tool's result comes from the ``qual-dep`` dependency, and a
    ``before_prompt_build`` hook adds a mark to every prompt.
    """
    directory = tmp_path / "plugins" / EXTERNAL
    directory.mkdir(parents=True)
    (directory / "openclaw.plugin.json").write_text(
        json.dumps(
            {
                "id": EXTERNAL,
                "name": EXTERNAL,
                "description": "NeMo Fabric plugin qualification probe",
                "contracts": {"tools": ["fabric_qual_probe"]},
                "configSchema": {"type": "object", "additionalProperties": False},
                "activation": {"onStartup": True},
            }
        )
    )
    (directory / "package.json").write_text(
        json.dumps(
            {
                "name": EXTERNAL,
                "version": "1.0.0",
                "private": True,
                "type": "module",
                "openclaw": {"extensions": ["./index.mjs"]},
                "dependencies": {"qual-dep": "1.0.0"},
            }
        )
    )
    dependency = directory / "node_modules" / "qual-dep"
    dependency.mkdir(parents=True)
    (dependency / "package.json").write_text(
        json.dumps(
            {
                "name": "qual-dep",
                "version": "1.0.0",
                "type": "module",
                "main": "index.js",
            }
        )
    )
    (dependency / "index.js").write_text(
        f"export function mark(word) {{ return `{DEPENDENCY_RESULT}:${{word}}`; }}\n"
    )
    (directory / "index.mjs").write_text(
        'import { mark } from "qual-dep";\n'
        "export default {\n"
        f'  id: "{EXTERNAL}",\n'
        f'  name: "{EXTERNAL}",\n'
        "  register(api) {\n"
        "    api.registerTool({\n"
        '      name: "fabric_qual_probe",\n'
        '      description: "Return the qualification probe marker for a word",\n'
        '      parameters: { type: "object", properties: { word: { type: "string" } },'
        ' required: ["word"], additionalProperties: false },\n'
        "      async execute(_id, params) {\n"
        '        return { content: [{ type: "text", text: mark(params.word) }] };\n'
        "      },\n"
        "    });\n"
        f'    api.on("before_prompt_build", () => ({{ prependContext: "{HOOK_MARK}" }}));\n'
        "  },\n"
        "};\n"
    )
    return directory


def plugin_settings(enabled: bool) -> dict[str, Any]:
    """The one plugin block every interface in the runtime shares."""
    return {
        "bundled": {"entries": {BUNDLED: {"enabled": enabled}}},
        "external": {
            # The path stays listed when the entry is off: a path alone
            # doesn't turn a plugin on.
            "paths": [f"plugins/{EXTERNAL}"],
            "entries": {
                EXTERNAL: {
                    "enabled": enabled,
                    "hooks": {"allowConversationAccess": True},
                }
            },
        },
    }


def _config(tmp_path: Path, api_server: str, port: int, enabled: bool) -> FabricConfig:
    os.environ["ADAPTER_PYTHON"] = sys.executable
    os.environ["FABRIC_TEST_MODEL_KEY"] = "plugin-qualification-key"
    return FabricConfig.from_mapping(
        {
            "metadata": {"name": "openclaw-plugin-qualification"},
            "harness": {
                "adapter_id": "nvidia.fabric.openclaw",
                "settings": {
                    "openclaw_command": os.environ["FABRIC_OPENCLAW_TEST_COMMAND"],
                    "agent_id": "qualifier",
                    "startup_timeout_seconds": 75,
                    "interfaces": {"dashboard": {"port": port}, "terminal": {}},
                    "plugins": plugin_settings(enabled),
                },
            },
            "models": {
                "default": {
                    "provider": "fabric-test",
                    "model": "fabric-echo",
                    "base_url": f"{api_server}/v1",
                    "api_key_env": "FABRIC_TEST_MODEL_KEY",
                    "max_tokens": 128,
                }
            },
            "environment": {
                "provider": "local",
                "workspace": str(tmp_path / "workspace"),
                "artifacts": str(tmp_path / "artifacts"),
            },
        }
    )


def scenario(interface: str, nonce: str) -> dict[str, Any]:
    """Search for both plugin tools, then call each one by id."""
    return {
        "tool_calls": [
            {
                "name": "tool_search",
                "arguments": {"query": "fabric_qual_probe wiki_status"},
            },
            {
                "name": "tool_call",
                "arguments": {"id": EXTERNAL_TOOL, "args": {"word": nonce}},
            },
            {"name": "tool_call", "arguments": {"id": BUNDLED_TOOL, "args": {}}},
        ],
        "response_text": f"{interface} plugin turn complete",
    }


def _tool_messages(request: dict[str, Any]) -> list[str]:
    return [
        json.dumps(message["content"])
        for message in request["messages"]
        if message["role"] == "tool"
    ]


def check_turn(
    requests: list[dict[str, Any]], nonce: str, enabled: bool
) -> dict[str, Any]:
    """Check one interface turn's model requests and return what they showed."""
    # One request per scripted step: search, two calls, and the final reply.
    # Requests without tools, such as title generation, aren't agent steps.
    turns = [request for request in requests if request.get("tools")]
    assert len(turns) == 4, f"expected 4 agent requests, got {len(turns)}"
    # The last request carries every tool result so far; drop the ones the
    # turn's first request already had, so only this turn's results count.
    tools = _tool_messages(turns[-1])[len(_tool_messages(turns[0])) :]
    prompts = json.dumps([turn["messages"] for turn in turns])
    searched = next((text for text in tools if "openclaw:" in text), "")
    seen = {
        "requests": len(turns),
        "external_tool_listed": EXTERNAL_TOOL in searched,
        "bundled_tool_listed": BUNDLED_TOOL in searched,
        "external_result": any(
            f"{DEPENDENCY_RESULT}:{nonce}" in text for text in tools
        ),
        "bundled_result": any(BUNDLED_RESULT in text for text in tools),
        # Per request, so a single request with the mark fails the "off" case.
        "hook_mark_in_prompt": [
            HOOK_MARK in json.dumps(turn["messages"]) for turn in turns
        ],
    }
    expected = {
        "external_tool_listed": enabled,
        "bundled_tool_listed": enabled,
        "external_result": enabled,
        "bundled_result": enabled,
        "hook_mark_in_prompt": [enabled] * len(turns),
    }
    assert searched, f"tool_search returned no tools: {tools}"
    assert {key: seen[key] for key in expected} == expected, (seen, tools)
    if not enabled:
        # Both calls reach the Gateway and fail; nothing from either plugin
        # appears, not even an earlier turn's result.
        assert sum("Unknown tool id" in text for text in tools) == 2, tools
        assert DEPENDENCY_RESULT not in prompts, tools
    return seen


async def _plugins_list(directory: Path) -> dict[str, dict[str, Any]]:
    """Ask the running Gateway for its plugins, as the terminal client would."""
    from nemo_fabric_adapters.openclaw import terminal

    command, environment, _ = terminal.attachment(directory / "gateway.json")
    process = await asyncio.create_subprocess_exec(
        command[0],
        "gateway",
        "call",
        "plugins.list",
        "--params",
        "{}",
        "--json",
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    assert process.returncode == 0, stderr.decode()
    listing = json.loads(stdout)
    return {
        plugin["id"]: {
            "origin": plugin.get("origin"),
            "enabled": plugin.get("enabled"),
            "state": (plugin.get("runtime") or {}).get("state"),
        }
        for plugin in listing.get("plugins", [])
        if plugin.get("id") in {BUNDLED, EXTERNAL}
    }


@pytest.mark.parametrize("enabled", [True, False], ids=["on", "off"])
@pytest.mark.usefixtures("external_plugin")
async def test_plugins_through_all_four_interfaces(
    tmp_path, api_server, unused_tcp_port, enabled
):
    pexpect = pytest.importorskip("pexpect")
    playwright_api = pytest.importorskip("playwright.async_api")

    (tmp_path / "workspace").mkdir()
    config = _config(tmp_path, api_server, unused_tcp_port, enabled)
    evidence: dict[str, Any] = {
        "openclaw_command": os.environ["FABRIC_OPENCLAW_TEST_COMMAND"],
        "plugins_setting": plugin_settings(enabled),
        "external_dependency": {"qual-dep": "1.0.0"},
        "turns": {},
    }

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        directory = (
            tmp_path
            / "artifacts/.fabric/openclaw/runtimes"
            / runtime.runtime_id
            / "interfaces"
        )
        token = (directory / "dashboard.token").read_text()
        listed = await _plugins_list(directory)
        evidence["plugins_list"] = listed
        if enabled:
            assert listed[BUNDLED]["state"] == "active", listed
            assert listed[EXTERNAL]["state"] == "active", listed
            assert listed[EXTERNAL]["origin"] != listed[BUNDLED]["origin"], listed
        else:
            for plugin in (BUNDLED, EXTERNAL):
                assert listed[plugin]["state"] != "active", listed

        async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
            seen_requests = 0

            async def new_requests() -> list[dict[str, Any]]:
                nonlocal seen_requests
                payloads = (await client.get(f"{api_server}/_requests")).json()
                fresh, seen_requests = payloads[seen_requests:], len(payloads)
                return fresh

            async def prepare(interface: str) -> str:
                nonce = f"{interface}-{uuid.uuid4().hex[:8]}"
                response = await client.post(
                    f"{api_server}/_scenario", json=scenario(interface, nonce)
                )
                response.raise_for_status()
                return nonce

            async def record(interface: str, nonce: str) -> None:
                fresh = await new_requests()
                (tmp_path / f"{interface}-model-requests.json").write_text(
                    json.dumps(fresh, indent=2)
                )
                evidence["turns"][interface] = {
                    "nonce": nonce,
                    **check_turn(fresh, nonce, enabled),
                }

            # SDK
            nonce = await prepare("sdk")
            result = await runtime.invoke(input="Run the plugin qualification turn")
            assert result.status == "succeeded", result.to_mapping()
            await record("sdk", nonce)

            # HTTP API
            nonce = await prepare("http")
            response = await client.post(
                f"http://127.0.0.1:{unused_tcp_port}/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "model": "openclaw/qualifier",
                    "user": "plugin-qualification-http",
                    "messages": [
                        {"role": "user", "content": "Run the plugin qualification turn"}
                    ],
                },
            )
            assert response.status_code == 200, response.text
            await record("http", nonce)

            # Browser dashboard
            async with playwright_api.async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    headless=True,
                    args=["--no-sandbox"],
                    executable_path=os.environ.get("FABRIC_TEST_CHROMIUM"),
                )
                page = await browser.new_page()
                try:
                    await page.goto(
                        f"http://127.0.0.1:{unused_tcp_port}/#token={token}"
                    )
                    composer = page.get_by_role(
                        "textbox", name="Chat composer", exact=True
                    )
                    await composer.wait_for(timeout=20000)
                    nonce = await prepare("browser")
                    await composer.fill("Run the plugin qualification turn")
                    await composer.press("Enter")
                    await (
                        page.get_by_role("paragraph")
                        .filter(has_text="browser plugin turn complete")
                        .wait_for(timeout=45000)
                    )
                    await page.wait_for_timeout(500)
                finally:
                    await page.screenshot(path=str(tmp_path / "browser.png"))
                    await browser.close()
            await record("browser", nonce)

            # Terminal TUI
            nonce = await prepare("terminal")

            def terminal_turn() -> None:
                child = pexpect.spawn(
                    str(directory / "terminal"),
                    ["--session", "plugin-qualification"],
                    encoding="utf-8",
                    timeout=60,
                    dimensions=(32, 120),
                )
                try:
                    with (tmp_path / "terminal.log").open("w") as transcript:
                        child.logfile_read = transcript
                        child.expect(r"(?<!dis)connected", timeout=30)
                        child.send("Run the plugin qualification turn\r")
                        child.expect("terminal plugin turn complete", timeout=60)
                        child.send("/quit\r")
                        child.expect(pexpect.EOF, timeout=15)
                    child.close()
                    # The pinned client may use its own forced-exit fallback.
                    assert child.exitstatus in {0, 130}
                finally:
                    if child.isalive():
                        child.terminate(force=True)

            await asyncio.to_thread(terminal_turn)
            await record("terminal", nonce)

    (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2))
    assert set(evidence["turns"]) == {"sdk", "http", "browser", "terminal"}

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification of Hermes plugins through the SDK and every native interface.

Hermes runs the API in a ``hermes gateway run`` child and the browser dashboard
and terminal in one shared TUI backend child; each child discovers plugins on
its own. One runtime with the same ``harness.settings.plugins`` serves them
all, with the SDK on or off. A scripted model calls the external plugin's tool
and writes a file the bundled ``security-guidance`` hook flags, so the
captured model requests show, per interface:

- whether the external tool was offered and returned a fresh result built from
  both of its dependencies, and which process ran it;
- whether the external plugin's ``pre_llm_call`` hook added its mark;
- whether ``security-guidance`` appended its warning to the write result.

The browser chat and the terminal client both talk to the TUI backend's
``/api/ws``. ``test_plugins_through_sdk_and_native_interfaces`` drives one turn
through that endpoint in place of both clients. ``test_plugins_through_real_browser_and_terminal_clients``
drives the real clients; it needs Hermes's built web and terminal assets,
Playwright, pexpect and pyte, and Chrome or Chromium in FABRIC_TEST_CHROMIUM.
Set FABRIC_HERMES_PLUGIN_QUALIFICATION=1 to also run the slower modes.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from nemo_fabric import Fabric, FabricConfig

pytestmark = [
    pytest.mark.usefixtures("requires_hermes_agent"),
    pytest.mark.skipif(os.name != "posix", reason="Hermes interfaces require POSIX"),
]

BUNDLED = "security-guidance"
EXTERNAL = "hermes-qual-probe"
TOOL = "hermes_qual_probe"
DEPENDENCY_RESULT = "QUAL-DEP-OK"
PATH_DEPENDENCY_RESULT = "QUAL-PATH-DEP-OK"
HOOK_MARK = "QUAL-HOOK-OK"
BUNDLED_WARNING = "Security guidance"
# Content that security-guidance flags when the agent writes it.
FLAGGED_CONTENT = (
    "import pickle\n\n\ndef load(path):\n    return pickle.load(open(path, 'rb'))\n"
)
PROMPT = "Run the plugin qualification turn"
MODEL_KEY = "plugin-qualification-key"


@pytest.fixture(name="external_plugin")
def external_plugin_fixture(tmp_path: Path) -> Path:
    """A prepared plugin folder with a vendored dependency.

    The tool's result combines the vendored ``qual_dep`` package, a
    ``qual_path_dep`` module found on ``PYTHONPATH``, and the name of the
    process that ran it. A ``pre_llm_call`` hook adds a mark to every turn.
    """
    directory = tmp_path / "mount" / EXTERNAL
    vendored = directory / "_vendor" / "qual_dep"
    vendored.mkdir(parents=True)
    (directory / "_vendor" / "__init__.py").write_text("")
    (vendored / "__init__.py").write_text(
        f"def mark(word):\n    return f'{DEPENDENCY_RESULT}:{{word}}'\n"
    )
    (directory / "plugin.yaml").write_text(
        f"name: {EXTERNAL}\n"
        "version: 1.0.0\n"
        'description: "NeMo Fabric plugin qualification probe"\n'
        f"provides_tools:\n  - {TOOL}\n"
        "provides_hooks:\n  - pre_llm_call\n"
    )
    (directory / "__init__.py").write_text(
        "import json\n"
        "import os\n"
        "import sys\n"
        "\n"
        "from ._vendor.qual_dep import mark\n"
        "\n"
        "\n"
        "def _handler(args, **_kwargs):\n"
        "    try:\n"
        "        import qual_path_dep\n"
        "\n"
        "        path_dependency = qual_path_dep.mark()\n"
        "    except ImportError:\n"
        "        path_dependency = 'QUAL-PATH-DEP-MISSING'\n"
        "    process = ' '.join(os.path.basename(arg) for arg in sys.argv[:2])\n"
        "    return json.dumps({'result': f\"{mark(args.get('word', ''))} \"\n"
        "                       f'{path_dependency} process={process}'})\n"
        "\n"
        "\n"
        "def _hook(**_kwargs):\n"
        f"    return {{'context': '{HOOK_MARK}'}}\n"
        "\n"
        "\n"
        "def register(ctx):\n"
        "    ctx.register_tool(\n"
        f"        name='{TOOL}',\n"
        f"        toolset='{TOOL}',\n"
        "        schema={\n"
        f"            'name': '{TOOL}',\n"
        "            'description': 'Return the qualification probe marker for a word',\n"
        "            'parameters': {\n"
        "                'type': 'object',\n"
        "                'properties': {'word': {'type': 'string'}},\n"
        "                'required': ['word'],\n"
        "            },\n"
        "        },\n"
        "        handler=_handler,\n"
        "    )\n"
        "    ctx.register_hook('pre_llm_call', _hook)\n"
    )
    return directory


@pytest.fixture(name="path_dependency")
def path_dependency_fixture(tmp_path: Path) -> Path:
    """A dependency provided on ``PYTHONPATH`` rather than in the plugin folder."""
    site = tmp_path / "site"
    package = site / "qual_path_dep"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f"def mark():\n    return '{PATH_DEPENDENCY_RESULT}'\n"
    )
    return site


def plugin_settings(
    plugin_path: Path, enabled: bool, isolation: str | None
) -> dict[str, Any]:
    """The one plugin block every interface in the runtime shares."""
    settings: dict[str, Any] = {
        "bundled": {"entries": {BUNDLED: {"enabled": enabled}}},
        "external": {
            # The path stays listed when the entry is off: a path alone
            # doesn't turn a plugin on.
            "paths": [str(plugin_path)],
            "entries": {EXTERNAL: {"enabled": enabled}},
        },
    }
    if isolation is not None:
        settings["isolation"] = isolation
    return settings


@pytest.fixture(name="qualification_config")
def qualification_config_fixture(
    tmp_path: Path,
    api_server: str,
    unused_tcp_port_factory: Callable[[], int],
    external_plugin: Path,
    path_dependency: Path,
) -> Callable[..., tuple[FabricConfig, dict[str, int]]]:
    """Build a runtime config with every Hermes interface and the plugin block."""
    os.environ["ADAPTER_PYTHON"] = sys.executable
    os.environ["FABRIC_TEST_MODEL_KEY"] = MODEL_KEY
    (tmp_path / "workspace").mkdir()

    def build(
        *, sdk: bool, enabled: bool, isolation: str | None = None
    ) -> tuple[FabricConfig, dict[str, int]]:
        ports = {"api": unused_tcp_port_factory(), "tui": unused_tcp_port_factory()}
        config = FabricConfig.from_mapping(
            {
                "metadata": {"name": "hermes-plugin-qualification"},
                "harness": {
                    "adapter_id": "nvidia.fabric.hermes",
                    "settings": {
                        "sdk": {"enabled": sdk},
                        "interfaces": {
                            "api": {"port": ports["api"]},
                            "dashboard": {"port": ports["tui"]},
                            "terminal": {"port": ports["tui"]},
                        },
                        "plugins": plugin_settings(external_plugin, enabled, isolation),
                    },
                },
                "models": {
                    "default": {
                        "provider": "custom",
                        "model": "fabric-echo",
                        "base_url": f"{api_server}/v1",
                        "api_key_env": "FABRIC_TEST_MODEL_KEY",
                    }
                },
                "environment": {
                    "provider": "local",
                    "workspace": str(tmp_path / "workspace"),
                    "artifacts": str(tmp_path / "artifacts"),
                    "env": {
                        "PYTHONPATH": str(path_dependency),
                        "FABRIC_TEST_MODEL_KEY": MODEL_KEY,
                    },
                },
                "runtime": {"artifacts": str(tmp_path / "artifacts")},
            }
        )
        return config, ports

    return build


def scenario(interface: str, nonce: str) -> dict[str, Any]:
    """Call the external tool through Hermes's deferred tools, then write a file."""
    return {
        "tool_calls": [
            {
                "name": "tool_call",
                "arguments": {"calls": [{"name": TOOL, "arguments": {"word": nonce}}]},
            },
            {
                "name": "write_file",
                "arguments": {"path": f"{nonce}.py", "content": FLAGGED_CONTENT},
            },
        ],
        "response_text": f"{interface} plugin turn complete",
    }


class ScriptedTurns:
    """Script one interface turn at a time and check its model requests."""

    def __init__(self, api_server: str, evidence_dir: Path, enabled: bool) -> None:
        self.api_server = api_server
        self.evidence_dir = evidence_dir
        self.enabled = enabled
        self.seen = 0
        self.turns: dict[str, Any] = {}

    def prepare(self, interface: str) -> str:
        nonce = f"{interface}-{uuid.uuid4().hex[:8]}"
        httpx.post(
            f"{self.api_server}/_scenario", json=scenario(interface, nonce), timeout=10
        ).raise_for_status()
        return nonce

    def record(self, interface: str, nonce: str, process: str) -> None:
        payloads = httpx.get(f"{self.api_server}/_requests", timeout=10).json()
        fresh, self.seen = payloads[self.seen :], len(payloads)
        (self.evidence_dir / f"{interface}-model-requests.json").write_text(
            json.dumps(fresh, indent=2)
        )
        self.turns[interface] = {
            "nonce": nonce,
            **check_turn(fresh, nonce, self.enabled, process),
        }


def check_turn(
    requests: list[dict[str, Any]], nonce: str, enabled: bool, process: str
) -> dict[str, Any]:
    """Check one interface turn's model requests and return what they showed."""
    # Hermes also sends tool-less auxiliary requests, such as title generation.
    turns = [request for request in requests if request.get("tools")]
    assert len(turns) == 3, f"expected 3 agent requests, got {len(turns)}"
    # The last request carries every tool result of the turn.
    probe, write = [
        message["content"]
        if isinstance(message["content"], str)
        else json.dumps(message["content"])
        for message in turns[-1]["messages"]
        if message["role"] == "tool"
    ]
    match = re.search(r"process=([^\"}]+)", probe)
    seen = {
        "requests": len(turns),
        "tool_offered": all(TOOL in json.dumps(turn["tools"]) for turn in turns),
        "tool_result": f"{DEPENDENCY_RESULT}:{nonce}" in probe,
        "path_dependency": PATH_DEPENDENCY_RESULT in probe,
        "process": match.group(1).strip() if match else None,
        "hook_mark": all(HOOK_MARK in json.dumps(turn["messages"]) for turn in turns),
        "bundled_warning": BUNDLED_WARNING in write,
    }
    expected = dict.fromkeys(
        (
            "tool_offered",
            "tool_result",
            "path_dependency",
            "hook_mark",
            "bundled_warning",
        ),
        enabled,
    )
    assert {key: seen[key] for key in expected} == expected, (seen, probe, write)
    # The write itself always runs; only the hook's warning depends on the plugin.
    assert '"bytes_written"' in write, write
    if enabled:
        assert seen["process"].startswith(process), seen
    else:
        assert "not a known tool name" in probe, probe
        assert DEPENDENCY_RESULT not in json.dumps(turns), probe
        assert HOOK_MARK not in json.dumps(turns), probe
    return seen


def _interfaces_dir(tmp_path: Path) -> Path:
    return next((tmp_path / "artifacts").rglob("interfaces/api.token")).parent


async def _api_turn(port: int, token: str) -> None:
    async with httpx.AsyncClient(trust_env=False, timeout=120) as client:
        response = await client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": "hermes-agent",
                "messages": [{"role": "user", "content": PROMPT}],
            },
        )
    assert response.status_code == 200, response.text


async def _tui_turn(port: int, token: str, workspace: Path) -> None:
    """One prompt through the TUI backend, as the browser chat and terminal do."""
    import aiohttp

    async with (
        aiohttp.ClientSession() as client,
        client.ws_connect(
            f"http://127.0.0.1:{port}/api/ws", params={"token": token}
        ) as ws,
    ):
        await ws.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session.create",
                "params": {"cwd": str(workspace)},
            }
        )
        while True:
            frame = await ws.receive_json(timeout=30)
            assert not (frame.get("id") == 1 and "error" in frame), frame
            if frame.get("id") == 1:
                session_id = frame["result"]["session_id"]
                break
        await ws.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "prompt.submit",
                "params": {"session_id": session_id, "text": PROMPT},
            }
        )
        while True:
            frame = await ws.receive_json(timeout=90)
            assert not (frame.get("id") == 2 and "error" in frame), frame
            if (
                frame.get("method") == "event"
                and frame["params"]["type"] == "message.complete"
            ):
                return


# Each mode starts three Hermes processes; the slower extra modes are opt-in.
FULL = pytest.mark.skipif(
    os.environ.get("FABRIC_HERMES_PLUGIN_QUALIFICATION") != "1",
    reason="Set FABRIC_HERMES_PLUGIN_QUALIFICATION=1 for every plugin mode",
)
MODES = [
    pytest.param(True, True, None, id="sdk-on-plugins-on"),
    pytest.param(True, False, None, id="sdk-on-plugins-off", marks=FULL),
    pytest.param(False, True, None, id="sdk-off-plugins-on"),
    pytest.param(False, False, None, id="sdk-off-plugins-off"),
    pytest.param(False, True, "host", id="sdk-off-plugin-host", marks=FULL),
]


@pytest.mark.parametrize(("sdk", "enabled", "isolation"), MODES)
async def test_plugins_through_sdk_and_native_interfaces(
    tmp_path, api_server, qualification_config, sdk, enabled, isolation
):
    if not os.environ.get("HERMES_WEB_DIST"):
        # The dashboard backend serves a frontend; the real browser client is
        # checked by the next test against Hermes's built one.
        dist = tmp_path / "web_dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html><body>fabric-test</body></html>")
        os.environ["HERMES_WEB_DIST"] = str(dist)
    config, ports = qualification_config(sdk=sdk, enabled=enabled, isolation=isolation)
    process = {"sdk": "adapter.py", "api": "hermes gateway", "tui": "hermes dashboard"}
    if isolation == "host":
        process = dict.fromkeys(process, "plugin_host_child.py")
    turns = ScriptedTurns(api_server, tmp_path, enabled)

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        interfaces = _interfaces_dir(tmp_path)
        native = (interfaces.parent / "config.yaml").read_text()
        # The adapter links the external folder into the Hermes home.
        assert (interfaces.parent / "plugins" / EXTERNAL).is_symlink()
        if sdk:
            nonce = turns.prepare("sdk")
            result = await runtime.invoke(input=PROMPT)
            assert result.status == "succeeded", result.to_mapping()
            turns.record("sdk", nonce, process["sdk"])
        else:
            result = await runtime.invoke(input=PROMPT)
            assert result.to_mapping()["error"]["code"] == "hermes_sdk_disabled", (
                result.to_mapping()
            )
        nonce = turns.prepare("api")
        await _api_turn(ports["api"], (interfaces / "api.token").read_text())
        turns.record("api", nonce, process["api"])
        nonce = turns.prepare("tui")
        await _tui_turn(
            ports["tui"], (interfaces / "tui.token").read_text(), tmp_path / "workspace"
        )
        turns.record("tui", nonce, process["tui"])

    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "sdk": sdk,
                "plugins_setting": plugin_settings(Path("<mount>"), enabled, isolation),
                "native_config": native,
                "turns": turns.turns,
            },
            indent=2,
        )
    )
    assert set(turns.turns) == ({"sdk"} if sdk else set()) | {"api", "tui"}
    if isolation == "host":
        assert "isolation: host" in native, native
    # Stop unlinks the external plugin from the Hermes home.
    assert not (interfaces.parent / "plugins" / EXTERNAL).exists()


def _wait_for_screen(child, screen, stream, text: str, timeout: float) -> str:
    """Feed the terminal client's output to an emulated screen until it shows text.

    The terminal client redraws only changed cells, so the raw byte stream
    doesn't hold the rendered words.
    """
    import pexpect

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            stream.feed(child.read_nonblocking(65536, timeout=0.5))
        except pexpect.TIMEOUT:
            pass
        rendered = "\n".join(screen.display)
        if text in rendered:
            return rendered
    raise AssertionError(
        f"terminal never showed {text!r}:\n" + "\n".join(screen.display)
    )


@pytest.mark.parametrize("enabled", [True, False], ids=["plugins-on", "plugins-off"])
async def test_plugins_through_real_browser_and_terminal_clients(
    tmp_path, api_server, qualification_config, enabled
):
    pexpect = pytest.importorskip("pexpect")
    pyte = pytest.importorskip("pyte")
    playwright_api = pytest.importorskip("playwright.async_api")
    import hermes_cli
    from nemo_fabric_adapters.hermes.terminal import frontend_command

    web_dist = Path(
        os.environ.get("HERMES_WEB_DIST")
        or Path(hermes_cli.__file__).parent / "web_dist"
    )
    if not (web_dist / "index.html").is_file():
        pytest.skip(f"Hermes's built dashboard frontend is needed at {web_dist}")
    try:
        frontend_command()
    except ValueError as error:
        pytest.skip(f"Hermes's built terminal client is needed: {error}")
    config, ports = qualification_config(sdk=False, enabled=enabled)
    turns = ScriptedTurns(api_server, tmp_path, enabled)

    async with await Fabric().start_runtime(config, base_dir=tmp_path):
        interfaces = _interfaces_dir(tmp_path)

        async with playwright_api.async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox"],
                executable_path=os.environ.get("FABRIC_TEST_CHROMIUM"),
            )
            page = await browser.new_page(viewport={"width": 1400, "height": 900})
            try:
                # On loopback the dashboard puts its session token in the page.
                await page.goto(f"http://127.0.0.1:{ports['tui']}/chat")
                composer = page.locator(".xterm-helper-textarea")
                await composer.wait_for(state="attached", timeout=30000)
                await page.wait_for_function(
                    "() => document.body.innerText.includes('Nous Research')",
                    timeout=30000,
                )
                await page.wait_for_timeout(3000)
                nonce = turns.prepare("browser")
                await composer.focus()
                await page.keyboard.type(PROMPT)
                await page.keyboard.press("Enter")
                await page.wait_for_function(
                    "() => document.body.innerText.includes('browser plugin turn complete')",
                    timeout=90000,
                )
                await page.wait_for_timeout(1000)
            finally:
                await page.screenshot(path=str(tmp_path / "browser.png"))
                await browser.close()
        turns.record("browser", nonce, "hermes dashboard")

        nonce = turns.prepare("terminal")

        def terminal_turn() -> None:
            screen = pyte.Screen(160, 45)
            stream = pyte.Stream(screen)
            child = pexpect.spawn(
                sys.executable,
                [
                    "-m",
                    "nemo_fabric_adapters.hermes.terminal",
                    "--connection",
                    str(interfaces / "tui.json"),
                ],
                encoding="utf-8",
                timeout=90,
                dimensions=(45, 160),
            )
            try:
                _wait_for_screen(child, screen, stream, "ready", 60)
                time.sleep(2)  # let the composer take input
                child.send(PROMPT + "\r")
                rendered = _wait_for_screen(
                    child, screen, stream, "terminal plugin turn complete", 90
                )
                (tmp_path / "terminal-screen.txt").write_text(rendered)
            finally:
                child.terminate(force=True)

        await asyncio.to_thread(terminal_turn)
        turns.record("terminal", nonce, "hermes dashboard")

    (tmp_path / "evidence.json").write_text(
        json.dumps({"sdk": False, "enabled": enabled, "turns": turns.turns}, indent=2)
    )
    assert set(turns.turns) == {"browser", "terminal"}

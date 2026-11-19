# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenClaw plugins against the pinned Gateway and a live model.

Opt-in: set FABRIC_OPENCLAW_TEST_COMMAND to the pinned ``openclaw`` command,
FABRIC_OPENCLAW_LIVE_BASE_URL to an OpenAI-compatible endpoint, and
FABRIC_OPENCLAW_LIVE_MODEL to a model that calls tools reliably. Set
FABRIC_OPENCLAW_LIVE_API_KEY when the endpoint needs a key.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric.errors import FabricRuntimeError

pytestmark = [
    pytest.mark.skipif(
        not (
            os.environ.get("FABRIC_OPENCLAW_TEST_COMMAND")
            and os.environ.get("FABRIC_OPENCLAW_LIVE_BASE_URL")
            and os.environ.get("FABRIC_OPENCLAW_LIVE_MODEL")
        ),
        reason="Set FABRIC_OPENCLAW_TEST_COMMAND, FABRIC_OPENCLAW_LIVE_BASE_URL "
        "and FABRIC_OPENCLAW_LIVE_MODEL for live OpenClaw plugin checks",
    ),
    pytest.mark.skipif(sys.platform == "win32", reason="Uses POSIX fixtures"),
]

PROBE_PLUGIN_MANIFEST = {
    "configSchema": {"type": "object", "additionalProperties": False},
    "activation": {"onStartup": True},
}


def _config(
    tmp_path: Path, plugins: dict[str, Any] | None, *, keep_state: bool = False
) -> FabricConfig:
    os.environ["ADAPTER_PYTHON"] = sys.executable
    os.environ.setdefault("FABRIC_OPENCLAW_LIVE_API_KEY", "not-used")
    settings: dict[str, Any] = {
        "openclaw_command": os.environ["FABRIC_OPENCLAW_TEST_COMMAND"],
        "startup_timeout_seconds": 75,
    }
    if plugins is not None:
        settings["plugins"] = plugins
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    mapping: dict[str, Any] = {
        "metadata": {"name": "openclaw-plugins-live"},
        "harness": {"adapter_id": "nvidia.fabric.openclaw", "settings": settings},
        "models": {
            "default": {
                "provider": "live",
                "model": os.environ["FABRIC_OPENCLAW_LIVE_MODEL"],
                "base_url": os.environ["FABRIC_OPENCLAW_LIVE_BASE_URL"],
                "api_key_env": "FABRIC_OPENCLAW_LIVE_API_KEY",
                "max_tokens": 4096,
            }
        },
        "environment": {"provider": "local", "workspace": str(workspace)},
    }
    if keep_state:
        mapping["runtime"] = {
            "keep_state": True,
            "artifacts": str(tmp_path / "artifacts"),
        }
    return FabricConfig.from_mapping(mapping)


def _transcript_events(state_dir: Path) -> list[dict[str, Any]]:
    """Return every transcript event in OpenClaw's per-agent session stores."""

    zstandard = pytest.importorskip("zstandard")
    events: list[dict[str, Any]] = []
    for database in state_dir.glob("agents/*/agent/openclaw-agent.sqlite"):
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "select event_json, event_zstd from transcript_events order by seq"
            ).fetchall()
        finally:
            connection.close()
        for text, compressed in rows:
            if text is None:
                text = zstandard.ZstdDecompressor().decompress(compressed).decode()
            events.append(json.loads(text))
    return events


def _tool_results(state_dir: Path, tool: str) -> list[str]:
    """Return the result text of every ``tool`` call the agent made.

    A direct call is a ``toolResult`` message; a call made through tool search
    is recorded as a nested tool under the ``tool_call`` result.
    """

    results: list[str] = []
    for event in _transcript_events(state_dir):
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        if message.get("role") == "toolResult" and message.get("toolName") == tool:
            results.append(json.dumps(message.get("content")))
        details = message.get("details")
        if (
            message.get("customType") == "openclaw.nested-tool.v1"
            and isinstance(details, dict)
            and details.get("toolName") == tool
        ):
            results.append(json.dumps(details.get("result")))
    return results


async def _plugins_list(config_path: Path, state_dir: Path) -> dict[str, dict]:
    process = await asyncio.create_subprocess_exec(
        os.environ["FABRIC_OPENCLAW_TEST_COMMAND"],
        "plugins",
        "list",
        "--json",
        env={
            **os.environ,
            "OPENCLAW_CONFIG_PATH": str(config_path),
            "OPENCLAW_STATE_DIR": str(state_dir),
            "OPENCLAW_CONFIG_READONLY": "1",
        },
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 120)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    assert process.returncode == 0, stderr.decode()
    return {plugin["id"]: plugin for plugin in json.loads(stdout)["plugins"]}


def _write_plugin(
    directory: Path, plugin_id: str, tool: str, body: str, dependency: bool
) -> None:
    directory.mkdir(parents=True)
    (directory / "openclaw.plugin.json").write_text(
        json.dumps(
            {
                "id": plugin_id,
                "name": plugin_id,
                "description": "NeMo Fabric test plugin",
                "contracts": {"tools": [tool]},
                **PROBE_PLUGIN_MANIFEST,
            }
        )
    )
    (directory / "package.json").write_text(
        json.dumps(
            {
                "name": plugin_id,
                "version": "1.0.0",
                "private": True,
                "type": "module",
                "openclaw": {"extensions": ["./index.mjs"]},
                **({"dependencies": {"probe-dep": "1.0.0"}} if dependency else {}),
            }
        )
    )
    (directory / "index.mjs").write_text(body)
    if dependency:
        package = directory / "node_modules" / "probe-dep"
        package.mkdir(parents=True)
        (package / "package.json").write_text(
            json.dumps(
                {
                    "name": "probe-dep",
                    "version": "1.0.0",
                    "type": "module",
                    "main": "index.js",
                }
            )
        )
        (package / "index.js").write_text(
            "export function mark(word) { return `MOUNTED-DEP-OK:${word}`; }\n"
        )


def _tool_plugin(
    directory: Path, plugin_id: str, tool: str, marker: str, *, dependency: bool
) -> Path:
    if dependency:
        result = "mark(params.word)"
        header = 'import { mark } from "probe-dep";\n'
    else:
        result = f"`{marker}:${{params.word}}`"
        header = ""
    body = (
        header
        + "export default {\n"
        + f'  id: "{plugin_id}",\n'
        + f'  name: "{plugin_id}",\n'
        + "  register(api) {\n"
        + "    api.registerTool({\n"
        + f'      name: "{tool}",\n'
        + '      description: "Return the probe marker for a word",\n'
        + '      parameters: { type: "object", properties: { word: { type: "string" } },'
        + ' required: ["word"], additionalProperties: false },\n'
        + "      async execute(_id, params) {\n"
        + f'        return {{ content: [{{ type: "text", text: {result} }}] }};\n'
        + "      },\n"
        + "    });\n"
        + "  },\n"
        + "};\n"
    )
    _write_plugin(directory, plugin_id, tool, body, dependency)
    return directory


# Bundled plugins


async def test_bundled_plugins_turn_on_and_off(tmp_path: Path, monkeypatch):
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp_root))
    config = _config(
        tmp_path,
        {
            "bundled": {
                "entries": {
                    "memory-wiki": {"enabled": True},
                    "github": {"enabled": False},
                }
            }
        },
    )

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        (config_path,) = temp_root.glob("nemo-fabric-openclaw-*/openclaw.json")
        state_dir = config_path.parent / "state"
        plugins = await _plugins_list(config_path, state_dir)
        assert plugins["memory-wiki"]["status"] == "loaded", plugins["memory-wiki"]
        assert plugins["github"]["status"] == "disabled", plugins["github"]

        result = await runtime.invoke(
            input="Call the wiki_status tool once, then reply with the single word DONE."
        )
        assert result.status == "succeeded", result.to_mapping()
        (status,) = _tool_results(state_dir, "wiki_status")
        assert "Vault: ready" in status, status


# External plugins


async def test_external_plugin_loads_with_its_own_dependencies(
    tmp_path: Path, monkeypatch
):
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setenv("TMPDIR", str(temp_root))
    # The volume carries built JavaScript and its node_modules; OpenClaw
    # installs nothing for a local-path plugin.
    _tool_plugin(
        tmp_path / "volume" / "plugins" / "fabric-mounted",
        "fabric-mounted",
        "fabric_mounted",
        "unused",
        dependency=True,
    )
    config = _config(
        tmp_path,
        {
            "external": {
                "paths": ["volume/plugins/fabric-mounted"],
                "entries": {"fabric-mounted": {"enabled": True}},
            }
        },
    )

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        (config_path,) = temp_root.glob("nemo-fabric-openclaw-*/openclaw.json")
        state_dir = config_path.parent / "state"
        result = await runtime.invoke(
            input="Call the fabric_mounted tool with the word heron, then reply "
            "with exactly the text the tool returned."
        )
        assert result.status == "succeeded", result.to_mapping()
        (output,) = _tool_results(state_dir, "fabric_mounted")
        assert "MOUNTED-DEP-OK:heron" in output, output


async def test_external_plugin_missing_a_dependency_fails_start(tmp_path: Path):
    plugin = _tool_plugin(
        tmp_path / "volume" / "plugins" / "fabric-broken",
        "fabric-broken",
        "fabric_broken",
        "unused",
        dependency=True,
    )
    for path in sorted((plugin / "node_modules").rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    (plugin / "node_modules").rmdir()
    config = _config(
        tmp_path,
        {
            "external": {
                "paths": ["volume/plugins/fabric-broken"],
                "entries": {"fabric-broken": {"enabled": True}},
            }
        },
    )

    # The running Gateway reports the plugin's own load error.
    with pytest.raises(FabricRuntimeError) as caught:
        await Fabric().start_runtime(config, base_dir=tmp_path)
    assert caught.value.code == "openclaw_plugin_unavailable", caught.value
    # The SDK carries the adapter metadata in the error message.
    message = str(caught.value)
    assert "harness.settings.plugins.external.entries.fabric-broken" in message, message
    assert "Plugin dependency probe-dep is missing" in message, message

    # A plugin id the Gateway doesn't know fails as not found.
    config = _config(
        tmp_path, {"bundled": {"entries": {"memory-wikki": {"enabled": True}}}}
    )
    with pytest.raises(FabricRuntimeError) as caught:
        await Fabric().start_runtime(config, base_dir=tmp_path)
    assert caught.value.code == "openclaw_plugin_not_found", caught.value
    assert "harness.settings.plugins.bundled.entries.memory-wikki" in str(caught.value)


# Agent-installed plugins


def _retained_config_path(tmp_path: Path) -> Path:
    (path,) = (tmp_path / "artifacts").rglob("fabric/openclaw.json")
    return path


async def test_agent_installed_plugin_survives_restart_until_disabled(
    tmp_path: Path,
):
    fixture = _tool_plugin(
        tmp_path / "downloads" / "fabric-probe",
        "fabric-probe",
        "fabric_probe",
        "PROBE-OK",
        dependency=False,
    )
    command = os.environ["FABRIC_OPENCLAW_TEST_COMMAND"]
    config = _config(tmp_path, None, keep_state=True)
    fabric = Fabric()

    async with await fabric.start_runtime(config, base_dir=tmp_path) as runtime:
        config_path = _retained_config_path(tmp_path)
        state_dir = config_path.parent.parent
        installed = await runtime.invoke(
            input=f"Use your exec tool to run exactly this shell command:\n"
            f"{shlex.join([command, 'plugins', 'install', str(fixture)])}\n"
            "Then reply with the last line of its output."
        )
        assert installed.status == "succeeded", installed.to_mapping()
        policy = json.loads((config_path.parent / "plugins.json").read_text())
        assert policy["entries"]["fabric-probe"]["enabled"] is True, policy
        assert json.loads(config_path.read_text())["plugins"] == {
            "$include": "./plugins.json"
        }
        # The running Gateway applies the install without a restart.
        used = await runtime.invoke(
            input="Call the fabric_probe tool with the word otter, then reply "
            "with exactly the text the tool returned."
        )
        assert used.status == "succeeded", used.to_mapping()
        (output,) = _tool_results(state_dir, "fabric_probe")
        assert "PROBE-OK:otter" in output, output

    # Fabric adds an entry of its own next to the agent's install.
    with_wiki = _config(
        tmp_path,
        {"bundled": {"entries": {"memory-wiki": {"enabled": True}}}},
        keep_state=True,
    )
    async with await fabric.start_runtime(with_wiki, base_dir=tmp_path) as runtime:
        config_path = _retained_config_path(tmp_path)
        policy = json.loads((config_path.parent / "plugins.json").read_text())
        assert policy["entries"]["memory-wiki"]["enabled"] is True, policy
        assert policy["entries"]["fabric-probe"]["enabled"] is True, policy
        plugins = await _plugins_list(config_path, state_dir)
        assert plugins["fabric-probe"]["status"] == "loaded", plugins["fabric-probe"]
        used = await runtime.invoke(
            input="Call the fabric_probe tool with the word lynx, then reply "
            "with exactly the text the tool returned."
        )
        assert used.status == "succeeded", used.to_mapping()
        outputs = _tool_results(state_dir, "fabric_probe")
        assert any("PROBE-OK:lynx" in output for output in outputs), outputs
        disabled = await runtime.invoke(
            input=f"Use your exec tool to run exactly this shell command:\n"
            f"{shlex.join([command, 'plugins', 'disable', 'fabric-probe'])}\n"
            "Then reply with the last line of its output."
        )
        assert disabled.status == "succeeded", disabled.to_mapping()
        policy = json.loads((config_path.parent / "plugins.json").read_text())
        assert policy["entries"]["fabric-probe"]["enabled"] is False, policy

    # Fabric drops its entry; the agent's disabled plugin stays as the agent left it.
    async with await fabric.start_runtime(config, base_dir=tmp_path) as runtime:
        config_path = _retained_config_path(tmp_path)
        policy = json.loads((config_path.parent / "plugins.json").read_text())
        assert "memory-wiki" not in policy["entries"], policy
        assert policy["entries"]["fabric-probe"]["enabled"] is False, policy
        plugins = await _plugins_list(config_path, state_dir)
        assert plugins["fabric-probe"]["status"] == "disabled", plugins["fabric-probe"]

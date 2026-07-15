# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Hermes's native and transparent Relay execution strategies."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

import nemo_fabric_adapters.common.utils as common_utils
from nemo_fabric_adapters.hermes import adapter


def _payload(
    tmp_path: Path, *, launch_mode: str = "native_plugin"
) -> dict[str, object]:
    return {
        "effective_config": {
            "config_root": str(tmp_path),
            "config": {
                "harness": {
                    "settings": {
                        "relay_launch_mode": launch_mode,
                        "model": "default",
                        "enabled_toolsets": ["terminal"],
                    }
                },
                "models": {
                    "default": {
                        "provider": "test",
                        "model": "test-model",
                        "api_key_env": "TEST_API_KEY",
                        "settings": {"base_url": "https://models.example/v1"},
                    }
                },
            },
        },
        "runtime_context": {
            "runtime_id": "runtime-123",
            "environment": {"workspace": str(tmp_path)},
        },
        "request": {"input": "contact alice@example.com"},
        "capability_plan": {"native": {}},
    }


def test_relay_launch_mode_defaults_and_rejects_unknown_values():
    assert adapter._relay_launch_mode({}) == adapter.NATIVE_PLUGIN_MODE
    assert (
        adapter._relay_launch_mode({"relay_launch_mode": "cli_wrapper"})
        == adapter.CLI_WRAPPER_MODE
    )
    with pytest.raises(ValueError, match="unsupported relay_launch_mode"):
        adapter._relay_launch_mode({"relay_launch_mode": "sidecar"})


async def test_cli_wrapper_requires_relay_telemetry(tmp_path: Path):
    payload = _payload(tmp_path, launch_mode="cli_wrapper")
    payload["telemetry_plan"] = {"providers": [], "relay_enabled": False}

    with pytest.raises(RuntimeError, match="requires Relay telemetry"):
        await adapter.run_hermes(payload)


def test_build_relay_hermes_command_preserves_order_and_forces_custom_provider(
    tmp_path: Path,
):
    launch = adapter.RelayCliLaunch(
        executable=tmp_path / "nemo-relay",
        config_path=tmp_path / "config.toml",
        plugin_config_path=tmp_path / "plugins.toml",
        env={},
        activation_receipt=[],
    )
    payload = _payload(tmp_path, launch_mode="cli_wrapper")
    settings = common_utils.settings_payload(payload)
    model_config = common_utils.selected_model_config(payload)

    command = adapter.build_relay_hermes_command(
        launch=launch,
        payload=payload,
        settings=settings,
        model_config=model_config,
        user_message="secret prompt",
    )

    assert command == [
        str(tmp_path / "nemo-relay"),
        "run",
        "--config",
        str(tmp_path / "config.toml"),
        "--agent",
        "hermes",
        "--plugin-config-path",
        str(tmp_path / "plugins.toml"),
        "--",
        "chat",
        "--quiet",
        "--query",
        "secret prompt",
        "--continue",
        "runtime-123",
        "--model",
        "test-model",
        "--provider",
        "custom",
        "--toolsets",
        "terminal",
    ]
    assert "secret prompt" not in adapter.redact_command(command)


def test_fake_cli_subprocess_proves_relay_starts_hermes(tmp_path: Path, monkeypatch):
    hermes_log = tmp_path / "hermes-args.json"
    hermes = tmp_path / "hermes"
    hermes.write_text(
        """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
Path(os.environ[\"FAKE_HERMES_LOG\"]).write_text(json.dumps(sys.argv[1:]))
print(\"fake hermes response\")
""",
        encoding="utf-8",
    )
    hermes.chmod(0o755)
    relay = tmp_path / "nemo-relay"
    relay.write_text(
        """#!/usr/bin/env python3
import subprocess, sys, tomllib
if \"--version\" in sys.argv:
    print(\"nemo-relay 0.6.0\")
    raise SystemExit(0)
args = sys.argv[1:]
config_path = args[args.index(\"--config\") + 1]
with open(config_path, \"rb\") as stream:
    config = tomllib.load(stream)
child = [config[\"agents\"][\"hermes\"][\"command\"], *args[args.index(\"--\") + 1:]]
raise SystemExit(subprocess.run(child).returncode)
""",
        encoding="utf-8",
    )
    relay.chmod(0o755)
    relay_wrapper = tmp_path / "relay.json"
    relay_wrapper.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FABRIC_RELAY_CONFIG_PATH", str(relay_wrapper))

    payload = _payload(tmp_path, launch_mode="cli_wrapper")
    settings = common_utils.settings_payload(payload)
    settings["relay_cli_command"] = str(relay)
    settings["hermes_command"] = str(hermes)
    settings["env"] = {"FAKE_HERMES_LOG": str(hermes_log)}
    launch = adapter.prepare_relay_cli_launch(
        payload=payload,
        settings=settings,
        model_config=common_utils.selected_model_config(payload),
        hermes_home=tmp_path / "hermes-home",
        hermes_config_path=tmp_path / "hermes-home" / "config.yaml",
        plugin_config={
            "version": 1,
            "components": [
                {
                    "kind": "pii_redaction",
                    "enabled": True,
                    "config": {"version": 1, "mode": "builtin", "mark": True},
                }
            ],
        },
        dynamic_plugins=[],
    )
    command = adapter.build_relay_hermes_command(
        launch=launch,
        payload=payload,
        settings=settings,
        model_config=common_utils.selected_model_config(payload),
        user_message="hello",
    )

    result, _, _ = adapter.invoke_relay_wrapped_hermes(
        command=command,
        cwd=tmp_path,
        env=launch.env,
    )

    assert result["completed"] is True
    assert result["response"] == "fake hermes response"
    child_args = json.loads(hermes_log.read_text(encoding="utf-8"))
    assert child_args[:4] == ["chat", "--quiet", "--query", "hello"]
    assert child_args[child_args.index("--provider") + 1] == "custom"
    with launch.config_path.open("rb") as stream:
        relay_config = tomllib.load(stream)
    assert relay_config["agents"]["hermes"]["command"] == str(hermes)
    assert relay_config["upstream"]["openai_base_url"] == "https://models.example/v1"


def test_native_plugin_config_layers_dynamic_components_in_order(
    tmp_path: Path, monkeypatch
):
    relay_wrapper = tmp_path / "relay.json"
    relay_wrapper.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FABRIC_RELAY_CONFIG_PATH", str(relay_wrapper))
    base = {
        "version": 1,
        "components": [{"kind": "observability", "config": {"version": 2}}],
    }

    path = adapter.write_native_relay_plugin_config(
        base,
        [
            {
                "plugin_id": "example.first",
                "kind": "rust_dynamic",
                "manifest_ref": "first.toml",
                "config": {"sequence": 1},
            },
            {
                "plugin_id": "example.second",
                "kind": "worker",
                "manifest_ref": "second.toml",
                "config": {"sequence": 2},
            },
        ],
    )

    with path.open("rb") as stream:
        document = tomllib.load(stream)
    assert [component["kind"] for component in document["components"]] == [
        "observability",
        "example.first",
        "example.second",
    ]
    assert document["components"][1]["config"] == {"sequence": 1}
    assert document["components"][2]["config"] == {"sequence": 2}
    assert base["components"] == [
        {"kind": "observability", "config": {"version": 2}}
    ]


def test_native_plugin_environment_is_invocation_scoped(tmp_path: Path, monkeypatch):
    name = "HERMES_NEMO_RELAY_PLUGINS_TOML"
    monkeypatch.setenv(name, "parent.toml")

    with adapter.native_relay_plugin_environment(tmp_path / "invocation.toml"):
        assert adapter.os.environ[name] == str(tmp_path / "invocation.toml")

    assert adapter.os.environ[name] == "parent.toml"


class _Activation:
    def __init__(self) -> None:
        self.report = {"diagnostics": []}
        self.entered = False
        self.closed = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *_args):
        self.closed = True


async def test_native_static_configuration_keeps_existing_plugin_context(
    tmp_path: Path,
    monkeypatch,
):
    relay_wrapper = tmp_path / "relay.json"
    relay_wrapper.write_text(
        json.dumps({"relay": {"config": {"version": 1, "components": []}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("FABRIC_RELAY_CONFIG_PATH", str(relay_wrapper))
    monkeypatch.setenv("TEST_API_KEY", "not-a-real-secret")
    payload = _payload(tmp_path)
    payload["telemetry_plan"] = {"providers": ["relay"], "relay_enabled": True}
    activation = _Activation()
    plugin_context = MagicMock(return_value=activation)
    fake_relay = ModuleType("nemo_relay")
    fake_relay.plugin = SimpleNamespace(plugin=plugin_context)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nemo_relay", fake_relay)
    monkeypatch.setattr(
        common_utils, "relay_api_plugin_config", lambda _config: "base-config"
    )
    monkeypatch.setattr(
        adapter,
        "_invoke_hermes",
        MagicMock(
            return_value=(
                {"response": "ok", "completed": True, "failed": False, "messages": []},
                [],
                [],
                "",
            )
        ),
    )

    output = await adapter.run_hermes(payload)

    plugin_context.assert_called_once_with("base-config")
    assert activation.entered is True
    assert activation.closed is True
    assert output["relay_launch_mode"] == "native_plugin"


async def test_native_dynamic_plugins_use_owned_activation_for_complete_call(
    tmp_path: Path,
    monkeypatch,
):
    relay_wrapper = tmp_path / "relay.json"
    relay_wrapper.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {"version": 1, "components": []},
                    "dynamic_plugins": [
                        {
                            "plugin_id": "example.fixture",
                            "kind": "rust_dynamic",
                            "manifest_ref": "fixture/relay-plugin.toml",
                            "config": {"mode": "test"},
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FABRIC_RELAY_CONFIG_PATH", str(relay_wrapper))
    monkeypatch.setenv("TEST_API_KEY", "not-a-real-secret")
    payload = _payload(tmp_path)
    payload["telemetry_plan"] = {"providers": ["relay"], "relay_enabled": True}
    activation = _Activation()
    initialize = AsyncMock(return_value=activation)
    fake_plugin = SimpleNamespace(initialize_with_dynamic_plugins=initialize)
    fake_relay = ModuleType("nemo_relay")
    fake_relay.plugin = fake_plugin  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nemo_relay", fake_relay)
    monkeypatch.setattr(
        common_utils, "relay_api_plugin_config", lambda _config: "base-config"
    )
    monkeypatch.setattr(
        common_utils, "relay_api_dynamic_plugins", lambda _specs: ["dynamic-spec"]
    )
    invoke = MagicMock(
        return_value=(
            {"response": "ok", "completed": True, "failed": False, "messages": []},
            [],
            [],
            "",
        )
    )
    monkeypatch.setattr(adapter, "_invoke_hermes", invoke)

    output = await adapter.run_hermes(payload)

    initialize.assert_awaited_once_with("base-config", ["dynamic-spec"])
    assert activation.entered is True
    assert activation.closed is True
    invoke.assert_called_once()
    assert output["relay_runtime"]["emitter"] == "hermes.observability/nemo_relay"
    assert output["relay_runtime"]["activation_report"] == {"diagnostics": []}

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenClaw native plugin settings: bundled, mounted and agent-installed plugins."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from _utils.configs import minimal_config
from nemo_fabric import Fabric
from nemo_fabric import FabricConfigError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.openclaw import adapter
from nemo_fabric_adapters.openclaw import plugins as openclaw_plugins

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="Runs the mock openclaw script directly"
)


@pytest.fixture(name="mock_openclaw")
def mock_openclaw_fixture(repo_root: Path) -> Path:
    return repo_root / "tests/_utils/mock_openclaw.py"


@pytest.fixture(name="capture")
def capture_fixture(tmp_path: Path) -> Path:
    os.environ["FAKE_OPENCLAW_CAPTURE"] = str(tmp_path / "config.json")
    os.environ["FAKE_OPENCLAW_REQUEST"] = str(tmp_path / "request.json")
    os.environ["FAKE_OPENCLAW_PLUGIN_COMMANDS"] = str(tmp_path / "plugin-commands")
    return tmp_path / "config.json"


def _context(workspace: Path, state_dir: Path | None = None) -> RuntimeContext:
    payload: dict[str, Any] = {
        "runtime_id": "openclaw-runtime",
        "invocation_id": "openclaw-invocation",
        "request_id": "openclaw-request",
        "environment": {
            "environment_id": "openclaw-environment",
            "provider": "local",
            "control_location": "in_env_control",
            "workspace": str(workspace),
            "env": {},
            "ownership": "caller_owned",
        },
        "artifacts": {},
    }
    if state_dir is not None:
        payload["state_dir"] = str(state_dir)
    return RuntimeContext.from_mapping(payload)


def _config(
    command: Path, plugins: dict[str, Any] | None = None, *, shared: bool = False
) -> AgentConfig:
    settings: dict[str, Any] = {"openclaw_command": str(command)}
    if plugins is not None:
        settings["plugins"] = plugins
    if shared:
        settings["interfaces"] = {"terminal": {}}
    return AgentConfig.from_mapping(
        {
            "harness": {"settings": settings},
            "instructions": {"system": {"content": "Be concise."}},
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )


def _generated(config: AgentConfig, tmp_path: Path) -> dict[str, Any]:
    return adapter._openclaw_config(
        config,
        _context(tmp_path),
        base_dir=tmp_path,
        port=20_000,
        token_env="OPENCLAW_GATEWAY_TOKEN",
    )


async def _start(
    runtime: adapter.OpenClawRuntime,
    tmp_path: Path,
    config: AgentConfig,
    *,
    state_dir: Path | None = None,
) -> None:
    await runtime.start(
        {
            "config": config,
            "runtime_context": _context(tmp_path, state_dir).to_mapping(),
            "base_dir": str(tmp_path),
        }
    )


async def _start_error(
    tmp_path: Path, config: AgentConfig, **options: Any
) -> adapter.lifecycle.LifecycleError:
    runtime = adapter.OpenClawRuntime()
    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await _start(runtime, tmp_path, config, **options)
    return caught.value


def _plan(tmp_path: Path, plugins: Any) -> None:
    Fabric().plan(
        minimal_config(
            {"plugins": plugins},
            name="openclaw-plugins",
            adapter_id="nvidia.fabric.openclaw",
            resolution="preinstalled",
        ),
        base_dir=tmp_path,
    )


# Bundled plugins


def _bundled(entries: dict) -> dict:
    return {"bundled": {"entries": entries}}


def _external(paths: list, entries: dict | None = None) -> dict:
    external: dict = {"paths": paths}
    if entries is not None:
        external["entries"] = entries
    return {"external": external}


@pytest.mark.parametrize(
    "plugins",
    [
        pytest.param(_bundled({"memory-wiki": {"enabled": True}}), id="enable"),
        pytest.param(_bundled({"github": {"enabled": False}}), id="disable"),
        pytest.param(
            _bundled(
                {
                    "memory-wiki": {
                        "enabled": True,
                        "config": {"vault": {"path": "wiki"}},
                        "hooks": {
                            "allowConversationAccess": False,
                            "allowPromptInjection": False,
                        },
                    }
                }
            ),
            id="config-and-hooks",
        ),
        pytest.param(
            _bundled(
                {
                    "memory-core": {
                        "enabled": True,
                        "config": {"dreaming": {"enabled": False}},
                    }
                }
            ),
            id="memory-core",
        ),
        pytest.param(
            {
                **_bundled({"memory-wiki": {"enabled": True}}),
                **_external(["plugins/x"], {"x": {"enabled": True}}),
            },
            id="both-sections",
        ),
    ],
)
def test_descriptor_accepts_plugin_settings(tmp_path: Path, plugins):
    _plan(tmp_path, plugins)


@pytest.mark.parametrize(
    ("plugins", "path"),
    [
        pytest.param(
            _bundled({"fabric-shared-config": {"enabled": True}}),
            "harness.settings.plugins.bundled.entries.fabric-shared-config",
            id="reserved-bundled",
        ),
        pytest.param(
            _external([], {"fabric-shared-config": {"enabled": True}}),
            "harness.settings.plugins.external.entries.fabric-shared-config",
            id="reserved-external",
        ),
        pytest.param(
            _bundled({"memory-wiki": {"enabled": "yes"}}),
            "harness.settings.plugins.bundled.entries.memory-wiki.enabled",
            id="enabled-type",
        ),
        pytest.param(
            _bundled({"memory-wiki": {"install": "npm"}}),
            "harness.settings.plugins.bundled.entries.memory-wiki.install",
            id="entry-unknown",
        ),
        pytest.param(
            _bundled({"memory-wiki": {"hooks": {"timeoutMs": 5}}}),
            "harness.settings.plugins.bundled.entries.memory-wiki.hooks.timeoutMs",
            id="hooks-unknown",
        ),
        pytest.param(
            {"bundled": {"paths": ["x"]}},
            "harness.settings.plugins.bundled.paths",
            id="bundled-paths",
        ),
        pytest.param(
            {"entries": {"memory-wiki": {"enabled": True}}},
            "harness.settings.plugins.entries",
            id="native-entries",
        ),
        pytest.param(
            {"allow": ["memory-wiki"]}, "harness.settings.plugins.allow", id="allow"
        ),
        pytest.param(
            {"load": {"paths": ["x"]}}, "harness.settings.plugins.load", id="load"
        ),
    ],
)
def test_descriptor_rejects_invalid_plugin_settings(tmp_path: Path, plugins, path):
    with pytest.raises(FabricConfigError) as caught:
        _plan(tmp_path, plugins)

    assert path in str(caught.value)


def test_both_sections_are_written_as_native_entries(
    mock_openclaw: Path, tmp_path: Path
):
    (tmp_path / "plugins" / "probe").mkdir(parents=True)
    plugins = {
        **_bundled(
            {
                "memory-wiki": {"enabled": True, "config": {"vault": {"path": "wiki"}}},
                "github": {"enabled": False},
            }
        ),
        **_external(["plugins/probe"], {"probe": {"enabled": True}}),
    }

    generated = _generated(_config(mock_openclaw, plugins), tmp_path)

    assert generated["plugins"] == {
        "entries": {
            "memory-wiki": {"enabled": True, "config": {"vault": {"path": "wiki"}}},
            "github": {"enabled": False},
            "probe": {"enabled": True},
        },
        "load": {"paths": [str((tmp_path / "plugins" / "probe").resolve())]},
    }


@pytest.mark.parametrize("section", ["bundled", "external"])
def test_reserved_plugin_id_is_refused_by_the_writer(
    mock_openclaw: Path, tmp_path: Path, section: str
):
    # The descriptor rejects the id; the writer refuses it too, with or
    # without the adapter's own plugin.
    for shared in (False, True):
        plugins = {section: {"entries": {"fabric-shared-config": {}}}}
        config = _config(mock_openclaw, plugins, shared=shared)
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            _generated(config, tmp_path)
        assert caught.value.code == "openclaw_invalid_configuration"
        assert caught.value.metadata == {
            "field": f"harness.settings.plugins.{section}.entries.fabric-shared-config"
        }


def test_plugin_in_both_sections_is_refused(mock_openclaw: Path, tmp_path: Path):
    plugins = {
        **_bundled({"probe": {"enabled": True}}),
        **_external([], {"probe": {"enabled": False}}),
    }

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        _generated(_config(mock_openclaw, plugins), tmp_path)

    assert caught.value.code == "openclaw_invalid_configuration"
    assert caught.value.metadata == {
        "field": "harness.settings.plugins.external.entries.probe"
    }


@posix_only
async def test_start_without_enabled_plugins_runs_no_plugin_commands(
    mock_openclaw: Path, tmp_path: Path, capture: Path
):
    config = _config(
        mock_openclaw,
        _bundled(
            {
                "github": {"enabled": False},
                "memory-core": {"config": {"dreaming": {"enabled": False}}},
            }
        ),
    )
    runtime = adapter.OpenClawRuntime()
    await _start(runtime, tmp_path, config)
    await runtime.stop()

    assert not (tmp_path / "plugin-commands").exists()


@posix_only
async def test_start_checks_every_enabled_plugin_loads(
    mock_openclaw: Path, tmp_path: Path, capture: Path
):
    config = _config(
        mock_openclaw,
        _bundled({"memory-wiki": {"enabled": True}, "github": {"enabled": False}}),
    )
    runtime = adapter.OpenClawRuntime()
    await _start(runtime, tmp_path, config)
    await runtime.stop()

    assert (tmp_path / "plugin-commands").read_text().splitlines() == [
        "plugins.list",
    ]
    assert json.loads(capture.read_text())["plugins"]["entries"] == {
        "memory-wiki": {"enabled": True},
        "github": {"enabled": False},
    }


@posix_only
@pytest.mark.parametrize(
    ("statuses", "errors", "code", "metadata"),
    [
        pytest.param(
            {"memory-wiki": "absent"},
            {},
            "openclaw_plugin_not_found",
            {},
            id="absent",
        ),
        pytest.param(
            {"memory-wiki": "catalog"},
            {},
            "openclaw_plugin_not_found",
            {},
            id="catalog-only",
        ),
        pytest.param(
            {"memory-wiki": "disabled"},
            {},
            "openclaw_plugin_unavailable",
            {"reason": "disabled"},
            id="not-loaded",
        ),
        pytest.param(
            {},
            {"memory-wiki": "Error [ERR_MODULE_NOT_FOUND]: missing"},
            "openclaw_plugin_unavailable",
            {"reason": "Error [ERR_MODULE_NOT_FOUND]: missing"},
            id="load-failed",
        ),
    ],
)
async def test_start_fails_when_an_enabled_plugin_does_not_load(
    mock_openclaw: Path,
    tmp_path: Path,
    capture: Path,
    statuses: dict[str, str],
    errors: dict[str, str],
    code: str,
    metadata: dict[str, str],
):
    os.environ["FAKE_OPENCLAW_PLUGINS"] = json.dumps(statuses)
    os.environ["FAKE_OPENCLAW_PLUGIN_ERRORS"] = json.dumps(
        {**errors, "unrelated": "ignored"}
    )
    os.environ["FAKE_OPENCLAW_STOPPED"] = str(tmp_path / "gateway.stopped")
    config = _config(mock_openclaw, _bundled({"memory-wiki": {"enabled": True}}))

    error = await _start_error(tmp_path, config)

    assert error.code == code
    assert error.metadata == {
        "field": "harness.settings.plugins.bundled.entries.memory-wiki",
        "plugin": "memory-wiki",
        **metadata,
    }
    # The check runs on the ready Gateway, which a failed start stops.
    assert (tmp_path / "gateway.stopped").read_text() == "stopped"


@posix_only
@pytest.mark.parametrize(
    ("plugins", "origin", "field"),
    [
        pytest.param(
            _external([], {"memory-wiki": {"enabled": True}}),
            "bundled",
            "harness.settings.plugins.external.entries.memory-wiki",
            id="external-from-bundled",
        ),
        pytest.param(
            _bundled({"memory-wiki": {"enabled": True}}),
            "global",
            "harness.settings.plugins.bundled.entries.memory-wiki",
            id="bundled-from-installed",
        ),
    ],
)
async def test_start_fails_when_a_plugin_loads_from_another_source(
    mock_openclaw: Path, tmp_path: Path, capture: Path, plugins, origin, field
):
    os.environ["FAKE_OPENCLAW_PLUGIN_ORIGINS"] = json.dumps({"memory-wiki": origin})

    error = await _start_error(tmp_path, _config(mock_openclaw, plugins))

    assert error.code == "openclaw_plugin_not_found"
    assert error.metadata == {"field": field, "plugin": "memory-wiki", "origin": origin}


@posix_only
@pytest.mark.parametrize(
    ("plugins", "origin"),
    [
        pytest.param(_bundled({"memory-wiki": {"enabled": True}}), "bundled"),
        pytest.param(_external([], {"memory-wiki": {"enabled": True}}), "config"),
    ],
)
async def test_start_accepts_a_plugin_from_its_own_source(
    mock_openclaw: Path, tmp_path: Path, capture: Path, plugins, origin
):
    os.environ["FAKE_OPENCLAW_PLUGIN_ORIGINS"] = json.dumps({"memory-wiki": origin})
    runtime = adapter.OpenClawRuntime()
    await _start(runtime, tmp_path, _config(mock_openclaw, plugins))
    await runtime.stop()


@posix_only
async def test_load_errors_for_plugins_not_requested_do_not_fail_start(
    mock_openclaw: Path, tmp_path: Path, capture: Path
):
    os.environ["FAKE_OPENCLAW_PLUGIN_ERRORS"] = json.dumps({"telegram": "no token"})
    config = _config(mock_openclaw, _bundled({"memory-wiki": {"enabled": True}}))
    runtime = adapter.OpenClawRuntime()
    await _start(runtime, tmp_path, config)
    await runtime.stop()


@posix_only
async def test_plugin_config_openclaw_rejects_names_the_setting(
    mock_openclaw: Path, tmp_path: Path, capture: Path
):
    os.environ["FAKE_OPENCLAW_CONFIG_ISSUES"] = json.dumps(
        [
            {
                "path": "plugins.entries.memory-wiki.config",
                "message": 'invalid config: must not have additional properties: "x"',
            }
        ]
    )
    config = _config(mock_openclaw, _bundled({"memory-wiki": {"config": {"x": 1}}}))

    error = await _start_error(tmp_path, config)

    assert error.code == "openclaw_invalid_configuration"
    assert error.metadata == {
        "field": "harness.settings.plugins.bundled.entries.memory-wiki.config",
        "detail": 'invalid config: must not have additional properties: "x"',
    }


@posix_only
async def test_other_config_issues_keep_the_command_error(
    mock_openclaw: Path, tmp_path: Path, capture: Path
):
    os.environ["FAKE_OPENCLAW_CONFIG_ISSUES"] = json.dumps(
        [{"path": "gateway.port", "message": "bad"}]
    )

    error = await _start_error(tmp_path, _config(mock_openclaw))

    assert error.code == "openclaw_command_failed"


# External plugins


def test_descriptor_accepts_external_paths(tmp_path: Path):
    _plan(tmp_path, _external(["/sandbox/plugins/probe", "plugins/x"]))


@pytest.mark.parametrize(
    ("plugins", "path"),
    [
        pytest.param(
            _external([""]), "harness.settings.plugins.external.paths.0", id="empty"
        ),
        pytest.param(
            {"external": {"paths": "plugins"}},
            "harness.settings.plugins.external.paths",
            id="not-a-list",
        ),
        pytest.param(
            {"external": {"watch": True}},
            "harness.settings.plugins.external.watch",
            id="unknown",
        ),
    ],
)
def test_descriptor_rejects_invalid_external_paths(tmp_path: Path, plugins, path):
    with pytest.raises(FabricConfigError) as caught:
        _plan(tmp_path, plugins)

    assert path in str(caught.value)


@pytest.mark.parametrize("shared", [False, True])
def test_external_paths_resolve_against_the_base_directory(
    mock_openclaw: Path, tmp_path: Path, shared: bool
):
    (tmp_path / "plugins" / "probe").mkdir(parents=True)
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    config = _config(
        mock_openclaw,
        _external(["plugins/probe", str(mounted)], {"probe": {"enabled": True}}),
        shared=shared,
    )

    generated = _generated(config, tmp_path)

    paths = generated["plugins"]["load"]["paths"]
    authored = [
        str((tmp_path / "plugins" / "probe").resolve()),
        str(mounted.resolve()),
    ]
    if shared:
        # The adapter's own plugin comes first and keeps its entry.
        assert paths[1:] == authored
        assert paths[0].endswith("native_config")
        assert set(generated["plugins"]["entries"]) == {"fabric-shared-config", "probe"}
    else:
        assert paths == authored
    assert config.harness.settings["plugins"]["external"]["paths"][0] == "plugins/probe"


def test_missing_external_path_fails_with_its_index(
    mock_openclaw: Path, tmp_path: Path
):
    (tmp_path / "present").mkdir()
    config = _config(mock_openclaw, _external(["present", "missing/plugin"]))

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        _generated(config, tmp_path)

    assert caught.value.code == "openclaw_plugin_not_found"
    assert caught.value.metadata == {
        "field": "harness.settings.plugins.external.paths[1]",
        "path": str(tmp_path / "missing" / "plugin"),
    }


@posix_only
async def test_missing_external_path_fails_start(
    mock_openclaw: Path, tmp_path: Path, capture: Path
):
    os.environ["FAKE_OPENCLAW_PID"] = str(tmp_path / "gateway.pid")
    config = _config(mock_openclaw, _external(["missing"]))

    error = await _start_error(tmp_path, config)

    assert error.code == "openclaw_plugin_not_found"
    assert error.metadata == {
        "field": "harness.settings.plugins.external.paths[0]",
        "path": str(tmp_path / "missing"),
    }
    assert not capture.exists()
    assert not (tmp_path / "gateway.pid").exists()


@posix_only
async def test_external_plugin_that_fails_to_load_fails_start(
    mock_openclaw: Path, tmp_path: Path, capture: Path
):
    (tmp_path / "plugins" / "probe").mkdir(parents=True)
    os.environ["FAKE_OPENCLAW_PLUGIN_ERRORS"] = json.dumps(
        {"probe": "Cannot find package 'probe-dep'"}
    )
    config = _config(
        mock_openclaw,
        _external(["plugins/probe"], {"probe": {"enabled": True}}),
    )

    error = await _start_error(tmp_path, config)

    assert error.code == "openclaw_plugin_unavailable"
    assert error.metadata["field"] == "harness.settings.plugins.external.entries.probe"
    assert error.metadata["reason"] == "Cannot find package 'probe-dep'"


async def test_attached_gateways_reject_external_paths(tmp_path: Path):
    os.environ["OPENCLAW_GATEWAY_TOKEN"] = "caller-owned-token"
    config = AgentConfig.from_mapping(
        {
            "harness": {"settings": {"plugins": _external(["plugins"])}},
            "models": {"default": {"provider": "test", "model": "fabric-echo"}},
        }
    )
    runtime = adapter.OpenClawRuntime()

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        await runtime.start(
            {
                "config": config,
                "runtime_context": _context(tmp_path).to_mapping(),
                "base_dir": str(tmp_path),
                "service": {
                    "operation": "attach",
                    "connection_path": str(tmp_path / "service-connection.json"),
                    "reference": {
                        "adapter_id": "nvidia.fabric.openclaw",
                        "service_type": "openclaw_gateway",
                        "connection": {
                            "gateway_url": "http://127.0.0.1:9",
                            "gateway_token_env": "OPENCLAW_GATEWAY_TOKEN",
                        },
                    },
                },
            }
        )

    assert caught.value.code == "openclaw_attach_configuration_unverifiable"
    assert caught.value.metadata == {"fields": ["harness.settings.plugins"]}


# Agent-installed plugins

merge = openclaw_plugins.merge_plugins


def test_merge_writes_fabric_leaves_over_the_agents():
    agent = {"entries": {"github": {"enabled": True, "config": {"token": "agent"}}}}
    fabric = {"entries": {"github": {"enabled": False}}}

    merged, owned = merge(agent, fabric, {})

    assert merged == {
        "entries": {"github": {"enabled": False, "config": {"token": "agent"}}}
    }
    assert owned == {
        "leaves": [
            {"path": ["entries", "github", "enabled"], "value": False, "shadowed": True}
        ],
        "items": {},
    }
    assert agent["entries"]["github"]["enabled"] is True  # not changed in place


def test_merge_restores_the_agents_value_when_fabric_stops_setting_it():
    agent = {"entries": {"github": {"enabled": True}}}
    merged, owned = merge(agent, {"entries": {"github": {"enabled": False}}}, {})
    merged, owned = merge(merged, {"entries": {"github": {"enabled": False}}}, owned)

    merged, owned = merge(merged, {}, owned)

    assert merged == agent
    assert owned == {"leaves": [], "items": {}}


def test_merge_keeps_an_agent_edit_made_after_fabric_set_the_value():
    fabric = {"entries": {"x": {"config": {"level": 1}}}}
    merged, owned = merge({"entries": {"x": {"config": {"level": 0}}}}, fabric, {})
    merged["entries"]["x"]["config"]["level"] = 5  # the agent changes it

    # NeMo Fabric still sets it: its value wins, the agent's edit is kept aside.
    again, again_owned = merge(merged, fabric, owned)
    assert again["entries"]["x"]["config"]["level"] == 1
    assert merge(again, {}, again_owned)[0] == merged

    # NeMo Fabric stops setting it: the agent's edit stays.
    assert merge(merged, {}, owned)[0] == merged


def test_merge_keeps_the_agents_own_settings():
    agent = {
        "allow": ["installed", "memory-wiki"],
        "deny": ["telegram"],
        "load": {"paths": ["/state/extensions/linked"]},
        "entries": {"installed": {"enabled": True, "config": {"mode": "x"}}},
        "slots": {"memory": "memory-core"},
    }
    fabric = {
        "allow": ["memory-wiki", "mounted"],
        "deny": ["github"],
        "load": {"paths": ["/sandbox/plugins/mounted"]},
        "entries": {"mounted": {"enabled": True}},
    }

    merged, owned = merge(agent, fabric, {})

    assert merged == {
        "allow": ["installed", "memory-wiki", "mounted"],
        "deny": ["telegram", "github"],
        "load": {"paths": ["/state/extensions/linked", "/sandbox/plugins/mounted"]},
        "entries": {
            "installed": {"enabled": True, "config": {"mode": "x"}},
            "mounted": {"enabled": True},
        },
        "slots": {"memory": "memory-core"},
    }
    # memory-wiki was the agent's already, so NeMo Fabric doesn't own it.
    assert owned["items"] == {
        "allow": ["mounted"],
        "deny": ["github"],
        "load.paths": ["/sandbox/plugins/mounted"],
    }


def test_merge_removes_what_fabric_no_longer_sets():
    first = {
        "allow": ["mounted"],
        "load": {"paths": ["/old/fabric-shared-config", "/sandbox/mounted"]},
        "entries": {
            "mounted": {"enabled": True, "config": {"level": 2}},
            "github": {"enabled": False},
        },
    }
    merged, owned = merge({}, first, {})
    # The agent installs a plugin, which OpenClaw adds to allow and entries.
    merged["allow"].append("installed")
    merged["entries"]["installed"] = {"enabled": True}
    merged["entries"]["mounted"]["hooks"] = {"allowPromptInjection": False}

    second = {
        "load": {"paths": ["/new/fabric-shared-config"]},
        "entries": {"mounted": {"enabled": True}},
    }
    merged, owned = merge(merged, second, owned)

    assert merged == {
        "allow": ["mounted", "installed"],
        "load": {"paths": ["/new/fabric-shared-config"]},
        "entries": {
            # The agent's own leaf under a Fabric entry is kept.
            "mounted": {"enabled": True, "hooks": {"allowPromptInjection": False}},
            "installed": {"enabled": True},
        },
    }
    assert owned == {
        "leaves": [{"path": ["entries", "mounted", "enabled"], "value": True}],
        "items": {"allow": ["mounted"], "load.paths": ["/new/fabric-shared-config"]},
    }


def test_merge_drops_list_fields_left_empty():
    merged, owned = merge({}, {"allow": ["x"], "deny": ["y"]}, {})
    merged, _ = merge(merged, {}, owned)

    assert merged == {}


@pytest.mark.parametrize("fabric_allow", [False, True])
def test_merge_keeps_the_adapter_plugin_in_an_allow_list(fabric_allow: bool):
    fabric: dict[str, Any] = {
        "load": {"paths": ["/pkg/native_config"]},
        "entries": {"fabric-shared-config": {"enabled": True}},
    }
    if fabric_allow:
        fabric["allow"] = ["mounted", "fabric-shared-config"]
    agent = {"allow": ["installed"]}

    merged, owned = merge(agent, fabric, {})

    assert "fabric-shared-config" in merged["allow"]
    assert "installed" in merged["allow"]
    # Without the adapter plugin next time, the id leaves allow again.
    merged, _ = merge(merged, {}, owned)
    assert merged == {"allow": ["installed"]}


def test_merge_does_not_add_an_allow_list():
    fabric = {"entries": {"fabric-shared-config": {"enabled": True}}}

    merged, owned = merge({}, fabric, {})

    assert "allow" not in merged
    assert owned["items"] == {}


def test_merge_keeps_agent_config_under_an_empty_fabric_object():
    agent = {"entries": {"probe": {"config": {"mode": "x"}}}}

    merged, owned = merge(agent, {"entries": {"probe": {"config": {}}}}, {})
    assert merged == agent

    merged, _ = merge(merged, {"entries": {"probe": {"config": {"level": 2}}}}, owned)
    assert merged == {"entries": {"probe": {"config": {"mode": "x", "level": 2}}}}


@pytest.mark.parametrize(
    ("agent", "fabric_value"),
    [
        pytest.param({}, None, id="null-on-missing"),
        pytest.param(
            {"entries": {"x": {"config": {"n": False}}}}, 0, id="false-vs-zero"
        ),
        pytest.param({"entries": {"x": {"config": {"n": True}}}}, 1, id="true-vs-one"),
    ],
)
def test_merge_applies_values_that_only_look_equal(agent: dict, fabric_value: Any):
    fabric = {"entries": {"x": {"config": {"n": fabric_value}}}}

    merged, owned = merge(agent, fabric, {})

    value = merged["entries"]["x"]["config"]["n"]
    assert value == fabric_value and type(value) is type(fabric_value)
    assert [leaf["path"] for leaf in owned["leaves"]] == [
        ["entries", "x", "config", "n"]
    ]


def test_merge_adds_enabled_plugins_to_an_agent_allow_list():
    merged, owned = merge(
        {"allow": ["installed"]}, {"entries": {"x": {"enabled": True}}}, {}
    )
    assert merged["allow"] == ["installed", "x"]

    merged, _ = merge(merged, {}, owned)
    assert merged == {"allow": ["installed"]}


def test_merge_does_not_own_an_agent_load_path():
    agent = {"load": {"paths": ["/sandbox/plugins/probe"]}}
    merged, owned = merge(agent, {"load": {"paths": ["/sandbox/plugins/probe"]}}, {})

    merged, _ = merge(merged, {}, owned)
    assert merged == agent


@pytest.mark.parametrize(
    ("name", "text"),
    [
        pytest.param("plugins.json", "{not json", id="invalid-json"),
        pytest.param("plugins.json", "[]", id="not-an-object"),
        pytest.param("plugins.owned.json", "{", id="invalid-record"),
        pytest.param("plugins.owned.json", '{"items": null}', id="record-shape"),
        pytest.param("plugins.owned.json", '{"leaves": [["a"]]}', id="leaf-shape"),
        pytest.param("plugins.json", b"\xff", id="invalid-utf8"),
    ],
)
def test_unreadable_plugin_state_fails_and_is_kept(
    tmp_path: Path, name: str, text: str
):
    folder = openclaw_plugins.state_folder(tmp_path)
    data = text if isinstance(text, bytes) else text.encode()
    (folder / name).write_bytes(data)

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        openclaw_plugins.prepare(folder, {"entries": {"x": {"enabled": True}}})

    assert caught.value.code == "openclaw_plugin_state_invalid"
    assert caught.value.metadata["path"] == str(folder / name)
    assert (folder / name).read_bytes() == data


@pytest.mark.parametrize(
    ("failing_write", "expected"),
    [
        # plugins.json was written: the retry removes NeMo Fabric's value.
        pytest.param(3, {}, id="record-write"),
        # plugins.json wasn't written: the agent's value stays.
        pytest.param(2, {"entries": {"x": {"enabled": True}}}, id="policy-write"),
        # The agent already has NeMo Fabric's value, so the policy is unchanged.
        pytest.param(-2, {"entries": {"x": {"enabled": True}}}, id="same-policy"),
    ],
)
def test_prepare_recovers_from_a_failed_write(
    tmp_path: Path, monkeypatch, failing_write: int, expected: dict
):
    folder = openclaw_plugins.state_folder(tmp_path)
    if failing_write in (2, -2):
        policy = {"entries": {"x": {"enabled": True}}}
        (folder / "plugins.json").write_text(json.dumps(policy, indent=2) + "\n")
    write = openclaw_plugins.local_interfaces.write_private_file
    calls = []

    def flaky_write(path, text):
        calls.append(path)
        if len(calls) == abs(failing_write):
            raise OSError("disk full")
        write(path, text)

    monkeypatch.setattr(
        openclaw_plugins.local_interfaces, "write_private_file", flaky_write
    )
    fabric = {"entries": {"x": {"enabled": failing_write != 2}}}
    with pytest.raises(OSError):
        openclaw_plugins.prepare(folder, fabric)
    monkeypatch.undo()

    assert openclaw_plugins.prepare(folder, {}) == expected


def test_remove_config_keeps_other_files(tmp_path: Path):
    config = tmp_path / "openclaw.json"
    for name in (
        "openclaw.json",
        "openclaw.json.bak",
        "openclaw.json.bak.1",
        "openclaw.json.last-good",
        "openclaw.json.clobbered.x",
        "openclaw.json.saved",
        "plugins.json",
    ):
        (tmp_path / name).write_text("{}")

    openclaw_plugins.remove_config(config)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "openclaw.json.saved",
        "plugins.json",
    ]


@pytest.mark.skipif(os.name != "posix", reason="Checks POSIX modes")
def test_prepare_writes_owner_only_files(tmp_path: Path):
    folder = openclaw_plugins.state_folder(tmp_path)
    (folder / "plugins.json").write_text('{"entries": {"installed": {}}}')
    (folder / "plugins.json").chmod(0o644)

    merged = openclaw_plugins.prepare(folder, {"entries": {"x": {"enabled": True}}})

    assert merged == {"entries": {"installed": {}, "x": {"enabled": True}}}
    assert json.loads((folder / "plugins.json").read_text()) == merged
    for name in ("plugins.json", "plugins.owned.json"):
        assert (folder / name).stat().st_mode & 0o777 == 0o600
    assert folder.stat().st_mode & 0o777 == 0o700
    assert sorted(path.name for path in folder.iterdir()) == [
        "plugins.json",
        "plugins.owned.json",
    ]


@posix_only
def test_state_folder_refuses_a_symlink(tmp_path: Path):
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "fabric").symlink_to(tmp_path / "elsewhere")

    with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
        openclaw_plugins.state_folder(tmp_path)

    assert caught.value.code == "session_incompatible"


@posix_only
async def test_without_keep_state_the_configuration_is_unchanged(
    mock_openclaw: Path, tmp_path: Path, capture: Path
):
    readonly = tmp_path / "readonly"
    os.environ["FAKE_OPENCLAW_CONFIG_READONLY"] = str(readonly)
    config = _config(
        mock_openclaw, _bundled({"memory-wiki": {"enabled": True}}), shared=True
    )
    runtime = adapter.OpenClawRuntime()
    await _start(runtime, tmp_path, config)
    await runtime.stop()

    generated = json.loads(capture.read_text())
    assert generated["plugins"] == _generated(config, tmp_path)["plugins"]
    assert readonly.read_text() == "1"
    assert not list(tmp_path.rglob("plugins.json"))


@posix_only
async def test_keep_state_moves_plugins_into_a_writable_include(
    mock_openclaw: Path, tmp_path: Path, capture: Path
):
    readonly = tmp_path / "readonly"
    os.environ["FAKE_OPENCLAW_CONFIG_READONLY"] = str(readonly)
    state = tmp_path / "state"
    folder = state / "fabric"
    config = _config(
        mock_openclaw,
        _bundled({"memory-wiki": {"enabled": True}, "github": {"enabled": False}}),
        shared=True,
    )
    runtime = adapter.OpenClawRuntime()
    await _start(runtime, tmp_path, config, state_dir=state)
    try:
        generated = json.loads(capture.read_text())
        assert generated["plugins"] == {"$include": "./plugins.json"}
        assert (folder / "openclaw.json").stat().st_mode & 0o777 == 0o600
        assert readonly.read_text() == ""
        plugins = json.loads((folder / "plugins.json").read_text())
        assert plugins == _generated(config, tmp_path)["plugins"]
        assert "fabric-shared-config" in plugins["entries"]
        assert (tmp_path / "plugin-commands").read_text().splitlines() == [
            "plugins.list",
        ]
        # OpenClaw keeps a copy of a writable configuration.
        (folder / "openclaw.json.last-good").write_text("{}")
    finally:
        await runtime.stop()
    # The configuration is regenerated at start; the plugin policy stays.
    assert not (folder / "openclaw.json").exists()
    assert not (folder / "openclaw.json.last-good").exists()
    assert (folder / "plugins.json").exists()
    assert (folder / "plugins.owned.json").exists()


@posix_only
async def test_keep_state_restart_keeps_agent_installed_plugins(
    mock_openclaw: Path, tmp_path: Path, capture: Path
):
    state = tmp_path / "state"
    folder = state / "fabric"
    first = _config(
        mock_openclaw,
        _bundled({"memory-wiki": {"enabled": True}, "github": {"enabled": False}}),
    )
    runtime = adapter.OpenClawRuntime()
    await _start(runtime, tmp_path, first, state_dir=state)
    await runtime.stop()
    # What `openclaw plugins install` and `enable`/`disable` write through.
    plugins = json.loads((folder / "plugins.json").read_text())
    plugins["entries"]["installed"] = {"enabled": True}
    plugins["entries"]["github"] = {"enabled": True}
    (folder / "plugins.json").write_text(json.dumps(plugins))

    second = _config(mock_openclaw, _bundled({"github": {"enabled": False}}))
    runtime = adapter.OpenClawRuntime()
    await _start(runtime, tmp_path, second, state_dir=state)
    await runtime.stop()

    assert json.loads((folder / "plugins.json").read_text()) == {
        "entries": {"github": {"enabled": False}, "installed": {"enabled": True}}
    }


@posix_only
async def test_keep_state_with_invalid_plugin_state_fails_start(
    mock_openclaw: Path, tmp_path: Path, capture: Path
):
    folder = tmp_path / "state" / "fabric"
    folder.mkdir(parents=True)
    (folder / "plugins.json").write_text("{broken")
    os.environ["FAKE_OPENCLAW_PID"] = str(tmp_path / "gateway.pid")

    error = await _start_error(
        tmp_path, _config(mock_openclaw), state_dir=tmp_path / "state"
    )

    assert error.code == "openclaw_plugin_state_invalid"
    assert (folder / "plugins.json").read_text() == "{broken"
    assert not (folder / "openclaw.json").exists()
    assert not (tmp_path / "gateway.pid").exists()

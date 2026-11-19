# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Hermes adapter's plugin settings and start checks."""

from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext

pytestmark = pytest.mark.usefixtures("requires_hermes_agent")

if importlib.util.find_spec("run_agent") is not None:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    from nemo_fabric_adapters.common.lifecycle import LifecycleError
    from nemo_fabric_adapters.hermes import adapter
    from nemo_fabric_adapters.hermes import configuration
    from nemo_fabric_adapters.hermes import plugins as hermes_plugins


MODEL = {"provider": "test", "model": "test-model", "api_key_env": "TEST_API_KEY"}


def _config(settings: dict[str, object]) -> AgentConfig:
    return AgentConfig.from_mapping(
        {"models": {"default": MODEL}, "harness": {"settings": settings}}
    )


def _payload(
    tmp_path: Path,
    settings: dict[str, object],
    *,
    runtime_id: str = "runtime-plugins",
    state_dir: Path | None = None,
) -> dict[str, object]:
    context: dict[str, object] = {
        "runtime_id": runtime_id,
        "invocation_id": "invocation-1",
        "request_id": "request-1",
        "environment": {
            "environment_id": "environment-1",
            "provider": "test",
            "control_location": "in_env_control",
            "ownership": "caller_owned",
            "workspace": str(tmp_path),
        },
        "artifacts": {"root": str(tmp_path / "artifacts")},
    }
    if state_dir is not None:
        context["state_dir"] = str(state_dir)
    return {
        "base_dir": str(tmp_path),
        "config": _config(settings),
        "runtime_context": RuntimeContext.from_mapping(context).to_mapping(),
    }


@pytest.fixture(name="mock_hermes_agent")
def mock_hermes_agent_fixture(monkeypatch) -> MagicMock:
    """Replace the embedded agent and session store; Hermes's plugin code stays real."""
    import hermes_state
    import run_agent

    mock_ai_agent_type = MagicMock(spec=AIAgent, return_value=MagicMock(spec=AIAgent))
    monkeypatch.setattr(
        mock_ai_agent_type.__init__.__func__,
        "__signature__",
        inspect.signature(AIAgent.__init__),
        raising=False,
    )
    monkeypatch.setattr(run_agent, "AIAgent", mock_ai_agent_type)
    monkeypatch.setattr(hermes_state, "SessionDB", MagicMock(spec=SessionDB))
    monkeypatch.setenv("TEST_API_KEY", "secret")
    for name in ("HOME", "HERMES_HOME"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    return mock_ai_agent_type


def _listed() -> list[dict[str, object]]:
    from hermes_cli.plugins import get_plugin_manager

    return get_plugin_manager().list_plugins()


def _entry(listed: list[dict[str, object]], key: str) -> dict[str, object]:
    return next(entry for entry in listed if entry["key"] == key)


# --- Bundled plugins ------------------------------------------------------


def _ids(*plugin_ids: str) -> list[tuple[str, str]]:
    return [
        (f"harness.settings.plugins.bundled.entries.{plugin_id}", plugin_id)
        for plugin_id in plugin_ids
    ]


def _bundled(**entries: object) -> dict[str, object]:
    return {"plugins": {"bundled": {"entries": entries}}}


def test_plugin_settings_translate_to_the_native_section():
    settings = {
        "plugins": {
            "bundled": {
                "entries": {
                    "security-guidance": {
                        "enabled": True,
                        "config": {"strict": True},
                        "capabilities": ["tools.override"],
                    },
                    "browser/firecrawl": {"enabled": False},
                }
            },
            "external": {
                "paths": ["/sandbox/plugins/acme"],
                "entries": {"acme": {"enabled": True, "config": {"greeting": "hi"}}},
            },
            "load_timeout_seconds": 20,
            "hook_callback_timeout": 5,
            "clone_timeout_seconds": 120,
            "auto_update_check_hours": 0,
            "auto_apply": False,
            "isolation": "host",
            "host": {"launcher": ["sandbox-run"]},
        }
    }
    config = configuration.build_hermes_config(
        _config(settings), workspace="/workspace"
    )

    assert config["plugins"] == {
        "load_timeout_seconds": 20,
        "hook_callback_timeout": 5,
        "clone_timeout_seconds": 120,
        "auto_update_check_hours": 0,
        "auto_apply": False,
        "isolation": "host",
        "host": {"launcher": ["sandbox-run"]},
        "enabled": ["security-guidance", "acme"],
        "disabled": ["browser/firecrawl"],
        "entries": {
            "security-guidance": {
                "settings": {"strict": True},
                "granted_capabilities": ["tools.override"],
            },
            "acme": {"settings": {"greeting": "hi"}},
        },
    }
    summary = configuration.summarize_hermes_config(config)
    assert summary["plugins"] == ["security-guidance", "acme"]
    assert summary["plugins_disabled"] == ["browser/firecrawl"]


def test_plugins_enabled_joins_the_enabled_entries_without_repeats():
    settings = {
        **_bundled(**{"security-guidance": {"enabled": True}}),
        "plugins_enabled": ["disk-cleanup", "security-guidance"],
    }
    config = configuration.build_hermes_config(
        _config(settings), workspace="/workspace"
    )

    assert config["plugins"] == {"enabled": ["security-guidance", "disk-cleanup"]}
    assert hermes_plugins.enabled_ids(settings) == [
        (
            "harness.settings.plugins.bundled.entries.security-guidance",
            "security-guidance",
        ),
        ("harness.settings.plugins_enabled[0]", "disk-cleanup"),
    ]


def test_a_disabled_entry_alone_writes_only_the_disabled_list():
    config = configuration.build_hermes_config(
        _config(_bundled(**{"web/exa": {"enabled": False}})), workspace="/workspace"
    )

    assert config["plugins"] == {"disabled": ["web/exa"]}


def test_an_id_in_both_sections_is_rejected():
    settings = {
        "plugins": {
            "bundled": {"entries": {"dup": {"enabled": True}}},
            "external": {"entries": {"dup": {"enabled": True}}},
        }
    }

    with pytest.raises(LifecycleError) as error:
        hermes_plugins.native_section(settings)

    assert error.value.code == "hermes_invalid_config"
    assert error.value.metadata == {
        "field": "harness.settings.plugins.external.entries.dup",
        "plugin": "dup",
    }


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({}, id="none"),
        pytest.param({"plugins": {}}, id="empty"),
        pytest.param({"plugins": {"bundled": {"entries": {"x": {}}}}}, id="bare-entry"),
    ],
)
def test_no_plugin_settings_write_no_plugins_section(settings):
    config = configuration.build_hermes_config(
        _config(settings), workspace="/workspace"
    )

    assert "plugins" not in config
    assert "plugins_disabled" not in configuration.summarize_hermes_config(config)


def test_check_bundled_rejects_a_plugin_hermes_does_not_ship():
    entries = hermes_plugins.plugin_entries(_bundled(mine={"enabled": True}))
    listed = [{"key": "mine", "name": "mine", "source": "user", "enabled": True}]

    with pytest.raises(LifecycleError) as error:
        hermes_plugins.check_bundled(entries, listed)

    assert error.value.code == "hermes_invalid_config"
    assert (
        error.value.metadata["field"] == "harness.settings.plugins.bundled.entries.mine"
    )
    assert "plugins.external" in error.value.message


def test_check_bundled_accepts_a_shipped_or_unlisted_plugin():
    entries = hermes_plugins.plugin_entries(
        _bundled(**{"security-guidance": {"enabled": True}, "absent": {}})
    )
    listed = [
        {"key": "security-guidance", "name": "security-guidance", "source": "bundled"}
    ]

    hermes_plugins.check_bundled(entries, listed)


def test_check_enabled_accepts_key_and_name_matches():
    listed = [
        {"key": "image_gen/openai", "name": "openai", "enabled": True, "error": None},
        {"key": "disk-cleanup", "name": "disk-cleanup", "enabled": True, "error": None},
    ]

    hermes_plugins.check_enabled(
        _ids("image_gen/openai", "openai", "disk-cleanup"), listed
    )


def test_check_enabled_rejects_an_unknown_id():
    listed = [{"key": "disk-cleanup", "name": "disk-cleanup", "enabled": True}]

    with pytest.raises(LifecycleError) as error:
        hermes_plugins.check_enabled(_ids("disk-cleanup", "no-such-plugin"), listed)

    assert error.value.code == "hermes_plugin_not_found"
    assert error.value.metadata == {
        "field": "harness.settings.plugins.bundled.entries.no-such-plugin",
        "plugin": "no-such-plugin",
    }


def test_check_enabled_rejects_a_plugin_that_is_also_disabled():
    listed = [{"key": "disk-cleanup", "name": "disk-cleanup", "enabled": True}]

    with pytest.raises(LifecycleError) as error:
        hermes_plugins.check_enabled(_ids("disk-cleanup"), listed, ["disk-cleanup"])

    assert error.value.code == "hermes_plugin_unavailable"
    assert error.value.metadata["reason"] == "its entry also sets enabled: false"


@pytest.mark.parametrize(
    ("entry", "reason"),
    [
        pytest.param(
            {"enabled": True, "error": "ImportError: boom"},
            "ImportError: boom",
            id="load-error",
        ),
        pytest.param(
            {"enabled": False, "error": "disabled via config"},
            "disabled via config",
            id="disabled",
        ),
        pytest.param({"enabled": False, "error": None}, "not enabled", id="no-error"),
    ],
)
def test_check_enabled_rejects_a_plugin_that_did_not_load(entry, reason):
    listed = [{"key": "broken", "name": "broken", **entry}]

    with pytest.raises(LifecycleError) as error:
        hermes_plugins.check_enabled(_ids("broken"), listed)

    assert error.value.code == "hermes_plugin_unavailable"
    assert error.value.metadata == {
        "field": "harness.settings.plugins.bundled.entries.broken",
        "plugin": "broken",
        "reason": reason,
    }
    assert reason in error.value.message


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_loads_an_enabled_bundled_plugin_and_disables_a_backend(
    tmp_path: Path,
):
    runtime = adapter.HermesRuntime()
    await runtime.start(
        _payload(
            tmp_path,
            _bundled(
                **{
                    "security-guidance": {"enabled": True},
                    "browser/firecrawl": {"enabled": False},
                }
            ),
        )
    )
    try:
        listed = _listed()
        assert _entry(listed, "security-guidance")["enabled"] is True
        assert _entry(listed, "security-guidance")["error"] is None
        firecrawl = _entry(listed, "browser/firecrawl")
        assert firecrawl["enabled"] is False
        assert firecrawl["error"] == "disabled via config"
        # A default-on backend that was not disabled still loads.
        assert _entry(listed, "browser/browserbase")["enabled"] is True
    finally:
        await runtime.stop()


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_start_fails_for_an_unknown_enabled_plugin(tmp_path: Path):
    runtime = adapter.HermesRuntime()

    with pytest.raises(LifecycleError) as error:
        await runtime.start(
            _payload(tmp_path, {"plugins_enabled": ["security-guidance", "nope"]})
        )

    assert error.value.code == "hermes_plugin_not_found"
    assert error.value.metadata["field"] == "harness.settings.plugins_enabled[1]"
    assert runtime._started is False


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_start_fails_for_a_plugin_both_enabled_and_turned_off(
    tmp_path: Path,
):
    runtime = adapter.HermesRuntime()

    with pytest.raises(LifecycleError) as error:
        await runtime.start(
            _payload(
                tmp_path,
                {
                    **_bundled(**{"security-guidance": {"enabled": False}}),
                    "plugins_enabled": ["security-guidance"],
                },
            )
        )

    assert error.value.code == "hermes_plugin_unavailable"
    assert "enabled: false" in error.value.message


# --- External plugins -----------------------------------------------------


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hermes_plugins"


def _external(
    paths: list[str], entries: dict[str, object] | None = None
) -> dict[str, object]:
    external: dict[str, object] = {"paths": paths}
    if entries is not None:
        external["entries"] = entries
    return {"plugins": {"external": external}}


def _plugin_folder(parent: Path, folder: str, name: str | None = None) -> Path:
    """A minimal plugin folder; ``name`` is the manifest name, if any."""
    path = parent / folder
    path.mkdir(parents=True)
    manifest = f"name: {name}\n" if name is not None else "version: 1.0.0\n"
    (path / "plugin.yaml").write_text(manifest, encoding="utf-8")
    (path / "__init__.py").write_text("def register(ctx):\n    pass\n")
    return path


def test_plan_mounted_reads_manifest_names_and_resolves_relative_paths(
    tmp_path: Path,
):
    _plugin_folder(tmp_path / "mount", "one", "first-plugin")
    unnamed = _plugin_folder(tmp_path / "mount", "unnamed")

    mounted = hermes_plugins.plan_mounted(
        _external(["mount/one", str(unnamed)]), tmp_path
    )

    assert [(plugin.index, plugin.name, plugin.path) for plugin in mounted] == [
        (0, "first-plugin", (tmp_path / "mount" / "one").resolve()),
        (1, "unnamed", unnamed.resolve()),
    ]


@pytest.mark.parametrize(
    ("make", "code", "reason"),
    [
        pytest.param(
            lambda root: root / "missing", "hermes_plugin_not_found", None, id="missing"
        ),
        pytest.param(
            lambda root: (root / "empty").mkdir() or root / "empty",
            "hermes_plugin_invalid",
            "missing_manifest",
            id="no-manifest",
        ),
        pytest.param(
            lambda root: (root / "file").write_text("x") and root / "file",
            "hermes_plugin_invalid",
            "not_a_folder",
            id="file",
        ),
        pytest.param(
            lambda root: _plugin_folder(root, "slash", "a/b"),
            "hermes_plugin_invalid",
            "invalid_name",
            id="invalid-name",
        ),
    ],
)
def test_plan_mounted_rejects_a_bad_entry(tmp_path: Path, make, code, reason):
    good = _plugin_folder(tmp_path, "good", "good")
    bad = make(tmp_path)

    with pytest.raises(LifecycleError) as error:
        hermes_plugins.plan_mounted(_external([str(good), str(bad)]), tmp_path)

    assert error.value.code == code
    assert error.value.metadata["field"] == "harness.settings.plugins.external.paths[1]"
    assert error.value.metadata.get("reason") == reason


def test_plan_mounted_rejects_a_plugin_invalid_manifest(tmp_path: Path):
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "plugin.yaml").write_text("name: [unclosed\n", encoding="utf-8")

    with pytest.raises(LifecycleError) as error:
        hermes_plugins.plan_mounted(_external([str(folder)]), tmp_path)

    assert error.value.code == "hermes_plugin_invalid"
    assert error.value.metadata["reason"] == "invalid_manifest"


def test_plan_mounted_rejects_two_entries_with_one_name(tmp_path: Path):
    first = _plugin_folder(tmp_path, "a", "same")
    second = _plugin_folder(tmp_path, "b", "same")

    with pytest.raises(LifecycleError) as error:
        hermes_plugins.plan_mounted(_external([str(first), str(second)]), tmp_path)

    assert error.value.code == "hermes_plugin_duplicate"
    assert error.value.metadata["field"] == "harness.settings.plugins.external.paths[1]"


def test_stage_mounted_creates_replaces_and_removes_links(tmp_path: Path):
    home = tmp_path / "home"
    plugin = _plugin_folder(tmp_path / "mount", "mine", "mine")
    stale_target = _plugin_folder(tmp_path / "old", "mine", "mine")
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / "mine").symlink_to(stale_target)
    mounted = hermes_plugins.plan_mounted(_external([str(plugin)]), tmp_path)

    links = hermes_plugins.stage_mounted(mounted, home)

    link = home / "plugins" / "mine"
    assert links == [link]
    assert link.is_symlink()
    assert link.resolve() == plugin.resolve()
    hermes_plugins.remove_mounted(links)
    assert not link.exists() and not link.is_symlink()
    assert (plugin / "plugin.yaml").is_file()


def test_stage_mounted_refuses_to_shadow_an_installed_plugin(tmp_path: Path):
    home = tmp_path / "home"
    _plugin_folder(home / "plugins", "taken", "taken")
    first = _plugin_folder(tmp_path / "mount", "first", "first")
    taken = _plugin_folder(tmp_path / "mount", "taken", "taken")
    mounted = hermes_plugins.plan_mounted(_external([str(first), str(taken)]), tmp_path)

    with pytest.raises(LifecycleError) as error:
        hermes_plugins.stage_mounted(mounted, home)

    assert error.value.code == "hermes_plugin_duplicate"
    assert error.value.metadata["field"] == "harness.settings.plugins.external.paths[1]"
    # The link made before the failure is removed; the installed plugin stays.
    assert not (home / "plugins" / "first").is_symlink()
    assert (home / "plugins" / "taken" / "plugin.yaml").is_file()


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_loads_a_mounted_plugin_and_unlinks_it_at_stop(tmp_path: Path):
    runtime = adapter.HermesRuntime()
    await runtime.start(
        _payload(
            tmp_path,
            _external(
                [str(FIXTURES / "hermes-mounted")],
                {"hermes-mounted": {"enabled": True}},
            ),
        )
    )
    home = runtime._hermes_home
    assert home is not None
    link = home / "plugins" / "hermes-mounted"
    try:
        assert link.is_symlink()
        entry = _entry(_listed(), "hermes-mounted")
        assert entry["enabled"] is True
        assert entry["source"] == "user"
        assert entry["tools"] == 1
    finally:
        await runtime.stop()

    assert not link.is_symlink()
    assert (FIXTURES / "hermes-mounted" / "plugin.yaml").is_file()


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_start_fails_for_a_missing_plugin_path(tmp_path: Path):
    runtime = adapter.HermesRuntime()

    with pytest.raises(LifecycleError) as error:
        await runtime.start(_payload(tmp_path, _external([str(tmp_path / "absent")])))

    assert error.value.code == "hermes_plugin_not_found"
    assert error.value.metadata["field"] == "harness.settings.plugins.external.paths[0]"


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_failed_start_unlinks_mounted_plugins(tmp_path: Path):
    runtime = adapter.HermesRuntime()
    state_dir = tmp_path / "state"

    with pytest.raises(LifecycleError) as error:
        await runtime.start(
            _payload(
                tmp_path,
                {
                    **_external(
                        [str(FIXTURES / "hermes-mounted")],
                        {"hermes-mounted": {"enabled": True}},
                    ),
                    "plugins_enabled": ["nope"],
                },
                state_dir=state_dir,
            )
        )

    assert error.value.code == "hermes_plugin_not_found"
    assert not (state_dir / "plugins" / "hermes-mounted").is_symlink()


def test_plan_mounted_rejects_an_external_entry_no_folder_provides(tmp_path: Path):
    plugin = _plugin_folder(tmp_path, "one", "one")

    with pytest.raises(LifecycleError) as error:
        hermes_plugins.plan_mounted(
            _external([str(plugin)], {"one": {"enabled": True}, "two": {}}), tmp_path
        )

    assert error.value.code == "hermes_plugin_not_found"
    assert error.value.metadata == {
        "field": "harness.settings.plugins.external.entries.two",
        "plugin": "two",
    }


# --- Agent-installed plugins (not supported) ------------------------------


def test_check_no_agent_installs_passes_without_an_installs_folder(tmp_path: Path):
    (tmp_path / "plugins").mkdir()

    hermes_plugins.check_no_agent_installs(tmp_path)


def test_check_no_agent_installs_ignores_hermes_bootstrap_records(tmp_path: Path):
    # Hermes's launcher writes these whenever it starts a native interface.
    bootstrap = tmp_path / "installs" / "0123456789abcdef" / "bootstrap"
    bootstrap.mkdir(parents=True)
    (bootstrap / "default.json").write_text("{}", encoding="utf-8")

    hermes_plugins.check_no_agent_installs(tmp_path)


def test_check_no_agent_installs_rejects_a_plugin_environment(tmp_path: Path):
    installs = tmp_path / "installs"
    environment = installs / "0123456789abcdef" / "environments" / "abc" / "venv"
    environment.mkdir(parents=True)

    with pytest.raises(LifecycleError) as error:
        hermes_plugins.check_no_agent_installs(tmp_path)

    assert error.value.code == "hermes_plugin_agent_installed_unsupported"
    assert error.value.metadata == {
        "path": str(installs),
        "hermes_home": str(tmp_path),
        "environments": [str(environment.parent)],
    }
    assert "plugins.external" in error.value.message
    assert installs.is_dir()


@pytest.mark.usefixtures("mock_hermes_agent")
async def test_runtime_start_fails_on_an_agent_plugin_install_in_a_kept_home(
    tmp_path: Path,
):
    state_dir = tmp_path / "state"
    (state_dir / "installs" / "0123456789abcdef" / "environments" / "abc").mkdir(
        parents=True
    )
    runtime = adapter.HermesRuntime()

    with pytest.raises(LifecycleError) as error:
        await runtime.start(_payload(tmp_path, {}, state_dir=state_dir))

    assert error.value.code == "hermes_plugin_agent_installed_unsupported"
    assert error.value.metadata["path"] == str(state_dir / "installs")
    assert runtime._started is False
    assert (state_dir / "installs").is_dir()

    # Once the user removes the folder, the same kept home starts.
    import shutil

    shutil.rmtree(state_dir / "installs")
    runtime = adapter.HermesRuntime()
    await runtime.start(_payload(tmp_path, {}, state_dir=state_dir))
    await runtime.stop()

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import json
import os
import sys
import tomllib
from io import StringIO
from pathlib import Path

import nemo_fabric_adapters.common.utils as common_utils
import pytest


@pytest.mark.parametrize(
    ("prefix", "base_prefix", "expected"),
    [
        ("/usr", "/usr", None),
        ("/workspace/.venv", "/usr", Path("/workspace/.venv")),
    ],
)
def test_current_virtualenv(
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
    base_prefix: str,
    expected: Path | None,
):
    monkeypatch.setattr(sys, "prefix", prefix)
    monkeypatch.setattr(sys, "base_prefix", base_prefix)

    assert common_utils.current_virtualenv() == expected


@pytest.mark.parametrize(
    ("os_name", "scripts_directory"),
    [("posix", "bin"), ("nt", "Scripts")],
)
def test_virtualenv_subprocess_env(
    monkeypatch: pytest.MonkeyPatch,
    os_name: str,
    scripts_directory: str,
):
    virtualenv = Path("/workspace/.venv")
    monkeypatch.setattr(common_utils, "current_virtualenv", lambda: virtualenv)
    monkeypatch.setattr(os, "name", os_name)
    os.environ["PATH"] = "/usr/bin"
    os.environ["PYTHONHOME"] = "/usr/lib/python"
    os.environ["FABRIC_TEST"] = "preserved"

    env = common_utils.virtualenv_subprocess_env()

    assert env["VIRTUAL_ENV"] == str(virtualenv)
    assert env["PATH"] == os.pathsep.join(
        (str(virtualenv / scripts_directory), "/usr/bin")
    )
    assert "PYTHONHOME" not in env
    assert env["FABRIC_TEST"] == "preserved"


def test_virtualenv_subprocess_env_preserves_environment_outside_virtualenv(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(common_utils, "current_virtualenv", lambda: None)
    os.environ["FABRIC_TEST"] = "preserved"

    env = common_utils.virtualenv_subprocess_env()

    assert env == os.environ
    assert env is not os.environ


def test_request_payload():
    assert common_utils.request_payload({"request": {"input": "hello"}}) == {"input": "hello"}
    assert common_utils.request_payload({}) == {}


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("nvidia", "https://integrate.api.nvidia.com/v1"),
        ("openai", None),
        (None, None),
    ],
)
def test_default_base_url(
    provider: str | None,
    expected: str | None,
):
    assert common_utils.default_base_url(provider) == expected


@pytest.mark.parametrize(
    ("settings", "model_config", "expected"),
    [
        (
            {"base_url": "https://settings.example/v1"},
            {"provider": "nvidia", "settings": {"base_url": "https://model.example/v1"}},
            "https://settings.example/v1",
        ),
        (
            {},
            {"provider": "openai", "settings": {"base_url": "https://model.example/v1"}},
            "https://model.example/v1",
        ),
        ({}, {"provider": "nvidia"}, "https://integrate.api.nvidia.com/v1"),
        ({}, {"provider": "other"}, None),
    ],
)
def test_get_base_url(
    settings: dict[str, object],
    model_config: dict[str, object],
    expected: str | None,
):
    assert common_utils.get_base_url(settings, model_config) == expected


@pytest.mark.parametrize(
    ("selected_model", "models", "expected"),
    [
        (
            "fast",
            {"fast": {"provider": "nvidia", "model": "fast-model"}},
            {"provider": "nvidia", "model": "fast-model"},
        ),
        (
            None,
            {"default": {"provider": "nvidia", "model": "default-model"}},
            {"provider": "nvidia", "model": "default-model"},
        ),
        ("bad", {"bad": "not-a-model-config"}, {}),
    ],
)
def test_selected_model_config(
    selected_model: str | None,
    models: dict[str, object],
    expected: dict[str, object],
):
    settings = {}
    if selected_model is not None:
        settings["model"] = selected_model
    payload = {
        "config": {
            "harness": {"settings": settings},
            "models": models,
        }
    }

    assert common_utils.selected_model_config(payload) == expected


def test_payload_accessors_use_canonical_plan_fields(tmp_path):
    base_dir = str(tmp_path / "outer")
    payload = {
        "agent_name": "outer-agent",
        "base_dir": base_dir,
        "request": {"input": "hello"},
        "environment": {"workspace": "/outer-workspace"},
        "settings": {"outer": True},
        "models": {"outer": {"model": "outer-model"}},
        "capabilities": {"outer": True},
        "runtime_context": {
            "environment": {"workspace": "/runtime-workspace"},
        },
        "config": {
            "harness": {"settings": {"inner": True}},
            "models": {"inner": {"model": "inner-model"}},
        },
        "capability_plan": {"native": {"skill_paths": ["skills"]}},
    }

    assert common_utils.fabric_config(payload) == payload["config"]
    assert common_utils.agent_name(payload) == "outer-agent"
    assert common_utils.base_dir(payload) == base_dir
    assert common_utils.runtime_context(payload) == payload["runtime_context"]
    assert common_utils.environment_payload(payload) == {"workspace": "/runtime-workspace"}
    assert common_utils.settings_payload(payload) == {"inner": True}
    assert common_utils.models_payload(payload) == {"inner": {"model": "inner-model"}}
    assert common_utils.capability_plan(payload) == {"native": {"skill_paths": ["skills"]}}


@pytest.mark.parametrize("value", [None, "", "relative/path"])
def test_base_dir_requires_an_absolute_path(value):
    with pytest.raises(ValueError, match="base_dir"):
        common_utils.base_dir({"base_dir": value})


def test_load_payload_reads_fabric_invocation(tmp_path: Path):
    invocation_path = tmp_path / "invocation.json"
    invocation_path.write_text(
        json.dumps({"request": {"input": "from file"}}),
        encoding="utf-8",
    )
    os.environ["FABRIC_INVOCATION"] = str(invocation_path)

    assert common_utils.load_payload() == {"request": {"input": "from file"}}


def test_load_payload_falls_back_to_stdin(
    monkeypatch: pytest.MonkeyPatch,
):
    os.environ.pop("FABRIC_INVOCATION", None)
    monkeypatch.setattr("sys.stdin", StringIO('{"request": {"input": "from stdin"}}'))

    assert common_utils.load_payload() == {"request": {"input": "from stdin"}}


@pytest.mark.parametrize(
    ("runtime_context", "expected"),
    [
        ({"runtime_id": "runtime-1"}, "runtime-1"),
    ],
)
def test_runtime_id_reads_required_runtime_context(
    runtime_context: dict[str, object],
    expected: str,
):
    assert common_utils.runtime_id({"runtime_context": runtime_context}) == expected


def test_runtime_id_requires_runtime_context():
    with pytest.raises(ValueError, match="runtime_context.runtime_id"):
        common_utils.runtime_id({"runtime_context": {}})


def test_runtime_state_directory_is_scoped_to_runtime(
    tmp_path: Path,
):
    first = common_utils.runtime_state_directory(
        tmp_path / "hermes-home",
        {"runtime_context": {"runtime_id": "runtime-1"}},
    )
    second = common_utils.runtime_state_directory(
        tmp_path / "hermes-home",
        {"runtime_context": {"runtime_id": "runtime-2"}},
    )

    assert first == tmp_path / "hermes-home" / "runtimes" / "runtime-1"
    assert second == tmp_path / "hermes-home" / "runtimes" / "runtime-2"


def test_dump_yaml_falls_back_to_json_when_yaml_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "yaml":
            raise ImportError("No module named yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert common_utils.dump_yaml({"model": {"default": "demo"}}) == json.dumps(
        {"model": {"default": "demo"}},
        indent=2,
        sort_keys=False,
    ) + "\n"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("git", ["git"]),
        (["git", 7, ""], ["git", "7"]),
        (42, ["42"]),
    ],
)
def test_normalize_list(value: object, expected: list[str]):
    assert common_utils.normalize_list(value) == expected


def test_without_none():
    assert common_utils.without_none({"a": 1, "b": None, "c": False}) == {"a": 1, "c": False}


def test_load_relay_plugin_config_wraps_and_normalizes_bare_observability_config(
    tmp_path: Path,
):
    config_path = tmp_path / "relay.json"
    config_path.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "atof": {
                            "enabled": True,
                            "output_directory": "custom-relay",
                        },
                        "atif": {"enabled": True},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(config_path)
    previous_atof_dir = tmp_path / "custom-relay" / "runtime-previous"
    previous_atif_dir = tmp_path / "artifacts" / "relay" / "runtime-previous"
    previous_atof_dir.mkdir(parents=True)
    previous_atif_dir.mkdir(parents=True)
    (previous_atof_dir / "events.atof.jsonl").write_text("{}", encoding="utf-8")
    (previous_atif_dir / "trajectory-old.atif.json").write_text(
        "{}", encoding="utf-8"
    )
    payload = {
        "agent_name": "review-agent",
        "base_dir": str(tmp_path),
        "config": {
            "harness": {"settings": {"model": "review"}},
            "models": {"review": {"model": "nvidia/review-model"}},
        },
        "runtime_context": {"runtime_id": "runtime-current"},
    }

    plugin_config = common_utils.load_relay_plugin_config(payload)
    observability = plugin_config["components"][0]["config"]

    assert plugin_config["version"] == 1
    assert plugin_config["components"][0]["kind"] == "observability"
    assert observability["atof"]["output_directory"] == str(
        tmp_path / "custom-relay" / "runtime-current"
    )
    assert observability["atof"]["filename"] == "events.atof.jsonl"
    assert observability["atof"]["mode"] == "overwrite"
    assert Path(observability["atof"]["output_directory"]).is_dir()
    assert observability["atif"]["output_directory"] == str(
        tmp_path / "artifacts" / "relay" / "runtime-current"
    )
    assert observability["atif"]["filename_template"] == "trajectory-{session_id}.atif.json"
    assert observability["atif"]["agent_name"] == "review-agent"
    assert observability["atif"]["model_name"] == "nvidia/review-model"
    assert Path(observability["atif"]["output_directory"]).is_dir()

    atof_file = Path(observability["atof"]["output_directory"]) / "events.atof.jsonl"
    atif_file = (
        Path(observability["atif"]["output_directory"])
        / "trajectory-current.atif.json"
    )
    atof_file.write_text("{}", encoding="utf-8")
    atif_file.write_text("{}", encoding="utf-8")

    assert common_utils.collect_relay_artifacts(plugin_config) == [
        {"kind": "atof", "path": str(atof_file)},
        {"kind": "atif", "path": str(atif_file)},
    ]


def test_collect_relay_artifacts(tmp_path: Path):
    atof_dir = tmp_path / "atof"
    atif_dir = tmp_path / "atif"
    atof_dir.mkdir()
    atif_dir.mkdir()
    atof_file = atof_dir / "events.atof.jsonl"
    atif_file = atif_dir / "trajectory-1.atif.json"
    ignored_file = atif_dir / "ignored.txt"
    atof_file.write_text("{}", encoding="utf-8")
    atif_file.write_text("{}", encoding="utf-8")
    ignored_file.write_text("ignored", encoding="utf-8")
    plugin_config = {
        "components": [
            {
                "kind": "observability",
                "config": {
                    "atof": {"enabled": True, "output_directory": str(atof_dir)},
                    "atif": {"enabled": True, "output_directory": str(atif_dir)},
                },
            }
        ]
    }

    assert common_utils.collect_relay_artifacts(plugin_config) == [
        {"kind": "atof", "path": str(atof_file)},
        {"kind": "atif", "path": str(atif_file)},
    ]


@pytest.mark.parametrize(
    ("relay_config", "plugin_config", "expected_names"),
    [
        ({"agents": {"codex": {"command": "codex"}}}, None, ("config.toml", None)),
        (None, {"version": 1, "components": []}, (None, "plugins.toml")),
        (
            {"agents": {"codex": {"command": "codex"}}},
            {"version": 1, "components": []},
            ("config.toml", "plugins.toml"),
        ),
    ],
)
def test_write_relay_configs(
    tmp_path: Path,
    relay_config: dict[str, object] | None,
    plugin_config: dict[str, object] | None,
    expected_names: tuple[str | None, str | None],
):
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(tmp_path / "nested" / "relay.json")

    paths = common_utils.write_relay_configs(
        relay_config=relay_config,
        plugin_config=plugin_config,
    )

    assert tuple(path.name if path else None for path in paths) == expected_names
    for path, config in zip(paths, (relay_config, plugin_config), strict=True):
        if path is not None:
            assert path.parent.name == "relay-config"
            with path.open("rb") as stream:
                assert tomllib.load(stream) == config


def test_write_relay_configs_migrates_atof_to_current_cli_contract(tmp_path: Path):
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(tmp_path / "relay.json")
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": True,
                        "output_directory": "/tmp/atof",
                        "filename": "events.jsonl",
                        "mode": "overwrite",
                        "endpoints": [
                            {
                                "url": "https://example.test/events",
                                "transport": "http_post",
                                "headers": {"x-test": "value"},
                                "header_env": {"authorization": "TOKEN"},
                                "timeout_millis": 1000,
                                "field_name_policy": "replace_dots",
                            }
                        ],
                    },
                    "atif": {"enabled": True, "output_directory": "/tmp/atif"},
                },
            }
        ],
    }

    _, plugin_path = common_utils.write_relay_configs(
        plugin_config=plugin_config,
        observability_version=2,
    )

    assert plugin_path is not None
    with plugin_path.open("rb") as stream:
        rendered = tomllib.load(stream)
    observability = rendered["components"][0]["config"]
    assert observability["version"] == 2
    assert observability["atof"] == {
        "enabled": True,
        "sinks": [
            {
                "type": "file",
                "output_directory": "/tmp/atof",
                "filename": "events.jsonl",
                "mode": "overwrite",
            },
            {
                "type": "stream",
                "url": "https://example.test/events",
                "transport": "http_post",
                "headers": {"x-test": "value"},
                "header_env": {"authorization": "TOKEN"},
                "timeout_millis": 1000,
                "field_name_policy": "replace_dots",
            },
        ],
    }
    assert observability["atif"] == {
        "enabled": True,
        "output_directory": "/tmp/atif",
    }
    assert plugin_config["components"][0]["config"]["version"] == 1
    assert "sinks" not in plugin_config["components"][0]["config"]["atof"]

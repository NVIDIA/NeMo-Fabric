# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import json
import types
from io import StringIO
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


@pytest.fixture(name="common_utils", scope="session")
def common_utils_fixture(adapters_common: str) -> types.ModuleType:
    import nemo_fabric_adapters.common.utils as common_utils  # noqa: E402

    return common_utils


def test_payload_accessors_prefer_effective_config(common_utils: types.ModuleType):
    payload = {
        "agent_name": "outer-agent",
        "config_root": "/outer",
        "request": {"input": "hello"},
        "environment": {"workspace": "/outer-workspace"},
        "settings": {"outer": True},
        "models": {"outer": {"model": "outer-model"}},
        "capabilities": {"outer": True},
        "runtime_context": {
            "environment": {"workspace": "/runtime-workspace"},
        },
        "effective_config": {
            "agent_name": "effective-agent",
            "config_root": "/effective",
            "config": {
                "harness": {"settings": {"inner": True}},
                "models": {"inner": {"model": "inner-model"}},
            },
        },
        "capability_plan": {"native": {"skill_paths": ["skills"]}},
    }

    assert common_utils.effective_config(payload) == payload["effective_config"]
    assert common_utils.fabric_config(payload) == payload["effective_config"]["config"]
    assert common_utils.agent_name(payload) == "effective-agent"
    assert common_utils.config_root(payload) == "/effective"
    assert common_utils.runtime_context(payload) == payload["runtime_context"]
    assert common_utils.environment_payload(payload) == {"workspace": "/runtime-workspace"}
    assert common_utils.settings_payload(payload) == {"inner": True}
    assert common_utils.models_payload(payload) == {"inner": {"model": "inner-model"}}
    assert common_utils.capability_plan(payload) == {"native": {"skill_paths": ["skills"]}}


def test_load_payload_reads_fabric_invocation(
    common_utils: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    invocation_path = tmp_path / "invocation.json"
    invocation_path.write_text(
        json.dumps({"request": {"input": "from file"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("FABRIC_INVOCATION", str(invocation_path))

    assert common_utils.load_payload() == {"request": {"input": "from file"}}


def test_load_payload_falls_back_to_stdin(
    common_utils: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FABRIC_INVOCATION", raising=False)
    monkeypatch.setattr("sys.stdin", StringIO('{"request": {"input": "from stdin"}}'))

    assert common_utils.load_payload() == {"request": {"input": "from stdin"}}


@pytest.mark.parametrize(
    ("runtime_context", "expected"),
    [
        ({"session_id": "caller-session", "runtime_id": "runtime-1"}, "caller-session"),
        ({"runtime_id": "runtime-1"}, "runtime-1"),
        ({}, None),
    ],
)
def test_runtime_session_id_prefers_caller_session_id(
    common_utils: types.ModuleType,
    runtime_context: dict[str, object],
    expected: str | None,
):
    assert common_utils.runtime_session_id({"runtime_context": runtime_context}) == expected


def test_dump_yaml_falls_back_to_json_when_yaml_is_unavailable(
    common_utils: types.ModuleType,
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
def test_normalize_list(common_utils: types.ModuleType, value: object, expected: list[str]):
    assert common_utils.normalize_list(value) == expected


def test_load_relay_plugin_config_wraps_and_normalizes_bare_observability_config(
    common_utils: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setenv("FABRIC_RELAY_CONFIG_PATH", str(config_path))
    previous_atof_dir = tmp_path / "custom-relay" / "runtime-previous"
    previous_atif_dir = tmp_path / "artifacts" / "relay" / "runtime-previous"
    previous_atof_dir.mkdir(parents=True)
    previous_atif_dir.mkdir(parents=True)
    (previous_atof_dir / "events.atof.jsonl").write_text("{}", encoding="utf-8")
    (previous_atif_dir / "trajectory-old.atif.json").write_text(
        "{}", encoding="utf-8"
    )
    payload = {
        "effective_config": {
            "agent_name": "review-agent",
            "config_root": str(tmp_path),
            "config": {
                "harness": {"settings": {"model": "review"}},
                "models": {"review": {"model": "nvidia/review-model"}},
            },
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


def test_collect_relay_artifacts(common_utils: types.ModuleType, tmp_path: Path):
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
    common_utils: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relay_config: dict[str, object] | None,
    plugin_config: dict[str, object] | None,
    expected_names: tuple[str | None, str | None],
):
    monkeypatch.setenv(
        "FABRIC_RELAY_CONFIG_PATH", str(tmp_path / "nested" / "relay.json")
    )

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

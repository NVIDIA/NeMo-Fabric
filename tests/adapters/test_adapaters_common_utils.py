# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import json
import os
import sys
import tomllib
from io import StringIO
from pathlib import Path
from typing import Any

import nemo_fabric_adapters.common.utils as common_utils
import pytest


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "X Foo",
        "X:Foo",
        "X-Föö",
        "X-Foo\0",
        "X-Foo\v",
        "X-Foo\r",
        "X-Foo\n",
    ],
)
def test_normalize_custom_headers_rejects_invalid_names(name):
    with pytest.raises(ValueError, match="Invalid HTTP header name"):
        common_utils.normalize_custom_headers("docs", {name: "bar"})


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "must not be blank"),
        (" \t ", "must not be blank"),
        (" value", "outer whitespace"),
        ("value\t", "outer whitespace"),
        ("bar\0", "control character"),
        ("bar\v", "control character"),
        ("bar\r", "control character"),
        ("bar\nX-Evil: injected", "control character"),
        ("bar\x7f", "control character"),
        ("Bearer 🔑", "not Latin-1 encodable"),
    ],
)
def test_normalize_custom_headers_rejects_invalid_values(value, message):
    with pytest.raises(ValueError, match=message):
        common_utils.normalize_custom_headers("docs", {"X-Foo": value})


def test_normalize_custom_headers_accepts_latin_1_and_embedded_tab():
    assert common_utils.normalize_custom_headers(
        "docs", {"X-Description": "café\tvalue"}
    ) == {"X-Description": "café\tvalue"}


def test_normalize_custom_headers_rejects_non_string_value():
    with pytest.raises(TypeError):
        common_utils.normalize_custom_headers("docs", {"X-Foo": None})


def test_normalize_custom_headers_expands_environment_variables():
    os.environ["FABRIC_HEADER_VALUE"] = "fabric"

    assert common_utils.normalize_custom_headers(
        "docs", {"X-Tenant": "${FABRIC_HEADER_VALUE}"}
    ) == {"X-Tenant": "fabric"}


@pytest.mark.parametrize("value", ["", " \t ", "fabric\v", "fabric\r\n"])
def test_normalize_custom_headers_rejects_invalid_values_after_expansion(value):
    os.environ["FABRIC_HEADER_VALUE"] = value

    with pytest.raises(ValueError, match="HTTP header value"):
        common_utils.normalize_custom_headers(
            "docs", {"X-Tenant": "${FABRIC_HEADER_VALUE}"}
        )


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
    assert common_utils.request_payload({"request": {"input": "hello"}}) == {
        "input": "hello"
    }
    assert common_utils.request_payload({}) == {}


@pytest.mark.parametrize(
    ("model_config", "expected"),
    [
        (
            {"provider": "nvidia", "base_url": "https://model.example/v1"},
            "https://model.example/v1",
        ),
        (
            {"provider": "openai", "base_url": "https://model.example/v1"},
            "https://model.example/v1",
        ),
        ({"provider": "nvidia"}, None),
        ({"provider": "other"}, None),
    ],
)
def test_get_base_url(
    model_config: dict[str, object],
    expected: str | None,
):
    assert common_utils.get_base_url(model_config) == expected


@pytest.mark.parametrize(
    ("models", "expected"),
    [
        (
            {"fast": {"provider": "nvidia", "model": "fast-model"}},
            {"provider": "nvidia", "model": "fast-model"},
        ),
        (
            {"default": {"provider": "nvidia", "model": "default-model"}},
            {"provider": "nvidia", "model": "default-model"},
        ),
        ({"bad": "not-a-model-config"}, {}),
        (
            {
                "fast": {"provider": "nvidia", "model": "fast-model"},
                "slow": {"provider": "nvidia", "model": "slow-model"},
            },
            {},
        ),
    ],
)
def test_selected_model_config(
    models: dict[str, object],
    expected: dict[str, object],
):
    payload = {
        "config": {
            "harness": {"settings": {}},
            "models": models,
        }
    }

    assert common_utils.selected_model_config(payload) == expected


def test_normalized_instruction_runtime_and_tool_accessors():
    payload = {
        "config": {
            "instructions": {
                "system": {"content": "Be concise.", "mode": "replace"}
            },
            "runtime": {"timeout_seconds": 12.5, "max_turns": 7},
            "tools": {
                "enabled": [],
                "blocked": ["Bash"],
            },
        },
        "runtime_context": {
            "environment": {"env": {"VISIBLE": "yes"}},
        },
    }

    assert common_utils.system_instruction(payload) == "Be concise."
    assert common_utils.max_turns(payload) == 7
    assert common_utils.timeout_seconds(payload, default=30) == 12.5
    assert common_utils.environment_env(payload) == {"VISIBLE": "yes"}
    assert common_utils.blocked_tools(payload) == ["Bash"]
    assert common_utils.enabled_tools(payload) == []


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
    assert common_utils.environment_payload(payload) == {
        "workspace": "/runtime-workspace"
    }
    assert common_utils.settings_payload(payload) == {"inner": True}
    assert common_utils.models_payload(payload) == {"inner": {"model": "inner-model"}}
    assert common_utils.capability_plan(payload) == {
        "native": {"skill_paths": ["skills"]}
    }


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


def test_dump_yaml_falls_back_to_json_when_yaml_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "yaml":
            raise ImportError("No module named yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert (
        common_utils.dump_yaml({"model": {"default": "demo"}})
        == json.dumps(
            {"model": {"default": "demo"}},
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )


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


def test_without_none_preserves_falsey_values():
    assert common_utils.without_none(
        {"zero": 0, "false": False, "empty": "", "missing": None}
    ) == {
        "zero": 0,
        "false": False,
        "empty": "",
    }


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
                            "sinks": [
                                {
                                    "type": "file",
                                    "output_directory": "custom-relay",
                                },
                                {
                                    "type": "stream",
                                    "url": "https://example.test/events",
                                },
                            ],
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
    (previous_atif_dir / "trajectory-old.atif.json").write_text("{}", encoding="utf-8")
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
    assert observability["version"] == 2
    file_sink, stream_sink = observability["atof"]["sinks"]
    assert file_sink["output_directory"] == str(
        tmp_path / "custom-relay" / "runtime-current"
    )
    assert file_sink["filename"] == "events.atof.jsonl"
    assert file_sink["mode"] == "overwrite"
    assert Path(file_sink["output_directory"]).is_dir()
    assert stream_sink == {
        "type": "stream",
        "url": "https://example.test/events",
    }
    assert observability["atif"]["output_directory"] == str(
        tmp_path / "artifacts" / "relay" / "runtime-current"
    )
    assert (
        observability["atif"]["filename_template"]
        == "trajectory-{session_id}.atif.json"
    )
    assert observability["atif"]["agent_name"] == "review-agent"
    assert observability["atif"]["model_name"] == "nvidia/review-model"
    assert Path(observability["atif"]["output_directory"]).is_dir()

    atof_file = Path(file_sink["output_directory"]) / "events.atof.jsonl"
    atif_file = (
        Path(observability["atif"]["output_directory"]) / "trajectory-current.atif.json"
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
    unrelated_atif = atif_dir / "config.json"
    atof_directory = atof_dir / "directory.jsonl"
    atif_directory = atif_dir / "trajectory-directory.atif.json"
    atof_file.write_text("{}", encoding="utf-8")
    atif_file.write_text("{}", encoding="utf-8")
    ignored_file.write_text("ignored", encoding="utf-8")
    unrelated_atif.write_text("{}", encoding="utf-8")
    atof_directory.mkdir()
    atif_directory.mkdir()
    plugin_config = {
        "components": [
            {
                "kind": "observability",
                "config": {
                    "atof": {
                        "enabled": True,
                        "sinks": [
                            {
                                "type": "file",
                                "output_directory": str(atof_dir),
                            },
                            {
                                "type": "stream",
                                "url": "https://example.test/events",
                            },
                        ],
                    },
                    "atif": {
                        "enabled": True,
                        "output_directory": str(atif_dir),
                        "filename_template": "trajectory-{session_id}.atif.json",
                    },
                },
            }
        ]
    }

    assert common_utils.collect_relay_artifacts(plugin_config) == [
        {"kind": "atof", "path": str(atof_file)},
        {"kind": "atif", "path": str(atif_file)},
    ]


@pytest.mark.parametrize("filename_template", [None, "", 123])
def test_collect_relay_artifacts_requires_valid_atif_filename_template(
    tmp_path: Path,
    filename_template: object,
):
    atif_dir = tmp_path / "atif"
    atif_dir.mkdir()
    (atif_dir / "config.json").write_text("{}", encoding="utf-8")
    atif_config: dict[str, Any] = {
        "enabled": True,
        "output_directory": str(atif_dir),
    }
    if filename_template is not None:
        atif_config["filename_template"] = filename_template
    plugin_config = {
        "components": [
            {
                "kind": "observability",
                "config": {"atif": atif_config},
            }
        ]
    }

    assert common_utils.collect_relay_artifacts(plugin_config) == []


def test_collect_relay_artifacts_treats_atif_template_as_literal(tmp_path: Path):
    atif_dir = tmp_path / "atif"
    atif_dir.mkdir()
    configured = atif_dir / "[team]-session.json"
    unrelated = atif_dir / "t-session.json"
    configured.write_text("{}", encoding="utf-8")
    unrelated.write_text("{}", encoding="utf-8")
    plugin_config = {
        "components": [
            {
                "kind": "observability",
                "config": {
                    "atif": {
                        "enabled": True,
                        "output_directory": str(atif_dir),
                        "filename_template": "[team]-{session_id}.json",
                    }
                },
            }
        ]
    }

    assert common_utils.collect_relay_artifacts(plugin_config) == [
        {"kind": "atif", "path": str(configured)}
    ]


def test_collect_relay_artifacts_ignores_missing_output_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "unrelated.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "unrelated.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    plugin_config = {
        "components": [
            {
                "kind": "observability",
                "config": {
                    "atof": {
                        "enabled": True,
                        "sinks": [{"type": "file"}],
                    },
                    "atif": {"enabled": True},
                },
            }
        ]
    }

    assert common_utils.collect_relay_artifacts(plugin_config) == []


def _atof_artifact_config(
    output_directory: object,
    *,
    filename: object | None = None,
) -> dict[str, Any]:
    sink: dict[str, Any] = {
        "type": "file",
        "output_directory": output_directory,
    }
    if filename is not None:
        sink["filename"] = filename
    return {
        "components": [
            {
                "kind": "observability",
                "config": {
                    "atof": {
                        "enabled": True,
                        "sinks": [sink],
                    }
                },
            }
        ]
    }


def test_collect_relay_artifacts_honors_configured_atof_filename(tmp_path: Path):
    atof_dir = tmp_path / "atof"
    atof_dir.mkdir()
    configured = atof_dir / "configured.jsonl"
    ignored = atof_dir / "ignored.jsonl"
    configured.write_text("{}", encoding="utf-8")
    ignored.write_text("{}", encoding="utf-8")
    plugin_config = _atof_artifact_config(atof_dir, filename=configured.name)

    assert common_utils.collect_relay_artifacts(plugin_config) == [
        {"kind": "atof", "path": str(configured)}
    ]


def test_collect_relay_artifacts_missing_configured_atof_file_returns_empty(
    tmp_path: Path,
):
    atof_dir = tmp_path / "atof"
    atof_dir.mkdir()
    (atof_dir / "unrelated.jsonl").write_text("{}", encoding="utf-8")
    plugin_config = _atof_artifact_config(atof_dir, filename="missing.jsonl")

    assert common_utils.collect_relay_artifacts(plugin_config) == []


def test_collect_relay_artifacts_configured_atof_directory_returns_empty(
    tmp_path: Path,
):
    atof_dir = tmp_path / "atof"
    atof_dir.mkdir()
    configured = atof_dir / "directory.jsonl"
    configured.mkdir()
    plugin_config = _atof_artifact_config(atof_dir, filename=configured.name)

    assert common_utils.collect_relay_artifacts(plugin_config) == []


def test_collect_relay_artifacts_ignores_path_resolution_runtime_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    artifact_dir = tmp_path / "atof"
    artifact_dir.mkdir()
    original_resolve = Path.resolve

    def resolve(path: Path, *, strict: bool = False) -> Path:
        if path.name == "loop":
            raise RuntimeError
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)

    assert common_utils.collect_relay_artifacts(
        _atof_artifact_config(tmp_path / "loop")
    ) == []
    assert common_utils.collect_relay_artifacts(
        _atof_artifact_config(artifact_dir, filename="loop")
    ) == []


def test_collect_relay_artifacts_non_string_atof_filename_uses_glob(tmp_path: Path):
    atof_dir = tmp_path / "atof"
    atof_dir.mkdir()
    artifact = atof_dir / "events.jsonl"
    artifact.write_text("{}", encoding="utf-8")
    plugin_config = _atof_artifact_config(atof_dir, filename=123)

    assert common_utils.collect_relay_artifacts(plugin_config) == [
        {"kind": "atof", "path": str(artifact)}
    ]


def test_collect_relay_artifacts_rejects_escaping_atof_filenames(tmp_path: Path):
    atof_dir = tmp_path / "atof"
    atof_dir.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}", encoding="utf-8")

    for filename in (str(outside), "../outside.jsonl"):
        plugin_config = _atof_artifact_config(atof_dir, filename=filename)
        assert common_utils.collect_relay_artifacts(plugin_config) == []


def test_collect_relay_artifacts_ignores_malformed_paths(tmp_path: Path):
    atof_dir = tmp_path / "atof"
    atof_dir.mkdir()
    plugin_config = _atof_artifact_config(
        atof_dir,
        filename="invalid\0name.jsonl",
    )
    plugin_config["components"][0]["config"]["atif"] = {
        "enabled": True,
        "output_directory": "invalid\0directory",
    }

    assert common_utils.collect_relay_artifacts(plugin_config) == []


def test_relay_validates_raw_v06_plugin_config():
    from nemo_relay import plugin

    os.environ["TOKEN"] = "test-token"
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 2,
                    "atof": {
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
                                "transport": "ndjson",
                                "headers": {"x-test": "value"},
                                "header_env": {"authorization": "TOKEN"},
                                "timeout_millis": 1000,
                                "field_name_policy": "replace_dots",
                                "name": "phoenix",
                            },
                        ],
                    },
                },
            }
        ],
    }

    assert plugin.validate(plugin_config)["diagnostics"] == []


def test_relay_validates_unknown_atof_sink_type():
    from nemo_relay import plugin

    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 2,
                    "atof": {
                        "enabled": True,
                        "sinks": [{"type": "unknown"}],
                    },
                },
            }
        ],
    }

    assert plugin.validate(plugin_config)["diagnostics"] == [
        {
            "code": "observability.invalid_plugin_config",
            "component": "observability",
            "level": "error",
            "message": (
                "invalid config: invalid observability plugin config: "
                "unknown variant `unknown`, expected `file` or `stream`"
            ),
        }
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


def test_write_relay_configs_preserves_current_cli_contract(tmp_path: Path):
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(tmp_path / "relay.json")
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 2,
                    "atof": {
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
    assert rendered == plugin_config

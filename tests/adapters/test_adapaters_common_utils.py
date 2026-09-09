# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import json
import os
import re
import sys
import tomllib
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import nemo_fabric_adapters.common.utils as common_utils
import pytest


@pytest.mark.parametrize(
    ("request_id", "expected_context"),
    [
        (
            "018f47a4-3af7-7d94-8e61-9f0f89b5d312",
            (
                "018f47a4-3af7-7d94-8e61-9f0f89b5d312",
                "018f47a4-3af7-7d94-8e61-9f0f89b5d312",
            ),
        ),
        ("request-1", None),
    ],
)
def test_relay_request_context(
    request_id: str,
    expected_context: tuple[str, str] | None,
    monkeypatch: pytest.MonkeyPatch,
):
    relay = ModuleType("nemo_relay")
    propagation_context = MagicMock(
        side_effect=lambda parent_uuid, root_uuid=None: SimpleNamespace(
            parent_uuid=parent_uuid,
            root_uuid=root_uuid,
        )
    )
    create_stack = MagicMock(side_effect=lambda context: context)
    used_stacks = []

    @contextmanager
    def use_stack(stack):
        used_stacks.append(stack)
        yield stack

    relay.PropagationContext = propagation_context
    relay.create_scope_stack_from_propagation = create_stack
    relay.use_scope_stack = use_stack
    monkeypatch.setitem(sys.modules, "nemo_relay", relay)

    request_context, metadata = common_utils.relay_request_context(request_id)
    with request_context:
        pass

    assert metadata == {"nemo_fabric_request_id": request_id}
    if expected_context is None:
        propagation_context.assert_not_called()
        create_stack.assert_not_called()
        assert used_stacks == []
    else:
        propagation_context.assert_called_once_with(
            expected_context[0], root_uuid=expected_context[1]
        )
        create_stack.assert_called_once()
        created_context = create_stack.call_args.args[0]
        assert created_context.parent_uuid == expected_context[0]
        assert created_context.root_uuid == expected_context[1]
        assert used_stacks == [created_context]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("hello", "hello"),
        ({"z": "last", "a": "first"}, '{"a": "first", "z": "last"}'),
        (None, ""),
    ],
)
def test_normalize_user_input(value: Any, expected: str):
    assert common_utils.normalize_user_input(value) == expected


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
def test_validate_http_headers_rejects_invalid_names(name):
    with pytest.raises(
        ValueError,
        match=re.escape(f"Invalid HTTP header name {name!r} for MCP server 'docs'"),
    ):
        common_utils.validate_http_headers("docs", {name: "bar"})


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
def test_validate_http_headers_rejects_invalid_values(value, message):
    with pytest.raises(
        ValueError,
        match=rf"HTTP header value for 'X-Foo' on MCP server 'docs' .*{message}",
    ):
        common_utils.validate_http_headers("docs", {"X-Foo": value})


def test_validate_http_headers_accepts_latin_1_and_embedded_tab():
    assert (
        common_utils.validate_http_headers("docs", {"X-Description": "café\tvalue"})
        is None
    )


def test_validate_http_headers_rejects_non_string_value():
    with pytest.raises(
        TypeError,
        match="HTTP header value for 'X-Foo' on MCP server 'docs' must be a string",
    ):
        common_utils.validate_http_headers("docs", {"X-Foo": None})


def test_expand_http_headers_expands_environment_variables_before_validation():
    os.environ["FABRIC_TEST_HEADER"] = "fabric"

    assert common_utils.expand_http_headers(
        "docs",
        {
            "X-Tenant": "${FABRIC_TEST_HEADER}",
            "X-Static": "static",
        },
    ) == {"X-Tenant": "fabric", "X-Static": "static"}


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
            "instructions": {"system": {"content": "Be concise.", "mode": "replace"}},
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


def test_ambient_relay_plugin_config_paths_prefers_xdg_and_nearest_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    xdg = tmp_path / "xdg"
    home = tmp_path / "home"
    project = tmp_path / "workspace"
    nested = project / "repos" / "service"
    user_config = xdg / "nemo-relay" / "plugins.toml"
    project_config = project / ".nemo-relay" / "plugins.toml"
    ignored_home_config = home / ".config" / "nemo-relay" / "plugins.toml"
    for path in (user_config, project_config, ignored_home_config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("version = 1\n", encoding="utf-8")
    nested.mkdir(parents=True)
    os.environ["XDG_CONFIG_HOME"] = str(xdg)
    os.environ["HOME"] = str(home)
    monkeypatch.chdir(nested)

    assert common_utils.ambient_relay_plugin_config_paths() == [
        user_config,
        project_config,
    ]


def test_ambient_relay_plugin_config_paths_falls_back_to_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    home = tmp_path / "home"
    user_config = home / ".config" / "nemo-relay" / "plugins.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text("version = 1\n", encoding="utf-8")
    os.environ.pop("XDG_CONFIG_HOME", None)
    os.environ["HOME"] = str(home)
    monkeypatch.chdir(tmp_path)

    assert common_utils.ambient_relay_plugin_config_paths() == [user_config]


@pytest.mark.parametrize("xdg_config_home", ["", "relative-config"])
def test_ambient_relay_plugin_config_paths_ignores_invalid_xdg_home(
    xdg_config_home: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    home = tmp_path / "home"
    user_config = home / ".config" / "nemo-relay" / "plugins.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text("version = 1\n", encoding="utf-8")
    os.environ["XDG_CONFIG_HOME"] = xdg_config_home
    os.environ["HOME"] = str(home)
    monkeypatch.chdir(tmp_path)

    assert common_utils.ambient_relay_plugin_config_paths() == [user_config]


def test_reject_ambient_relay_plugin_config_allows_clean_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "xdg")
    monkeypatch.chdir(tmp_path)

    common_utils.reject_ambient_relay_plugin_config()


def test_reject_ambient_relay_plugin_config_reports_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    project_config = tmp_path / ".nemo-relay" / "plugins.toml"
    project_config.parent.mkdir()
    project_config.write_text("version = 1\n", encoding="utf-8")
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "xdg")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match=re.escape(str(project_config))):
        common_utils.reject_ambient_relay_plugin_config()


def test_reject_inherited_relay_plugin_config_allows_system_policy():
    common_utils.reject_inherited_relay_plugin_config(
        {
            "diagnostics": [
                {
                    "level": "warning",
                    "code": "plugin.configuration_inherited",
                    "message": (
                        "inherited plugin configuration from discovered file: "
                        "/etc/nemo-relay/plugins.toml"
                    ),
                }
            ]
        }
    )


@pytest.mark.parametrize(
    "message",
    [
        (
            "inherited plugin configuration from discovered file: "
            "/workspace/.nemo-relay/plugins.toml"
        ),
        (
            "inherited plugin configuration from discovered file: "
            "/etc/nemo-relay/../nemo-relay/plugins.toml"
        ),
        "inherited plugin configuration from an unknown source",
    ],
)
def test_reject_inherited_relay_plugin_config_rejects_unmanaged_sources(message):
    with pytest.raises(RuntimeError, match="user or project files"):
        common_utils.reject_inherited_relay_plugin_config(
            {
                "diagnostics": [
                    {
                        "level": "warning",
                        "code": "plugin.configuration_inherited",
                        "message": message,
                    }
                ]
            }
        )


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


def test_load_relay_plugin_config_wraps_and_normalizes_bare_v3_observability_config(
    tmp_path: Path,
):
    config_path = tmp_path / "relay.json"
    config_path.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "version": 3,
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
    assert observability["version"] == 3
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


def test_load_relay_plugin_config_keeps_empty_config_component_free(tmp_path: Path):
    config_path = tmp_path / "relay.json"
    config_path.write_text(json.dumps({"relay": {"config": {}}}), encoding="utf-8")
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(config_path)

    plugin_config = common_utils.load_relay_plugin_config(
        {
            "base_dir": str(tmp_path),
            "runtime_context": {"runtime_id": "runtime-current"},
        }
    )

    assert plugin_config == {"version": 1, "components": []}
    assert not (tmp_path / "artifacts").exists()


def test_load_relay_plugin_config_accepts_implicit_v3_without_inserting_version(
    tmp_path: Path,
):
    config_path = tmp_path / "relay.json"
    config_path.write_text(
        json.dumps({"relay": {"config": {"atif": {"enabled": False}}}}),
        encoding="utf-8",
    )
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(config_path)

    plugin_config = common_utils.load_relay_plugin_config(
        {
            "base_dir": str(tmp_path),
            "runtime_context": {"runtime_id": "runtime-current"},
        }
    )

    observability = plugin_config["components"][0]["config"]
    assert observability == {"atif": {"enabled": False}}
    assert "version" not in observability


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

    assert (
        common_utils.collect_relay_artifacts(_atof_artifact_config(tmp_path / "loop"))
        == []
    )
    assert (
        common_utils.collect_relay_artifacts(
            _atof_artifact_config(artifact_dir, filename="loop")
        )
        == []
    )


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


def test_relay_0_7_validates_v3_plugin_config():
    from nemo_relay import plugin

    os.environ["TOKEN"] = "test-token"
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 3,
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

    common_utils.validate_relay_observability_v3(plugin_config)
    assert plugin.validate(plugin_config)["diagnostics"] == []


async def test_relay_0_7_initializes_v3_atof_atif_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Regress the initialize-only failure that prompted the Relay 0.6 pin."""

    from nemo_relay import plugin

    os.environ["XDG_CONFIG_HOME"] = str(tmp_path / "xdg")
    monkeypatch.chdir(tmp_path)
    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 3,
                    "atof": {
                        "enabled": True,
                        "sinks": [
                            {
                                "type": "file",
                                "output_directory": str(artifact_directory),
                                "filename": "events.atof.jsonl",
                                "mode": "append",
                            }
                        ],
                    },
                    "atif": {
                        "enabled": True,
                        "output_directory": str(artifact_directory),
                        "filename_template": "trajectory-{session_id}.atif.json",
                        "agent_name": "fabric-agent",
                        "agent_version": "test",
                        "model_name": "test-model",
                    },
                },
            }
        ],
    }
    common_utils.validate_relay_observability_v3(plugin_config)
    assert plugin.validate(plugin_config)["diagnostics"] == []
    async with plugin.plugin(plugin_config) as activation_report:
        assert activation_report["diagnostics"] == []


def test_relay_0_7_validates_all_v3_otlp_fields():
    from nemo_relay import plugin

    os.environ["OTEL_TOKEN"] = "test-token"
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 3,
                    "atif": {
                        "enabled": True,
                        "output_directory": "/tmp/atif",
                        "filename_template": "trajectory-{session_id}.atif.json",
                    },
                    "opentelemetry": {
                        "enabled": True,
                        "endpoints": [
                            {
                                "type": "full",
                                "endpoint": "http://localhost:4318/v1/traces",
                                "mark_projection": "tool",
                                "mark_exclude_names": ["llm.chunk"],
                                "attribute_mappings": [
                                    {
                                        "key": "gen_ai.request.model",
                                        "alias": "llm.model",
                                    }
                                ],
                                "transport": "http_binary",
                                "headers": {"x-tenant": "evaluation"},
                                "header_env": {"authorization": "OTEL_TOKEN"},
                                "resource_attributes": {
                                    "deployment.environment": "test"
                                },
                                "service_name": "fabric",
                                "service_namespace": "platform",
                                "service_version": "0.4.0",
                                "instrumentation_scope": "fabric.relay",
                                "timeout_millis": 1000,
                            },
                            {
                                "type": "openinference",
                                "endpoint": "http://localhost:6006/v1/traces",
                                "transport": "grpc",
                            },
                        ],
                    },
                },
            }
        ],
    }

    common_utils.validate_relay_observability_v3(plugin_config)
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
                    "version": 3,
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


def test_validate_relay_observability_v3_accepts_v3_and_implicit_v3_without_mutation():
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {"version": 3, "atif": {"enabled": False}},
            },
            {
                "kind": "observability",
                "enabled": False,
                "config": {"atof": {"enabled": False}},
            },
            {
                "kind": "model_pricing",
                "enabled": True,
                "config": {"version": 2, "currency": "USD"},
            },
        ],
    }
    original = json.loads(json.dumps(plugin_config))

    common_utils.validate_relay_observability_v3(plugin_config)

    assert plugin_config == original


def test_validate_relay_observability_v3_matches_relay_implicit_version():
    from nemo_relay import plugin

    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {},
            }
        ],
    }

    common_utils.validate_relay_observability_v3(plugin_config)

    assert plugin.validate(plugin_config)["diagnostics"] == []


@pytest.mark.parametrize(
    ("version", "enabled"),
    [
        (1, True),
        (2, True),
        (2, False),
        (4, True),
        (True, True),
        (False, True),
        ("3", True),
        (3.0, True),
        (None, True),
    ],
)
def test_validate_relay_observability_v3_rejects_explicit_non_v3_versions(
    version: object,
    enabled: bool,
):
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": enabled,
                "config": {"version": version},
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match=r"unsupported NeMo Relay observability config version .*expected version 3",
    ):
        common_utils.validate_relay_observability_v3(plugin_config)


def test_validate_relay_observability_v3_reports_version_before_v3_shape():
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 2,
                    "opentelemetry": {
                        "enabled": True,
                        "endpoint": "http://localhost:4318/v1/traces",
                    },
                },
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match=r"unsupported NeMo Relay observability config version 2",
    ):
        common_utils.validate_relay_observability_v3(plugin_config)


@pytest.mark.parametrize(
    ("opentelemetry", "valid"),
    [
        ({"enabled": True}, False),
        ({"enabled": True, "endpoints": []}, False),
        ({"enabled": False, "endpoints": []}, True),
    ],
)
def test_validate_relay_observability_v3_requires_endpoint_when_enabled(
    opentelemetry: dict[str, object],
    valid: bool,
):
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 3,
                    "opentelemetry": opentelemetry,
                },
            }
        ],
    }

    if valid:
        common_utils.validate_relay_observability_v3(plugin_config)
    else:
        with pytest.raises(ValueError, match="requires at least one endpoint"):
            common_utils.validate_relay_observability_v3(plugin_config)


@pytest.mark.parametrize(
    ("opentelemetry", "message"),
    [
        (False, "opentelemetry config must be an object"),
        ({"enabled": "true"}, r"opentelemetry\.enabled must be a boolean"),
        ({"endpoints": None}, r"opentelemetry\.endpoints must be a list"),
        ({"endpoints": "endpoint"}, r"opentelemetry\.endpoints must be a list"),
        (
            {"endpoints": [False]},
            r"endpoint must be an object for opentelemetry\.endpoints\[0\]",
        ),
        (
            {"endpoints": [{"endpoint": "http://localhost:4318/v1/traces"}]},
            r"endpoint type must be one of .*opentelemetry\.endpoints\[0\]\.type",
        ),
        (
            {
                "endpoints": [
                    {
                        "type": "zipkin",
                        "endpoint": "http://localhost:4318/v1/traces",
                    }
                ]
            },
            r"endpoint type must be one of .*opentelemetry\.endpoints\[0\]\.type",
        ),
        (
            {"endpoints": [{"type": "full"}]},
            r"endpoint must be a non-empty string for opentelemetry\.endpoints\[0\]",
        ),
    ],
)
def test_validate_relay_observability_v3_rejects_malformed_opentelemetry(
    opentelemetry: object,
    message: str,
):
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {"version": 3, "opentelemetry": opentelemetry},
            }
        ],
    }

    with pytest.raises(ValueError, match=message):
        common_utils.validate_relay_observability_v3(plugin_config)


@pytest.mark.parametrize("endpoint_type", ["full", "gen_ai", "openinference"])
def test_validate_relay_observability_v3_accepts_endpoint_types(endpoint_type: str):
    common_utils.validate_relay_observability_v3(
        {
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": True,
                    "config": {
                        "version": 3,
                        "opentelemetry": {
                            "endpoints": [
                                {
                                    "type": endpoint_type,
                                    "endpoint": "http://localhost:4318/v1/traces",
                                }
                            ]
                        },
                    },
                }
            ],
        }
    )


@pytest.mark.parametrize("endpoint", [None, 42, "", " \t "])
def test_validate_relay_observability_v3_rejects_invalid_endpoint(endpoint: object):
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 3,
                    "opentelemetry": {
                        "enabled": True,
                        "endpoints": [{"type": "full", "endpoint": endpoint}],
                    },
                },
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match=r"non-empty string for opentelemetry\.endpoints\[0\]",
    ):
        common_utils.validate_relay_observability_v3(plugin_config)


@pytest.mark.parametrize("config", [None, [], "version = 3", 3])
def test_validate_relay_observability_v3_requires_object_config(config: object):
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": False,
                "config": config,
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match="NeMo Relay observability component config must be an object",
    ):
        common_utils.validate_relay_observability_v3(plugin_config)


@pytest.mark.parametrize(
    "config",
    [
        {
            "version": 3,
            "openinference": {
                "enabled": True,
                "endpoint": "http://localhost:6006/v1/traces",
            },
        },
        {
            "version": 3,
            "opentelemetry": {
                "enabled": True,
                "endpoint": "http://localhost:4318/v1/traces",
            },
        },
    ],
)
def test_validate_relay_observability_v3_rejects_legacy_exporter_shapes(config):
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": config,
            }
        ],
    }

    with pytest.raises(ValueError, match="observability config version 3"):
        common_utils.validate_relay_observability_v3(plugin_config)


def test_normalize_relay_output_dirs_validates_all_components_before_mutation(
    tmp_path: Path,
):
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 3,
                    "atof": {
                        "enabled": True,
                        "sinks": [
                            {
                                "type": "file",
                                "output_directory": "first",
                            }
                        ],
                    },
                },
            },
            {
                "kind": "observability",
                "enabled": False,
                "config": {
                    "version": 2,
                    "atif": {"enabled": True},
                },
            },
        ],
    }
    original = json.loads(json.dumps(plugin_config))

    with pytest.raises(ValueError, match="config version 2"):
        common_utils.normalize_relay_output_dirs(
            plugin_config,
            {
                "base_dir": str(tmp_path),
                "runtime_context": {"runtime_id": "runtime-current"},
            },
        )

    assert plugin_config == original
    assert not (tmp_path / "first").exists()
    assert not (tmp_path / "artifacts").exists()


def test_load_relay_plugin_config_rejects_v2_before_creating_directories(
    tmp_path: Path,
):
    config_path = tmp_path / "relay.json"
    config_path.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "version": 1,
                        "components": [
                            {
                                "kind": "observability",
                                "enabled": True,
                                "config": {
                                    "version": 2,
                                    "atif": {"enabled": True},
                                },
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(config_path)

    with pytest.raises(ValueError, match="config version 2"):
        common_utils.load_relay_plugin_config(
            {
                "base_dir": str(tmp_path),
                "runtime_context": {"runtime_id": "runtime-current"},
            }
        )

    assert not (tmp_path / "artifacts").exists()


def test_write_relay_configs_preserves_v3_plugin_config_exactly(tmp_path: Path):
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(tmp_path / "relay.json")
    plugin_config = {
        "version": 1,
        "policy": {"unknown_component": "warn"},
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 3,
                    "atif": {"enabled": False},
                    "opentelemetry": {
                        "enabled": True,
                        "endpoints": [
                            {
                                "type": "openinference",
                                "endpoint": "http://localhost:6006/v1/traces",
                            }
                        ],
                    },
                },
            },
            {
                "kind": "model_pricing",
                "enabled": True,
                "config": {"version": 2, "currency": "USD"},
            },
        ],
    }
    original = json.loads(json.dumps(plugin_config))

    _, plugin_path = common_utils.write_relay_configs(plugin_config=plugin_config)

    assert plugin_path is not None
    with plugin_path.open("rb") as stream:
        assert tomllib.load(stream) == original
    assert plugin_config == original


def test_write_relay_configs_rejects_v2_before_creating_config_directory(
    tmp_path: Path,
):
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(tmp_path / "nested" / "relay.json")
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {"version": 2},
            }
        ],
    }

    with pytest.raises(ValueError, match="config version 2"):
        common_utils.write_relay_configs(
            relay_config={"agents": {}},
            plugin_config=plugin_config,
        )

    assert not (tmp_path / "nested" / "relay-config").exists()


def test_write_relay_configs_rejects_null_endpoints_before_creating_directory(
    tmp_path: Path,
):
    os.environ["FABRIC_RELAY_CONFIG_PATH"] = str(tmp_path / "nested" / "relay.json")
    plugin_config = {
        "version": 1,
        "components": [
            {
                "kind": "observability",
                "enabled": True,
                "config": {
                    "version": 3,
                    "opentelemetry": {"endpoints": None},
                },
            }
        ],
    }

    with pytest.raises(ValueError, match=r"opentelemetry\.endpoints must be a list"):
        common_utils.write_relay_configs(plugin_config=plugin_config)

    assert not (tmp_path / "nested" / "relay-config").exists()

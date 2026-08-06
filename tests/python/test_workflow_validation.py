# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planning validation for descriptor-owned workflow schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from nemo_fabric import Fabric
from nemo_fabric import FabricConfig
from nemo_fabric import FabricConfigError
from nemo_fabric import HarnessConfig
from nemo_fabric import MetadataConfig
from nemo_fabric import ToolsConfig
from nemo_fabric import WorkflowConfig
from nemo_fabric import WorkflowEntrypointConfig


def _workflow_schema(*, optional: bool = False) -> dict[str, Any]:
    workflow: dict[str, Any] = {
        "type": "object",
        "properties": {
            "entrypoint": {
                "type": "object",
                "properties": {
                    "kind": {"const": "workflow_registry"},
                    "ref": {"type": "string", "minLength": 1},
                },
                "required": ["kind", "ref"],
                "additionalProperties": False,
            },
            "settings": {
                "type": "object",
                "properties": {"llm_name": {"type": "string"}},
                "additionalProperties": False,
            },
        },
        "required": ["entrypoint"],
        "additionalProperties": False,
    }
    if optional:
        return {"anyOf": [workflow, {"type": "null"}]}
    return workflow


def _write_descriptor(base_dir: Path, *, optional: bool = False) -> Path:
    path = base_dir / "adapters/workflow/fabric-adapter.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "contract_version": "fabric.adapter/v1alpha1",
                "adapter_id": "test.fabric.workflow",
                "harness": "workflow-test",
                "adapter_kind": "python",
                "runner": {"module": "test.fabric.workflow"},
                "workflow_schema": _workflow_schema(optional=optional),
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_static_contract_descriptor(
    base_dir: Path, *, accepts_blocked_tools: bool = True
) -> Path:
    path = base_dir / "adapters/workflow/fabric-adapter.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    injection_points: dict[str, Any] = {}
    accepts: list[str] = []
    if accepts_blocked_tools:
        injection_points["tools.blocked"] = {"name": "context.tools"}
        accepts.append("tools.blocked")
    path.write_text(
        json.dumps(
            {
                "contract_version": "fabric.adapter/v1alpha1",
                "adapter_id": "test.fabric.workflow",
                "harness": "workflow-test",
                "adapter_kind": "python",
                "runner": {"module": "test.fabric.workflow"},
                "config": {"accepts": accepts},
                "workflow_contracts": [
                    {
                        "entrypoint": {
                            "kind": "langgraph_factory",
                            "ref": "examples.review:build_graph",
                        },
                        "settings_schema": {
                            "type": "object",
                            "properties": {
                                "review_mode": {
                                    "enum": ["security", "quality"]
                                }
                            },
                            "required": ["review_mode"],
                            "additionalProperties": False,
                        },
                        "injection_points": injection_points,
                        "execution_constraints": {"state_owner": "application"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _config(*, workflow: WorkflowConfig | None = None) -> FabricConfig:
    return FabricConfig(
        metadata=MetadataConfig(name="workflow-test"),
        harness=HarnessConfig(
            adapter_id="test.fabric.workflow",
            resolution="preinstalled",
        ),
        workflow=workflow,
    )


def _workflow(**settings: Any) -> WorkflowConfig:
    return WorkflowConfig(
        entrypoint=WorkflowEntrypointConfig(
            kind="workflow_registry",
            ref="test_agent",
        ),
        settings=settings,
    )


def test_workflow_schema_validates_and_preserves_config(tmp_path: Path):
    descriptor = _write_descriptor(tmp_path)
    config = _config(workflow=_workflow(llm_name="default"))

    plan = Fabric().plan(config, base_dir=tmp_path)

    assert plan.config.workflow == config.to_mapping()["workflow"]
    assert plan["adapter_descriptor"]["descriptor"]["workflow_schema"] == (
        _workflow_schema()
    )
    assert Path(plan["adapter_descriptor"]["path"]).samefile(descriptor)


def test_workflow_schema_reports_exact_invalid_setting_path(tmp_path: Path):
    _write_descriptor(tmp_path)

    with pytest.raises(FabricConfigError, match=r"workflow\.settings\.llm_name"):
        Fabric().plan(
            _config(workflow=_workflow(llm_name=7)),
            base_dir=tmp_path,
        )


def test_workflow_schema_controls_whether_workflow_is_required(tmp_path: Path):
    _write_descriptor(tmp_path)

    with pytest.raises(FabricConfigError, match=r"at `workflow`"):
        Fabric().plan(_config(), base_dir=tmp_path)

    _write_descriptor(tmp_path, optional=True)
    plan = Fabric().plan(_config(), base_dir=tmp_path)
    assert plan.config.workflow is None


def test_adapter_without_workflow_schema_rejects_workflow(tmp_path: Path):
    config = FabricConfig(
        metadata=MetadataConfig(name="unsupported-workflow"),
        harness=HarnessConfig(
            adapter_id="nvidia.fabric.claude",
            resolution="preinstalled",
        ),
        workflow=_workflow(llm_name="default"),
    )

    with pytest.raises(
        FabricConfigError,
        match="does not declare a workflow_schema",
    ):
        Fabric().plan(config, base_dir=tmp_path)


def test_static_workflow_contract_is_selected_and_exposed_in_plan(tmp_path: Path):
    _write_static_contract_descriptor(tmp_path)
    config = _config(
        workflow=WorkflowConfig(
            entrypoint=WorkflowEntrypointConfig(
                kind="langgraph_factory",
                ref="examples.review:build_graph",
            ),
            settings={"review_mode": "security"},
        )
    )
    config.tools = ToolsConfig(blocked=["calculator__divide"])

    plan = Fabric().plan(config, base_dir=tmp_path)

    assert plan.workflow_contract is not None
    assert plan.workflow_contract.entrypoint.kind == "langgraph_factory"
    assert plan.workflow_contract.entrypoint.ref == "examples.review:build_graph"
    assert plan.workflow_contract.accepted_fields == ("tools.blocked",)
    assert plan.workflow_contract.injection_points == {
        "tools.blocked": {"name": "context.tools"}
    }
    assert plan.workflow_contract.digest.startswith("sha256:")
    assert plan["capability_plan"]["native"]["tools_configured"] is True


def test_static_workflow_contract_rejects_unknown_settings_before_runtime(tmp_path: Path):
    _write_static_contract_descriptor(tmp_path)
    config = _config(
        workflow=WorkflowConfig(
            entrypoint=WorkflowEntrypointConfig(
                kind="langgraph_factory",
                ref="examples.review:build_graph",
            ),
            settings={"review_mode": "security", "unknown": True},
        )
    )

    with pytest.raises(FabricConfigError, match=r"workflow\.settings\.unknown"):
        Fabric().plan(config, base_dir=tmp_path)


def test_static_workflow_contract_rejects_undeclared_tool_policy(tmp_path: Path):
    _write_static_contract_descriptor(tmp_path, accepts_blocked_tools=False)
    config = _config(
        workflow=WorkflowConfig(
            entrypoint=WorkflowEntrypointConfig(
                kind="langgraph_factory",
                ref="examples.review:build_graph",
            ),
            settings={"review_mode": "security"},
        )
    )
    config.tools = ToolsConfig(blocked=["calculator__divide"])

    with pytest.raises(FabricConfigError, match=r"tools\.blocked"):
        Fabric().plan(config, base_dir=tmp_path)

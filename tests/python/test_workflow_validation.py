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

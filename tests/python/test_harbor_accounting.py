# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

pytestmark = pytest.mark.usefixtures("requires_harbor")


@pytest.fixture(name="result_path")
def result_path_fixture(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "agent_name": "accounting",
                "harness": "custom-agent",
                "adapter_kind": "python",
                "adapter_id": "acme.agent",
                "runtime_id": "runtime-1",
                "invocation_id": "invocation-1",
                "request_id": "request-1",
                "output": {"response": "done"},
                "events": [],
                "metadata": {},
                "artifacts": {"artifacts": []},
                "telemetry": [],
                "usage": {
                    "input_tokens": 20,
                    "input_tokens_include_cache": True,
                    "output_tokens": 5,
                    "cost_usd": 0.25,
                },
            }
        )
    )
    return path


def test_normalized_usage_reaches_harbor_without_telemetry(result_path):
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor.fabric_agent import (
        FabricAgent,
    )

    agent = FabricAgent(logs_dir=result_path.parent, fabric_adapter_id="acme.agent")
    agent._result_path = result_path
    context = AgentContext()
    agent.populate_context_post_run(context)

    assert context.n_input_tokens == 20
    assert context.n_output_tokens == 5
    assert context.n_cache_tokens is None
    assert context.cost_usd == 0.25
    assert context.metadata["fabric"]["usage"]["input_tokens"] == 20


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
@pytest.mark.parametrize(
    ("usage", "expected_input", "expected_cache"),
    [
        (
            {
                "input_tokens": 20,
                "cached_input_tokens": 5,
                "input_tokens_include_cache": True,
            },
            20,
            5,
        ),
        (
            {
                "input_tokens": 20,
                "cached_input_tokens": 5,
                "input_tokens_include_cache": False,
            },
            25,
            5,
        ),
        (
            {
                "input_tokens": 20,
                "cached_input_tokens": 0,
                "input_tokens_include_cache": False,
            },
            20,
            0,
        ),
        ({"input_tokens": 20, "input_tokens_include_cache": False}, None, None),
        ({"cached_input_tokens": 5, "input_tokens_include_cache": False}, None, 5),
        ({"input_tokens": 20, "cached_input_tokens": 5}, None, 5),
        ({"input_tokens": 20}, None, None),
        ({"input_tokens": 20, "cached_input_tokens": 0}, 20, 0),
        (
            {
                "input_tokens": 0,
                "input_tokens_include_cache": True,
                "output_tokens": 0,
                "cost_usd": 0,
            },
            0,
            None,
        ),
        ({"metadata": {"estimated_cost_usd": 0.25}}, None, None),
    ],
)
def test_cache_semantics_and_unknown_values(
    result_path, status, usage, expected_input, expected_cache
):
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor.fabric_agent import (
        populate_context_from_result,
    )

    document = json.loads(result_path.read_text())
    document.update(status=status, usage=usage)
    result_path.write_text(json.dumps(document))
    context = AgentContext()

    populate_context_from_result(context, result_path)

    assert context.n_input_tokens == expected_input
    assert context.n_cache_tokens == expected_cache
    assert context.n_output_tokens == usage.get("output_tokens")
    assert context.cost_usd == usage.get("cost_usd")
    assert context.metadata["fabric"]["status"] == status
    assert context.metadata["fabric"]["usage"] == {"metadata": {}, **usage}


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"metadata": {"estimated_cost_usd": 9}}, (12, 3, 4, 0.25)),
        ({"input_tokens": 20, "cached_input_tokens": 5}, (12, 5, 4, 0.25)),
        ({"input_tokens": 20, "input_tokens_include_cache": False}, (12, 3, 4, 0.25)),
        (
            {
                "input_tokens": 0,
                "input_tokens_include_cache": True,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0,
            },
            (0, 0, 0, 0),
        ),
    ],
)
def test_unknown_usage_preserves_existing_context(result_path, usage, expected):
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor.fabric_agent import (
        populate_context_from_result,
    )

    document = json.loads(result_path.read_text())
    document["usage"] = usage
    result_path.write_text(json.dumps(document))
    context = AgentContext(
        n_input_tokens=12, n_cache_tokens=3, n_output_tokens=4, cost_usd=0.25
    )

    populate_context_from_result(context, result_path)
    populate_context_from_result(context, result_path)

    assert (
        context.n_input_tokens,
        context.n_cache_tokens,
        context.n_output_tokens,
        context.cost_usd,
    ) == expected


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {"input_tokens": 0, "input_tokens_include_cache": True, "cost_usd": 0},
            (0, 3, 4, 0),
        ),
        (
            {
                "input_tokens": 20,
                "cached_input_tokens": 5,
                "input_tokens_include_cache": True,
                "output_tokens": 8,
                "cost_usd": 1.25,
            },
            (20, 5, 8, 1.25),
        ),
        (None, (12, 3, 4, 0.25)),
    ],
)
def test_atif_only_fills_missing_accounting(result_path, tmp_path, usage, expected):
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor.fabric_agent import (
        populate_context_from_result,
        populate_context_from_trajectory,
    )

    document = json.loads(result_path.read_text())
    document["usage"] = usage
    result_path.write_text(json.dumps(document))
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": "runtime-1",
                "agent": {"name": "fabric", "version": "0.1.0"},
                "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
                "final_metrics": {
                    "total_prompt_tokens": 12,
                    "total_cached_tokens": 3,
                    "total_completion_tokens": 4,
                    "total_cost_usd": 0.25,
                },
            }
        )
    )
    context = AgentContext()
    populate_context_from_result(context, result_path)
    populate_context_from_trajectory(context, trajectory)
    populate_context_from_trajectory(context, trajectory)

    assert (
        context.n_input_tokens,
        context.n_cache_tokens,
        context.n_output_tokens,
        context.cost_usd,
    ) == expected

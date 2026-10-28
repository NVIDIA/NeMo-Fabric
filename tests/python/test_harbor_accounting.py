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


def test_cline_reported_usage_reaches_harbor_with_evidence(result_path):
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor.fabric_agent import (
        populate_context_from_result,
    )

    evidence = {
        "observed": {"input_tokens": 20},
        "gaps": ["cache_breakdown_unreported"],
        "complete_source": "nvidia_openai_compatible_http",
    }
    document = json.loads(result_path.read_text())
    document["adapter_id"] = "nvidia.fabric.cline"
    document["adapter_kind"] = "typescript"
    document["usage"] = {
        "input_tokens": 20,
        "input_tokens_include_cache": True,
        "output_tokens": 0,
        "total_tokens": 20,
        "extensions": {"nvidia.fabric.cline.usage_evidence.v1": evidence},
    }
    result_path.write_text(json.dumps(document))
    context = AgentContext()

    populate_context_from_result(context, result_path)

    assert (context.n_input_tokens, context.n_output_tokens) == (20, 0)
    assert context.n_cache_tokens is None
    assert context.cost_usd is None
    assert (
        context.metadata["fabric"]["usage"]["extensions"][
            "nvidia.fabric.cline.usage_evidence.v1"
        ]
        == evidence
    )


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
    ("detailed_usage", "expected_input", "expected_output", "expected_cache"),
    [
        (
            {
                "input_tokens": 20,
                "cached_input_tokens": 5,
                "cache_write_input_tokens": 3,
                "input_tokens_include_cache": False,
                "output_tokens": 5,
                "reasoning_tokens": 4,
                "output_tokens_include_reasoning": False,
            },
            28,
            9,
            5,
        ),
        (
            {
                "input_tokens": 20,
                "cached_input_tokens": 5,
                "cache_write_input_tokens": 3,
                "input_tokens_include_cache": True,
                "output_tokens": 5,
                "reasoning_tokens": 4,
                "output_tokens_include_reasoning": True,
            },
            20,
            5,
            5,
        ),
        (
            {
                "input_tokens": 20,
                "cached_input_tokens": 5,
                "input_tokens_include_cache": False,
                "output_tokens": 5,
                "reasoning_tokens": 4,
                "output_tokens_include_reasoning": False,
            },
            None,
            9,
            5,
        ),
        (
            {
                "input_tokens": 20,
                "cached_input_tokens": 5,
                "cache_write_input_tokens": 3,
                "input_tokens_include_cache": False,
                "output_tokens": 5,
                "output_tokens_include_reasoning": False,
            },
            28,
            None,
            5,
        ),
        (
            {
                "input_tokens": 20,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
                "output_tokens": 5,
                "reasoning_tokens": 0,
            },
            20,
            5,
            0,
        ),
        pytest.param(
            {"cache_write_input_tokens": 3},
            None,
            999,
            None,
            id="cache-write-only-selects-detailed-input",
        ),
        pytest.param(
            {"reasoning_tokens": 4},
            999,
            None,
            999,
            id="reasoning-only-selects-detailed-output",
        ),
        ({"models": []}, 999, 999, 999),
    ],
)
def test_detailed_usage_is_authoritative_per_dimension(
    result_path,
    detailed_usage,
    expected_input,
    expected_output,
    expected_cache,
):
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor.fabric_agent import (
        populate_context_from_result,
    )

    document = json.loads(result_path.read_text())
    document["usage"] = {
        "input_tokens": 999,
        "cached_input_tokens": 999,
        "input_tokens_include_cache": True,
        "output_tokens": 999,
        "metadata": {"nemo_fabric.detailed_token_usage.v1": detailed_usage},
    }
    result_path.write_text(json.dumps(document))
    context = AgentContext()

    populate_context_from_result(context, result_path)

    assert context.n_input_tokens == expected_input
    assert context.n_output_tokens == expected_output
    assert context.n_cache_tokens == expected_cache
    assert context.metadata["fabric"]["usage"]["metadata"] == {
        "nemo_fabric.detailed_token_usage.v1": detailed_usage
    }


ATIF_FINAL_METRICS = {
    "total_prompt_tokens": 12,
    "total_completion_tokens": 4,
    "total_cached_tokens": 3,
    "total_cost_usd": 0.25,
    "total_steps": None,
    "extra": None,
}


@pytest.fixture(name="trajectory_path")
def trajectory_path_fixture(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(
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
    return path


@pytest.fixture(name="failure_path")
def failure_path_fixture(tmp_path):
    path = tmp_path / "failure.json"
    path.write_text(
        json.dumps(
            {
                "runner_error": {
                    "stage": "start",
                    "code": "configuration_failed",
                    "message": "startup failure",
                    "retryable": False,
                }
            }
        )
    )
    return path


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"metadata": {"estimated_cost_usd": 9}}, (None, None, None, None)),
        ({"input_tokens": 20, "cached_input_tokens": 5}, (None, 5, None, None)),
        (
            {"input_tokens": 20, "input_tokens_include_cache": False},
            (None, None, None, None),
        ),
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
        pytest.param(None, (None, None, None, None), id="usage-null"),
        pytest.param(
            {
                "input_tokens": 20,
                "cached_input_tokens": 5,
                "input_tokens_include_cache": True,
                "output_tokens": 7,
                "cost_usd": 1.0,
                "metadata": {
                    "nemo_fabric.detailed_token_usage.v1": {"reasoning_tokens": 4}
                },
            },
            (20, 5, None, 1.0),
            id="detailed-usage-makes-output-unavailable",
        ),
    ],
)
def test_fabric_usage_replaces_existing_context(result_path, usage, expected):
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor.fabric_agent import (
        populate_context_from_result,
    )

    document = json.loads(result_path.read_text())
    document["usage"] = usage
    result_path.write_text(json.dumps(document))
    # Values an earlier ATIF fallback could have left behind.
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


@pytest.mark.parametrize("trajectory_first", [False, True])
@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {"input_tokens": 0, "input_tokens_include_cache": True, "cost_usd": 0},
            (0, None, None, 0),
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
        (None, (None, None, None, None)),
        pytest.param(
            {
                "input_tokens": 5,
                "output_tokens": 90,
                "metadata": {
                    "nemo_fabric.detailed_token_usage.v1": {
                        "input_tokens": 5,
                        "cached_input_tokens": 3000,
                        "cache_write_input_tokens": 400,
                        "input_tokens_include_cache": False,
                        "output_tokens": 90,
                        "output_tokens_include_reasoning": True,
                    }
                },
            },
            (3405, 3000, 90, None),
            id="complete-detailed-input-leaves-unreported-cost-unknown",
        ),
        pytest.param(
            {
                "input_tokens": 5,
                "output_tokens": 90,
                "metadata": {
                    "nemo_fabric.detailed_token_usage.v1": {
                        "output_tokens": 90,
                        "output_tokens_include_reasoning": True,
                    }
                },
            },
            (None, None, 90, None),
            id="output-only-detailed-leaves-input-and-cache-unknown",
        ),
    ],
)
def test_atif_never_fills_fabric_accounting(
    result_path, trajectory_path, usage, expected, trajectory_first
):
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor.fabric_agent import (
        populate_context_from_result,
        populate_context_from_trajectory,
    )

    document = json.loads(result_path.read_text())
    document["usage"] = usage
    result_path.write_text(json.dumps(document))
    context = AgentContext()
    if trajectory_first:
        populate_context_from_trajectory(context, trajectory_path)
    populate_context_from_result(context, result_path)
    populate_context_from_trajectory(context, trajectory_path)
    populate_context_from_trajectory(context, trajectory_path)

    assert (
        context.n_input_tokens,
        context.n_cache_tokens,
        context.n_output_tokens,
        context.cost_usd,
    ) == expected
    assert context.metadata["fabric"]["atif_final_metrics"] == {
        "source": "atif",
        "coverage": "unverified",
        "final_metrics": ATIF_FINAL_METRICS,
    }


@pytest.mark.parametrize(
    "passes",
    [
        ("trajectory", "result"),
        ("result", "trajectory", "result"),
        ("trajectory", "failure"),
        ("failure", "trajectory", "result"),
        ("result", "trajectory", "failure"),
    ],
)
def test_result_pass_keeps_atif_evidence(
    result_path, failure_path, trajectory_path, passes
):
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor.fabric_agent import (
        populate_context_from_result,
        populate_context_from_trajectory,
    )

    paths = {"result": result_path, "failure": failure_path}
    context = AgentContext()
    for name in passes:
        if name == "trajectory":
            populate_context_from_trajectory(context, trajectory_path)
        else:
            populate_context_from_result(context, paths[name])

    fabric = context.metadata["fabric"]
    assert fabric["atif_final_metrics"] == {
        "source": "atif",
        "coverage": "unverified",
        "final_metrics": ATIF_FINAL_METRICS,
    }
    assert fabric["harbor_atif_validation"] == {"status": "succeeded", "error": None}
    # Only the last pass's result fields remain.
    if passes[-1] == "result":
        assert fabric["invocation_id"] == "invocation-1"
        assert "runner_error" not in fabric
    else:
        assert fabric["runner_error"]["code"] == "configuration_failed"
        assert "invocation_id" not in fabric


@pytest.mark.parametrize("last_pass", ["result", "failure"])
def test_result_pass_keeps_telemetry_validation(
    result_path, failure_path, tmp_path, last_pass
):
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor.fabric_agent import (
        populate_context_from_result,
        populate_context_from_telemetry_summary,
    )

    summary = {"status": "succeeded", "atof": {"records": 7}}
    summary_path = tmp_path / "telemetry-validation.json"
    summary_path.write_text(json.dumps(summary))
    context = AgentContext()
    populate_context_from_result(context, result_path)
    populate_context_from_telemetry_summary(context, summary_path)

    populate_context_from_result(
        context, {"result": result_path, "failure": failure_path}[last_pass]
    )

    assert context.metadata["fabric"]["telemetry_validation"] == summary


@pytest.mark.parametrize("prepopulated", [False, True])
@pytest.mark.parametrize("runner_failure_document", [False, True])
def test_no_fabric_result_keeps_atif_out_of_accounting(
    failure_path, trajectory_path, runner_failure_document, prepopulated
):
    from harbor.models.agent.context import AgentContext
    from nemo_fabric.integrations.harbor.fabric_agent import FabricAgent

    agent = FabricAgent(logs_dir=failure_path.parent, fabric_adapter_id="acme.agent")
    if runner_failure_document:
        agent._result_path = failure_path
    context = (
        AgentContext(
            n_input_tokens=12, n_cache_tokens=3, n_output_tokens=4, cost_usd=0.25
        )
        if prepopulated
        else AgentContext()
    )

    agent.populate_context_post_run(context)

    assert (
        context.n_input_tokens,
        context.n_cache_tokens,
        context.n_output_tokens,
        context.cost_usd,
    ) == (None, None, None, None)
    fabric = context.metadata["fabric"]
    assert fabric["atif_final_metrics"] == {
        "source": "atif",
        "coverage": "unverified",
        "final_metrics": ATIF_FINAL_METRICS,
    }
    assert fabric["harbor_atif_validation"] == {"status": "succeeded", "error": None}
    assert ("runner_error" in fabric) is runner_failure_document

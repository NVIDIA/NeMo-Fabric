# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import json
from pathlib import Path
from typing import get_type_hints

import pytest

from nemo_fabric.accounting import summarize_usage
from nemo_fabric.errors import FabricConfigError
from nemo_fabric.types import AppliedPricing, ModelUsage, PerModelUsage, TokenCounts


def test_per_model_complete_counts_have_the_public_token_types():
    counts = get_type_hints(TokenCounts)
    per_model = get_type_hints(PerModelUsage)
    for field in (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
        "total_tokens",
    ):
        assert per_model.get(field) == counts[field] == int | None, field


@pytest.fixture(name="usage")
def usage_fixture():
    return {
        "input_tokens": 100,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 5,
        "total_tokens": 105,
        "observed": {},
        "gaps": [],
        "models_exhaustive": False,
        "cost": {"reported": {}, "estimated": {}},
    }


def test_flat_counts_preserve_reported_zero_and_independent_unknown(usage):
    usage["cached_input_tokens"] = None
    parsed = ModelUsage.from_mapping(usage)
    assert parsed.input_tokens == 100
    assert parsed.cached_input_tokens is None
    assert parsed.cache_write_input_tokens == 0
    assert parsed.to_mapping()["input_tokens"] == 100


@pytest.mark.parametrize("value", [-1, True, 1.5, 1 << 64, {"value": 100}])
def test_invalid_or_old_measurement_count_is_rejected(usage, value):
    usage["input_tokens"] = value
    with pytest.raises(FabricConfigError):
        ModelUsage.from_mapping(usage)


def test_gap_conflict_and_independent_provider_total_are_rejected(usage):
    usage["gaps"] = [{"dimensions": ["input_tokens"], "reason": "missing_usage"}]
    with pytest.raises(FabricConfigError, match="conflicts"):
        ModelUsage.from_mapping(usage)
    usage["gaps"] = []
    usage["total_tokens"] = 999
    with pytest.raises(FabricConfigError, match="input plus output"):
        ModelUsage.from_mapping(usage)


def test_complete_partial_and_missing_preserve_coverage_and_observations(usage):
    partial = deepcopy(usage)
    partial.update(input_tokens=None, total_tokens=None)
    partial["observed"] = {"input_tokens": 50}
    partial["gaps"] = [
        {"dimensions": ["input_tokens", "total_tokens"], "reason": "unfinished_work"}
    ]
    rows = [
        {"runtime_id": "r", "invocation_id": name, "model_usage": value}
        for name, value in [("complete", usage), ("partial", partial)]
    ]
    result = summarize_usage(
        rows, expected=[("r", "complete"), ("r", "partial"), ("r", "missing")]
    )
    assert result["dimensions"]["input_tokens"] == {
        "expected_count": 3,
        "complete_count": 1,
        "incomplete_count": 1,
        "missing_result_count": 1,
        "complete_total": None,
        "observed_sum": 150,
        "reasons": {"missing_result": 1, "unfinished_work": 1},
    }
    assert result["dimensions"]["output_tokens"]["observed_sum"] == 10


def test_unsupported_pricing_preserves_valid_token_accounting(usage):
    usage["cost"]["estimated"] = {"amount_nano_usd": 200, "observed_nano_usd": 200}
    usage["cost"]["pricing"] = {
        "basis": "future_basis",
        "currency": "USD",
        "normalization_version": "future",
        "assumption_policy": "future",
        "bundled_catalog_sha256": "sha256:" + "a" * 64,
    }
    dimensions = summarize_usage(
        [{"runtime_id": "r", "invocation_id": "i", "model_usage": usage}],
        expected=[("r", "i")],
    )["dimensions"]
    assert dimensions["input_tokens"]["complete_total"] == 100
    assert dimensions["estimated_cost"]["complete_total"] is None
    assert dimensions["estimated_cost"]["observed_sum"] is None
    assert dimensions["estimated_cost"]["reasons"] == {"pricing_unverified": 1}


@pytest.fixture(name="priced_usage")
def priced_usage_fixture(usage):
    usage["cost"]["estimated"] = {
        "amount_nano_usd": 14_000,
        "observed_nano_usd": 14_000,
    }
    usage["cost"]["pricing"] = {
        "basis": "catalog",
        "currency": "USD",
        "normalization_version": "cache_reasoning_inclusive_v1",
        "assumption_policy": "applicable_tariff_no_missing_counts_v1",
        "bundled_catalog_sha256": "sha256:" + "a" * 64,
    }
    return usage


@pytest.mark.parametrize("observed_only", [False, True])
def test_unpriced_different_catalog_preserves_cost_and_full_coverage(
    priced_usage, observed_only
):
    unpriced = deepcopy(priced_usage)
    unpriced["cost"]["pricing"]["bundled_catalog_sha256"] = "sha256:" + "b" * 64
    unpriced["cost"]["estimated"] = {}
    unpriced["gaps"] = [{"dimensions": ["estimated_cost"], "reason": "missing_usage"}]
    if observed_only:
        priced_usage["cost"]["estimated"]["amount_nano_usd"] = None
        priced_usage["gaps"] = [
            {"dimensions": ["estimated_cost"], "reason": "unfinished_work"}
        ]
    summary = summarize_usage(
        [
            {"runtime_id": "r", "invocation_id": name, "model_usage": value}
            for name, value in [("priced", priced_usage), ("unpriced", unpriced)]
        ],
        expected=[("r", "priced"), ("r", "unpriced"), ("r", "missing")],
    )
    cost = summary["dimensions"]["estimated_cost"]
    assert cost["expected_count"] == 3
    assert cost["complete_count"] == (0 if observed_only else 1)
    assert cost["incomplete_count"] == (2 if observed_only else 1)
    assert cost["missing_result_count"] == 1
    assert cost["complete_total"] is None
    assert cost["observed_sum"] == 14_000
    expected_reasons = {"missing_usage": 1, "missing_result": 1}
    if observed_only:
        expected_reasons["unfinished_work"] = 1
    assert cost["reasons"] == expected_reasons
    assert "aggregation_error" not in cost
    assert summary["pricing"] == priced_usage["cost"]["pricing"]
    assert "pricing_groups" not in summary
    assert summary["dimensions"]["input_tokens"]["observed_sum"] == 200


def test_two_contributing_catalogs_withhold_sum_and_keep_unpriced_group_coverage(
    priced_usage,
):
    other = deepcopy(priced_usage)
    other["cost"]["pricing"]["bundled_catalog_sha256"] = "sha256:" + "b" * 64
    other["cost"]["estimated"] = {
        "amount_nano_usd": 20_000,
        "observed_nano_usd": 20_000,
    }
    unpriced = deepcopy(priced_usage)
    unpriced["cost"]["estimated"] = {}
    unpriced["gaps"] = [{"dimensions": ["estimated_cost"], "reason": "missing_usage"}]
    rows = [
        {"runtime_id": "r", "invocation_id": name, "model_usage": value}
        for name, value in [
            ("first", priced_usage),
            ("second", other),
            ("unpriced", unpriced),
        ]
    ]
    summary = summarize_usage(
        rows, expected=[("r", row["invocation_id"]) for row in rows]
    )
    cost = summary["dimensions"]["estimated_cost"]
    assert cost["complete_total"] is None
    assert cost["observed_sum"] is None
    assert cost["aggregation_error"] == "incompatible_pricing"
    assert cost["expected_count"] == 3
    assert cost["complete_count"] == 2
    assert cost["incomplete_count"] == 1
    assert cost["reasons"] == {"missing_usage": 1}
    groups = {
        group["pricing"]["bundled_catalog_sha256"]: group["dimensions"][
            "estimated_cost"
        ]
        for group in summary["pricing_groups"]
    }
    first = groups["sha256:" + "a" * 64]
    assert first["expected_count"] == 2
    assert first["incomplete_count"] == 1
    assert first["complete_total"] is None
    assert first["observed_sum"] == 14_000
    assert first["reasons"] == {"missing_usage": 1}
    assert groups["sha256:" + "b" * 64]["complete_total"] == 20_000
    assert summary["dimensions"]["input_tokens"]["complete_total"] == 300


@pytest.mark.integration
async def test_native_runtime_reaches_flat_sdk_and_summary(accounting_runtime):
    root = Path(__file__).resolve().parents[2]
    cases = json.loads(
        (root / "tests/fixtures/usage_accounting/conformance.json").read_text()
    )
    fields = {
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
        "total_tokens",
    }
    for case in cases:
        raw = {
            **case["usage"],
            "extensions": {
                **case["usage"].get("extensions", {}),
                "nemo_fabric.usage_accounting.v1": case["accounting"],
            },
        }
        result = await accounting_runtime.invoke(input=raw)
        assert result.status == "succeeded", case["name"]
        parsed = result.model_usage
        assert isinstance(parsed, ModelUsage), case["name"]
        summary = summarize_usage(
            [result], expected=[(result.runtime_id, result.invocation_id)]
        )
        for field, expected in case["expected"].items():
            if field in fields:
                assert parsed[field] == expected["complete"], (case["name"], field)
                assert parsed.observed.get(field) == expected["observed"], (
                    case["name"],
                    field,
                )
                assert (
                    summary["dimensions"][field]["complete_total"]
                    == expected["complete"]
                ), (case["name"], field)
                assert (
                    summary["dimensions"][field]["observed_sum"] == expected["observed"]
                ), (case["name"], field)


@pytest.mark.parametrize("scope", ["invocation", "model"])
@pytest.mark.parametrize(
    "field,observed",
    [("input_tokens", 101), ("cached_input_tokens", 1), ("output_tokens", 6)],
)
def test_saved_observations_cannot_exceed_a_complete_dimension(
    usage, scope, field, observed
):
    node = usage
    if scope == "model":
        node = {
            key: deepcopy(value)
            for key, value in usage.items()
            if key != "models_exhaustive"
        }
        node["model"] = "fixture-model"
        usage["models"] = [node]
    node["observed"] = {field: observed}
    with pytest.raises(FabricConfigError, match="observation exceeds complete"):
        ModelUsage.from_mapping(usage)


@pytest.mark.parametrize(
    "observed",
    [
        {"input_tokens": 1, "output_tokens": 2, "total_tokens": 999},
        {"input_tokens": 1, "total_tokens": 3},
        {"input_tokens": (1 << 64) - 1, "output_tokens": 1, "total_tokens": 0},
    ],
)
def test_saved_observed_total_requires_matching_known_components(usage, observed):
    usage.update(input_tokens=None, output_tokens=None, total_tokens=None)
    usage["observed"] = observed
    with pytest.raises(FabricConfigError, match="observed total"):
        ModelUsage.from_mapping(usage)


@pytest.mark.parametrize("kind", ["reported", "estimated"])
def test_saved_observed_cost_cannot_exceed_complete_cost(usage, kind):
    usage["cost"][kind] = {"amount_nano_usd": 10, "observed_nano_usd": 11}
    with pytest.raises(FabricConfigError, match="observation exceeds complete"):
        ModelUsage.from_mapping(usage)


def test_saved_subset_observations_are_valid_without_claiming_a_lower_bound(usage):
    usage["observed"] = {"input_tokens": 50, "output_tokens": 2, "total_tokens": 52}
    usage["cost"]["reported"] = {"amount_nano_usd": 10, "observed_nano_usd": 5}
    parsed = ModelUsage.from_mapping(usage)
    assert parsed.input_tokens == 100
    assert parsed.observed.input_tokens == 50
    assert parsed.cost.reported.observed_nano_usd == 5


@pytest.mark.parametrize("per_model", [False, True])
@pytest.mark.parametrize("cached,written", [(11, None), (None, 11), (7, 4)])
def test_complete_cache_subsets_cannot_exceed_inclusive_input(
    usage, per_model, cached, written
):
    usage.update(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cached_input_tokens=cached,
        cache_write_input_tokens=written,
    )
    if per_model:
        from nemo_fabric.types import PerModelUsage

        usage.pop("models_exhaustive")
        usage["model"] = "test-model"
        reader = PerModelUsage.from_mapping
    else:
        reader = ModelUsage.from_mapping
    with pytest.raises(FabricConfigError, match="cache.*input"):
        reader(usage)


@pytest.mark.parametrize("per_model", [False, True])
def test_independent_cache_counts_survive_unknown_input(usage, per_model):
    usage.update(
        input_tokens=None,
        total_tokens=None,
        cached_input_tokens=100,
        cache_write_input_tokens=200,
    )
    if per_model:
        from nemo_fabric.types import PerModelUsage

        usage.pop("models_exhaustive")
        usage["model"] = "test-model"
        parsed = PerModelUsage.from_mapping(usage)
    else:
        parsed = ModelUsage.from_mapping(usage)
    assert parsed.input_tokens is None
    assert parsed.cached_input_tokens == 100
    assert parsed.cache_write_input_tokens == 200


def test_observed_cache_dimensions_do_not_imply_a_shared_subset(usage):
    usage.update(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cached_input_tokens=None,
        cache_write_input_tokens=None,
    )
    usage["observed"] = {"input_tokens": 1, "cached_input_tokens": 100}
    parsed = ModelUsage.from_mapping(usage)
    assert parsed.observed.cached_input_tokens == 100
    assert parsed.observed.input_tokens == 1


@pytest.fixture(name="per_model_usage")
def per_model_usage_fixture(usage):
    model = {
        key: deepcopy(value)
        for key, value in usage.items()
        if key != "models_exhaustive"
    }
    model["model"] = "fixture-model"
    return model


@pytest.mark.parametrize(
    "exhaustive,complete,observed",
    [(False, 101, None), (True, 101, None), (True, 90, None), (False, None, 101)],
)
def test_model_input_contradicting_complete_invocation_is_rejected(
    usage, per_model_usage, exhaustive, complete, observed
):
    usage["models_exhaustive"] = exhaustive
    per_model_usage.update(
        input_tokens=complete,
        total_tokens=None if complete is None else complete + 5,
    )
    if observed is not None:
        per_model_usage["observed"] = {"input_tokens": observed}
    usage["models"] = [per_model_usage]
    with pytest.raises(FabricConfigError, match="model token"):
        ModelUsage.from_mapping(usage)


@pytest.mark.parametrize(
    "field,value",
    [("cached_input_tokens", 1), ("cache_write_input_tokens", 1), ("output_tokens", 6)],
)
def test_independent_model_dimension_exceeding_invocation_is_rejected(
    usage, per_model_usage, field, value
):
    per_model_usage[field] = value
    if field == "output_tokens":
        per_model_usage["total_tokens"] = 100 + value
    usage["models"] = [per_model_usage]
    with pytest.raises(FabricConfigError, match="model token"):
        ModelUsage.from_mapping(usage)


@pytest.mark.parametrize(
    "root,parts", [(100, [70, 40]), ((1 << 64) - 1, [(1 << 64) - 1, 1])]
)
def test_summed_model_tokens_cannot_exceed_complete_invocation_or_uint64(
    usage, per_model_usage, root, parts
):
    usage.update(input_tokens=root, output_tokens=None, total_tokens=None)
    usage["models"] = []
    for index, value in enumerate(parts):
        model = deepcopy(per_model_usage)
        model.update(
            model=f"model-{index}",
            input_tokens=value,
            output_tokens=None,
            total_tokens=None,
        )
        usage["models"].append(model)
    with pytest.raises(FabricConfigError, match="model token"):
        ModelUsage.from_mapping(usage)


@pytest.mark.parametrize("exhaustive", [False, True])
def test_incomplete_model_dimensions_do_not_invalidate_independent_complete_root(
    usage, per_model_usage, exhaustive
):
    usage["models_exhaustive"] = exhaustive
    per_model_usage.update(input_tokens=None, total_tokens=None)
    per_model_usage["observed"] = {"input_tokens": 50}
    usage["models"] = [per_model_usage]
    assert ModelUsage.from_mapping(usage).input_tokens == 100


def test_unknown_root_input_keeps_independent_cache_dimensions(usage, per_model_usage):
    usage.update(
        input_tokens=None,
        total_tokens=None,
        cached_input_tokens=100,
        cache_write_input_tokens=200,
    )
    per_model_usage.update(
        input_tokens=None,
        total_tokens=None,
        cached_input_tokens=90,
        cache_write_input_tokens=150,
    )
    usage["models"] = [per_model_usage]
    parsed = ModelUsage.from_mapping(usage)
    assert parsed.cached_input_tokens == 100
    assert parsed.cache_write_input_tokens == 200


def test_independently_rounded_provider_cost_is_not_a_token_reconciliation_error(
    usage, per_model_usage
):
    usage["models_exhaustive"] = True
    usage["cost"]["reported"] = {"amount_nano_usd": 10}
    per_model_usage["cost"]["reported"] = {"amount_nano_usd": 9}
    usage["models"] = [per_model_usage]
    assert ModelUsage.from_mapping(usage).cost.reported.amount_nano_usd == 10


@pytest.mark.parametrize("cache_rates", [{}, {"cache_read": 0}, {"cache_write": 0}])
def test_applied_pricing_accepts_schema_optional_cache_rates(cache_rates):
    parsed = AppliedPricing.from_mapping(
        {
            "catalog_sha256": "sha256:" + "a" * 64,
            "provider": "fixture-provider",
            "model_id": "fixture-model",
            "pricing_as_of": "2026-10-22",
            "pricing_source": "fixture-catalog",
            "rates": {"input": 10, "output": 20, **cache_rates},
        }
    )
    expected = {
        "input": 10,
        "output": 20,
        "cache_read": cache_rates.get("cache_read"),
        "cache_write": cache_rates.get("cache_write"),
    }
    rates = parsed.rates
    assert json.loads(json.dumps(rates)) == expected
    rates["input"] = 999
    assert parsed.rates == expected


@pytest.mark.parametrize(
    "provider", [{}, {"provider": None}, {"provider": "fixture-provider"}]
)
def test_per_model_optional_provider_attribute_preserves_source_presence(
    per_model_usage, provider
):
    from nemo_fabric.types import PerModelUsage

    per_model_usage.update(provider)
    parsed = PerModelUsage.from_mapping(per_model_usage)
    assert parsed.provider == provider.get("provider")
    serialized = parsed.to_mapping()
    assert ("provider" in serialized) == ("provider" in provider)
    if "provider" in provider:
        assert serialized["provider"] == provider["provider"]


@pytest.mark.parametrize(
    "user_catalog",
    [{}, {"user_catalog_sha256": None}, {"user_catalog_sha256": "sha256:" + "b" * 64}],
)
def test_pricing_basis_optional_user_catalog_attribute_preserves_source_presence(
    user_catalog,
):
    from nemo_fabric.types import PricingBasis

    source = {
        "basis": "catalog",
        "currency": "USD",
        "normalization_version": "cache_reasoning_inclusive_v1",
        "assumption_policy": "applicable_tariff_no_missing_counts_v1",
        "bundled_catalog_sha256": "sha256:" + "a" * 64,
        **user_catalog,
    }
    parsed = PricingBasis.from_mapping(source)
    assert parsed.user_catalog_sha256 == user_catalog.get("user_catalog_sha256")
    assert parsed.to_mapping() == source

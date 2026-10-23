// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Invocation-owned normalized counts and separately identified costs.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::AgentUsage;
use crate::usage_accounting::{AccountingError, GapReason, UsageAccounting};
use crate::usage_normalization as normalized;

/// Token counts for one fixed invocation scope. Missing values never imply zero.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TokenCounts {
    /// Cache-inclusive input.
    #[schemars(range(max = u64::MAX))]
    pub input_tokens: Option<u64>,
    /// Independently known cache reads.
    #[schemars(range(max = u64::MAX))]
    pub cached_input_tokens: Option<u64>,
    /// Independently known cache writes.
    #[schemars(range(max = u64::MAX))]
    pub cache_write_input_tokens: Option<u64>,
    /// Reasoning-inclusive output.
    #[schemars(range(max = u64::MAX))]
    pub output_tokens: Option<u64>,
    /// Input plus output; the independent provider total remains raw evidence.
    #[schemars(range(max = u64::MAX))]
    pub total_tokens: Option<u64>,
}

/// Dimensions whose complete values can be unavailable independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UsageDimension {
    /// Input tokens.
    InputTokens,
    /// Cache reads.
    CachedInputTokens,
    /// Cache writes.
    CacheWriteInputTokens,
    /// Output tokens.
    OutputTokens,
    /// Input plus output.
    TotalTokens,
    /// Provider-reported cost.
    ReportedCost,
    /// Applicable catalog estimate.
    EstimatedCost,
}

/// A completeness gap. Diagnostics never change the dimension or reason semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct UsageGap {
    /// Nonempty set of affected dimensions.
    #[schemars(schema_with = "gap_dimensions_schema")]
    pub dimensions: Vec<UsageDimension>,
    /// Why complete accounting is unavailable.
    pub reason: GapReason,
    /// Bounded stable diagnostic; never an exception message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(max = 64))]
    pub code: Option<String>,
}

fn gap_dimensions_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
    let mut schema = Vec::<UsageDimension>::json_schema(generator);
    schema.insert("minItems".into(), 1.into());
    schema.insert("uniqueItems".into(), true.into());
    schema
}

/// Complete and observed amounts in integer nano-USD.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CostAmount {
    /// Complete amount for this invocation/model scope.
    #[schemars(range(max = u64::MAX))]
    pub amount_nano_usd: Option<u64>,
    /// Valid observed contributions, not a guaranteed bound or invoice.
    #[schemars(range(max = u64::MAX))]
    pub observed_nano_usd: Option<u64>,
}

/// Provider-reported amounts stay separate from reproducible catalog estimates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TokenCost {
    /// Provider-reported amounts, never a harness estimate.
    pub reported: CostAmount,
    /// Catalog estimates; incomplete tariff evidence affects only these amounts.
    pub estimated: CostAmount,
    /// Required basis for any estimated amount.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<crate::pricing::PricingBasis>,
    /// Distinct selected rate sets, rather than one duplicate snapshot per call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub applied_rates: Vec<crate::pricing::AppliedPricing>,
}

/// Normalized usage for one literal model/provider identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PerModelUsage {
    /// Source-established accounting model identity. A configured identity
    /// requires proof that all included work used that model; an outer requested
    /// alias alone is insufficient. This is not served-model provenance.
    pub model: String,
    /// Literal provider identity when established.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Complete normalized token dimensions.
    #[serde(flatten)]
    pub tokens: TokenCounts,
    /// Available deduplicated observations, possibly incomplete.
    pub observed: TokenCounts,
    /// Dimension-specific gaps.
    pub gaps: Vec<UsageGap>,
    /// Provider-reported and catalog-estimated cost.
    pub cost: TokenCost,
}

/// Normalized accounting frozen at return. Late usage never moves to a later invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModelUsage {
    /// Complete normalized token dimensions.
    #[serde(flatten)]
    pub tokens: TokenCounts,
    /// Available deduplicated observations; these are not guaranteed lower bounds.
    pub observed: TokenCounts,
    /// Dimension-specific completeness gaps.
    pub gaps: Vec<UsageGap>,
    /// Whether all model identities used by owned work are represented.
    pub models_exhaustive: bool,
    /// Literal per-model breakdown, when the source establishes attribution.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<PerModelUsage>,
    /// Separately typed provider-reported amounts and catalog estimates.
    pub cost: TokenCost,
}

/// Normalize supported evidence. Legacy usage remains raw rather than gaining
/// a completeness guarantee merely because a newer core read it.
pub(crate) fn normalize_model_usage(
    usage: &AgentUsage,
) -> Result<Option<ModelUsage>, AccountingError> {
    let Some(_) = UsageAccounting::from_usage(usage)? else {
        return Ok(None);
    };
    Ok(Some(view(normalized::describe_model_usage(usage)?)))
}

fn tokens(values: &normalized::ModelMeasurements) -> TokenCounts {
    TokenCounts {
        input_tokens: values.input_tokens.value,
        cached_input_tokens: values.cached_input_tokens.value,
        cache_write_input_tokens: values.cache_write_input_tokens.value,
        output_tokens: values.output_tokens.value,
        total_tokens: values.total_tokens.value,
    }
}

fn observations(values: &normalized::ObservedUsage) -> TokenCounts {
    TokenCounts {
        input_tokens: values.input_tokens,
        cached_input_tokens: values.cached_input_tokens,
        cache_write_input_tokens: values.cache_write_input_tokens,
        output_tokens: values.output_tokens,
        total_tokens: values.total_tokens,
    }
}

fn cost(value: &normalized::TokenCost, observed: &normalized::ObservedUsage) -> TokenCost {
    let mut applied_rates = Vec::new();
    for rate in value
        .applied
        .iter()
        .chain(value.applied_calls.iter().flatten())
    {
        if !applied_rates.contains(rate) {
            applied_rates.push(rate.clone());
        }
    }
    TokenCost {
        reported: CostAmount {
            amount_nano_usd: value.reported.value,
            observed_nano_usd: observed.reported_cost_nano_usd,
        },
        estimated: CostAmount {
            amount_nano_usd: value.estimated.value,
            observed_nano_usd: observed.estimated_cost_nano_usd,
        },
        pricing: value.pricing.clone(),
        applied_rates,
    }
}

fn gaps(
    raw: &[normalized::AccountingGap],
    values: &normalized::ModelMeasurements,
) -> Vec<UsageGap> {
    use normalized::AccountingDimension as Raw;
    let mut result = Vec::new();
    for gap in raw {
        for dimension in &gap.dimensions {
            let (dimension, measurement) = match dimension {
                Raw::InputTokens => (UsageDimension::InputTokens, &values.input_tokens),
                Raw::CachedInputTokens => (
                    UsageDimension::CachedInputTokens,
                    &values.cached_input_tokens,
                ),
                Raw::CacheWriteInputTokens => (
                    UsageDimension::CacheWriteInputTokens,
                    &values.cache_write_input_tokens,
                ),
                Raw::OutputTokens => (UsageDimension::OutputTokens, &values.output_tokens),
                Raw::TotalTokens => (UsageDimension::TotalTokens, &values.total_tokens),
                Raw::ReportedCost => (UsageDimension::ReportedCost, &values.token_cost.reported),
                Raw::EstimatedCost => (UsageDimension::EstimatedCost, &values.token_cost.estimated),
                Raw::ReasoningTokens | Raw::PeakRequestInputTokens => continue,
            };
            let next = UsageGap {
                dimensions: vec![dimension],
                reason: gap.reason,
                code: measurement.reason.clone(),
            };
            if !result.contains(&next) {
                result.push(next);
            }
        }
    }
    result
}

fn view(value: normalized::ModelUsage) -> ModelUsage {
    let measurements = normalized::ModelMeasurements {
        input_tokens: value.input_tokens,
        cached_input_tokens: value.cached_input_tokens,
        cache_write_input_tokens: value.cache_write_input_tokens,
        output_tokens: value.output_tokens,
        reasoning_tokens: value.reasoning_tokens,
        total_tokens: value.total_tokens,
        peak_request_input_tokens: value.peak_request_input_tokens,
        token_cost: value.token_cost,
    };
    let accounting = value.accounting.expect("supported evidence was validated");
    ModelUsage {
        tokens: tokens(&measurements),
        observed: observations(&accounting.observed),
        gaps: gaps(&accounting.gaps, &measurements),
        models_exhaustive: accounting.models_exhaustive,
        models: value
            .models
            .into_iter()
            .map(|model| PerModelUsage {
                model: model.model,
                provider: model.provider,
                tokens: tokens(&model.usage),
                observed: observations(&model.accounting.observed),
                gaps: gaps(&model.accounting.gaps, &model.usage),
                cost: cost(&model.usage.token_cost, &model.accounting.observed),
            })
            .collect(),
        cost: cost(&measurements.token_cost, &accounting.observed),
    }
}

#[cfg(test)]
mod accounting_tests {
    use super::normalize_model_usage;
    use crate::AgentUsage;
    use serde_json::{Value, json};

    fn normalize(base: Value, accounting: Value) -> Value {
        let mut usage: AgentUsage = serde_json::from_value(base).unwrap();
        usage
            .extensions
            .insert("nemo_fabric.usage_accounting.v1".into(), accounting);
        serde_json::to_value(normalize_model_usage(&usage).unwrap().unwrap()).unwrap()
    }

    #[test]
    fn shared_source_vectors_keep_the_reduced_complete_and_observed_contract() {
        let cases: Vec<Value> = serde_json::from_str(include_str!(
            "../../../tests/fixtures/usage_accounting/conformance.json"
        ))
        .unwrap();
        let dimensions = [
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
            "total_tokens",
        ];
        for case in cases {
            let result = normalize(case["usage"].clone(), case["accounting"].clone());
            for (field, expected) in case["expected"].as_object().unwrap() {
                if dimensions.contains(&field.as_str()) {
                    assert_eq!(
                        result[field], expected["complete"],
                        "{} {field}",
                        case["name"]
                    );
                    assert_eq!(
                        result["observed"][field], expected["observed"],
                        "{} {field}",
                        case["name"]
                    );
                }
            }
            assert!(result.get("accounting").is_none());
            assert!(result.get("reasoning_tokens").is_none());
            assert!(result.get("peak_request_input_tokens").is_none());
            assert!(result["input_tokens"].is_number() || result["input_tokens"].is_null());
        }
    }

    #[test]
    fn legacy_usage_does_not_gain_a_completeness_guarantee() {
        let usage: AgentUsage = serde_json::from_value(json!({"input_tokens": 100})).unwrap();
        assert!(normalize_model_usage(&usage).unwrap().is_none());
    }

    #[test]
    fn normalized_total_does_not_replace_the_independent_raw_provider_total() {
        let mut usage: AgentUsage = serde_json::from_value(json!({
            "input_tokens": 100, "input_tokens_include_cache": true,
            "output_tokens": 5, "total_tokens": 999,
            "extensions": {"nemo_fabric.detailed_token_usage.v1": {"output_tokens": 5, "output_tokens_include_reasoning": true}}
        })).unwrap();
        usage.extensions.insert(
            "nemo_fabric.usage_accounting.v1".into(),
            json!({"models_exhaustive": false, "gaps": []}),
        );
        let normalized = normalize_model_usage(&usage).unwrap().unwrap();
        assert_eq!(normalized.tokens.total_tokens, Some(105));
        assert_eq!(usage.total_tokens, Some(999));
    }

    #[test]
    fn overflow_does_not_erase_independent_valid_counts() {
        let result = normalize(
            json!({}),
            json!({"models_exhaustive": true, "gaps": [], "models": [{
            "model": "x", "gaps": [], "reported": {"input_tokens": u64::MAX, "input_tokens_include_cache": true,
                "output_tokens": 1, "output_tokens_include_reasoning": true}}]}),
        );
        assert_eq!(result["input_tokens"], json!(u64::MAX));
        assert_eq!(result["output_tokens"], 1);
        assert!(result["total_tokens"].is_null());
    }

    fn cost_with_requests(
        aggregate: f64,
        requests: &[Option<f64>],
        exhaustive: bool,
    ) -> AgentUsage {
        let counts = json!({"input_tokens": 100, "input_tokens_include_cache": true,
            "output_tokens": 5, "output_tokens_include_reasoning": true, "cost_usd": aggregate});
        let mut usage = AgentUsage {
            input_tokens: Some(100),
            input_tokens_include_cache: Some(true),
            output_tokens: Some(5),
            cost_usd: Some(aggregate),
            ..Default::default()
        };
        usage
            .extensions
            .insert("nemo_fabric.detailed_token_usage.v1".into(), counts.clone());
        usage.extensions.insert("nemo_fabric.usage_accounting.v1".into(), json!({
            "models_exhaustive": true, "gaps": [], "models": [{
                "provider": "test", "model": "test-model", "gaps": [], "reported": counts,
                "requests_exhaustive": exhaustive,
                "requests": requests.iter().map(|cost| json!({"usage": {"cost_usd": cost}})).collect::<Vec<_>>()
            }]
        }));
        usage
    }

    #[test]
    fn contradictory_request_reported_costs_are_rejected() {
        for (requests, exhaustive) in [
            (vec![Some(0.2)], true),
            (vec![Some(0.2)], false),
            (vec![Some(0.04)], true),
            (vec![Some(10_000_000_000.0); 2], true),
        ] {
            let usage = cost_with_requests(0.1, &requests, exhaustive);
            assert!(
                normalize_model_usage(&usage).is_err(),
                "contradictory cost must not become complete: {requests:?}, exhaustive={exhaustive}"
            );
        }
    }

    #[test]
    fn independently_rounded_request_costs_keep_fractional_nanos_and_large_values() {
        for (aggregate, requests) in [
            (1.6e-9, vec![Some(0.4e-9); 4]),
            (
                200_000_000.72337076,
                vec![Some(100_000_000.65093447), Some(100_000_000.07243629)],
            ),
        ] {
            let usage = cost_with_requests(aggregate, &requests, true);
            let result = normalize_model_usage(&usage).unwrap().unwrap();
            assert_eq!(result.tokens.input_tokens, Some(100));
            assert_eq!(result.tokens.total_tokens, Some(105));
            assert!(result.cost.reported.amount_nano_usd.is_some());
        }
    }

    #[test]
    fn partial_or_missing_request_costs_preserve_independent_complete_dimensions() {
        for (requests, exhaustive) in [
            (vec![Some(0.04)], false),
            (vec![Some(0.04), None], true),
            (vec![None], true),
            (vec![Some(0.0)], false),
        ] {
            let usage = cost_with_requests(0.1, &requests, exhaustive);
            let result = normalize_model_usage(&usage).unwrap().unwrap();
            assert_eq!(result.tokens.input_tokens, Some(100));
            assert_eq!(result.tokens.output_tokens, Some(5));
            assert_eq!(result.tokens.total_tokens, Some(105));
            assert_eq!(result.cost.reported.amount_nano_usd, Some(100_000_000));
        }
    }

    fn omit_model_cost(usage: &mut AgentUsage) {
        usage
            .extensions
            .get_mut("nemo_fabric.usage_accounting.v1")
            .unwrap()["models"][0]["reported"]
            .as_object_mut()
            .unwrap()
            .remove("cost_usd");
    }

    #[test]
    fn request_costs_without_model_cost_cannot_certify_a_contradictory_root() {
        let mut usage = cost_with_requests(0.1, &[Some(0.2)], true);
        omit_model_cost(&mut usage);
        let result = normalize_model_usage(&usage).unwrap().unwrap();
        assert_eq!(result.cost.reported.amount_nano_usd, None);
        assert_eq!(result.cost.reported.observed_nano_usd, None);
        assert_eq!(result.tokens.total_tokens, Some(105));
    }

    #[test]
    fn exhaustive_requests_recover_cost_without_losing_legitimate_root_rounding() {
        for (aggregate, requests) in [
            (0.1, vec![Some(0.04), Some(0.06)]),
            (
                200_000_000.72337076,
                vec![Some(100_000_000.65093447), Some(100_000_000.07243629)],
            ),
        ] {
            let mut usage = cost_with_requests(aggregate, &requests, true);
            omit_model_cost(&mut usage);
            let with_root = normalize_model_usage(&usage).unwrap().unwrap();
            assert_eq!(
                with_root.cost.reported.amount_nano_usd,
                Some((aggregate * 1e9).round() as u64)
            );
            assert_eq!(with_root.tokens.total_tokens, Some(105));
            usage.cost_usd = None;
            let recovered = normalize_model_usage(&usage).unwrap().unwrap();
            assert!(recovered.cost.reported.amount_nano_usd.is_some());
            assert_eq!(
                recovered.cost.reported.amount_nano_usd,
                recovered.models[0].cost.reported.amount_nano_usd
            );
        }
    }

    #[test]
    fn incomplete_requests_recover_only_observed_cost_when_no_complete_cost_exists() {
        for (requests, exhaustive) in [(vec![Some(0.04)], false), (vec![Some(0.04), None], true)] {
            let mut usage = cost_with_requests(0.1, &requests, exhaustive);
            omit_model_cost(&mut usage);
            usage.cost_usd = None;
            let result = normalize_model_usage(&usage).unwrap().unwrap();
            assert_eq!(result.cost.reported.amount_nano_usd, None);
            assert_eq!(result.cost.reported.observed_nano_usd, Some(40_000_000));
            assert_eq!(result.tokens.total_tokens, Some(105));
        }
    }

    #[test]
    fn broad_request_gap_cannot_certify_complete_model_or_invocation_counts() {
        let counts = json!({"input_tokens": 100, "input_tokens_include_cache": true,
            "output_tokens": 5, "output_tokens_include_reasoning": true});
        for gap in ["unfinished_work", "missing_usage", "scope_unverified"] {
            for root_complete in [false, true] {
                let mut usage = AgentUsage::default();
                if root_complete {
                    usage.input_tokens = Some(100);
                    usage.input_tokens_include_cache = Some(true);
                    usage.output_tokens = Some(5);
                    usage
                        .extensions
                        .insert("nemo_fabric.detailed_token_usage.v1".into(), counts.clone());
                }
                usage.extensions.insert(
                    "nemo_fabric.usage_accounting.v1".into(),
                    json!({
                        "models_exhaustive": true, "gaps": [], "models": [{
                            "model": "model", "gaps": [],
                            "reported": if root_complete { json!({}) } else { counts.clone() },
                            "requests_exhaustive": true,
                            "requests": [{"usage": {}, "gap": gap}]
                        }]
                    }),
                );
                assert!(
                    normalize_model_usage(&usage).is_err(),
                    "broad request gap must conflict with complete counts: {gap}, root={root_complete}"
                );
            }
        }
    }

    #[test]
    fn broad_request_gap_keeps_observations_when_no_complete_value_is_claimed() {
        let observed = json!({"input_tokens": 100, "input_tokens_include_cache": true,
            "output_tokens": 5, "output_tokens_include_reasoning": true});
        let result = normalize(
            json!({}),
            json!({
                "models_exhaustive": false, "gaps": [], "models": [{
                    "model": "model", "gaps": [], "observed": observed,
                    "observed_tokens_aligned": true, "requests_exhaustive": true,
                    "requests": [{"usage": observed, "gap": "unfinished_work", "tokens_aligned": true}]
                }]
            }),
        );
        assert!(result["input_tokens"].is_null());
        assert!(result["output_tokens"].is_null());
        assert_eq!(result["observed"]["input_tokens"], 100);
        assert_eq!(result["observed"]["output_tokens"], 5);
        for scope in [&result, &result["models"][0]] {
            for dimension in [
                "input_tokens",
                "cached_input_tokens",
                "cache_write_input_tokens",
                "output_tokens",
                "total_tokens",
                "reported_cost",
            ] {
                assert!(
                    scope["gaps"].as_array().unwrap().iter().any(|gap| {
                        gap["dimensions"] == json!([dimension])
                            && gap["reason"] == "unfinished_work"
                    }),
                    "request reason missing for {dimension}: {}",
                    scope["gaps"]
                );
            }
        }

        let mixed = normalize(
            json!({}),
            json!({
                "models_exhaustive": true, "gaps": [], "models": [{
                    "model": "pending", "gaps": [{"dimensions": ["output_tokens"], "reason": "missing_usage"}],
                    "observed": observed, "observed_tokens_aligned": true,
                    "requests_exhaustive": true,
                    "requests": [
                        {"usage": observed, "gap": "unfinished_work", "tokens_aligned": true},
                        {"usage": {}, "gap": "scope_unverified"},
                        {"usage": {}, "gap": "unfinished_work"}
                    ]
                }, {"model": "settled", "reported": observed, "gaps": []}]
            }),
        );
        assert!(mixed["input_tokens"].is_null());
        assert_eq!(mixed["observed"]["input_tokens"], 200);
        assert_eq!(mixed["observed"]["output_tokens"], 10);
        assert_eq!(mixed["models"][1]["total_tokens"], 105);
        assert!(
            !mixed["models"][1]["gaps"]
                .as_array()
                .unwrap()
                .iter()
                .any(|gap| {
                    gap["reason"] == "unfinished_work" || gap["reason"] == "scope_unverified"
                })
        );
        for scope in [&mixed, &mixed["models"][0]] {
            let gaps = scope["gaps"].as_array().unwrap();
            for reason in ["unfinished_work", "scope_unverified"] {
                assert_eq!(
                    gaps.iter()
                        .filter(|gap| {
                            gap["dimensions"] == json!(["output_tokens"]) && gap["reason"] == reason
                        })
                        .count(),
                    1,
                    "each request reason must survive once: {gaps:?}"
                );
            }
        }
        assert!(
            mixed["models"][0]["gaps"]
                .as_array()
                .unwrap()
                .iter()
                .any(|gap| {
                    gap["dimensions"] == json!(["output_tokens"])
                        && gap["reason"] == "missing_usage"
                })
        );
    }

    #[test]
    fn dimension_scoped_request_omission_keeps_independent_complete_output() {
        let result = normalize(
            json!({}),
            json!({
                "models_exhaustive": true, "gaps": [], "models": [{
                    "model": "model", "gaps": [{"dimensions": ["input_tokens"], "reason": "missing_usage"}],
                    "reported": {"output_tokens": 5, "output_tokens_include_reasoning": true},
                    "requests_exhaustive": true,
                    "requests": [{"usage": {"output_tokens": 5, "output_tokens_include_reasoning": true}, "tokens_aligned": true}]
                }]
            }),
        );
        assert!(result["input_tokens"].is_null());
        assert_eq!(result["output_tokens"], 5);
        assert!(result["total_tokens"].is_null());
    }
}

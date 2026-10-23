// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reduced public accounting and honest partial pricing.

use nemo_fabric_core::AgentUsage;
use nemo_fabric_core::model_usage::normalize_model_usage;
use nemo_fabric_core::price_catalog::UserPriceCatalog;
use serde_json::{Value, json};

fn normalize(base: Value, accounting: Value) -> Value {
    let mut usage: AgentUsage = serde_json::from_value(base).unwrap();
    usage
        .extensions
        .insert("nemo_fabric.usage_accounting.v1".into(), accounting);
    serde_json::to_value(normalize_model_usage(&usage, None, false).unwrap().unwrap()).unwrap()
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
    assert!(
        normalize_model_usage(&usage, None, false)
            .unwrap()
            .is_none()
    );
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
    let normalized = normalize_model_usage(&usage, None, false).unwrap().unwrap();
    assert_eq!(normalized.tokens.total_tokens, Some(105));
    assert_eq!(usage.total_tokens, Some(999));
}

fn catalog(provider: &str) -> UserPriceCatalog {
    catalog_with_cache_rates(provider, 0.1, 1.25)
}

fn catalog_with_cache_rates(provider: &str, read: f64, write: f64) -> UserPriceCatalog {
    UserPriceCatalog::from_bytes("test-catalog.json".into(), &serde_json::to_vec(&json!({
        "version": 1, "entries": [{"provider": provider, "model_id": "model", "currency": "USD", "unit": "per_token",
            "rates": {"input_per_million": 1.0, "output_per_million": 2.0, "cache_read_per_million": read, "cache_write_per_million": write},
            "prompt_cache": {"read_accounting": "included_in_prompt_tokens"},
            "pricing_as_of": "2026-10-21", "pricing_source": "test fixture"}]
    })).unwrap()).unwrap()
}

fn priced(provider: &str, counts: Value) -> nemo_fabric_core::model_usage::ModelUsage {
    let mut usage = AgentUsage::default();
    usage.extensions.insert(
        "nemo_fabric.usage_accounting.v1".into(),
        json!({
            "models_exhaustive": true, "gaps": [], "models": [{
                "model": "model", "provider": provider, "reported": counts, "gaps": []}]
        }),
    );
    normalize_model_usage(&usage, Some(&catalog(provider)), false)
        .unwrap()
        .unwrap()
}

#[test]
fn nonzero_anthropic_writes_without_tariff_keep_tokens_and_only_observed_cost() {
    let result = priced(
        "anthropic",
        json!({
            "input_tokens": 100, "cached_input_tokens": 20, "cache_write_input_tokens": 10,
            "input_tokens_include_cache": true, "output_tokens": 5, "output_tokens_include_reasoning": true
        }),
    );
    assert_eq!(result.tokens.input_tokens, Some(100));
    assert_eq!(result.tokens.total_tokens, Some(105));
    assert_eq!(result.cost.estimated.amount_nano_usd, None);
    assert_eq!(result.cost.estimated.observed_nano_usd, Some(82_000));
    assert!(
        result.models[0]
            .gaps
            .iter()
            .any(|gap| gap.code.as_deref() == Some("cache_write_tariff_unverified"))
    );
}

#[test]
fn zero_anthropic_writes_need_no_expiry_evidence() {
    let result = priced(
        "anthropic",
        json!({
            "input_tokens": 100, "cached_input_tokens": 20, "cache_write_input_tokens": 0,
            "input_tokens_include_cache": true, "output_tokens": 5, "output_tokens_include_reasoning": true
        }),
    );
    assert_eq!(result.cost.estimated.amount_nano_usd, Some(92_000));
    assert_eq!(result.cost.estimated.observed_nano_usd, Some(92_000));
}

#[test]
fn output_only_observations_are_priceable_without_claiming_complete_spend() {
    let result = priced(
        "openai",
        json!({"output_tokens": 20, "output_tokens_include_reasoning": true}),
    );
    assert_eq!(result.tokens.output_tokens, Some(20));
    assert_eq!(result.tokens.input_tokens, None);
    assert_eq!(result.cost.estimated.amount_nano_usd, None);
    assert_eq!(result.cost.estimated.observed_nano_usd, Some(40_000));
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

#[test]
fn equal_catalog_rates_do_not_prove_an_unknown_anthropic_write_tariff() {
    let mut usage = AgentUsage::default();
    usage.extensions.insert("nemo_fabric.usage_accounting.v1".into(), json!({
        "models_exhaustive": true, "gaps": [], "models": [{"model": "model", "provider": "anthropic", "gaps": [],
        "reported": {"input_tokens": 100, "cached_input_tokens": 20, "cache_write_input_tokens": 10,
            "input_tokens_include_cache": true, "output_tokens": 5, "output_tokens_include_reasoning": true}}]}));
    let result = normalize_model_usage(
        &usage,
        Some(&catalog_with_cache_rates("anthropic", 1.0, 1.0)),
        false,
    )
    .unwrap()
    .unwrap();
    assert_eq!(result.cost.estimated.amount_nano_usd, None);
    // Uncached input 70 + reads 20 + output 5*2; writes remain unpriced.
    assert_eq!(result.cost.estimated.observed_nano_usd, Some(100_000));
}

#[test]
fn explicit_provider_never_uses_another_providers_catalog_rate() {
    let mut usage = AgentUsage::default();
    usage.extensions.insert(
        "nemo_fabric.usage_accounting.v1".into(),
        json!({"models_exhaustive": true, "gaps": [], "models": [{
            "model": "model", "provider": "custom-route", "gaps": [],
            "reported": {"input_tokens": 100, "cached_input_tokens": 0,
                "input_tokens_include_cache": true, "output_tokens": 5,
                "output_tokens_include_reasoning": true}
        }]}),
    );
    let result = normalize_model_usage(&usage, Some(&catalog("openai")), false)
        .unwrap()
        .unwrap();
    assert_eq!(result.tokens.input_tokens, Some(100));
    assert_eq!(result.tokens.total_tokens, Some(105));
    assert_eq!(result.cost.estimated.amount_nano_usd, None);
    assert_eq!(result.cost.estimated.observed_nano_usd, None);
    assert!(
        result.models[0]
            .gaps
            .iter()
            .any(|gap| gap.code.as_deref() == Some("model_not_in_catalog"))
    );
}

#[test]
fn unverified_route_cannot_use_native_prices_for_a_bundled_model() {
    let mut usage = AgentUsage::default();
    usage.extensions.insert(
        "nemo_fabric.usage_accounting.v1".into(),
        json!({"models_exhaustive": true, "gaps": [], "models": [{
            "model": "gpt-5.4", "gaps": [],
            "reported": {"input_tokens": 100, "cached_input_tokens": 0,
                "input_tokens_include_cache": true, "output_tokens": 5,
                "output_tokens_include_reasoning": true}
        }]}),
    );
    // Global permission for bundled prices cannot establish an individual route.
    let custom = normalize_model_usage(&usage, None, true).unwrap().unwrap();
    assert_eq!(custom.tokens.input_tokens, Some(100));
    assert_eq!(custom.tokens.total_tokens, Some(105));
    assert_eq!(custom.cost.estimated.amount_nano_usd, None);
    assert_eq!(custom.cost.estimated.observed_nano_usd, None);
    assert!(
        custom.models[0]
            .gaps
            .iter()
            .any(|gap| gap.code.as_deref() == Some("provider_price_unverified"))
    );

    // The same bundled model is priceable when the source establishes OpenAI.
    usage
        .extensions
        .get_mut("nemo_fabric.usage_accounting.v1")
        .unwrap()["models"][0]["provider"] = json!("openai");
    let native = normalize_model_usage(&usage, None, true).unwrap().unwrap();
    assert!(native.cost.estimated.amount_nano_usd.is_some());
    assert_eq!(native.tokens.total_tokens, Some(105));
}

fn cost_with_requests(aggregate: f64, requests: &[Option<f64>], exhaustive: bool) -> AgentUsage {
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
            normalize_model_usage(&usage, None, false).is_err(),
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
        let result = normalize_model_usage(&usage, None, false).unwrap().unwrap();
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
        let result = normalize_model_usage(&usage, None, false).unwrap().unwrap();
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
    let result = normalize_model_usage(&usage, None, false).unwrap().unwrap();
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
        let with_root = normalize_model_usage(&usage, None, false).unwrap().unwrap();
        assert_eq!(
            with_root.cost.reported.amount_nano_usd,
            Some((aggregate * 1e9).round() as u64)
        );
        assert_eq!(with_root.tokens.total_tokens, Some(105));
        usage.cost_usd = None;
        let recovered = normalize_model_usage(&usage, None, false).unwrap().unwrap();
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
        let result = normalize_model_usage(&usage, None, false).unwrap().unwrap();
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
                normalize_model_usage(&usage, None, false).is_err(),
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
                    gap["dimensions"] == json!([dimension]) && gap["reason"] == "unfinished_work"
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
                gap["dimensions"] == json!(["output_tokens"]) && gap["reason"] == "missing_usage"
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

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
pub fn normalize_model_usage(
    usage: &AgentUsage,
    catalog: Option<&crate::price_catalog::UserPriceCatalog>,
    bundled_pricing_allowed: bool,
) -> Result<Option<ModelUsage>, AccountingError> {
    normalize_model_usage_for_routes(usage, catalog, |_, _| bundled_pricing_allowed)
}

pub(crate) fn normalize_model_usage_for_routes(
    usage: &AgentUsage,
    catalog: Option<&crate::price_catalog::UserPriceCatalog>,
    bundled_pricing_allowed: impl Fn(Option<&str>, &str) -> bool,
) -> Result<Option<ModelUsage>, AccountingError> {
    let Some(evidence) = UsageAccounting::from_usage(usage)? else {
        return Ok(None);
    };
    let mut result = normalized::describe_model_usage(usage)?;
    crate::pricing::apply_pricing_with_evidence(
        &mut result,
        catalog,
        bundled_pricing_allowed,
        Some(&evidence),
    );
    Ok(Some(view(result)))
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

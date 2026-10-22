// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared normalization and completeness policy for invocation model usage.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::AgentUsage;
use crate::measurement::{
    Attribution, Availability, Measurement, MeasurementSource, MeasurementUnit,
};
use crate::usage_accounting::{
    AccountingError, GapReason, ReportedUsage, UsageAccounting, UsageDimension, UsageGap,
    reported_usage, usd_to_nano_usd,
};

/// Normalized invocation usage. Complete values and valid observations are
/// separate. Without `accounting`, legacy values have no new coverage guarantee.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelUsage {
    /// Cache-inclusive input.
    pub input_tokens: Measurement,
    /// Independently reported cache reads.
    pub cached_input_tokens: Measurement,
    /// Independently reported cache writes.
    pub cache_write_input_tokens: Measurement,
    /// Reasoning-inclusive output.
    pub output_tokens: Measurement,
    /// Independently reported reasoning tokens.
    pub reasoning_tokens: Measurement,
    /// Normalized input plus output, not the provider's independent raw total.
    pub total_tokens: Measurement,
    /// Largest request's cache-inclusive input.
    pub peak_request_input_tokens: Measurement,
    /// Provider-reported cost and separately identified catalog estimate.
    pub token_cost: TokenCost,
    /// One entry per literal model/provider identity.
    #[serde(default)]
    pub models: Vec<PerModelUsage>,
    /// Present only after validating the supported accounting extension.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accounting: Option<InvocationAccounting>,
}

/// Monetary values are checked integer nano-USD, with distinct reported and catalog sources.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TokenCost {
    /// Selected rates by source request index; null entries are unpriced.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub applied_calls: Vec<Option<crate::pricing::AppliedPricing>>,
    /// Provider-reported amount, never a harness estimate or an invoice guarantee.
    pub reported: Measurement,
    /// Central catalog estimate; absent when scope or price inputs are unverified.
    pub estimated: Measurement,
    /// Required comparison basis for catalog prices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<crate::pricing::PricingBasis>,
    /// Exact selected entry and integer rates for one model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub applied: Option<crate::pricing::AppliedPricing>,
}

/// Complete and observed usage for one model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PerModelUsage {
    /// Literal source model identity.
    pub model: String,
    /// Literal source provider identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Normalized values for this model, excluding nested model entries.
    #[serde(flatten)]
    pub usage: ModelMeasurements,
    /// Independently validated observations and gaps for this model.
    pub accounting: ModelAccounting,
}

/// Measurements shared by an invocation and its per-model breakdown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelMeasurements {
    /// Cache-inclusive input.
    pub input_tokens: Measurement,
    /// Cache reads.
    pub cached_input_tokens: Measurement,
    /// Cache writes.
    pub cache_write_input_tokens: Measurement,
    /// Reasoning-inclusive output.
    pub output_tokens: Measurement,
    /// Reasoning tokens.
    pub reasoning_tokens: Measurement,
    /// Input plus output.
    pub total_tokens: Measurement,
    /// Largest request input.
    pub peak_request_input_tokens: Measurement,
    /// Costs for this model.
    pub token_cost: TokenCost,
}

/// A normalized dimension. Cost units are nano-USD, not raw floating-point USD.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AccountingDimension {
    /// Input tokens.
    InputTokens,
    /// Cache reads.
    CachedInputTokens,
    /// Cache writes.
    CacheWriteInputTokens,
    /// Output tokens.
    OutputTokens,
    /// Reasoning tokens.
    ReasoningTokens,
    /// Input plus output.
    TotalTokens,
    /// Largest request input.
    PeakRequestInputTokens,
    /// Provider-reported nano-USD.
    ReportedCost,
    /// Catalog-estimated nano-USD.
    EstimatedCost,
}

/// A normalized dimension affected by missing evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AccountingGap {
    /// Affected dimensions.
    pub dimensions: Vec<AccountingDimension>,
    /// Why these are incomplete.
    pub reason: GapReason,
}

/// Independently valid normalized observations. These are not promised bounds.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ObservedUsage {
    /// Cache-inclusive observed input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub input_tokens: Option<u64>,
    /// Independently observed cache reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub cached_input_tokens: Option<u64>,
    /// Independently observed cache writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub cache_write_input_tokens: Option<u64>,
    /// Reasoning-inclusive observed output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub output_tokens: Option<u64>,
    /// Independently observed reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub reasoning_tokens: Option<u64>,
    /// Sum of available normalized input and output observations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub total_tokens: Option<u64>,
    /// Largest observed request input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub peak_request_input_tokens: Option<u64>,
    /// Observed provider-reported cost.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub reported_cost_nano_usd: Option<u64>,
    /// Catalog price of verifiable observed contributions; not a complete cost.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub estimated_cost_nano_usd: Option<u64>,
}

/// Native invocation accounting. Empty gaps do not turn absent measurements into zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct InvocationAccounting {
    /// Whether the model list covers all identities used by owned work.
    pub models_exhaustive: bool,
    /// Available deduplicated observations; never added to complete totals again.
    pub observed: ObservedUsage,
    /// Explicit and normalization-derived gaps.
    pub gaps: Vec<AccountingGap>,
}

/// Native per-model accounting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelAccounting {
    /// Observed token dimensions retain one verified source-call subset.
    #[serde(default)]
    pub observed_tokens_aligned: bool,
    /// Available deduplicated observations.
    pub observed: ObservedUsage,
    /// Explicit and normalization-derived gaps.
    pub gaps: Vec<AccountingGap>,
}

type Count = Result<u64, &'static str>;
const INPUT: usize = UsageDimension::InputTokens as usize;
const OUTPUT: usize = UsageDimension::OutputTokens as usize;
const TOTAL: usize = UsageDimension::TotalTokens as usize;

// Request lists may be partial, but their usable subtotal cannot exceed a
// complete model total. Exhaustive settled lists must agree; costs allow only
// numeric conversion roundoff, while token comparisons are exact.
fn validate_request_totals(
    model: &crate::usage_accounting::AccountedModel,
    complete: &[Count; 8],
    root_complete: &[Count; 8],
    requests: &[[Count; 8]],
) -> Result<(), AccountingError> {
    // A request-wide gap covers every dimension. Scoped omissions instead
    // leave only their affected request fields absent.
    if model.requests.iter().any(|request| request.gap.is_some())
        && complete.iter().chain(root_complete).any(Result::is_ok)
    {
        return Err(AccountingError(
            "complete value conflicts with request scope gap".into(),
        ));
    }
    for (request, counts) in model.requests.iter().zip(requests) {
        if (request.gap.is_none() || request.tokens_aligned)
            && let (Some(peak), Ok(input)) =
                (request.usage.peak_request_input_tokens, counts[INPUT])
            && peak != input
        {
            return Err(AccountingError(
                "single-request peak contradicts input".into(),
            ));
        }
    }
    for (index, dimension) in UsageDimension::ALL.into_iter().enumerate() {
        let Ok(aggregate) = complete[index] else {
            continue;
        };
        let parts: Vec<_> = requests
            .iter()
            .filter_map(|counts| counts[index].ok())
            .collect();
        if parts.is_empty() {
            continue;
        }
        let subtotal = combine_values(&parts, dimension);
        let exhaustive = model.requests_exhaustive
            && model.requests.iter().all(|request| request.gap.is_none())
            && parts.len() == requests.len();
        let contradictory = subtotal.is_none_or(|sum| {
            let tolerance = if dimension == UsageDimension::CostUsd {
                request_cost_rounding_tolerance(aggregate.max(sum), &parts)
            } else {
                0
            };
            sum.saturating_sub(aggregate) > tolerance
                || (exhaustive && sum.abs_diff(aggregate) > tolerance)
        });
        if contradictory {
            return Err(AccountingError(
                "request totals contradict model evidence".into(),
            ));
        }
    }
    Ok(())
}

// Costs are additive evidence, but each raw f64 USD value is independently
// converted to rounded nano-USD. One ULP in the rounded amount bounds its f64
// spacing; two ULPs plus one nano conservatively cover USD scaling and nearest
// integer rounding. An n-term source sum can round n times at the total's
// magnitude. Use saturating allowances, never a percentage or billing policy.
fn request_cost_rounding_tolerance(total: u64, parts: &[u64]) -> u64 {
    let allowance = |value: u64| {
        let exponent = value.checked_ilog2().unwrap_or(0);
        let ulp = 1_u64 << exponent.saturating_sub(52);
        ulp.saturating_mul(2).saturating_add(1)
    };
    let terms = u64::try_from(parts.len()).unwrap_or(u64::MAX);
    parts
        .iter()
        .fold(allowance(total).saturating_mul(terms), |sum, part| {
            sum.saturating_add(allowance(*part))
        })
}

/// Validate and normalize one adapter result's usage. Callers must also enforce
/// descriptor declarations at the normal runtime extension-validation boundary.
pub fn describe_model_usage(usage: &AgentUsage) -> Result<ModelUsage, AccountingError> {
    let evidence = UsageAccounting::from_usage(usage)?;
    let raw = reported_usage(usage)?;
    let mut counts = normalize(&raw, false);
    if evidence.is_none() {
        // Legacy adapters can put client-side estimates in cost_usd. Preserve
        // that raw field without relabeling it as a provider-reported amount.
        counts[UsageDimension::CostUsd as usize] = Err("legacy_cost_unverified");
    }
    let mut observations = evidence
        .as_ref()
        .and_then(|e| e.observed.as_ref())
        .map(|raw| normalize(raw, true))
        .unwrap_or([Err("not_reported"); 8]);
    let gaps = evidence
        .as_ref()
        .map(|e| e.gaps.as_slice())
        .unwrap_or_default();
    apply_gaps(&mut counts, &raw, gaps);
    let mut models = Vec::new();
    if let Some(evidence) = &evidence {
        let mut source_models: Vec<_> = evidence.models.iter().collect();
        source_models.sort_by_key(|model| (&model.provider, &model.model));
        for model in source_models {
            let mut complete = normalize(&model.reported, false);
            apply_gaps(&mut complete, &model.reported, &model.gaps);
            let requests: Vec<_> = model
                .requests
                .iter()
                .map(|request| {
                    normalize_request(
                        &request.usage,
                        request.gap.is_some() && !request.tokens_aligned,
                    )
                })
                .collect();
            validate_request_totals(model, &complete, &counts, &requests)?;
            let mut observed = model
                .observed
                .as_ref()
                .map(|v| normalize(v, !model.observed_tokens_aligned))
                .unwrap_or([Err("not_reported"); 8]);
            // Per-call conventions can differ even for one model. Normalize
            // before combining them; an inexpressible raw aggregate is not an
            // information loss when its complete request evidence is retained.
            for (index, dimension) in UsageDimension::ALL.into_iter().enumerate() {
                if requests.is_empty() {
                    continue;
                }
                let parts: Vec<_> = requests
                    .iter()
                    .filter_map(|counts| counts[index].ok())
                    .collect();
                if recoverable_observation(observed[index]) && !parts.is_empty() {
                    observed[index] =
                        combine_values(&parts, dimension).ok_or("token_count_out_of_range");
                }
                if complete[index] == Err("not_reported")
                    && model.requests_exhaustive
                    && model.requests.iter().all(|request| request.gap.is_none())
                    && parts.len() == requests.len()
                {
                    complete[index] =
                        combine_values(&parts, dimension).ok_or("token_count_out_of_range");
                }
            }
            let mut aligned = model.observed_tokens_aligned;
            for (index, dimension) in UsageDimension::ALL.into_iter().enumerate() {
                let tolerance = if dimension == UsageDimension::CostUsd {
                    let parts: Vec<_> = requests
                        .iter()
                        .filter_map(|counts| counts[index].ok())
                        .collect();
                    request_cost_rounding_tolerance(
                        complete[index]
                            .unwrap_or(0)
                            .max(observed[index].unwrap_or(0)),
                        &parts,
                    )
                    .max(1)
                } else {
                    0
                };
                if complete[index] == Err("inconsistent_usage")
                    || complete[index].is_ok_and(|value| {
                        observed[index]
                            .is_ok_and(|observed| observed > value.saturating_add(tolerance))
                    })
                {
                    complete[index] = Err("inconsistent_usage");
                    observed[index] = Err("inconsistent_usage");
                } else if complete[index].is_ok() {
                    // Replacing a subset observation with a complete dimension
                    // can mix call sets, even though each value is valid alone.
                    if dimension != UsageDimension::CostUsd && observed[index] != complete[index] {
                        aligned = false;
                    }
                    observed[index] = complete[index];
                }
            }
            complete[TOTAL] = add(complete[INPUT], complete[OUTPUT]);
            observed[TOTAL] = add(observed[INPUT], observed[OUTPUT]);
            models.push((model, complete, observed, aligned));
        }
        for (index, dimension) in UsageDimension::ALL.into_iter().enumerate() {
            if dimension == UsageDimension::TotalTokens {
                continue;
            }
            let complete_parts: Vec<_> = models.iter().map(|(_, c, _, _)| c[index]).collect();
            let observed_parts: Vec<_> = models
                .iter()
                .filter_map(|(_, _, o, _)| o[index].ok())
                .collect();
            let combined_complete = combine(&complete_parts, dimension);
            let combined_observed = combine_values(&observed_parts, dimension);
            let exhaustive = evidence.models_exhaustive
                && !models.is_empty()
                && complete_parts.iter().all(Result::is_ok);
            let contradiction = counts[index] == Err("inconsistent_usage")
                || complete_parts.contains(&Err("inconsistent_usage"))
                || if let Ok(value) = counts[index] {
                    let tolerance = if dimension == UsageDimension::CostUsd {
                        let request_costs: Vec<_> = evidence
                            .models
                            .iter()
                            .flat_map(|model| &model.requests)
                            .filter_map(|request| {
                                request
                                    .usage
                                    .cost_usd
                                    .and_then(|cost| usd_to_nano_usd(cost).ok())
                            })
                            .collect();
                        let magnitude = value
                            .max(combined_observed.unwrap_or(0))
                            .max(observations[index].unwrap_or(0));
                        request_cost_rounding_tolerance(magnitude, &observed_parts)
                            .saturating_add(request_cost_rounding_tolerance(
                                magnitude,
                                &request_costs,
                            ))
                            .max(models.len() as u64)
                    } else {
                        0
                    };
                    combined_observed.is_some_and(|sum| sum > value.saturating_add(tolerance))
                        || (!observed_parts.is_empty() && combined_observed.is_none())
                        || (exhaustive
                            && combined_complete.is_ok_and(|sum| value.abs_diff(sum) > tolerance))
                        || observations[index]
                            .is_ok_and(|observed| observed > value.saturating_add(tolerance))
                } else {
                    false
                };
            if contradiction {
                counts[index] = Err("inconsistent_usage");
                observations[index] = Err("inconsistent_usage");
                continue;
            }
            if counts[index] == Err("not_reported") && exhaustive {
                counts[index] = combined_complete;
            }
            if let Ok(value) = counts[index] {
                observations[index] = Ok(value);
            } else if recoverable_observation(observations[index]) && !observed_parts.is_empty() {
                observations[index] = combined_observed.ok_or("token_count_out_of_range");
            }
        }
    }
    counts[TOTAL] = add(counts[INPUT], counts[OUTPUT]);
    observations[TOTAL] = add(observations[INPUT], observations[OUTPUT]);
    let values = measurements(counts);
    Ok(ModelUsage {
        input_tokens: values.input_tokens,
        cached_input_tokens: values.cached_input_tokens,
        cache_write_input_tokens: values.cache_write_input_tokens,
        output_tokens: values.output_tokens,
        reasoning_tokens: values.reasoning_tokens,
        total_tokens: values.total_tokens,
        peak_request_input_tokens: values.peak_request_input_tokens,
        token_cost: values.token_cost,
        models: models
            .into_iter()
            .map(|(model, complete, observed, aligned)| PerModelUsage {
                model: model.model.clone(),
                provider: model.provider.clone(),
                usage: measurements(complete),
                accounting: ModelAccounting {
                    observed_tokens_aligned: aligned,
                    observed: observed_usage(observed),
                    gaps: native_gaps(
                        complete,
                        &model.gaps,
                        model.requests.iter().filter_map(|request| request.gap),
                    ),
                },
            })
            .collect(),
        accounting: evidence.map(|evidence| InvocationAccounting {
            models_exhaustive: evidence.models_exhaustive,
            observed: observed_usage(observations),
            gaps: native_gaps(
                counts,
                &evidence.gaps,
                evidence
                    .models
                    .iter()
                    .flat_map(|model| &model.requests)
                    .filter_map(|request| request.gap),
            ),
        }),
    })
}

// An unconvertible aggregate must not erase a qualified model/request subset.
// Contradictions and overflow stay errors; these reasons only lack a usable
// normalized observation. Recovery never certifies a complete invocation.
fn recoverable_observation(value: Count) -> bool {
    matches!(
        value,
        Err("not_reported"
            | "observation_scope_unverified"
            | "cache_components_unreported"
            | "cache_inclusion_unknown"
            | "reasoning_tokens_unreported"
            | "reasoning_inclusion_unknown")
    )
}

fn normalize(raw: &ReportedUsage, observation: bool) -> [Count; 8] {
    let mut input = raw.input_tokens.ok_or("not_reported").and_then(|input| {
        match raw.input_tokens_include_cache {
            Some(true) => Ok(input),
            Some(false) if !observation => add(
                add(
                    Ok(input),
                    raw.cached_input_tokens.ok_or("cache_components_unreported"),
                ),
                raw.cache_write_input_tokens
                    .ok_or("cache_components_unreported"),
            ),
            Some(false) => Err("observation_scope_unverified"),
            None if raw.cached_input_tokens == Some(0)
                && raw.cache_write_input_tokens == Some(0) =>
            {
                Ok(input)
            }
            None => Err("cache_inclusion_unknown"),
        }
    });
    let mut output = raw.output_tokens.ok_or("not_reported").and_then(|output| {
        match raw.output_tokens_include_reasoning {
            Some(true) => Ok(output),
            Some(false) if !observation => add(
                Ok(output),
                raw.reasoning_tokens.ok_or("reasoning_tokens_unreported"),
            ),
            Some(false) => Err("observation_scope_unverified"),
            None if raw.reasoning_tokens == Some(0) => Ok(output),
            None => Err("reasoning_inclusion_unknown"),
        }
    });
    // Complete or explicitly aligned dimensions share scope. Other observations may cover
    // different subsets, so these component comparisons cannot validate them.
    if !observation {
        if let Ok(value) = input {
            let cache = raw
                .cached_input_tokens
                .unwrap_or(0)
                .checked_add(raw.cache_write_input_tokens.unwrap_or(0));
            if cache.is_none_or(|cache| cache > value)
                || raw
                    .peak_request_input_tokens
                    .is_some_and(|peak| peak > value)
            {
                input = Err("inconsistent_usage");
            }
        }
        if let Ok(value) = output
            && raw
                .reasoning_tokens
                .is_some_and(|reasoning| reasoning > value)
        {
            output = Err("inconsistent_usage");
        }
    }
    [
        input,
        raw.cached_input_tokens.ok_or("not_reported"),
        raw.cache_write_input_tokens.ok_or("not_reported"),
        output,
        raw.reasoning_tokens.ok_or("not_reported"),
        add(input, output),
        raw.peak_request_input_tokens.ok_or("not_reported"),
        raw.cost_usd.ok_or("not_reported").and_then(usd_to_nano_usd),
    ]
}

fn normalize_request(raw: &ReportedUsage, observation: bool) -> [Count; 8] {
    let mut counts = normalize(raw, observation);
    if raw.peak_request_input_tokens.is_none() {
        counts[UsageDimension::PeakRequestInputTokens as usize] = counts[INPUT];
    }
    counts
}

fn add(left: Count, right: Count) -> Count {
    left.and_then(|left| {
        right.and_then(|right| left.checked_add(right).ok_or("token_count_out_of_range"))
    })
}

fn apply_gaps(counts: &mut [Count; 8], raw: &ReportedUsage, gaps: &[UsageGap]) {
    for gap in gaps {
        for dimension in &gap.dimensions {
            let index = UsageDimension::ALL
                .iter()
                .position(|d| d == dimension)
                .expect("known dimension");
            if *dimension != UsageDimension::TotalTokens {
                counts[index] = Err(gap_reason(gap.reason));
            }
            if matches!(
                dimension,
                UsageDimension::CachedInputTokens | UsageDimension::CacheWriteInputTokens
            ) && raw.input_tokens_include_cache == Some(false)
            {
                counts[INPUT] = Err(gap_reason(gap.reason));
            }
            if *dimension == UsageDimension::ReasoningTokens
                && raw.output_tokens_include_reasoning == Some(false)
            {
                counts[OUTPUT] = Err(gap_reason(gap.reason));
            }
        }
    }
    counts[TOTAL] = add(counts[INPUT], counts[OUTPUT]);
}

fn gap_reason(reason: GapReason) -> &'static str {
    match reason {
        GapReason::UnfinishedWork => "unfinished_work",
        GapReason::MissingUsage => "missing_usage",
        GapReason::ScopeUnverified => "scope_unverified",
    }
}

fn combine(parts: &[Count], dimension: UsageDimension) -> Count {
    let values = parts.iter().copied().collect::<Result<Vec<_>, _>>()?;
    combine_values(&values, dimension).ok_or("token_count_out_of_range")
}

fn combine_values(values: &[u64], dimension: UsageDimension) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    if dimension == UsageDimension::PeakRequestInputTokens {
        values.iter().copied().max()
    } else {
        values
            .iter()
            .try_fold(0_u64, |sum, value| sum.checked_add(*value))
    }
}

fn measured(count: Count, unit: MeasurementUnit) -> Measurement {
    Measurement {
        availability: if count.is_ok() {
            Availability::Measured
        } else {
            Availability::Unavailable
        },
        value: count.ok(),
        unit,
        source: MeasurementSource::AdapterUsage,
        attribution: if count.is_ok() {
            Attribution::Invocation
        } else {
            Attribution::None
        },
        reason: count.err().map(str::to_owned),
    }
}

fn measurements(counts: [Count; 8]) -> ModelMeasurements {
    let tokens = |index| measured(counts[index], MeasurementUnit::Tokens);
    ModelMeasurements {
        input_tokens: tokens(0),
        cached_input_tokens: tokens(1),
        cache_write_input_tokens: tokens(2),
        output_tokens: tokens(3),
        reasoning_tokens: tokens(4),
        total_tokens: tokens(5),
        peak_request_input_tokens: tokens(6),
        token_cost: TokenCost {
            applied_calls: Vec::new(),
            reported: measured(counts[7], MeasurementUnit::NanoUsd),
            pricing: None,
            applied: None,
            estimated: Measurement {
                availability: Availability::Unavailable,
                value: None,
                unit: MeasurementUnit::NanoUsd,
                source: MeasurementSource::FabricPriceCatalog,
                attribution: Attribution::None,
                reason: Some("catalog_pricing_unavailable".into()),
            },
        },
    }
}

fn observed_usage(counts: [Count; 8]) -> ObservedUsage {
    ObservedUsage {
        input_tokens: counts[0].ok(),
        cached_input_tokens: counts[1].ok(),
        cache_write_input_tokens: counts[2].ok(),
        output_tokens: counts[3].ok(),
        reasoning_tokens: counts[4].ok(),
        total_tokens: counts[5].ok(),
        peak_request_input_tokens: counts[6].ok(),
        reported_cost_nano_usd: counts[7].ok(),
        estimated_cost_nano_usd: None,
    }
}

fn native_gaps(
    counts: [Count; 8],
    raw: &[UsageGap],
    request_reasons: impl Iterator<Item = GapReason>,
) -> Vec<AccountingGap> {
    let request_reasons: Vec<_> = request_reasons.collect();
    let dimensions = [
        AccountingDimension::InputTokens,
        AccountingDimension::CachedInputTokens,
        AccountingDimension::CacheWriteInputTokens,
        AccountingDimension::OutputTokens,
        AccountingDimension::ReasoningTokens,
        AccountingDimension::TotalTokens,
        AccountingDimension::PeakRequestInputTokens,
        AccountingDimension::ReportedCost,
    ];
    let mut result = Vec::new();
    for (index, dimension) in dimensions.into_iter().enumerate() {
        if let Err(reason) = counts[index] {
            let category = match reason {
                "unfinished_work" => GapReason::UnfinishedWork,
                "not_reported"
                | "missing_usage"
                | "cache_components_unreported"
                | "reasoning_tokens_unreported" => GapReason::MissingUsage,
                _ => GapReason::ScopeUnverified,
            };
            let mut add_reason = |reason| {
                let gap = AccountingGap {
                    dimensions: vec![dimension],
                    reason,
                };
                if !result.contains(&gap) {
                    result.push(gap);
                }
            };
            // A request-wide gap explains absent complete values at both scopes.
            // Preserve independent normalization or explicitly scoped reasons too.
            if reason != "not_reported" || request_reasons.is_empty() {
                add_reason(category);
            }
            for reason in &request_reasons {
                add_reason(*reason);
            }
            // Derived total may be affected by several independently actionable gaps.
            if dimension == AccountingDimension::TotalTokens {
                for gap in raw {
                    if gap.dimensions.iter().any(|d| {
                        matches!(
                            d,
                            UsageDimension::InputTokens | UsageDimension::OutputTokens
                        )
                    }) && gap.reason != category
                    {
                        add_reason(gap.reason);
                    }
                }
            }
        }
    }
    result.push(AccountingGap {
        dimensions: vec![AccountingDimension::EstimatedCost],
        reason: GapReason::MissingUsage,
    });
    result
}

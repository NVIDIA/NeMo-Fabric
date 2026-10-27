// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validated accounting evidence carried through the existing usage extension.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::AgentUsage;

/// Descriptor-declared transport key. This does not change the closed AgentUsage wire shape.
pub const USAGE_ACCOUNTING_EXTENSION: &str = "nemo_fabric.usage_accounting.v1";

/// Raw counts in provider/harness conventions. Missing values are never zero.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReportedUsage {
    /// Raw input count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub input_tokens: Option<u64>,
    /// Cache reads, independently reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub cached_input_tokens: Option<u64>,
    /// Cache writes, independently reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub cache_write_input_tokens: Option<u64>,
    /// Whether input includes both cache categories.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens_include_cache: Option<bool>,
    /// Raw output count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub output_tokens: Option<u64>,
    /// Reasoning count, independently reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub reasoning_tokens: Option<u64>,
    /// Whether output includes reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_include_reasoning: Option<bool>,
    /// Provider-reported total, independent of normalized input plus output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub total_tokens: Option<u64>,
    /// Largest request's cache-inclusive input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub peak_request_input_tokens: Option<u64>,
    /// Provider-reported USD, never a client-side price estimate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0))]
    pub cost_usd: Option<f64>,
}

/// A dimension in raw adapter evidence.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
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
    /// Reasoning tokens.
    ReasoningTokens,
    /// Provider-reported total.
    TotalTokens,
    /// Largest request input.
    PeakRequestInputTokens,
    /// Provider-reported USD.
    CostUsd,
}

/// Categories that affect whether a consumer can treat a value as complete.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum GapReason {
    /// Owned work was still running at the return boundary.
    UnfinishedWork,
    /// A completed call did not provide usable usage.
    MissingUsage,
    /// The supported source does not establish coverage of owned work.
    ScopeUnverified,
}

/// Missing evidence for one or more raw dimensions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UsageGap {
    /// Nonempty, unique set of affected dimensions.
    #[schemars(schema_with = "gap_dimensions_schema")]
    pub dimensions: Vec<UsageDimension>,
    /// Why those dimensions are incomplete.
    pub reason: GapReason,
}

fn gap_dimensions_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
    let mut schema = Vec::<UsageDimension>::json_schema(generator);
    schema.insert("minItems".into(), 1.into());
    schema.insert("uniqueItems".into(), true.into());
    schema
}

/// One model call retained for request-tier pricing and evidence reconciliation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UsageRequest {
    /// One deduplicated model-call snapshot, never a session/model aggregate.
    pub usage: ReportedUsage,
    /// Known incompleteness of this call's source coverage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gap: Option<GapReason>,
    /// Present observed fields describe the same request/subset.
    #[serde(default)]
    pub tokens_aligned: bool,
}

/// Evidence for one literal provider/model identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AccountedModel {
    /// Optional per-call evidence for exact request-tier pricing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requests: Vec<UsageRequest>,
    /// The request list covers every source unit contributing to this model.
    #[serde(default)]
    pub requests_exhaustive: bool,
    /// Model identity as reported by the source.
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub model: String,
    /// Literal provider identity, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1), regex(pattern = r"\S"))]
    pub provider: Option<String>,
    /// Complete per-model evidence; absent fields are unreported.
    #[serde(default)]
    pub reported: ReportedUsage,
    /// Valid deduplicated observations, potentially incomplete.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<ReportedUsage>,
    /// Every present observed token dimension describes the same source-call subset.
    /// False unless the producer establishes this, including peak and normalization components.
    #[serde(default)]
    pub observed_tokens_aligned: bool,
    /// Gaps affecting this model.
    pub gaps: Vec<UsageGap>,
}

/// The declared adapter transport for invocation accounting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UsageAccounting {
    /// Every model identity used by owned work is represented; this alone does
    /// not certify that any dimension is complete.
    pub models_exhaustive: bool,
    /// Valid deduplicated observations, not a promised mathematical lower bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<ReportedUsage>,
    /// Invocation-wide gaps.
    pub gaps: Vec<UsageGap>,
    /// One entry per literal provider/model pair.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<AccountedModel>,
}

/// Invalid accounting evidence uses the existing invalid-result path.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("invalid usage accounting: {0}")]
pub struct AccountingError(pub String);

impl UsageDimension {
    pub(crate) const ALL: [Self; 8] = [
        Self::InputTokens,
        Self::CachedInputTokens,
        Self::CacheWriteInputTokens,
        Self::OutputTokens,
        Self::ReasoningTokens,
        Self::TotalTokens,
        Self::PeakRequestInputTokens,
        Self::CostUsd,
    ];

    pub(crate) fn present(self, usage: &ReportedUsage) -> bool {
        match self {
            Self::InputTokens => usage.input_tokens.is_some(),
            Self::CachedInputTokens => usage.cached_input_tokens.is_some(),
            Self::CacheWriteInputTokens => usage.cache_write_input_tokens.is_some(),
            Self::OutputTokens => usage.output_tokens.is_some(),
            Self::ReasoningTokens => usage.reasoning_tokens.is_some(),
            Self::TotalTokens => usage.total_tokens.is_some(),
            Self::PeakRequestInputTokens => usage.peak_request_input_tokens.is_some(),
            Self::CostUsd => usage.cost_usd.is_some(),
        }
    }
}

impl UsageAccounting {
    /// Decode and validate the typed extension, if present. Descriptor schema
    /// validation remains required at the runtime boundary as well.
    pub fn from_usage(usage: &AgentUsage) -> Result<Option<Self>, AccountingError> {
        let Some(value) = usage.extensions.get(USAGE_ACCOUNTING_EXTENSION) else {
            return Ok(None);
        };
        let evidence: Self = serde_json::from_value(value.clone())
            .map_err(|error| AccountingError(error.to_string()))?;
        let reported = reported_usage(usage)?;
        validate_record(&reported, evidence.observed.as_ref(), &evidence.gaps)?;
        if evidence.models_exhaustive
            && evidence.models.is_empty()
            && std::iter::once(&reported)
                .chain(evidence.observed.as_ref())
                .any(|counts| {
                    [
                        counts.input_tokens,
                        counts.cached_input_tokens,
                        counts.cache_write_input_tokens,
                        counts.output_tokens,
                        counts.reasoning_tokens,
                        counts.total_tokens,
                        counts.peak_request_input_tokens,
                    ]
                    .into_iter()
                    .flatten()
                    .any(|count| count > 0)
                })
        {
            return Err(AccountingError(
                "exhaustive empty model list conflicts with positive tokens".into(),
            ));
        }
        let mut identities = BTreeSet::new();
        for model in &evidence.models {
            if model.model.trim().is_empty()
                || model
                    .provider
                    .as_ref()
                    .is_some_and(|provider| provider.trim().is_empty())
                || !identities.insert((model.provider.as_deref(), model.model.as_str()))
            {
                return Err(AccountingError("blank or duplicate model identity".into()));
            }
            validate_record(&model.reported, model.observed.as_ref(), &model.gaps)?;
            if model.requests_exhaustive && model.requests.is_empty() {
                return Err(AccountingError(
                    "exhaustive requests must be nonempty".into(),
                ));
            }
            for request in &model.requests {
                validate_record(&ReportedUsage::default(), Some(&request.usage), &[])?;
            }
        }
        Ok(Some(evidence))
    }
}

fn validate_record(
    reported: &ReportedUsage,
    observed: Option<&ReportedUsage>,
    gaps: &[UsageGap],
) -> Result<(), AccountingError> {
    for counts in std::iter::once(reported).chain(observed) {
        if counts
            .cost_usd
            .is_some_and(|cost| usd_to_nano_usd(cost).is_err())
        {
            return Err(AccountingError(
                "cost must fit finite nonnegative nano-USD".into(),
            ));
        }
    }
    for gap in gaps {
        let unique: BTreeSet<_> = gap.dimensions.iter().collect();
        if unique.is_empty() || unique.len() != gap.dimensions.len() {
            return Err(AccountingError(
                "gap dimensions must be nonempty and unique".into(),
            ));
        }
        if gap
            .dimensions
            .iter()
            .any(|dimension| dimension.present(reported))
        {
            return Err(AccountingError(
                "complete value conflicts with explicit gap".into(),
            ));
        }
    }
    Ok(())
}

/// Recover the complete raw record without changing legacy AgentUsage. A
/// detailed extension selects its whole input/output dimension, never a mix.
pub(crate) fn reported_usage(usage: &AgentUsage) -> Result<ReportedUsage, AccountingError> {
    let mut result = ReportedUsage {
        input_tokens: usage.input_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        input_tokens_include_cache: usage.input_tokens_include_cache,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
        cost_usd: usage.cost_usd,
        ..Default::default()
    };
    if let Some(detail) = usage.extensions.get("nemo_fabric.detailed_token_usage.v1") {
        // Existing descriptors may declare extra detailed fields (for example
        // legacy model breakdowns). Preserve those as raw evidence rather than
        // rejecting a previously valid result while consuming known counters.
        let detail = detail
            .as_object()
            .ok_or_else(|| AccountingError("detailed usage must be an object".into()))?;
        let known = [
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "input_tokens_include_cache",
            "output_tokens",
            "reasoning_tokens",
            "output_tokens_include_reasoning",
            "total_tokens",
            "peak_request_input_tokens",
            "cost_usd",
        ];
        let detail: ReportedUsage = serde_json::from_value(serde_json::Value::Object(
            detail
                .iter()
                .filter(|(key, _)| known.contains(&key.as_str()))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        ))
        .map_err(|error| AccountingError(error.to_string()))?;
        result.cost_usd = result.cost_usd.or(detail.cost_usd);
        if detail.input_tokens.is_some()
            || detail.cached_input_tokens.is_some()
            || detail.cache_write_input_tokens.is_some()
            || detail.input_tokens_include_cache.is_some()
        {
            result.input_tokens = detail.input_tokens;
            result.cached_input_tokens = detail.cached_input_tokens;
            result.cache_write_input_tokens = detail.cache_write_input_tokens;
            result.input_tokens_include_cache = detail.input_tokens_include_cache;
        }
        if detail.output_tokens.is_some()
            || detail.reasoning_tokens.is_some()
            || detail.output_tokens_include_reasoning.is_some()
        {
            result.output_tokens = detail.output_tokens;
            result.reasoning_tokens = detail.reasoning_tokens;
            result.output_tokens_include_reasoning = detail.output_tokens_include_reasoning;
        }
        result.peak_request_input_tokens = detail.peak_request_input_tokens;
    }
    Ok(result)
}

pub(crate) fn usd_to_nano_usd(usd: f64) -> Result<u64, &'static str> {
    let nanos = (usd * 1e9).round();
    if usd.is_finite() && usd >= 0.0 && nanos < 18_446_744_073_709_551_616.0 {
        Ok(nanos as u64)
    } else {
        Err("cost_out_of_range")
    }
}

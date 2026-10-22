// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Persisted pricing-basis types for the stable model-usage result contract.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Fields that must agree before a report combines catalog estimates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PricingBasis {
    /// `catalog`; provider-reported cost has a different basis.
    pub basis: String,
    /// `USD` in this implementation.
    pub currency: String,
    /// Version of normalization and monetary rounding.
    pub normalization_version: String,
    /// No missing count is assumed zero.
    pub assumption_policy: String,
    /// Exact bundled catalog bytes.
    pub bundled_catalog_sha256: String,
    /// Exact override bytes, when supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_catalog_sha256: Option<String>,
}

/// Integer rates selected for a model, in nano-USD per million tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RateSnapshot {
    /// Uncached input rate.
    #[schemars(range(max = u64::MAX))]
    pub input: u64,
    /// Reasoning-inclusive output rate.
    #[schemars(range(max = u64::MAX))]
    pub output: u64,
    /// Cache read rate; absent is not zero.
    #[schemars(range(max = u64::MAX))]
    pub cache_read: Option<u64>,
    /// Cache write rate; absent is not zero.
    #[schemars(range(max = u64::MAX))]
    pub cache_write: Option<u64>,
}

/// Persisted entry and selected rates, so saved prices remain reproducible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AppliedPricing {
    /// Exact catalog containing this entry.
    pub catalog_sha256: String,
    /// Entry's provider, without rewriting the observed identity.
    pub provider: String,
    /// Entry's model identity, which may match an explicit alias.
    pub model_id: String,
    /// Date recorded by the catalog author.
    pub pricing_as_of: String,
    /// Source recorded by the catalog author.
    pub pricing_source: String,
    /// One selected rate set; distinct selections are retained in `model_usage::TokenCost.applied_rates`.
    pub rates: RateSnapshot,
}

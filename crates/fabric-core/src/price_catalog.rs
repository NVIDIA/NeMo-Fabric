// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model price catalogs for estimating token cost.
//!
//! A catalog uses the NeMo Relay model pricing catalog format, version 1, so
//! one file can configure both tools. NVIDIA NeMo Fabric parses the format itself and
//! does not depend on NeMo Relay. Names are trimmed and compared case-insensitively.
//! An explicit provider restricts aliases and model-path suffixes to that provider.
//! Without a provider, lookup tries provider-qualified keys from the model path
//! before model-only suffixes. For each candidate key, the first matching
//! catalog entry wins.
//!
//! NeMo Fabric ships one catalog, embedded at build time, with the list prices
//! of models that its presets, docs, and examples use. Each entry cites the
//! provider pricing page it was taken from and the date it was checked. A
//! caller can add a catalog of their own, whose entries take precedence.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::LazyLock;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The only catalog format version NeMo Fabric reads.
const PRICING_CATALOG_VERSION: u32 = 1;

/// Rates must be below one million US dollars per million tokens. NeMo Relay
/// sets no upper bound; this one keeps every rate exact in nano-USD per
/// million tokens and every cost sum within 128 bits.
const MAX_RATE_PER_MILLION: f64 = 1_000_000.0;

/// Nano-USD per US dollar.
const NANO_USD_PER_USD: f64 = 1e9;

/// The catalog NeMo Fabric ships.
const BUNDLED_CATALOG_JSON: &str = include_str!("../data/model-pricing.json");

static BUNDLED_CATALOG: LazyLock<BundledCatalog> = LazyLock::new(|| BundledCatalog {
    // A unit test checks that the shipped file is valid, so `None` only
    // guards against a build that skipped the tests.
    catalog: PricingCatalog::from_json(BUNDLED_CATALOG_JSON.as_bytes()).ok(),
    sha256: sha256_fingerprint(BUNDLED_CATALOG_JSON.as_bytes()),
});

struct BundledCatalog {
    catalog: Option<PricingCatalog>,
    sha256: String,
}

/// A model price catalog in the NeMo Relay model pricing catalog format,
/// version 1.
///
/// NVIDIA NeMo Fabric reads and writes the format unchanged, with one extra check:
/// every rate must be below one million US dollars per million tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PricingCatalog {
    /// Catalog format version. Must be `1`.
    pub version: u32,
    /// Price entries. For each lookup key, the first matching entry wins.
    pub entries: Vec<ModelPricing>,
}

/// Prices of one model, in `currency` per million tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelPricing {
    /// Provider that sets the price, such as `anthropic` or `openai`.
    pub provider: String,
    /// Canonical model identifier.
    pub model_id: String,
    /// Other identifiers of the same model, such as dated snapshots.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// ISO 4217 currency of the rates. NVIDIA NeMo Fabric estimates cost only from
    /// `USD` entries.
    #[serde(default = "default_currency")]
    pub currency: String,
    /// What a rate is charged per. NeMo Fabric estimates cost only from
    /// `per_token` entries.
    #[serde(default)]
    pub unit: PricingUnit,
    /// Flat rates. Ignored when `rate_schedule` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rates: Option<TokenPricingRates>,
    /// Rates that depend on the number of prompt tokens in a request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_schedule: Option<TokenRateSchedule>,
    /// How the provider counts cached prompt tokens. NeMo Relay uses it to
    /// split provider usage; NeMo Fabric prices usage that is already
    /// normalized, so it does not read it.
    pub prompt_cache: PromptCachePricing,
    /// Date the price was taken from `pricing_source`, as `YYYY-MM-DD` in
    /// the bundled catalog.
    pub pricing_as_of: String,
    /// Where the price was taken from, such as the provider's pricing page.
    pub pricing_source: String,
}

/// What a model price is charged per.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PricingUnit {
    /// Per token.
    #[default]
    PerToken,
    /// Per request.
    PerRequest,
    /// Per second.
    PerSecond,
    /// Per GPU hour, for self-hosted models.
    GpuHour,
}

/// Token rates in currency units per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenPricingRates {
    /// Rate for input tokens that were neither read from nor written to the
    /// prompt cache.
    pub input_per_million: f64,
    /// Rate for output tokens, including reasoning tokens.
    pub output_per_million: f64,
    /// Rate for input tokens read from the prompt cache. Without it, cached
    /// reads cannot be priced; set `0.0` when they are free.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_per_million: Option<f64>,
    /// Rate for input tokens written to the prompt cache. Without it, cache
    /// writes cannot be priced; set `0.0` when they are free.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_per_million: Option<f64>,
}

/// Rates that depend on the size of a request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, tag = "type", rename_all = "snake_case")]
pub enum TokenRateSchedule {
    /// One tier's rates apply to a whole request, chosen by the request's
    /// prompt tokens.
    PromptTokenThreshold {
        /// How the chosen tier's rates apply.
        #[serde(default)]
        applies_to: RateScheduleApplication,
        /// Tiers in the order they are tried. The first tier whose bounds
        /// contain a request's prompt tokens applies to it.
        tiers: Vec<TokenRateTier>,
    },
}

/// How a chosen rate tier applies to a request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RateScheduleApplication {
    /// The tier's rates apply to every token of the request.
    #[default]
    FullRequest,
}

/// Rates for requests whose prompt tokens fall within inclusive bounds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TokenRateTier {
    /// Smallest number of prompt tokens in the tier, or no lower bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_prompt_tokens: Option<u64>,
    /// Largest number of prompt tokens in the tier, or no upper bound.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_prompt_tokens: Option<u64>,
    /// Rates of the tier.
    pub rates: TokenPricingRates,
}

/// How a provider counts cached prompt tokens in its usage reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PromptCachePricing {
    /// Whether the provider's prompt token count includes cached reads.
    pub read_accounting: CacheReadAccounting,
}

/// Whether a provider's prompt token count includes cached reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CacheReadAccounting {
    /// Prompt tokens include cached reads.
    IncludedInPromptTokens,
    /// Prompt tokens exclude cached reads.
    Separate,
}

/// A price catalog the caller supplied, read and validated when the run plan
/// is resolved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct UserPriceCatalog {
    /// Absolute path the catalog was read from.
    pub path: PathBuf,
    /// `sha256:` followed by the lowercase hexadecimal SHA-256 of the file.
    pub sha256: String,
    /// The validated catalog.
    pub catalog: PricingCatalog,
}

/// Why a catalog is invalid. The messages match NeMo Relay's.
#[derive(Debug, thiserror::Error)]
pub enum PricingCatalogError {
    /// The catalog is not JSON of the catalog format.
    #[error("invalid model pricing catalog JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The catalog format version is not 1.
    #[error("unsupported model pricing catalog version {version}")]
    UnsupportedVersion {
        /// Version in the catalog.
        version: u32,
    },
    /// Two entries or aliases of one provider name the same model.
    #[error("duplicate model pricing alias '{model}'")]
    DuplicateModelAlias {
        /// Normalized `provider/model` key.
        model: String,
    },
    /// A required text field is blank.
    #[error("model pricing entry {entry_index} has empty {field}")]
    EmptyField {
        /// Zero-based entry index.
        entry_index: usize,
        /// Field name.
        field: String,
    },
    /// A rate or a tier bound is out of range.
    #[error("model pricing entry {entry_index} has invalid {field}: {value}")]
    InvalidRate {
        /// Zero-based entry index.
        entry_index: usize,
        /// Field name.
        field: String,
        /// Invalid value.
        value: f64,
    },
}

impl UserPriceCatalog {
    /// Validate bytes and retain their exact catalog fingerprint.
    pub fn from_bytes(path: PathBuf, bytes: &[u8]) -> Result<Self, PricingCatalogError> {
        Ok(Self {
            path,
            sha256: sha256_fingerprint(bytes),
            catalog: PricingCatalog::from_json(bytes)?,
        })
    }
}

impl PricingCatalog {
    /// Parse and validate a catalog.
    pub fn from_json(json: &[u8]) -> Result<Self, PricingCatalogError> {
        let catalog: Self = serde_json::from_slice(json)?;
        catalog.validate()?;
        Ok(catalog)
    }

    /// Find `model`, keeping an explicit provider on every alias or suffix.
    /// Without a provider, try the model path's provider prefix before model-only
    /// suffixes. The first matching entry in listed order wins for each key.
    pub(crate) fn pricing_for(&self, provider: Option<&str>, model: &str) -> Option<&ModelPricing> {
        lookup_keys(provider, model)
            .iter()
            .find_map(|key| self.entries.iter().find(|entry| entry.matches(key)))
    }

    fn validate(&self) -> Result<(), PricingCatalogError> {
        if self.version != PRICING_CATALOG_VERSION {
            return Err(PricingCatalogError::UnsupportedVersion {
                version: self.version,
            });
        }
        let mut seen = HashSet::new();
        for (entry_index, entry) in self.entries.iter().enumerate() {
            entry.validate(entry_index)?;
            let provider = normalized_provider(&entry.provider);
            for model in entry.model_keys() {
                let key = format!("{provider}/{model}");
                if !seen.insert(key.clone()) {
                    return Err(PricingCatalogError::DuplicateModelAlias { model: key });
                }
            }
        }
        Ok(())
    }
}

impl ModelPricing {
    pub(crate) fn rates_at(&self, prompt_tokens: u64) -> Option<TokenPricingRates> {
        match &self.rate_schedule {
            None => self.rates,
            Some(TokenRateSchedule::PromptTokenThreshold { tiers, .. }) => tiers
                .iter()
                .find(|tier| {
                    tier.min_prompt_tokens.unwrap_or(0) <= prompt_tokens
                        && tier
                            .max_prompt_tokens
                            .is_none_or(|max| prompt_tokens <= max)
                })
                .map(|tier| tier.rates),
        }
    }
    /// Rates that apply to every request of a model whose largest request
    /// had `peak_prompt_tokens` prompt tokens. `None` when the entry has a
    /// rate schedule and the peak is unknown or no single tier covers every
    /// request up to it.
    pub(crate) fn rates_up_to(&self, peak_prompt_tokens: Option<u64>) -> Option<TokenPricingRates> {
        match &self.rate_schedule {
            Some(schedule) => schedule.rates_up_to(peak_prompt_tokens?),
            None => self.rates,
        }
    }

    fn matches(&self, key: &LookupKey) -> bool {
        key.provider
            .as_deref()
            .is_none_or(|provider| normalized_provider(&self.provider) == provider)
            && self.model_keys().any(|model| model == key.model)
    }

    /// The normalized model identifier and aliases, without blank ones.
    fn model_keys(&self) -> impl Iterator<Item = String> + '_ {
        std::iter::once(&self.model_id)
            .chain(&self.aliases)
            .map(|model| normalized_model(model))
            .filter(|model| !model.is_empty())
    }

    fn validate(&self, entry_index: usize) -> Result<(), PricingCatalogError> {
        for (field, value) in [
            ("provider", &self.provider),
            ("model_id", &self.model_id),
            ("currency", &self.currency),
            ("pricing_as_of", &self.pricing_as_of),
            ("pricing_source", &self.pricing_source),
        ] {
            if value.trim().is_empty() {
                return Err(empty_field(entry_index, field));
            }
        }
        if self.unit == PricingUnit::PerToken
            && self.rates.is_none()
            && self.rate_schedule.is_none()
        {
            return Err(empty_field(entry_index, "rates or rate_schedule"));
        }
        if let Some(rates) = &self.rates {
            rates.validate(entry_index, "rates")?;
        }
        if let Some(schedule) = &self.rate_schedule {
            schedule.validate(entry_index)?;
        }
        Ok(())
    }
}

impl TokenPricingRates {
    fn validate(&self, entry_index: usize, prefix: &str) -> Result<(), PricingCatalogError> {
        for (field, value) in [
            ("input_per_million", Some(self.input_per_million)),
            ("output_per_million", Some(self.output_per_million)),
            ("cache_read_per_million", self.cache_read_per_million),
            ("cache_write_per_million", self.cache_write_per_million),
        ] {
            if let Some(value) = value
                && nano_usd_per_million_tokens(value).is_none()
            {
                return Err(PricingCatalogError::InvalidRate {
                    entry_index,
                    field: format!("{prefix}.{field}"),
                    value,
                });
            }
        }
        Ok(())
    }
}

impl TokenRateSchedule {
    fn rates_up_to(&self, peak_prompt_tokens: u64) -> Option<TokenPricingRates> {
        match self {
            // NeMo Relay prices each request with the first tier whose bounds
            // contain its prompt tokens. Every request had between zero and
            // the peak prompt tokens, so one tier's rates apply to all of them
            // exactly when the first tier that overlaps that range contains
            // all of it.
            Self::PromptTokenThreshold { tiers, .. } => {
                let tier = tiers
                    .iter()
                    .find(|tier| tier.overlaps_up_to(peak_prompt_tokens))?;
                tier.contains_up_to(peak_prompt_tokens)
                    .then_some(tier.rates)
            }
        }
    }

    fn validate(&self, entry_index: usize) -> Result<(), PricingCatalogError> {
        let Self::PromptTokenThreshold { tiers, .. } = self;
        if tiers.is_empty() {
            return Err(empty_field(entry_index, "rate_schedule.tiers"));
        }
        for (tier_index, tier) in tiers.iter().enumerate() {
            if let (Some(min), Some(max)) = (tier.min_prompt_tokens, tier.max_prompt_tokens)
                && min > max
            {
                return Err(PricingCatalogError::InvalidRate {
                    entry_index,
                    field: "rate_schedule.tiers.prompt_tokens".to_string(),
                    // NeMo Relay reports the lower bound in the same way.
                    value: min as f64,
                });
            }
            tier.rates.validate(
                entry_index,
                &format!("rate_schedule.tiers[{tier_index}].rates"),
            )?;
        }
        Ok(())
    }
}

impl TokenRateTier {
    /// Whether the tier contains any prompt token count from zero to `peak`.
    fn overlaps_up_to(&self, peak: u64) -> bool {
        self.min_prompt_tokens.unwrap_or(0) <= peak
    }

    /// Whether the tier contains every prompt token count from zero to `peak`.
    fn contains_up_to(&self, peak: u64) -> bool {
        self.min_prompt_tokens.unwrap_or(0) == 0
            && self.max_prompt_tokens.is_none_or(|max| peak <= max)
    }
}

fn empty_field(entry_index: usize, field: &str) -> PricingCatalogError {
    PricingCatalogError::EmptyField {
        entry_index,
        field: field.to_string(),
    }
}

fn normalized_model(model: &str) -> String {
    model.trim().to_ascii_lowercase()
}

fn normalized_provider(provider: &str) -> String {
    provider.trim().trim_matches('/').to_ascii_lowercase()
}

/// A lookup key: a normalized model, optionally qualified by a normalized
/// provider.
#[derive(Debug, PartialEq, Eq)]
struct LookupKey {
    provider: Option<String>,
    model: String,
}

/// Prefer exact model keys, preserving an explicitly established provider.
fn lookup_keys(provider: Option<&str>, model: &str) -> Vec<LookupKey> {
    let model = normalized_model(model);
    if model.is_empty() {
        return Vec::new();
    }
    let parts: Vec<&str> = model
        .split('/')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    let last = parts.last().copied().unwrap_or(model.as_str());
    let mut keys = Vec::new();
    let mut push = |provider: Option<String>, model: String| {
        let key = LookupKey { provider, model };
        if !key.model.is_empty() && !keys.contains(&key) {
            keys.push(key);
        }
    };
    let explicit_provider = provider
        .map(normalized_provider)
        .filter(|provider| !provider.is_empty());
    match &explicit_provider {
        Some(provider) => {
            push(Some(provider.clone()), model.clone());
            push(Some(provider.clone()), last.to_string());
        }
        None if parts.len() > 1 => {
            push(Some(parts[..parts.len() - 1].join("/")), last.to_string());
        }
        None => {}
    }
    for start in 0..parts.len() {
        push(explicit_provider.clone(), parts[start..].join("/"));
    }
    keys
}

/// Where a price came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PriceOrigin {
    /// The catalog NeMo Fabric ships.
    Bundled,
    /// The catalog the caller supplied.
    User,
}

/// The price catalogs an estimate consults, the caller's first.
pub(crate) struct PriceCatalogs<'a> {
    bundled: Option<&'a PricingCatalog>,
    bundled_sha256: &'a str,
    user: Option<&'a UserPriceCatalog>,
}

impl<'a> PriceCatalogs<'a> {
    /// The shipped catalog, preceded by the caller's catalog when given.
    pub(crate) fn with_user_catalog(user: Option<&'a UserPriceCatalog>) -> Self {
        Self::from_parts(
            BUNDLED_CATALOG.catalog.as_ref(),
            &BUNDLED_CATALOG.sha256,
            user,
        )
    }

    pub(crate) fn from_parts(
        bundled: Option<&'a PricingCatalog>,
        bundled_sha256: &'a str,
        user: Option<&'a UserPriceCatalog>,
    ) -> Self {
        Self {
            bundled,
            bundled_sha256,
            user,
        }
    }

    /// Fingerprint of the shipped catalog.
    pub(crate) fn bundled_sha256(&self) -> &'a str {
        self.bundled_sha256
    }

    /// Fingerprint of the caller's catalog, when one is configured.
    pub(crate) fn user_sha256(&self) -> Option<&'a str> {
        self.user.map(|user| user.sha256.as_str())
    }

    /// Whether the shipped catalog parsed.
    pub(crate) fn bundled_is_valid(&self) -> bool {
        self.bundled.is_some()
    }

    /// The entry for a model, from the caller's catalog first. With
    /// `user_only`, the shipped catalog is not consulted.
    pub(crate) fn find(
        &self,
        provider: Option<&str>,
        model: &str,
        user_only: bool,
    ) -> Option<(&'a ModelPricing, PriceOrigin)> {
        let user = self
            .user
            .and_then(|user| user.catalog.pricing_for(provider, model))
            .map(|entry| (entry, PriceOrigin::User));
        if user.is_some() || user_only {
            return user;
        }
        self.bundled
            .and_then(|bundled| bundled.pricing_for(provider, model))
            .map(|entry| (entry, PriceOrigin::Bundled))
    }
}

/// Nano-USD per million tokens of a rate in US dollars per million tokens,
/// rounded to the nearest whole number. `None` for a rate no valid catalog
/// holds: negative, not finite, or not below one million.
pub(crate) fn nano_usd_per_million_tokens(rate: f64) -> Option<u128> {
    (rate.is_finite() && (0.0..MAX_RATE_PER_MILLION).contains(&rate))
        // In range, so the rounded value is a whole number below 10^15 and
        // the conversion is exact.
        .then(|| (rate * NANO_USD_PER_USD).round() as u128)
}

/// `sha256:` followed by the lowercase hexadecimal SHA-256 of `bytes`.
pub(crate) fn sha256_fingerprint(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut text = String::with_capacity("sha256:".len() + digest.len() * 2);
    text.push_str("sha256:");
    for byte in digest.iter() {
        // Writing to a String cannot fail.
        let _ = write!(text, "{byte:02x}");
    }
    text
}

fn default_currency() -> String {
    "USD".to_string()
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    fn catalog(entries: Value) -> PricingCatalog {
        PricingCatalog::from_json(
            json!({"version": 1, "entries": entries})
                .to_string()
                .as_bytes(),
        )
        .expect("valid catalog")
    }

    fn catalog_error(entries: Value) -> String {
        PricingCatalog::from_json(
            json!({"version": 1, "entries": entries})
                .to_string()
                .as_bytes(),
        )
        .expect_err("invalid catalog")
        .to_string()
    }

    fn flat_entry(provider: &str, model_id: &str, input: f64, output: f64) -> Value {
        json!({
            "provider": provider,
            "model_id": model_id,
            "pricing_as_of": "2026-06-04",
            "pricing_source": format!("https://example.test/{provider}"),
            "rates": {"input_per_million": input, "output_per_million": output},
            "prompt_cache": {"read_accounting": "separate"}
        })
    }

    fn threshold_entry(tiers: Value) -> Value {
        json!({
            "provider": "threshold-ai",
            "model_id": "threshold-model",
            "pricing_as_of": "2026-06-05",
            "pricing_source": "https://example.test/pricing",
            "rate_schedule": {"type": "prompt_token_threshold", "tiers": tiers},
            "prompt_cache": {"read_accounting": "separate"}
        })
    }

    fn rates(input: f64) -> Value {
        json!({"input_per_million": input, "output_per_million": input * 2.0})
    }

    fn found<'a>(
        catalog: &'a PricingCatalog,
        provider: Option<&str>,
        model: &str,
    ) -> (&'a str, &'a str) {
        let entry = catalog
            .pricing_for(provider, model)
            .unwrap_or_else(|| panic!("no entry for {provider:?} {model}"));
        (entry.provider.as_str(), entry.model_id.as_str())
    }

    // The shipped catalog.

    #[test]
    fn bundled_catalog_is_valid_and_cites_a_dated_provider_page_for_every_entry() {
        let catalog = PricingCatalog::from_json(BUNDLED_CATALOG_JSON.as_bytes())
            .expect("the shipped catalog is valid");

        assert!(!catalog.entries.is_empty());
        for entry in &catalog.entries {
            let name = format!("{}/{}", entry.provider, entry.model_id);
            assert_eq!(entry.currency, "USD", "{name}");
            assert_eq!(entry.unit, PricingUnit::PerToken, "{name}");
            assert!(
                entry.pricing_source.starts_with("https://"),
                "{name}: {}",
                entry.pricing_source
            );
            assert!(
                is_iso_date(&entry.pricing_as_of),
                "{name}: {}",
                entry.pricing_as_of
            );
        }
        assert!(BUNDLED_CATALOG.catalog.is_some());
    }

    /// `YYYY-MM-DD` with a plausible month and day. The test never compares
    /// the date with today's, so an old date cannot fail CI.
    fn is_iso_date(text: &str) -> bool {
        let parts: Vec<&str> = text.split('-').collect();
        let [year, month, day] = parts.as_slice() else {
            return false;
        };
        let number = |part: &str, digits: usize| {
            (part.len() == digits && part.bytes().all(|byte| byte.is_ascii_digit()))
                .then(|| part.parse::<u32>().ok())
                .flatten()
        };
        matches!(
            (number(year, 4), number(month, 2), number(day, 2)),
            (Some(_), Some(1..=12), Some(1..=31))
        )
    }

    #[test]
    fn bundled_catalog_keeps_native_provider_boundaries_for_routed_models() {
        let catalog = BUNDLED_CATALOG.catalog.as_ref().expect("bundled catalog");
        assert!(
            catalog
                .pricing_for(Some("nvidia"), "aws/anthropic/claude-opus-4-5")
                .is_none()
        );
        assert!(
            catalog
                .pricing_for(Some("nvidia"), "azure/openai/gpt-5.4")
                .is_none()
        );
        assert_eq!(
            found(catalog, Some("anthropic"), "aws/anthropic/claude-opus-4-5"),
            ("anthropic", "claude-opus-4-5")
        );
        assert_eq!(
            found(catalog, Some("openai"), "azure/openai/gpt-5.4"),
            ("openai", "gpt-5.4")
        );
        assert_eq!(
            found(catalog, Some("anthropic"), "claude-sonnet-4-5-20250929"),
            ("anthropic", "claude-sonnet-4-5")
        );
    }

    #[test]
    fn catalog_fingerprint_is_the_sha256_of_the_file_bytes() {
        assert_eq!(
            sha256_fingerprint(b"abc"),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let fingerprint = PriceCatalogs::with_user_catalog(None).bundled_sha256();
        assert_eq!(
            fingerprint,
            sha256_fingerprint(BUNDLED_CATALOG_JSON.as_bytes())
        );
        assert_eq!(fingerprint.len(), "sha256:".len() + 64);
    }

    // Lookup, mirroring NeMo Relay's model pricing tests.

    #[test]
    fn lookup_matches_aliases_case_insensitively() {
        let catalog = catalog(json!([{
            "provider": "configured",
            "model_id": "configured-model",
            "aliases": ["configured-model-2026-06-04"],
            "pricing_as_of": "2026-06-04",
            "pricing_source": "file:///tmp/pricing.json",
            "rates": {"input_per_million": 0.15, "output_per_million": 0.60},
            "prompt_cache": {"read_accounting": "included_in_prompt_tokens"}
        }]));

        let entry = catalog
            .pricing_for(None, "CONFIGURED-MODEL-2026-06-04")
            .expect("alias");

        assert_eq!(entry.model_id, "configured-model");
        assert_eq!(entry.currency, "USD");
        assert_eq!(entry.unit, PricingUnit::PerToken);
    }

    #[test]
    fn lookup_matches_routed_model_names_by_suffix() {
        let catalog = catalog(json!([flat_entry("openai", "gpt-4o-mini", 0.15, 0.6)]));

        assert_eq!(
            found(&catalog, None, "azure/openai/gpt-4o-mini"),
            ("openai", "gpt-4o-mini")
        );
        assert_eq!(
            found(&catalog, None, "openai/openai/gpt-4o-mini"),
            ("openai", "gpt-4o-mini")
        );
    }

    #[test]
    fn lookup_prefers_the_exact_routed_model_before_a_suffix() {
        let catalog = catalog(json!([
            flat_entry("openai", "gpt-4o-mini", 0.15, 0.6),
            flat_entry("azure-openai", "azure/openai/gpt-4o-mini", 1.0, 2.0)
        ]));

        assert_eq!(
            found(&catalog, None, "azure/openai/gpt-4o-mini"),
            ("azure-openai", "azure/openai/gpt-4o-mini")
        );
    }

    #[test]
    fn lookup_tells_apart_one_model_id_of_two_providers() {
        let catalog = catalog(json!([
            flat_entry("openai", "gpt-4o-mini", 0.15, 0.6),
            flat_entry("azure/openai", "gpt-4o-mini", 1.0, 2.0)
        ]));

        assert_eq!(
            found(&catalog, Some("openai"), "gpt-4o-mini"),
            ("openai", "gpt-4o-mini")
        );
        assert_eq!(
            found(&catalog, Some("azure/openai"), "gpt-4o-mini"),
            ("azure/openai", "gpt-4o-mini")
        );
    }

    #[test]
    fn lookup_normalizes_providers_and_ignores_blank_or_unknown_models() {
        let catalog = catalog(json!([{
            "provider": "provider-a",
            "model_id": "priced-model",
            "aliases": ["alias-model"],
            "pricing_as_of": "2026-06-04",
            "pricing_source": "test",
            "rates": {"input_per_million": 1.0, "output_per_million": 2.0},
            "prompt_cache": {"read_accounting": "included_in_prompt_tokens"}
        }]));

        assert_eq!(
            found(&catalog, Some("/provider-a/"), "provider-a/priced-model"),
            ("provider-a", "priced-model")
        );
        assert_eq!(
            found(&catalog, None, " ALIAS-MODEL "),
            ("provider-a", "priced-model")
        );
        assert!(catalog.pricing_for(Some("other"), "priced-model").is_none());
        assert_eq!(
            found(&catalog, Some("provider-a"), "path/alias-model"),
            ("provider-a", "priced-model")
        );
        assert!(catalog.pricing_for(None, " ").is_none());
        assert!(catalog.pricing_for(None, "missing-model").is_none());
    }

    #[test]
    fn caller_catalog_takes_precedence_and_alone_prices_self_hosted_models() {
        let bundled = catalog(json!([
            flat_entry("openai", "gpt-4o-mini", 0.15, 0.6),
            flat_entry("openai", "gpt-4o", 2.5, 10.0)
        ]));
        let user = UserPriceCatalog {
            path: PathBuf::from("/prices.json"),
            sha256: "sha256:user".to_string(),
            catalog: catalog(json!([flat_entry("openai", "gpt-4o-mini", 1.0, 2.0)])),
        };
        let prices = PriceCatalogs::from_parts(Some(&bundled), "sha256:bundled", Some(&user));

        let origin = |model: &str, user_only: bool| {
            prices
                .find(Some("openai"), model, user_only)
                .map(|(entry, origin)| (entry.provider.as_str(), origin))
        };

        assert_eq!(
            origin("gpt-4o-mini", false),
            Some(("openai", PriceOrigin::User))
        );
        assert_eq!(
            origin("gpt-4o", false),
            Some(("openai", PriceOrigin::Bundled))
        );
        assert_eq!(origin("gpt-4o", true), None);
        assert_eq!(
            origin("gpt-4o-mini", true),
            Some(("openai", PriceOrigin::User))
        );
        assert_eq!(prices.bundled_sha256(), "sha256:bundled");
        assert_eq!(prices.user_sha256(), Some("sha256:user"));
    }

    #[test]
    fn unrelated_user_provider_cannot_shadow_matching_bundled_price() {
        let bundled = catalog(json!([flat_entry("openai", "model", 1.0, 2.0)]));
        let user = UserPriceCatalog {
            path: PathBuf::from("/prices.json"),
            sha256: "sha256:user".to_string(),
            catalog: catalog(json!([flat_entry("custom-route", "model", 3.0, 4.0)])),
        };
        let prices = PriceCatalogs::from_parts(Some(&bundled), "sha256:bundled", Some(&user));
        let (entry, origin) = prices.find(Some("openai"), "model", false).unwrap();
        assert_eq!(entry.provider, "openai");
        assert_eq!(origin, PriceOrigin::Bundled);
        let (entry, origin) = prices.find(Some("custom-route"), "model", true).unwrap();
        assert_eq!(entry.provider, "custom-route");
        assert_eq!(origin, PriceOrigin::User);
    }

    // Validation, mirroring NeMo Relay's catalog validation tests.

    #[test]
    fn validation_rejects_an_unsupported_version() {
        let error = PricingCatalog::from_json(br#"{"version": 2, "entries": []}"#)
            .expect_err("version 2")
            .to_string();

        assert!(
            error.contains("unsupported model pricing catalog version 2"),
            "{error}"
        );
    }

    #[test]
    fn validation_rejects_duplicate_model_aliases_of_one_provider() {
        let mut other = flat_entry("a", "other-model", 1.0, 1.0);
        other["aliases"] = json!(["SAME-MODEL"]);

        let error = catalog_error(json!([flat_entry("a", "same-model", 1.0, 1.0), other]));

        assert!(
            error.contains("duplicate model pricing alias 'a/same-model'"),
            "{error}"
        );
    }

    #[test]
    fn validation_rejects_blank_required_fields() {
        for field in [
            "provider",
            "model_id",
            "currency",
            "pricing_as_of",
            "pricing_source",
        ] {
            let mut entry = flat_entry("configured", "configured-model", 1.0, 2.0);
            entry[field] = json!(" ");

            let error = catalog_error(json!([entry]));

            assert!(
                error.contains(&format!("empty {field}")),
                "{field}: {error}"
            );
        }
    }

    #[test]
    fn validation_rejects_negative_or_too_large_rates() {
        for (field, value) in [
            ("input_per_million", -0.1),
            ("output_per_million", -0.1),
            ("cache_read_per_million", -0.1),
            ("cache_write_per_million", -0.1),
            ("input_per_million", 1_000_000.0),
        ] {
            let mut entry = flat_entry("configured", "configured-model", 1.0, 2.0);
            entry["rates"][field] = json!(value);

            let error = catalog_error(json!([entry]));

            assert!(
                error.contains(&format!("rates.{field}")),
                "{field}: {error}"
            );
        }
    }

    #[test]
    fn validation_requires_rates_for_per_token_entries() {
        let mut entry = flat_entry("broken", "missing-rates", 1.0, 2.0);
        entry.as_object_mut().expect("entry").remove("rates");

        let error = catalog_error(json!([entry]));

        assert!(error.contains("empty rates or rate_schedule"), "{error}");
    }

    #[test]
    fn non_token_units_are_valid_without_rates() {
        let mut entry = flat_entry("self-hosted", "nemotron-owned", 1.0, 2.0);
        entry.as_object_mut().expect("entry").remove("rates");
        entry["unit"] = json!("gpu_hour");

        let catalog = catalog(json!([entry]));

        assert_eq!(catalog.entries[0].unit, PricingUnit::GpuHour);
    }

    #[test]
    fn validation_rejects_invalid_rate_schedules() {
        let empty = catalog_error(json!([threshold_entry(json!([]))]));
        let reversed = catalog_error(json!([threshold_entry(json!([
            {"min_prompt_tokens": 20, "max_prompt_tokens": 10, "rates": rates(1.0)}
        ]))]));
        let negative = catalog_error(json!([threshold_entry(json!([{"rates": rates(-1.0)}]))]));

        assert!(empty.contains("rate_schedule.tiers"), "{empty}");
        assert!(
            reversed.contains("rate_schedule.tiers.prompt_tokens"),
            "{reversed}"
        );
        assert!(
            negative.contains("rate_schedule.tiers[0].rates.input_per_million"),
            "{negative}"
        );
    }

    #[test]
    fn validation_rejects_unknown_fields_at_every_level() {
        let entry = flat_entry("configured", "configured-model", 1.0, 2.0);
        let with = |pointer: &str, field: &str| {
            let mut catalog = json!({"version": 1, "entries": [entry.clone()]});
            catalog
                .pointer_mut(pointer)
                .and_then(Value::as_object_mut)
                .expect("object")
                .insert(field.to_string(), json!(true));
            catalog
        };
        let mut tiers = json!({"version": 1, "entries": [threshold_entry(json!([
            {"min_prompt_tokens": 0, "rates": rates(1.0)}
        ]))]});
        tiers["entries"][0]["rate_schedule"]["tierz"] = json!([]);
        let tier = json!({"version": 1, "entries": [threshold_entry(json!([
            {"min_prompt_toknes": 0, "rates": rates(1.0)}
        ]))]});

        for (catalog, field) in [
            (with("", "unexpected_catalog"), "unexpected_catalog"),
            (with("/entries/0", "unexpected_entry"), "unexpected_entry"),
            (
                with("/entries/0/rates", "cache_writ_per_million"),
                "cache_writ_per_million",
            ),
            (
                with("/entries/0/prompt_cache", "read_accountng"),
                "read_accountng",
            ),
            (tiers, "tierz"),
            (tier, "min_prompt_toknes"),
        ] {
            let error = PricingCatalog::from_json(catalog.to_string().as_bytes())
                .expect_err("unknown field")
                .to_string();

            assert!(
                error.contains(&format!("unknown field `{field}`")),
                "{field}: {error}"
            );
        }
    }

    // Rate tiers.

    #[test]
    fn a_rate_schedule_resolves_only_when_one_tier_covers_every_request() {
        let catalog = catalog(json!([threshold_entry(json!([
            {"max_prompt_tokens": 200_000, "rates": rates(1.0)},
            {"min_prompt_tokens": 200_001, "rates": rates(10.0)}
        ]))]));
        let entry = &catalog.entries[0];
        let input =
            |peak: Option<u64>| entry.rates_up_to(peak).map(|rates| rates.input_per_million);

        assert_eq!(input(Some(0)), Some(1.0));
        assert_eq!(input(Some(200_000)), Some(1.0));
        assert_eq!(input(Some(200_001)), None);
        assert_eq!(input(None), None);
    }

    #[test]
    fn a_rate_schedule_follows_tier_order_like_nemo_relay() {
        let catalog = catalog(json!([threshold_entry(json!([
            {"min_prompt_tokens": 100, "rates": rates(10.0)},
            {"max_prompt_tokens": 99, "rates": rates(1.0)}
        ]))]));
        let gap = catalog_with_first_tier_from(1);
        let entry = &catalog.entries[0];

        assert_eq!(
            entry
                .rates_up_to(Some(99))
                .map(|rates| rates.input_per_million),
            Some(1.0)
        );
        assert_eq!(entry.rates_up_to(Some(100)), None);
        assert_eq!(gap.entries[0].rates_up_to(Some(10)), None);
    }

    fn catalog_with_first_tier_from(min_prompt_tokens: u64) -> PricingCatalog {
        catalog(json!([threshold_entry(json!([
            {"min_prompt_tokens": min_prompt_tokens, "rates": rates(1.0)}
        ]))]))
    }

    #[test]
    fn flat_rates_apply_whatever_the_peak() {
        let catalog = catalog(json!([flat_entry("openai", "gpt-4o-mini", 0.15, 0.6)]));

        for peak in [None, Some(0), Some(u64::MAX)] {
            assert_eq!(
                catalog.entries[0]
                    .rates_up_to(peak)
                    .map(|rates| rates.input_per_million),
                Some(0.15)
            );
        }
    }

    // Rate conversion.

    #[test]
    fn rates_convert_to_whole_nano_usd_per_million_tokens() {
        for (rate, expected) in [
            (0.0, Some(0)),
            (0.075, Some(75_000_000)),
            (0.3, Some(300_000_000)),
            (2.5, Some(2_500_000_000)),
            (3.75, Some(3_750_000_000)),
            (1e-10, Some(0)),
            (6e-10, Some(1)),
            (999_999.999_999_999, Some(999_999_999_999_999)),
            (1_000_000.0, None),
            (-0.000_001, None),
            (f64::NAN, None),
            (f64::INFINITY, None),
        ] {
            assert_eq!(nano_usd_per_million_tokens(rate), expected, "rate {rate}");
        }
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Catalog prices are estimates, separate from provider-reported amounts.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::measurement::{
    Attribution, Availability, Measurement, MeasurementSource, MeasurementUnit,
};
use crate::price_catalog::{
    PriceCatalogs, PriceOrigin, PricingUnit, TokenPricingRates, UserPriceCatalog,
    nano_usd_per_million_tokens,
};
use crate::usage_accounting::GapReason;
use crate::usage_normalization::{AccountingDimension, AccountingGap, ModelUsage, ObservedUsage};

pub(crate) const OPENAI_ENDPOINT_ENV_KEYS: [&str; 2] = ["OPENAI_BASE_URL", "OPENAI_API_BASE"];
pub(crate) const ANTHROPIC_ENDPOINT_ENV_KEYS: [&str; 1] = ["ANTHROPIC_BASE_URL"];

/// Configured and inherited endpoint overrides must still identify a native
/// endpoint before provider list prices apply. An adapter can inherit URL
/// settings even when models.<role>.base_url is absent.
pub(crate) fn native_catalog_route(
    model: &crate::ModelConfig,
    forwarded: &std::collections::BTreeMap<String, String>,
    inherited: impl Fn(&str) -> Option<String>,
) -> bool {
    native_catalog_endpoint(
        &model.provider,
        model.base_url.as_deref(),
        forwarded,
        inherited,
    )
}

fn native_catalog_endpoint(
    provider: &str,
    base_url: Option<&str>,
    forwarded: &std::collections::BTreeMap<String, String>,
    inherited: impl Fn(&str) -> Option<String>,
) -> bool {
    let (keys, urls): (&[&str], &[&str]) = match provider {
        "openai" => (&OPENAI_ENDPOINT_ENV_KEYS, &["https://api.openai.com/v1"]),
        "anthropic" => (
            &ANTHROPIC_ENDPOINT_ENV_KEYS,
            &["https://api.anthropic.com", "https://api.anthropic.com/v1"],
        ),
        _ => return false,
    };
    let native = |url: &str| urls.contains(&url.trim_end_matches('/'));
    if let Some(url) = base_url {
        return native(url);
    }
    keys.iter().all(|key| {
        forwarded
            .get(*key)
            .cloned()
            .or_else(|| inherited(key))
            .is_none_or(|url| native(&url))
    })
}

/// Startup route evidence, qualified for the provider/model actually accounted.
/// An unused custom role must not disable a different model's native tariff.
#[derive(Debug, Clone)]
pub(crate) struct NativePricingRoutes {
    models: Vec<NativeModelRoute>,
    endpoints: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
struct NativeModelRoute {
    provider: String,
    model: String,
    native: bool,
}

impl NativePricingRoutes {
    pub(crate) fn from_models(
        models: &std::collections::BTreeMap<String, crate::ModelConfig>,
        endpoints: &std::collections::BTreeMap<String, String>,
    ) -> Self {
        Self {
            models: models
                .values()
                .map(|model| NativeModelRoute {
                    provider: model.provider.clone(),
                    model: model.model.clone(),
                    native: native_catalog_route(model, endpoints, |_| None),
                })
                .collect(),
            endpoints: endpoints.clone(),
        }
    }

    pub(crate) fn allows(&self, provider: Option<&str>, model: &str) -> bool {
        let mut exact = self
            .models
            .iter()
            .filter(|route| Some(route.provider.as_str()) == provider && route.model == model)
            .peekable();
        if exact.peek().is_some() {
            return exact.all(|route| route.native);
        }
        // A native provider route still supports a served-model alias. Mixed
        // routes for that provider cannot establish which endpoint served it.
        let mut provider_routes = self
            .models
            .iter()
            .filter(|route| Some(route.provider.as_str()) == provider)
            .peekable();
        if provider_routes.peek().is_some() {
            return provider_routes.all(|route| route.native);
        }
        // A child provider's inherited endpoint can differ from configured roles.
        let accounted_endpoint_native = provider.is_none_or(|provider| {
            native_catalog_endpoint(provider, None, &self.endpoints, |_| None)
        });
        accounted_endpoint_native
            && !self.models.is_empty()
            && self.models.iter().all(|route| route.native)
    }
}

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

impl RateSnapshot {
    pub(crate) fn from_rates(rates: TokenPricingRates) -> Result<Self, &'static str> {
        let convert = |rate| {
            nano_usd_per_million_tokens(rate)
                .and_then(|value| u64::try_from(value).ok())
                .ok_or("price_catalog_invalid")
        };
        Ok(Self {
            input: convert(rates.input_per_million)?,
            output: convert(rates.output_per_million)?,
            cache_read: rates.cache_read_per_million.map(convert).transpose()?,
            cache_write: rates.cache_write_per_million.map(convert).transpose()?,
        })
    }

    fn price(
        &self,
        input: Option<u64>,
        cached: Option<u64>,
        written: Option<u64>,
        output: Option<u64>,
    ) -> Result<u64, &'static str> {
        rounded_price(self.charges(input, cached, written, output)?)
    }

    fn charges(
        &self,
        input: Option<u64>,
        cached: Option<u64>,
        written: Option<u64>,
        output: Option<u64>,
    ) -> Result<u128, &'static str> {
        let input = input.ok_or("pricing_input_unreported")?;
        let output = output.ok_or("pricing_output_unreported")?;
        // A missing cache category cannot affect the price only when it is
        // charged exactly like ordinary input, or the entire input is zero.
        let component = |count: Option<u64>, rate: Option<u64>| match count {
            Some(count) => match rate {
                Some(rate) => Ok((count, rate)),
                None if count == 0 => Ok((0, 0)),
                None => Err("cache_rate_unreported"),
            },
            None if input == 0 || rate == Some(self.input) => Ok((0, self.input)),
            None => Err("pricing_cache_unreported"),
        };
        let (cached, read_rate) = component(cached, self.cache_read)?;
        let (written, write_rate) = component(written, self.cache_write)?;
        let uncached = input
            .checked_sub(cached)
            .and_then(|v| v.checked_sub(written))
            .ok_or("inconsistent_usage")?;
        let charges = [
            (uncached, self.input),
            (cached, read_rate),
            (written, write_rate),
            (output, self.output),
        ];
        let sum = charges
            .into_iter()
            .try_fold(0_u128, |sum, (count, rate)| {
                sum.checked_add(u128::from(count).checked_mul(u128::from(rate))?)
            })
            .ok_or("cost_out_of_range")?;
        Ok(sum)
    }

    fn observed_charges(
        &self,
        observed: &ObservedUsage,
        aligned: bool,
        write_tariff_known: bool,
    ) -> Option<u128> {
        // Categories are disjoint. Without aligned input/cache observations,
        // retain independently priceable categories instead of subtracting
        // counters from different source-call subsets.
        let mut charges = 0_u128;
        let mut any = false;
        let mut add = |count: Option<u64>, rate: Option<u64>| -> Option<()> {
            if let (Some(count), Some(rate)) = (count, rate) {
                charges = charges.checked_add(u128::from(count).checked_mul(u128::from(rate))?)?;
                any = true;
            }
            Some(())
        };
        add(observed.output_tokens, Some(self.output))?;
        let equal_rates =
            self.cache_read == Some(self.input) && self.cache_write == Some(self.input);
        if let Some(input) = observed
            .input_tokens
            .filter(|_| equal_rates && write_tariff_known)
        {
            add(Some(input), Some(self.input))?;
        } else {
            if aligned {
                let cached = observed
                    .cached_input_tokens
                    .or((self.cache_read == Some(self.input)).then_some(0));
                let written = observed
                    .cache_write_input_tokens
                    .or((write_tariff_known && self.cache_write == Some(self.input)).then_some(0));
                if let (Some(input), Some(cached), Some(written)) =
                    (observed.input_tokens, cached, written)
                {
                    add(
                        input
                            .checked_sub(cached)
                            .and_then(|v| v.checked_sub(written)),
                        Some(self.input),
                    )?;
                }
            }
            add(observed.cached_input_tokens, self.cache_read)?;
            if write_tariff_known {
                add(observed.cache_write_input_tokens, self.cache_write)?;
            }
        }
        any.then_some(charges)
    }

    fn observed_price(
        &self,
        observed: &ObservedUsage,
        aligned: bool,
        write_tariff_known: bool,
    ) -> Option<u64> {
        rounded_price(self.observed_charges(observed, aligned, write_tariff_known)?).ok()
    }
}

fn rounded_price(charges: u128) -> Result<u64, &'static str> {
    u64::try_from(charges.checked_add(500_000).ok_or("cost_out_of_range")? / 1_000_000)
        .map_err(|_| "cost_out_of_range")
}

fn estimate(value: Result<u64, &'static str>, source: MeasurementSource) -> Measurement {
    Measurement {
        availability: if value.is_ok() {
            Availability::Measured
        } else {
            Availability::Unavailable
        },
        value: value.ok(),
        unit: MeasurementUnit::NanoUsd,
        source,
        attribution: if value.is_ok() {
            Attribution::Invocation
        } else {
            Attribution::None
        },
        reason: value.err().map(str::to_owned),
    }
}

fn update_price_gaps(gaps: &mut Vec<AccountingGap>, cost: &Measurement) {
    gaps.retain(|gap| gap.dimensions != [AccountingDimension::EstimatedCost]);
    if cost.value.is_some() {
        return;
    }
    let mut reasons: std::collections::BTreeSet<_> = gaps
        .iter()
        .filter(|gap| {
            gap.dimensions.iter().any(|dimension| {
                matches!(
                    dimension,
                    AccountingDimension::InputTokens | AccountingDimension::OutputTokens
                )
            })
        })
        .map(|gap| gap.reason)
        .collect();
    if reasons.is_empty() {
        reasons.insert(match cost.reason.as_deref() {
            Some("model_scope_unverified" | "inconsistent_usage" | "provider_price_unverified") => {
                GapReason::ScopeUnverified
            }
            _ => GapReason::MissingUsage,
        });
    }
    gaps.extend(reasons.into_iter().map(|reason| AccountingGap {
        dimensions: vec![AccountingDimension::EstimatedCost],
        reason,
    }));
}

/// Price validated normalized usage with a route decision for each accounted model.
/// An unqualified route permits only explicitly supplied prices.
///
/// Price with validated original request boundaries when an aggregate spans tiers.
/// Keep this evidence explicit: normalized aggregates cannot recover request sizes.
pub(crate) fn apply_pricing_with_evidence(
    usage: &mut ModelUsage,
    user: Option<&UserPriceCatalog>,
    bundled_allowed: impl Fn(Option<&str>, &str) -> bool,
    evidence: Option<&crate::usage_accounting::UsageAccounting>,
) {
    let prices = PriceCatalogs::with_user_catalog(user);
    if !prices.bundled_is_valid() {
        usage.token_cost.estimated = estimate(
            Err("price_catalog_invalid"),
            MeasurementSource::FabricPriceCatalog,
        );
        return;
    }
    let basis = PricingBasis {
        basis: "catalog".into(),
        currency: "USD".into(),
        normalization_version: "cache_reasoning_inclusive_v1".into(),
        assumption_policy: "applicable_tariff_no_missing_counts_v1".into(),
        bundled_catalog_sha256: prices.bundled_sha256().into(),
        user_catalog_sha256: prices.user_sha256().map(str::to_owned),
    };
    usage.token_cost.pricing = Some(basis.clone());
    if usage.accounting.is_none() {
        usage.token_cost.estimated = estimate(
            Err("legacy_unverified"),
            MeasurementSource::FabricPriceCatalog,
        );
        return;
    }
    let mut any_user = false;
    let mut root_cache_read_required = false;
    let mut root_cache_write_required = false;
    for model in &mut usage.models {
        let cost = &mut model.usage.token_cost;
        cost.applied = None;
        cost.applied_calls.clear();
        model.accounting.observed.estimated_cost_nano_usd = None;
        cost.pricing = Some(basis.clone());
        let selected = prices.find(
            model.provider.as_deref(),
            &model.model,
            !bundled_allowed(model.provider.as_deref(), &model.model),
        );
        let Some((entry, origin)) = selected else {
            cost.estimated = estimate(
                Err("model_not_in_catalog"),
                MeasurementSource::FabricPriceCatalog,
            );
            update_price_gaps(&mut model.accounting.gaps, &cost.estimated);
            continue;
        };
        let source = match origin {
            PriceOrigin::User => MeasurementSource::UserPriceCatalog,
            PriceOrigin::Bundled => MeasurementSource::FabricPriceCatalog,
        };
        any_user |= origin == PriceOrigin::User;
        if origin == PriceOrigin::Bundled
            && !model
                .provider
                .as_ref()
                .is_some_and(|provider| provider.eq_ignore_ascii_case(&entry.provider))
        {
            cost.estimated = estimate(Err("provider_price_unverified"), source);
            update_price_gaps(&mut model.accounting.gaps, &cost.estimated);
            continue;
        }
        let mut result = (|| {
            if entry.unit != PricingUnit::PerToken || !entry.currency.eq_ignore_ascii_case("USD") {
                return Err("price_currency_or_unit_unsupported");
            }
            // A complete invocation's total input is also a safe upper bound
            // on each request; a subset peak never prices a larger complete set.
            let complete_peak = model
                .usage
                .peak_request_input_tokens
                .value
                .or(model.usage.input_tokens.value);
            let complete_rates = entry
                .rates_up_to(complete_peak)
                .map(RateSnapshot::from_rates)
                .transpose()?;
            let observed_peak = if model.accounting.observed_tokens_aligned {
                model
                    .accounting
                    .observed
                    .peak_request_input_tokens
                    .or(model.accounting.observed.input_tokens)
            } else {
                None
            };
            let observed_rates = match &complete_rates {
                Some(rates) => Some(rates.clone()),
                None => entry
                    .rates_up_to(observed_peak)
                    .map(RateSnapshot::from_rates)
                    .transpose()?,
            };
            let write_tariff_known = !entry.provider.eq_ignore_ascii_case("anthropic")
                || model.usage.cache_write_input_tokens.value == Some(0);
            let complete = match &complete_rates {
                Some(rates) => {
                    root_cache_read_required |= rates.cache_read != Some(rates.input);
                    root_cache_write_required |= rates.cache_write != Some(rates.input);
                    if !write_tariff_known && model.usage.input_tokens.value != Some(0) {
                        Err("cache_write_tariff_unverified")
                    } else {
                        rates.price(
                            model.usage.input_tokens.value,
                            model.usage.cached_input_tokens.value,
                            model.usage.cache_write_input_tokens.value,
                            model.usage.output_tokens.value,
                        )
                    }
                }
                None => Err("tiered_price_unresolvable"),
            };
            let observed = complete.ok().or_else(|| {
                observed_rates.as_ref().and_then(|rates| {
                    rates.observed_price(
                        &model.accounting.observed,
                        model.accounting.observed_tokens_aligned
                            || (model.usage.input_tokens.value.is_some()
                                && model.usage.cached_input_tokens.value.is_some()
                                && model.usage.cache_write_input_tokens.value.is_some()),
                        !entry.provider.eq_ignore_ascii_case("anthropic")
                            || model.accounting.observed.cache_write_input_tokens == Some(0),
                    )
                })
            });
            model.accounting.observed.estimated_cost_nano_usd = observed;
            let rates = if complete.is_ok() {
                complete_rates
            } else {
                observed_rates.or(complete_rates)
            };
            let Some(rates) = rates else {
                return complete;
            };
            cost.applied = Some(AppliedPricing {
                catalog_sha256: match origin {
                    PriceOrigin::Bundled => prices.bundled_sha256(),
                    PriceOrigin::User => prices.user_sha256().expect("selected user catalog"),
                }
                .into(),
                provider: entry.provider.clone(),
                model_id: entry.model_id.clone(),
                pricing_as_of: entry.pricing_as_of.clone(),
                pricing_source: entry.pricing_source.clone(),
                rates,
            });
            complete
        })();
        if let Some(raw) = evidence
            .and_then(|e| {
                e.models
                    .iter()
                    .find(|raw| raw.model == model.model && raw.provider == model.provider)
            })
            .filter(|raw| {
                !raw.requests.is_empty()
                    && entry.unit == PricingUnit::PerToken
                    && entry.currency.eq_ignore_ascii_case("USD")
            })
        {
            let mut charges = Some(0_u128);
            let mut complete = raw.requests_exhaustive;
            let mut any = false;
            for request in &raw.requests {
                let aligned = request.gap.is_none() || request.tokens_aligned;
                let normalized =
                    crate::usage_normalization::normalized_request(&request.usage, !aligned);
                let priced: Result<(u128, RateSnapshot, bool), &'static str> = (|| {
                    let selected = if entry.rate_schedule.is_some() {
                        entry.rates_at(
                            normalized
                                .input_tokens
                                .value
                                .ok_or("pricing_input_unreported")?,
                        )
                    } else {
                        entry.rates
                    };
                    let rates =
                        RateSnapshot::from_rates(selected.ok_or("tiered_price_unresolvable")?)?;
                    let write_tariff_known = !entry.provider.eq_ignore_ascii_case("anthropic")
                        || normalized.cache_write_input_tokens.value == Some(0)
                        || normalized.input_tokens.value == Some(0);
                    let complete_charge = if aligned && write_tariff_known {
                        rates
                            .charges(
                                normalized.input_tokens.value,
                                normalized.cached_input_tokens.value,
                                normalized.cache_write_input_tokens.value,
                                normalized.output_tokens.value,
                            )
                            .ok()
                    } else {
                        None
                    };
                    let observed = ObservedUsage {
                        input_tokens: normalized.input_tokens.value,
                        cached_input_tokens: normalized.cached_input_tokens.value,
                        cache_write_input_tokens: normalized.cache_write_input_tokens.value,
                        output_tokens: normalized.output_tokens.value,
                        ..Default::default()
                    };
                    let charge = complete_charge
                        .or_else(|| rates.observed_charges(&observed, aligned, write_tariff_known))
                        .ok_or("pricing_usage_unavailable")?;
                    Ok((charge, rates, complete_charge.is_some()))
                })();
                match priced {
                    Ok((charge, rates, all_parts_priced)) => {
                        any = true;
                        charges = charges.and_then(|total| total.checked_add(charge));
                        root_cache_read_required |= rates.cache_read != Some(rates.input);
                        root_cache_write_required |= rates.cache_write != Some(rates.input);
                        cost.applied_calls.push(Some(AppliedPricing {
                            catalog_sha256: match origin {
                                PriceOrigin::Bundled => prices.bundled_sha256(),
                                PriceOrigin::User => {
                                    prices.user_sha256().expect("selected catalog")
                                }
                            }
                            .into(),
                            provider: entry.provider.clone(),
                            model_id: entry.model_id.clone(),
                            pricing_as_of: entry.pricing_as_of.clone(),
                            pricing_source: entry.pricing_source.clone(),
                            rates,
                        }));
                        complete &= request.gap.is_none() && all_parts_priced;
                    }
                    Err(_) => {
                        complete = false;
                        cost.applied_calls.push(None);
                    }
                }
            }
            let amount = charges
                .and_then(|charges| rounded_price(charges).ok())
                .filter(|_| any);
            if result.is_err()
                && complete
                && model.usage.input_tokens.value.is_some()
                && model.usage.output_tokens.value.is_some()
            {
                result = amount.ok_or("cost_out_of_range");
                cost.applied = None;
            }
            if let Some(amount) = result.as_ref().ok().copied().or(amount) {
                model.accounting.observed.estimated_cost_nano_usd = Some(amount);
            }
        }
        cost.estimated = estimate(result, source);
        update_price_gaps(&mut model.accounting.gaps, &cost.estimated);
    }
    let inconsistent = [
        &usage.input_tokens,
        &usage.cached_input_tokens,
        &usage.cache_write_input_tokens,
        &usage.output_tokens,
    ]
    .iter()
    .any(|m| m.reason.as_deref() == Some("inconsistent_usage"));
    let accounting = usage.accounting.as_mut().expect("checked accounting");
    let complete = if !accounting.models_exhaustive || usage.models.is_empty() {
        Err("model_scope_unverified")
    } else if inconsistent {
        Err("inconsistent_usage")
    } else if usage.models.iter().any(|model| {
        model.usage.token_cost.estimated.reason.as_deref() == Some("cache_write_tariff_unverified")
    }) {
        Err("cache_write_tariff_unverified")
    } else if usage.input_tokens.value.is_none()
        || usage.output_tokens.value.is_none()
        || (root_cache_read_required
            && usage.cached_input_tokens.value.is_none()
            && usage.input_tokens.value != Some(0))
        || (root_cache_write_required
            && usage.cache_write_input_tokens.value.is_none()
            && usage.input_tokens.value != Some(0))
    {
        Err("pricing_usage_incomplete")
    } else {
        usage.models.iter().try_fold(0_u64, |sum, model| {
            sum.checked_add(
                model
                    .usage
                    .token_cost
                    .estimated
                    .value
                    .ok_or("unpriced_model")?,
            )
            .ok_or("cost_out_of_range")
        })
    };
    let observed = if inconsistent {
        None
    } else {
        let values: Vec<_> = usage
            .models
            .iter()
            .filter_map(|model| model.accounting.observed.estimated_cost_nano_usd)
            .collect();
        if values.is_empty() {
            None
        } else {
            values.into_iter().try_fold(0_u64, u64::checked_add)
        }
    };
    accounting.observed.estimated_cost_nano_usd = complete.ok().or(observed);
    usage.token_cost.estimated = estimate(
        complete,
        if any_user {
            MeasurementSource::UserPriceCatalog
        } else {
            MeasurementSource::FabricPriceCatalog
        },
    );
    update_price_gaps(&mut accounting.gaps, &usage.token_cost.estimated);
}

#[cfg(test)]
mod tests {
    use super::native_catalog_route;
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn native_pricing_routes_preserve_ambiguity_and_served_aliases() {
        let mut models = serde_json::from_value(json!({
            "default": {"provider": "openai", "model": "gpt-5.4"},
            "auxiliary": {"provider": "openai", "model": "gpt-5.4",
                "base_url": "https://custom.invalid/v1"}
        }))
        .unwrap();
        let routes = super::NativePricingRoutes::from_models(&models, &BTreeMap::new());
        assert!(!routes.allows(Some("openai"), "gpt-5.4"));
        assert!(!routes.allows(Some("openai"), "served-alias"));
        assert!(!routes.allows(Some("anthropic"), "child-model"));

        models.remove("auxiliary");
        let routes = super::NativePricingRoutes::from_models(&models, &BTreeMap::new());
        assert!(routes.allows(Some("openai"), "served-alias"));
        assert!(routes.allows(Some("anthropic"), "source-qualified-child"));

        models.get_mut("default").unwrap().base_url = Some("https://custom.invalid/v1".into());
        models.insert(
            "unused_native".into(),
            serde_json::from_value(json!({
                "provider": "openai", "model": "other-model"
            }))
            .unwrap(),
        );
        let routes = super::NativePricingRoutes::from_models(&models, &BTreeMap::new());
        assert!(!routes.allows(Some("openai"), "gpt-5.4"));
        assert!(routes.allows(Some("openai"), "other-model"));
        assert!(!routes.allows(Some("openai"), "served-alias"));

        let empty = super::NativePricingRoutes::from_models(&BTreeMap::new(), &BTreeMap::new());
        assert!(!empty.allows(Some("openai"), "gpt-5.4"));
    }

    #[test]
    fn child_provider_pricing_checks_its_own_endpoint() {
        let models = serde_json::from_value(json!({
            "default": {"provider": "openai", "model": "gpt-5.4"}
        }))
        .unwrap();
        for (endpoint, expected) in [
            (None, true),
            (Some("https://api.anthropic.com/"), true),
            (Some("https://proxy.invalid"), false),
        ] {
            let endpoints = endpoint
                .map(|url| BTreeMap::from([("ANTHROPIC_BASE_URL".into(), url.into())]))
                .unwrap_or_default();
            let routes = super::NativePricingRoutes::from_models(&models, &endpoints);
            assert_eq!(
                routes.allows(Some("anthropic"), "source-qualified-child"),
                expected,
                "child endpoint: {endpoint:?}"
            );
            assert!(routes.allows(Some("openai"), "served-alias"));
            assert!(routes.allows(None, "source-qualified-child"));
        }
        let routes = super::NativePricingRoutes::from_models(&models, &BTreeMap::new());
        assert!(!routes.allows(Some("unknown-provider"), "source-qualified-child"));
    }

    #[test]
    fn inherited_and_forwarded_custom_routes_require_explicit_prices() {
        for (configured, inherited, forwarded, expected) in [
            (None, None, None, true),
            (
                Some("https://api.openai.com/v1/"),
                Some("https://proxy.invalid/v1"),
                None,
                true,
            ),
            (Some("https://proxy.invalid/v1"), None, None, false),
            (None, Some("https://proxy.invalid/v1"), None, false),
            (None, None, Some("https://proxy.invalid/v1"), false),
            (None, Some("https://api.openai.com/v1"), None, true),
        ] {
            let model = serde_json::from_value(
                json!({"provider":"openai","model":"gpt-5.4","base_url":configured}),
            )
            .unwrap();
            let env = forwarded
                .map(|url| BTreeMap::from([("OPENAI_BASE_URL".into(), url.into())]))
                .unwrap_or_default();
            assert_eq!(
                native_catalog_route(&model, &env, |_| inherited.map(str::to_owned)),
                expected
            );
        }
    }
}

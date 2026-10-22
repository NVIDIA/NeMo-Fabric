// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared vocabulary for values and their provenance.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// One value and the context needed to interpret it. Availability alone does
/// not certify accounting coverage; read the enclosing accounting record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Measurement {
    /// Whether a value was obtained.
    pub availability: Availability,
    /// Value in `unit`; measured zero is a real zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(max = u64::MAX))]
    pub value: Option<u64>,
    /// Unit of value.
    pub unit: MeasurementUnit,
    /// Source of value, or the source that would have supplied it.
    pub source: MeasurementSource,
    /// What the value describes.
    pub attribution: Attribution,
    /// Stable reason for an unavailable measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Availability of a measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Availability {
    /// A value is present.
    Measured,
    /// Supported, but no usable value is available.
    Unavailable,
    /// Not supplied by this implementation.
    Unsupported,
    /// Preserve future values when reading old results.
    #[serde(untagged)]
    Other(String),
}

/// Unit of a measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementUnit {
    /// Model tokens.
    Tokens,
    /// Billionths of a US dollar.
    NanoUsd,
    /// Preserve future units.
    #[serde(untagged)]
    Other(String),
}

/// Source of a measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementSource {
    /// Harness or provider evidence collected by the adapter.
    AdapterUsage,
    /// Fabric's bundled price catalog.
    FabricPriceCatalog,
    /// A user-supplied price catalog.
    UserPriceCatalog,
    /// Preserve future sources.
    #[serde(untagged)]
    Other(String),
}

/// What a measurement describes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Attribution {
    /// Work initiated by this invocation, including its descendants.
    Invocation,
    /// No value is available.
    None,
    /// Preserve future attribution categories.
    #[serde(untagged)]
    Other(String),
}

// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What executed for one invocation, and how NeMo Fabric learned each fact.
//!
//! [`WhatRan`] rides on [`RunResult`](crate::RunResult). Each fact is a
//! [`Fact`]: a value, or a [`FactState`] with a reason, and in both cases
//! the [`FactSource`], the component that collected it and the method it
//! used. A few names fixed by the run plan stay plain strings: the adapter
//! ID, each model role's provider, the environment provider, and each MCP
//! server's origin and transport.
//!
//! A string value longer than 256 characters, in a string fact or as an
//! element of a string-list fact, keeps its first 240 characters, then `~`,
//! then the first 15 hex digits of its SHA-256. Map keys and the plain names
//! above are never cut.
//!
//! `models` keeps at most 16 roles and `mcp.servers` at most 32 servers, in
//! sorted key order. `plan` servers take their places first, and servers
//! only the harness reports fill the places left. An entry whose key is
//! longer than 128 characters is dropped. While the compact JSON of the
//! whole object is larger than 128 KiB, MCP server entries are dropped from
//! the end of the sorted order, `harness` entries before `plan` entries,
//! then model roles. Dropped entries are counted in a [`TruncatedEntries`]
//! marker next to the map: `models_truncated` and `mcp.servers_truncated`.
//! If the object is still too large with no MCP server and no model role,
//! it is replaced by a minimal one that holds only the schema name, the
//! version and the `what_ran_truncated` marker.
//!
//! An adapter reports its own facts through the report keys
//! [`VERSIONS_REPORT_KEY`] and [`MCP_REPORT_KEY`] in its result's
//! extensions, once its descriptor declares them. A reported list keeps at
//! most 64 elements in sorted order, counted in a marker beside it, such as
//! `tool_names_truncated`.
//!
//! `what_ran` is evidence. It never decides which runs are compared.
//!
//! Compatibility rules for readers:
//!
//! - `source` and `state` are closed sets. A new value changes
//!   [`WHAT_RAN_VERSION`].
//! - `collected_by`, `method` and `reason` are open. Readers accept values
//!   they do not know.
//! - Readers accept unknown properties. Adding an optional property keeps
//!   [`WHAT_RAN_VERSION`]; removing one or changing its meaning changes it.
//!   The Rust types drop unknown properties when they decode, so a host
//!   that relays a task core's result drops them too.
//! - A fact holds `value`, or `state` and `reason`, never both. A
//!   `RunResult` reader that finds a fact breaking this, any other value
//!   the schema rejects, or another schema or version reads `what_ran` as
//!   absent and still decodes the result.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::de::{DeserializeOwned, Error as _};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::adapter_contract::AdapterExtensionPoint;
use crate::config::{McpServerPlan, McpTransport, ModelConfig, RunPlan};
use crate::endpoint::{endpoint_identity, executable_name};

/// Schema name written into every [`WhatRan`].
pub const WHAT_RAN_SCHEMA: &str = "nemo.fabric.what_ran";

/// Contract version written into every [`WhatRan`].
pub const WHAT_RAN_VERSION: &str = "v1alpha1";

/// How NeMo Fabric learned a fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FactSource {
    /// NeMo Fabric core saw it itself.
    Observed,
    /// The adapter, the harness, an MCP server or a remote service said so.
    Reported,
    /// It comes from the resolved run plan.
    Configured,
}

/// Why a fact has no value. A collector picks the first state that matches,
/// in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FactState {
    /// The fact cannot exist for this run.
    NotApplicable,
    /// The adapter or environment provider has no source for the fact.
    Unsupported,
    /// A source exists, but it failed, timed out, returned an invalid report
    /// or never returned.
    CollectionFailed,
    /// NeMo Fabric holds no value and no evidence for the other states.
    Unavailable,
}

/// Pattern of the open codes `collected_by`, `method` and `reason`.
const CODE_PATTERN: &str = "^[a-z0-9_.:#+/-]{1,128}$";

/// Who collected a fact and how.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FactOrigin {
    /// How NeMo Fabric learned the fact.
    pub source: FactSource,
    /// Component that collected the fact, such as `host_core`, `task_core`
    /// or `adapter`.
    #[schemars(extend("pattern" = CODE_PATTERN))]
    pub collected_by: String,
    /// How the collector got the fact, such as `crate_version`,
    /// `resolved_plan` or `adapter_report`.
    #[schemars(extend("pattern" = CODE_PATTERN))]
    pub method: String,
}

/// A fact with a value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(
    rename = "KnownFact_for_{T}",
    extend("not" = {"anyOf": [{"required": ["state"]}, {"required": ["reason"]}]})
)]
pub struct KnownFact<T> {
    /// The fact itself: a string, a boolean or a list of strings.
    pub value: T,
    /// Who collected the fact and how.
    #[serde(flatten)]
    pub origin: FactOrigin,
}

/// A fact without a value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(extend("not" = {"required": ["value"]}))]
pub struct MissingFact {
    /// Why the fact has no value.
    pub state: FactState,
    /// Short code that explains the state, such as `key_not_declared`.
    #[schemars(extend("pattern" = CODE_PATTERN))]
    pub reason: String,
    /// Who would have collected the fact and how.
    #[serde(flatten)]
    pub origin: FactOrigin,
}

/// One fact: a `value`, or a `state` with a `reason`, never both.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(untagged)]
#[schemars(rename = "Fact_for_{T}")]
pub enum Fact<T> {
    /// The fact has a value.
    Known(KnownFact<T>),
    /// The fact has no value.
    Missing(MissingFact),
}

impl<'de, T: DeserializeOwned> Deserialize<'de> for Fact<T> {
    /// Decode a fact, rejecting one that holds `value` together with
    /// `state` or `reason`, or an open code outside its pattern. Unknown
    /// properties are ignored.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let fact = serde_json::Map::<String, Value>::deserialize(deserializer)?;
        let fact = if fact.contains_key("value") {
            if fact.contains_key("state") || fact.contains_key("reason") {
                return Err(D::Error::custom(
                    "a fact holds `value`, or `state` and `reason`, never both",
                ));
            }
            KnownFact::deserialize(Value::Object(fact)).map(Self::Known)
        } else {
            MissingFact::deserialize(Value::Object(fact)).map(Self::Missing)
        }
        .map_err(D::Error::custom)?;
        let origin = fact.origin();
        let reason = match &fact {
            Self::Known(_) => None,
            Self::Missing(missing) => Some(missing.reason.as_str()),
        };
        if [origin.collected_by.as_str(), origin.method.as_str()]
            .into_iter()
            .chain(reason)
            .all(is_code)
        {
            Ok(fact)
        } else {
            Err(D::Error::custom(format!(
                "`collected_by`, `method` and `reason` must match {CODE_PATTERN}"
            )))
        }
    }
}

/// Whether `code` matches [`CODE_PATTERN`].
fn is_code(code: &str) -> bool {
    (1..=128).contains(&code.len())
        && code.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'_' | b'.' | b':' | b'#' | b'+' | b'/' | b'-')
        })
}

/// Entries core dropped to keep `what_ran` within its bounds: next to a
/// map for the entries dropped from it, or as `what_ran_truncated` for the
/// whole object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct TruncatedEntries {
    /// Why the entries were dropped: `key_too_long` for a key longer than
    /// 128 characters, `entry_cap` for an entry past the map's cap, or
    /// `byte_budget` for an entry dropped to keep the whole object within
    /// 128 KiB. One marker covers a map: when more than one rule dropped
    /// entries from it, `reason` names the last rule applied, in that
    /// order, and `dropped` counts them all. An open code like a fact's
    /// `reason`.
    #[schemars(extend("pattern" = CODE_PATTERN))]
    pub reason: String,
    /// How many entries were dropped. A JSON integer: the Rust and Python
    /// readers reject `1.0`, although a JSON Schema validator may accept it.
    #[schemars(range(min = 1, max = u64::MAX))]
    pub dropped: u64,
}

impl<'de> Deserialize<'de> for TruncatedEntries {
    /// Decode a marker, rejecting a `reason` outside the code pattern or a
    /// `dropped` count of zero. Unknown properties are ignored.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Marker {
            reason: String,
            dropped: u64,
        }
        let Marker { reason, dropped } = Marker::deserialize(deserializer)?;
        if !is_code(&reason) {
            return Err(D::Error::custom(format!(
                "a truncation `reason` must match {CODE_PATTERN}"
            )));
        }
        if dropped == 0 {
            return Err(D::Error::custom(
                "a truncation `dropped` count is at least 1",
            ));
        }
        Ok(Self { reason, dropped })
    }
}

/// What executed for one invocation.
///
/// Every section is present, unless `what_ran_truncated` is present: then
/// the object is a minimal one that the byte budget left with only its
/// schema name and version, and every section may be absent.
///
/// Decoding a `WhatRan` directly checks the shape of every section and
/// fact, and that no section is `null`. `RunResult` also checks the schema
/// name and version, that every section is present unless
/// `what_ran_truncated` is, and that every `plan` MCP entry names its
/// transport.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(extend("anyOf" = [
    {"required": ["what_ran_truncated"]},
    {"required": ["fabric", "adapter", "harness", "models", "mcp", "environment"]}
]))]
pub struct WhatRan {
    /// Schema name, `nemo.fabric.what_ran`.
    #[schemars(extend("const" = WHAT_RAN_SCHEMA))]
    pub schema: String,
    /// Contract version, `v1alpha1`.
    #[schemars(extend("const" = WHAT_RAN_VERSION))]
    pub version: String,
    /// NeMo Fabric core that ran the adapter.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(with = "FabricFacts")]
    pub fabric: Option<FabricFacts>,
    /// Adapter that ran.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(with = "AdapterFacts")]
    pub adapter: Option<AdapterFacts>,
    /// Harness behind the adapter.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(with = "HarnessFacts")]
    pub harness: Option<HarnessFacts>,
    /// Model facts keyed by the role names of the plan's `models` settings.
    /// At most 16 roles, in sorted key order, each key at most 128
    /// characters.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(with = "BTreeMap<String, ModelFacts>")]
    pub models: Option<BTreeMap<String, ModelFacts>>,
    /// Model roles dropped from `models`. Absent when none was dropped,
    /// never `null`.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(with = "TruncatedEntries")]
    pub models_truncated: Option<TruncatedEntries>,
    /// MCP server facts.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(with = "McpFacts")]
    pub mcp: Option<McpFacts>,
    /// Environment the runtime ran in.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(with = "EnvironmentFacts")]
    pub environment: Option<EnvironmentFacts>,
    /// Present only on a minimal object: every section was dropped to keep
    /// the object within 128 KiB, with reason `byte_budget` and a dropped
    /// count of 1.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(with = "TruncatedEntries")]
    pub what_ran_truncated: Option<TruncatedEntries>,
}

/// Decode a property that may be absent but is never `null`.
fn present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

/// NeMo Fabric core facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FabricFacts {
    /// Version of the core that ran the adapter.
    pub core_version: Fact<String>,
    /// Version of the host core that relayed the result from a task core
    /// in another environment. `not_applicable` for a local run.
    pub host_core_version: Fact<String>,
}

/// Adapter facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AdapterFacts {
    /// Adapter ID from the plan, when an adapter was resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Version of the adapter package, as the adapter reports it.
    pub package_version: Fact<String>,
}

/// Harness facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct HarnessFacts {
    /// Name of the harness program that ran, as the adapter reports it.
    /// Core never derives it from the adapter ID.
    pub name: Fact<String>,
    /// Version of the harness program that ran, as the adapter reports it.
    pub version: Fact<String>,
}

/// Facts about one model role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ModelFacts {
    /// Provider from the plan.
    pub provider: String,
    /// Model the plan requested.
    pub requested: Fact<String>,
    /// Model that served, as the harness or provider reports it.
    pub served: Fact<String>,
}

/// MCP facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct McpFacts {
    /// Servers keyed by name. Empty when the plan configures no MCP server
    /// and the harness reports none. At most 32 servers, in sorted key
    /// order, each key at most 128 characters.
    pub servers: BTreeMap<String, McpServerFacts>,
    /// Servers dropped from `servers`. Absent when none was dropped, never
    /// `null`.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(with = "TruncatedEntries")]
    pub servers_truncated: Option<TruncatedEntries>,
}

/// Facts about one MCP server.
///
/// An entry with origin `harness` is a server the harness reports and the
/// plan does not configure. It has the same shape: its `endpoint` reads
/// `not_applicable` with reason `not_configured`, and runtime facts the
/// adapter did not collect read `unavailable` with reason `not_collected`.
/// Only a `plan` entry must name its transport. An entry of any other
/// origin may leave it out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[schemars(extend(
    "if" = {"properties": {"origin": {"const": ORIGIN_PLAN}}},
    "then" = {"required": ["transport"], "properties": {"transport": {"type": "string"}}}
))]
pub struct McpServerFacts {
    /// Where the entry comes from: `plan` for a server the plan configures,
    /// `harness` for one only the harness reports.
    pub origin: String,
    /// Transport, such as `stdio` or `streamable-http`. Required for a
    /// `plan` entry, and optional for any other origin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<String>,
    /// Configured endpoint without secrets: the lowercase scheme and host
    /// and the port for a network server, or the executable's file name for
    /// a `stdio` server.
    pub endpoint: Fact<String>,
    /// What the harness learned when it connected, as the adapter reports it.
    pub runtime: McpRuntimeFacts,
}

/// What the harness learned about one MCP server when it connected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct McpRuntimeFacts {
    /// Whether the harness connected to the server.
    pub connected: Fact<bool>,
    /// MCP protocol version negotiated with the server.
    pub protocol_version: Fact<String>,
    /// Name the server reports for itself.
    pub server_name: Fact<String>,
    /// Version the server reports for itself.
    pub server_version: Fact<String>,
    /// Raw tool names the server lists, each bounded by the string length
    /// rule. At most 64 names, in sorted order.
    pub tool_names: Fact<Vec<String>>,
    /// Tool names dropped from `tool_names`, by the adapter's helper or by
    /// core. Absent when none was dropped, never `null`.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    #[schemars(with = "TruncatedEntries")]
    pub tool_names_truncated: Option<TruncatedEntries>,
}

/// Environment facts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EnvironmentFacts {
    /// Environment provider, such as `local`.
    pub provider: String,
}

const HOST_CORE: &str = "host_core";
const TASK_CORE: &str = "task_core";
const ADAPTER: &str = "adapter";
const ORIGIN_PLAN: &str = "plan";
const METHOD_CRATE_VERSION: &str = "crate_version";
const METHOD_RESOLVED_PLAN: &str = "resolved_plan";
const METHOD_ADAPTER_REPORT: &str = "adapter_report";
const REASON_KEY_NOT_DECLARED: &str = "key_not_declared";
const REASON_LOCAL_RUN: &str = "local_run";
const REASON_NOT_CONFIGURED: &str = "not_configured";
const REASON_ENDPOINT_UNPARSED: &str = "endpoint_unparsed";
const REASON_EXECUTABLE_UNPARSED: &str = "executable_unparsed";
const REASON_ENTRY_CAP: &str = "entry_cap";
const REASON_KEY_TOO_LONG: &str = "key_too_long";
const REASON_BYTE_BUDGET: &str = "byte_budget";

/// Most model roles `what_ran` keeps.
const MAX_MODEL_ROLES: usize = 16;
/// Most MCP servers `what_ran` keeps.
const MAX_MCP_SERVERS: usize = 32;
/// Longest map key `what_ran` keeps, in characters.
const MAX_KEY_CHARS: usize = 128;
/// Largest compact JSON of a whole `what_ran`, in bytes.
const MAX_WHAT_RAN_BYTES: usize = 128 * 1024;

/// Longest string value kept whole, in characters.
const MAX_VALUE_CHARS: usize = 256;
/// Characters a longer string value keeps before `~` and its digest prefix.
const TRUNCATED_PREFIX_CHARS: usize = 240;
/// Hex digits of the SHA-256 that end a truncated string value.
const TRUNCATED_DIGEST_HEX_DIGITS: usize = 15;
const _: () = assert!(TRUNCATED_PREFIX_CHARS + 1 + TRUNCATED_DIGEST_HEX_DIGITS == MAX_VALUE_CHARS);

impl Fact<String> {
    /// A known string fact, with the string length rule applied.
    fn text(value: &str, source: FactSource, collected_by: &str, method: &str) -> Self {
        Self::Known(KnownFact {
            value: bounded_text(value),
            origin: FactOrigin {
                source,
                collected_by: collected_by.to_string(),
                method: method.to_string(),
            },
        })
    }
}

impl<T> Fact<T> {
    fn origin(&self) -> &FactOrigin {
        match self {
            Self::Known(known) => &known.origin,
            Self::Missing(missing) => &missing.origin,
        }
    }

    fn origin_mut(&mut self) -> &mut FactOrigin {
        match self {
            Self::Known(known) => &mut known.origin,
            Self::Missing(missing) => &mut missing.origin,
        }
    }

    fn missing(
        state: FactState,
        reason: &str,
        source: FactSource,
        collected_by: &str,
        method: &str,
    ) -> Self {
        Self::Missing(MissingFact {
            state,
            reason: reason.to_string(),
            origin: FactOrigin {
                source,
                collected_by: collected_by.to_string(),
                method: method.to_string(),
            },
        })
    }

    /// An adapter-reported fact while core reads no adapter report key.
    fn report_key_not_declared() -> Self {
        reported_missing(FactState::Unsupported, REASON_KEY_NOT_DECLARED)
    }
}

impl WhatRan {
    /// The origin of every fact, so a host can relabel the facts a task
    /// core collected.
    fn origins_mut(&mut self) -> Vec<&mut FactOrigin> {
        let Self {
            fabric,
            adapter,
            harness,
            models,
            mcp,
            ..
        } = self;
        let mut origins = Vec::new();
        if let Some(fabric) = fabric {
            origins.extend([
                fabric.core_version.origin_mut(),
                fabric.host_core_version.origin_mut(),
            ]);
        }
        if let Some(adapter) = adapter {
            origins.push(adapter.package_version.origin_mut());
        }
        if let Some(harness) = harness {
            origins.extend([harness.name.origin_mut(), harness.version.origin_mut()]);
        }
        for model in models.iter_mut().flat_map(BTreeMap::values_mut) {
            origins.extend([model.requested.origin_mut(), model.served.origin_mut()]);
        }
        for server in mcp.iter_mut().flat_map(|mcp| mcp.servers.values_mut()) {
            let runtime = &mut server.runtime;
            origins.extend([
                server.endpoint.origin_mut(),
                runtime.connected.origin_mut(),
                runtime.protocol_version.origin_mut(),
                runtime.server_name.origin_mut(),
                runtime.server_version.origin_mut(),
                runtime.tool_names.origin_mut(),
            ]);
        }
        origins
    }

    /// Keep the compact JSON of the whole object within 128 KiB. While it
    /// is larger, drop the last MCP server in sorted key order, a server of
    /// any other origin before a `plan` server, then the last model role,
    /// counting each in its map's marker with reason `byte_budget`. With
    /// neither left and the object still too large, replace it with
    /// [`WhatRan::minimal`].
    fn apply_byte_budget(&mut self) {
        while self.serialized_len() > MAX_WHAT_RAN_BYTES {
            if let Some(mcp) = self.mcp.as_mut()
                && pop_last_server(&mut mcp.servers).is_some()
            {
                mcp.servers_truncated = Some(one_more_byte_budget_drop(&mcp.servers_truncated));
            } else if let Some(models) = self.models.as_mut()
                && models.pop_last().is_some()
            {
                self.models_truncated = Some(one_more_byte_budget_drop(&self.models_truncated));
            } else {
                *self = Self::minimal();
                return;
            }
        }
    }

    /// Apply the entry caps and the key rule to an object another core
    /// collected, adding what they drop to the existing markers.
    fn apply_entry_caps(&mut self) {
        if let Some(models) = self.models.take() {
            let (kept, dropped) = bounded_entries(&models, MAX_MODEL_ROLES, ModelFacts::clone);
            self.models = Some(kept);
            self.models_truncated = merged_marker(self.models_truncated.take(), dropped);
        }
        if let Some(mcp) = self.mcp.as_mut() {
            let (kept, dropped) = bounded_servers(&mcp.servers);
            mcp.servers = kept;
            mcp.servers_truncated = merged_marker(mcp.servers_truncated.take(), dropped);
        }
    }

    /// The object left when every section had to go: the schema name, the
    /// version and a `what_ran_truncated` marker.
    fn minimal() -> Self {
        Self {
            schema: WHAT_RAN_SCHEMA.to_string(),
            version: WHAT_RAN_VERSION.to_string(),
            fabric: None,
            adapter: None,
            harness: None,
            models: None,
            models_truncated: None,
            mcp: None,
            environment: None,
            what_ran_truncated: Some(TruncatedEntries {
                reason: REASON_BYTE_BUDGET.to_string(),
                dropped: 1,
            }),
        }
    }

    fn serialized_len(&self) -> usize {
        serde_json::to_vec(self)
            .expect("what_ran has only string keys and is always JSON serializable")
            .len()
    }

    /// Whether every `plan` MCP entry names its transport. An entry of any
    /// other origin may leave it out.
    fn plan_servers_have_transports(&self) -> bool {
        self.mcp
            .iter()
            .flat_map(|mcp| mcp.servers.values())
            .all(|server| server.origin != ORIGIN_PLAN || server.transport.is_some())
    }

    /// Whether every section is present, or the object is a minimal one.
    fn has_every_section_unless_truncated(&self) -> bool {
        self.what_ran_truncated.is_some()
            || (self.fabric.is_some()
                && self.adapter.is_some()
                && self.harness.is_some()
                && self.models.is_some()
                && self.mcp.is_some()
                && self.environment.is_some())
    }
}

/// Collect the facts the core that runs the adapter knows before it sends
/// the request: its own version and the plan's adapter ID, models, MCP
/// servers and environment provider. The model and MCP server maps follow
/// the entry caps and the byte budget, and collection never fails.
///
/// Every adapter-reported fact reads `unsupported` with reason
/// `key_not_declared` until [`ingest_adapter_reports`] reads the report keys
/// the result carries. The collecting core
/// runs the adapter in a local environment, so `host_core_version` reads
/// `not_applicable`. A host that relays the result from a task core calls
/// [`attribute_to_task_core`].
pub(crate) fn collect_before_request(
    plan: &RunPlan,
    adapter_id: Option<String>,
    environment_provider: &str,
) -> WhatRan {
    let (models, models_truncated) =
        bounded_entries(&plan.config.models, MAX_MODEL_ROLES, model_facts);
    let (servers, servers_truncated) = bounded_entries(
        &plan.capability_plan.mcp_servers,
        MAX_MCP_SERVERS,
        mcp_server_facts,
    );
    let mut what_ran = WhatRan {
        schema: WHAT_RAN_SCHEMA.to_string(),
        version: WHAT_RAN_VERSION.to_string(),
        fabric: Some(FabricFacts {
            core_version: core_version_fact(),
            host_core_version: Fact::missing(
                FactState::NotApplicable,
                REASON_LOCAL_RUN,
                FactSource::Observed,
                HOST_CORE,
                METHOD_CRATE_VERSION,
            ),
        }),
        adapter: Some(AdapterFacts {
            id: adapter_id,
            package_version: Fact::report_key_not_declared(),
        }),
        harness: Some(HarnessFacts {
            name: Fact::report_key_not_declared(),
            version: Fact::report_key_not_declared(),
        }),
        models: Some(models),
        models_truncated,
        mcp: Some(McpFacts {
            servers,
            servers_truncated,
        }),
        environment: Some(EnvironmentFacts {
            provider: environment_provider.to_string(),
        }),
        what_ran_truncated: None,
    };
    what_ran.apply_byte_budget();
    what_ran
}

/// The facts of at most `cap` entries, taken in sorted key order, and a
/// marker that counts the dropped entries.
///
/// An entry whose key is longer than 128 characters is dropped and takes no
/// place under the cap. The marker reason is `entry_cap` when the cap
/// dropped an entry, and `key_too_long` when only long keys were dropped.
/// [`BTreeMap`] iteration gives the sorted key order, whatever order the
/// plan's entries were written in.
fn bounded_entries<P, F>(
    entries: &BTreeMap<String, P>,
    cap: usize,
    facts: impl Fn(&P) -> F,
) -> (BTreeMap<String, F>, Option<TruncatedEntries>) {
    let mut kept = BTreeMap::new();
    let (mut key_too_long, mut over_cap) = (0_u64, 0_u64);
    for (key, entry) in entries {
        if key.chars().count() > MAX_KEY_CHARS {
            key_too_long += 1;
        } else if kept.len() == cap {
            over_cap += 1;
        } else {
            kept.insert(key.clone(), facts(entry));
        }
    }
    let reason = if over_cap > 0 {
        REASON_ENTRY_CAP
    } else {
        REASON_KEY_TOO_LONG
    };
    let dropped = key_too_long + over_cap;
    let truncated = (dropped > 0).then(|| TruncatedEntries {
        reason: reason.to_string(),
        dropped,
    });
    (kept, truncated)
}

/// At most 32 MCP servers, `plan` servers first: the cap and the key rule
/// of [`bounded_entries`] apply to the `plan` servers, and servers of any
/// other origin, such as `harness`, fill only the places left, so a report
/// never pushes out a server the plan configures. One marker counts every
/// dropped server.
fn bounded_servers(
    servers: &BTreeMap<String, McpServerFacts>,
) -> (BTreeMap<String, McpServerFacts>, Option<TruncatedEntries>) {
    let (plan, others): (BTreeMap<_, _>, BTreeMap<_, _>) = servers
        .iter()
        .map(|(name, server)| (name.clone(), server.clone()))
        .partition(|(_, server)| server.origin == ORIGIN_PLAN);
    let (mut kept, plan_dropped) = bounded_entries(&plan, MAX_MCP_SERVERS, McpServerFacts::clone);
    let (others_kept, others_dropped) =
        bounded_entries(&others, MAX_MCP_SERVERS - kept.len(), McpServerFacts::clone);
    kept.extend(others_kept);
    let dropped = match (plan_dropped, others_dropped) {
        (Some(plan), Some(others)) => Some(TruncatedEntries {
            reason: if plan.reason == REASON_ENTRY_CAP {
                plan.reason
            } else {
                others.reason
            },
            dropped: plan.dropped.saturating_add(others.dropped),
        }),
        (plan, others) => plan.or(others),
    };
    (kept, dropped)
}

/// Remove the last server in sorted key order whose origin is not `plan`,
/// or the last server when every one is a `plan` server.
fn pop_last_server(servers: &mut BTreeMap<String, McpServerFacts>) -> Option<McpServerFacts> {
    let name = servers
        .iter()
        .rev()
        .find(|(_, server)| server.origin != ORIGIN_PLAN)
        .or_else(|| servers.last_key_value())
        .map(|(name, _)| name.clone())?;
    servers.remove(&name)
}

/// `marker` with one more dropped entry and reason `byte_budget`.
fn one_more_byte_budget_drop(marker: &Option<TruncatedEntries>) -> TruncatedEntries {
    TruncatedEntries {
        reason: REASON_BYTE_BUDGET.to_string(),
        dropped: marker
            .as_ref()
            .map_or(0, |marker| marker.dropped)
            .saturating_add(1),
    }
}

/// An `earlier` marker followed by a `later` one: the later reason, and the
/// two counts added. A task core can send any count, so the sum saturates.
fn merged_marker(
    earlier: Option<TruncatedEntries>,
    later: Option<TruncatedEntries>,
) -> Option<TruncatedEntries> {
    match (earlier, later) {
        (Some(earlier), Some(later)) => Some(TruncatedEntries {
            reason: later.reason,
            dropped: earlier.dropped.saturating_add(later.dropped),
        }),
        (earlier, later) => later.or(earlier),
    }
}

fn core_version_fact() -> Fact<String> {
    Fact::text(
        crate::version(),
        FactSource::Observed,
        HOST_CORE,
        METHOD_CRATE_VERSION,
    )
}

fn model_facts(model: &ModelConfig) -> ModelFacts {
    ModelFacts {
        provider: model.provider.clone(),
        requested: Fact::text(
            &model.model,
            FactSource::Configured,
            HOST_CORE,
            METHOD_RESOLVED_PLAN,
        ),
        served: Fact::report_key_not_declared(),
    }
}

fn mcp_server_facts(server: &McpServerPlan) -> McpServerFacts {
    McpServerFacts {
        origin: ORIGIN_PLAN.to_string(),
        transport: Some(server.transport.as_str().to_string()),
        endpoint: configured_endpoint(server.transport, &server.url),
        runtime: McpRuntimeFacts {
            connected: Fact::report_key_not_declared(),
            protocol_version: Fact::report_key_not_declared(),
            server_name: Fact::report_key_not_declared(),
            server_version: Fact::report_key_not_declared(),
            tool_names: Fact::report_key_not_declared(),
            tool_names_truncated: None,
        },
    }
}

/// The configured endpoint of one MCP server, without secrets. `url` holds
/// the server URL, or the command for a `stdio` server.
///
/// An empty `url` reads `unavailable` with reason `not_configured`. A value
/// core cannot reduce to its safe form reads `collection_failed`, and the
/// raw value never enters the fact.
fn configured_endpoint(transport: McpTransport, url: &str) -> Fact<String> {
    let missing = |state, reason| {
        Fact::missing(
            state,
            reason,
            FactSource::Configured,
            HOST_CORE,
            METHOD_RESOLVED_PLAN,
        )
    };
    if url.trim().is_empty() {
        return missing(FactState::Unavailable, REASON_NOT_CONFIGURED);
    }
    let endpoint = match transport {
        McpTransport::Stdio => executable_name(url).ok_or(REASON_EXECUTABLE_UNPARSED),
        McpTransport::Sse | McpTransport::StreamableHttp => {
            endpoint_identity(url).ok_or(REASON_ENDPOINT_UNPARSED)
        }
    };
    match endpoint {
        Ok(endpoint) => Fact::text(
            &endpoint,
            FactSource::Configured,
            HOST_CORE,
            METHOD_RESOLVED_PLAN,
        ),
        Err(reason) => missing(FactState::CollectionFailed, reason),
    }
}

/// Attribute a task core's `what_ran` on the host that relayed its result.
///
/// The task core runs in a projected `local` environment and labels its
/// own facts `host_core`. The host relabels them `task_core`, records its
/// own version as `host_core_version`, and replaces the environment
/// provider with its own, such as `openshell`. A minimal object stays
/// minimal. The task core may run another Fabric version, and the host's
/// own facts can grow the object, so the entry caps, the key rule and the
/// byte budget apply again.
pub(crate) fn attribute_to_task_core(what_ran: &mut WhatRan, environment_provider: &str) {
    for origin in what_ran.origins_mut() {
        if origin.collected_by == HOST_CORE {
            origin.collected_by = TASK_CORE.to_string();
        }
    }
    if let Some(fabric) = what_ran.fabric.as_mut() {
        fabric.host_core_version = core_version_fact();
    }
    if let Some(environment) = what_ran.environment.as_mut() {
        environment.provider = environment_provider.to_string();
    }
    what_ran.apply_entry_caps();
    what_ran.apply_byte_budget();
}

/// Report key for the adapter package version, the harness name and version,
/// and the served model of each role.
pub const VERSIONS_REPORT_KEY: &str = "nemo_fabric.versions.v1";
/// Report key for the `runtime` block of each MCP server, and for servers
/// the harness reports that the plan does not configure.
pub const MCP_REPORT_KEY: &str = "nemo_fabric.mcp.v1";

/// Largest compact JSON of one report key, in bytes.
const MAX_REPORT_BYTES: usize = 64 * 1024;
/// Most elements a list fact keeps.
const MAX_LIST_ENTRIES: usize = 64;
const ORIGIN_HARNESS: &str = "harness";
const REASON_KEY_OMITTED: &str = "key_omitted";
const REASON_FACT_OMITTED: &str = "fact_omitted";
const REASON_INVALID_REPORT: &str = "invalid_report";
const REASON_TOO_LARGE: &str = "too_large";

/// Move the adapter's declared report keys from a result's extensions into
/// `what_ran`.
///
/// Core calls this after the descriptor's extension schema accepted
/// `extensions`, so ingestion never rejects a result. A key is declared
/// only when the descriptor's `run_result` extension schema names it
/// literally under `properties`; the facts of an undeclared key keep
/// `unsupported` with reason `key_not_declared`, and the key stays in the
/// extensions. A declared key that is absent reads `unavailable` with
/// reason `key_omitted`, and one that is not an object reads
/// `collection_failed` with reason `invalid_report` and stays in the
/// extensions. A declared object is removed from the extensions. Past
/// 64 KiB of compact JSON, every fact of that key reads `collection_failed`
/// with reason `too_large`.
///
/// Inside a report, a key may hold only adapter-owned facts, each with
/// source `reported` and collector `adapter`: the adapter package version,
/// the harness name and version, each plan role's served model, each MCP
/// server's `runtime` block with its `tool_names_truncated` marker, a
/// `harness` server's transport, and the `servers_truncated` marker beside
/// `servers`, which counts only harness-only servers the helper dropped. A malformed or forged fact reads
/// `collection_failed` with reason `invalid_report`, and an omitted one
/// reads `unavailable` with reason `fact_omitted`. Core ignores every other
/// property, so a report never changes a fact core owns. The string length
/// rule, the 64-entry cap on `tool_names`, the entry caps and the byte
/// budget then apply to the result.
pub(crate) fn ingest_adapter_reports(
    what_ran: &mut WhatRan,
    plan: &RunPlan,
    extensions: &mut BTreeMap<String, Value>,
) {
    let run_result_schema = plan.adapter_descriptor.as_ref().and_then(|adapter| {
        adapter
            .descriptor
            .extension_schemas
            .get(&AdapterExtensionPoint::RunResult)
    });
    for key in [VERSIONS_REPORT_KEY, MCP_REPORT_KEY] {
        if !declares_report_key(run_result_schema, key) {
            continue;
        }
        let removed = match extensions.get(key) {
            Some(Value::Object(_)) => extensions.remove(key),
            _ => None,
        };
        let report = match &removed {
            Some(Value::Object(object)) if compact_json_len(object) > MAX_REPORT_BYTES => {
                Report::missing(FactState::CollectionFailed, REASON_TOO_LARGE)
            }
            Some(Value::Object(object)) => Report::Object(object),
            _ if extensions.contains_key(key) => {
                Report::missing(FactState::CollectionFailed, REASON_INVALID_REPORT)
            }
            _ => Report::missing(FactState::Unavailable, REASON_KEY_OMITTED),
        };
        if key == VERSIONS_REPORT_KEY {
            apply_versions_report(what_ran, &report);
        } else {
            apply_mcp_report(what_ran, plan, &report);
        }
    }
    what_ran.apply_entry_caps();
    what_ran.apply_byte_budget();
}

fn compact_json_len(object: &serde_json::Map<String, Value>) -> usize {
    serde_json::to_vec(object)
        .expect("a JSON object is always serializable")
        .len()
}

/// Whether a `run_result` extension schema names `key` literally under
/// `properties`. Core never resolves `$ref`, `allOf` or patterns.
fn declares_report_key(schema: Option<&serde_json::Map<String, Value>>, key: &str) -> bool {
    schema
        .and_then(|schema| schema.get("properties"))
        .and_then(Value::as_object)
        .is_some_and(|properties| properties.contains_key(key))
}

/// One object of an adapter report, or the state every fact under it reads
/// when the object is missing or malformed.
enum Report<'a> {
    Object(&'a serde_json::Map<String, Value>),
    Missing(MissingFact),
}

impl<'a> Report<'a> {
    fn missing(state: FactState, reason: &str) -> Self {
        match reported_missing::<()>(state, reason) {
            Fact::Missing(missing) => Self::Missing(missing),
            Fact::Known(_) => unreachable!("reported_missing builds a missing fact"),
        }
    }

    /// The object under `key`: `fact_omitted` when absent, `invalid_report`
    /// when not an object.
    fn child(&self, key: &str) -> Report<'a> {
        match self {
            Self::Object(object) => match object.get(key) {
                Some(Value::Object(child)) => Report::Object(child),
                Some(_) => Report::missing(FactState::CollectionFailed, REASON_INVALID_REPORT),
                None => Report::missing(FactState::Unavailable, REASON_FACT_OMITTED),
            },
            Self::Missing(missing) => Report::Missing(missing.clone()),
        }
    }

    /// The reported fact under `key`, kept only when it decodes and names
    /// source `reported` and collector `adapter`.
    fn fact<T: DeserializeOwned>(&self, key: &str) -> Fact<T> {
        let value = match self {
            Self::Missing(missing) => return Fact::Missing(missing.clone()),
            Self::Object(object) => object.get(key),
        };
        match value.map(Fact::<T>::deserialize) {
            None => reported_missing(FactState::Unavailable, REASON_FACT_OMITTED),
            Some(Ok(fact))
                if fact.origin().source == FactSource::Reported
                    && fact.origin().collected_by == ADAPTER =>
            {
                fact
            }
            Some(_) => reported_missing(FactState::CollectionFailed, REASON_INVALID_REPORT),
        }
    }
}

/// A reported fact without a value, collected by the adapter's report.
fn reported_missing<T>(state: FactState, reason: &str) -> Fact<T> {
    Fact::missing(
        state,
        reason,
        FactSource::Reported,
        ADAPTER,
        METHOD_ADAPTER_REPORT,
    )
}

fn apply_versions_report(what_ran: &mut WhatRan, report: &Report<'_>) {
    if let Some(adapter) = what_ran.adapter.as_mut() {
        adapter.package_version = bounded_string(report.child("adapter").fact("package_version"));
    }
    if let Some(harness) = what_ran.harness.as_mut() {
        let reported = report.child("harness");
        harness.name = bounded_string(reported.fact("name"));
        harness.version = bounded_string(reported.fact("version"));
    }
    let models = report.child("models");
    for (role, model) in what_ran.models.iter_mut().flatten() {
        model.served = bounded_string(models.child(role).fact("served"));
    }
}

/// Fill each plan server's `runtime` block, and add a `harness` entry for
/// each reported server the plan does not configure.
///
/// When a helper drops server entries to keep `nemo_fabric.mcp.v1` under
/// 64 KiB, the report's `servers_truncated` counts only dropped names absent
/// from the full capability plan supplied at lifecycle start. Core adds that
/// count to `what_ran.mcp.servers_truncated`. A dropped plan server keeps its
/// entry, with runtime facts marked `unavailable` for reason `fact_omitted`.
/// A malformed marker
/// makes the whole `servers` report read `invalid_report`, as a malformed
/// `tool_names_truncated` does for `tool_names`.
fn apply_mcp_report(what_ran: &mut WhatRan, plan: &RunPlan, report: &Report<'_>) {
    let Some(mcp) = what_ran.mcp.as_mut() else {
        return;
    };
    let mut servers = report.child("servers");
    let mut helper_marker = None;
    if let (Report::Object(_), Report::Object(object)) = (&servers, report)
        && let Some(marker) = object.get("servers_truncated")
    {
        match TruncatedEntries::deserialize(marker) {
            Ok(marker) => helper_marker = Some(marker),
            Err(_) => {
                servers = Report::missing(FactState::CollectionFailed, REASON_INVALID_REPORT);
            }
        }
    }
    if let Report::Object(_) = servers {
        mcp.servers_truncated = merged_marker(mcp.servers_truncated.take(), helper_marker);
    }
    let plan_servers = &plan.capability_plan.mcp_servers;
    for (name, server) in &mut mcp.servers {
        server.runtime = runtime_facts(&servers.child(name).child("runtime"));
    }
    let Report::Object(reported) = servers else {
        return;
    };
    for (name, entry) in reported {
        if plan_servers.contains_key(name) {
            continue;
        }
        let entry_report = servers.child(name);
        mcp.servers.insert(
            name.clone(),
            McpServerFacts {
                origin: ORIGIN_HARNESS.to_string(),
                transport: entry
                    .get("transport")
                    .and_then(Value::as_str)
                    .map(str::to_lowercase),
                endpoint: Fact::missing(
                    FactState::NotApplicable,
                    REASON_NOT_CONFIGURED,
                    FactSource::Configured,
                    HOST_CORE,
                    METHOD_RESOLVED_PLAN,
                ),
                runtime: runtime_facts(&entry_report.child("runtime")),
            },
        );
    }
}

/// One server's `runtime` block from its report. `tool_names` keeps at most
/// 64 names in sorted order, and its marker adds core's drops to the
/// helper's.
fn runtime_facts(report: &Report<'_>) -> McpRuntimeFacts {
    let (mut tool_names, core_marker) = bounded_list(report.fact("tool_names"));
    let mut helper_marker = None;
    if let (Fact::Known(_), Report::Object(object)) = (&tool_names, report)
        && let Some(marker) = object.get("tool_names_truncated")
    {
        match TruncatedEntries::deserialize(marker) {
            Ok(marker) => helper_marker = Some(marker),
            Err(_) => {
                tool_names = reported_missing(FactState::CollectionFailed, REASON_INVALID_REPORT);
            }
        }
    }
    let tool_names_truncated = match tool_names {
        Fact::Known(_) => merged_marker(helper_marker, core_marker),
        Fact::Missing(_) => None,
    };
    McpRuntimeFacts {
        connected: report.fact("connected"),
        protocol_version: bounded_string(report.fact("protocol_version")),
        server_name: bounded_string(report.fact("server_name")),
        server_version: bounded_string(report.fact("server_version")),
        tool_names,
        tool_names_truncated,
    }
}

/// `fact` with the string length rule applied to its value.
fn bounded_string(fact: Fact<String>) -> Fact<String> {
    match fact {
        Fact::Known(known) => Fact::Known(KnownFact {
            value: bounded_text(&known.value),
            origin: known.origin,
        }),
        missing => missing,
    }
}

/// `fact` with each element bounded by the string length rule, sorted by
/// code point, and cut to 64 elements, with a marker that counts the cut.
fn bounded_list(fact: Fact<Vec<String>>) -> (Fact<Vec<String>>, Option<TruncatedEntries>) {
    let Fact::Known(known) = fact else {
        return (fact, None);
    };
    let mut value: Vec<String> = known.value.iter().map(|item| bounded_text(item)).collect();
    value.sort();
    let dropped = value.len().saturating_sub(MAX_LIST_ENTRIES);
    value.truncate(MAX_LIST_ENTRIES);
    let marker = (dropped > 0).then(|| TruncatedEntries {
        reason: REASON_ENTRY_CAP.to_string(),
        dropped: dropped as u64,
    });
    (
        Fact::Known(KnownFact {
            value,
            origin: known.origin,
        }),
        marker,
    )
}

/// `text` whole when it has at most 256 characters. A longer string keeps
/// its first 240 characters, then `~`, then the first 15 hex digits of the
/// whole string's SHA-256, so two long values collide only on a 60-bit
/// hash collision. Characters are Unicode scalar values, and the digest
/// covers the UTF-8 bytes.
fn bounded_text(text: &str) -> String {
    if text.chars().count() <= MAX_VALUE_CHARS {
        return text.to_string();
    }
    let mut bounded: String = text.chars().take(TRUNCATED_PREFIX_CHARS).collect();
    bounded.push('~');
    let digest_hex: String = Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    bounded.push_str(&digest_hex[..TRUNCATED_DIGEST_HEX_DIGITS]);
    bounded
}

/// Decode `RunResult.what_ran`, reading it as absent when it breaks the
/// schema or names another schema or version, so the rest of the result
/// still decodes.
pub(crate) fn deserialize_lenient<'de, D>(deserializer: D) -> Result<Option<WhatRan>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value
        .and_then(|value| serde_json::from_value::<WhatRan>(value).ok())
        .filter(|what_ran| {
            what_ran.schema == WHAT_RAN_SCHEMA
                && what_ran.version == WHAT_RAN_VERSION
                && what_ran.has_every_section_unless_truncated()
                && what_ran.plan_servers_have_transports()
        }))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use super::*;
    use crate::config::{
        FabricConfig, ResolveContext, resolve_run_plan_from_config_with_adapter_directories,
    };

    fn repository_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn resolve_plan(config: Value) -> RunPlan {
        let root = repository_root();
        let config: FabricConfig = serde_json::from_value(config).expect("typed config");
        resolve_run_plan_from_config_with_adapter_directories(
            config,
            ResolveContext::new(&root),
            &[root.join("adapters")],
        )
        .expect("resolve plan")
    }

    fn config_for(adapter_id: &str) -> Value {
        json!({
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "what-ran-agent"},
            "harness": {"adapter_id": adapter_id},
            "runtime": {}
        })
    }

    /// A Claude Code plan whose MCP servers hold a secret in every place a
    /// configured server can: URL user info, path, query and fragment,
    /// headers, and a `stdio` command's folder, arguments and environment.
    fn plan_with_secrets() -> RunPlan {
        let mut config = config_for("nvidia.fabric.claude");
        config["models"] = json!({
            "default": {"provider": "anthropic", "model": "claude-sonnet-4-5"},
        });
        config["mcp"] = json!({"servers": {
            "github": {
                "transport": "streamable-http",
                "url": "https://user:SECRET_PASSWORD@API.GitHubCopilot.com/mcp/SECRET_PATH/?token=SECRET_QUERY#SECRET_FRAGMENT",
                "custom_headers": {"Authorization": "Bearer SECRET_HEADER"},
                "exposure": "harness_native"
            },
            "events": {
                "transport": "sse",
                "url": "http://localhost:8931/sse",
                "exposure": "harness_native"
            },
            "files": {
                "transport": "stdio",
                "url": "/home/SECRET_USER/.venv/bin/files-mcp",
                "args": ["--api-key", "SECRET_ARGUMENT"],
                "env": {"FILES_TOKEN": "SECRET_ENVIRONMENT"},
                "exposure": "harness_native"
            }
        }});
        resolve_plan(config)
    }

    fn collect(plan: &RunPlan) -> Value {
        let adapter_id = plan
            .adapter_descriptor
            .as_ref()
            .map(|adapter| adapter.descriptor.adapter_id.clone());
        serde_json::to_value(collect_before_request(plan, adapter_id, "local"))
            .expect("serialize what_ran")
    }

    fn not_declared() -> Value {
        json!({"state": "unsupported", "reason": "key_not_declared", "source": "reported",
               "collected_by": "adapter", "method": "adapter_report"})
    }

    fn configured(value: &str) -> Value {
        json!({"value": value, "source": "configured", "collected_by": "host_core",
               "method": "resolved_plan"})
    }

    fn mcp_server(transport: &str, endpoint: Value) -> Value {
        let missing = not_declared();
        json!({
            "origin": "plan",
            "transport": transport,
            "endpoint": endpoint,
            "runtime": {
                "connected": missing,
                "protocol_version": missing,
                "server_name": missing,
                "server_version": missing,
                "tool_names": missing
            }
        })
    }

    /// Every JSON object that carries `source` is a fact.
    fn facts(value: &Value) -> Vec<&serde_json::Map<String, Value>> {
        match value {
            Value::Object(map) if map.contains_key("source") => vec![map],
            Value::Object(map) => map.values().flat_map(facts).collect(),
            _ => Vec::new(),
        }
    }

    #[test]
    fn collected_facts_read_the_plan_and_mark_adapter_facts_key_not_declared() {
        let what_ran = collect(&plan_with_secrets());
        let not_declared = not_declared();

        assert_eq!(
            what_ran,
            json!({
                "schema": "nemo.fabric.what_ran",
                "version": "v1alpha1",
                "fabric": {
                    "core_version": {"value": crate::version(), "source": "observed",
                        "collected_by": "host_core", "method": "crate_version"},
                    "host_core_version": {"state": "not_applicable", "reason": "local_run",
                        "source": "observed", "collected_by": "host_core",
                        "method": "crate_version"}
                },
                "adapter": {"id": "nvidia.fabric.claude", "package_version": not_declared},
                "harness": {"name": not_declared, "version": not_declared},
                "models": {"default": {
                    "provider": "anthropic",
                    "requested": configured("claude-sonnet-4-5"),
                    "served": not_declared
                }},
                "mcp": {"servers": {
                    "events": mcp_server("sse", configured("http://localhost:8931")),
                    "files": mcp_server("stdio", configured("files-mcp")),
                    "github": mcp_server(
                        "streamable-http",
                        configured("https://api.githubcopilot.com:443"),
                    )
                }},
                "environment": {"provider": "local"}
            })
        );
    }

    #[test]
    fn collected_facts_hold_no_configured_secret() {
        let text = serde_json::to_string(&collect(&plan_with_secrets())).expect("JSON");

        assert!(!text.contains("SECRET"), "{text}");
    }

    #[test]
    fn every_fact_has_a_value_or_a_state_with_a_reason_never_both() {
        let what_ran = collect(&plan_with_secrets());
        let facts = facts(&what_ran);

        // 2 fabric + 1 adapter + 2 harness + 2 model + 3 servers x (endpoint + 5 runtime).
        assert_eq!(facts.len(), 25);
        for fact in facts {
            assert_ne!(
                fact.contains_key("value"),
                fact.contains_key("state"),
                "{fact:?}"
            );
            assert_eq!(
                fact.contains_key("state"),
                fact.contains_key("reason"),
                "{fact:?}"
            );
            for key in ["collected_by", "method"] {
                assert!(fact.get(key).is_some_and(Value::is_string), "{fact:?}");
            }
        }
    }

    /// `plan_with_secrets` with `models` and MCP servers under the given
    /// keys, each a copy of its `default` model or its `files` server.
    fn plan_with_entries(model_roles: &[String], server_names: &[String]) -> RunPlan {
        let mut plan = plan_with_secrets();
        let model = plan.config.models["default"].clone();
        let server = plan.capability_plan.mcp_servers["files"].clone();
        plan.config.models = model_roles
            .iter()
            .map(|role| (role.clone(), model.clone()))
            .collect();
        plan.capability_plan.mcp_servers = server_names
            .iter()
            .map(|name| (name.clone(), server.clone()))
            .collect();
        plan
    }

    fn numbered_keys(prefix: &str, count: usize) -> Vec<String> {
        (0..count)
            .map(|index| format!("{prefix}-{index:02}"))
            .collect()
    }

    fn map_keys(map: &Value) -> Vec<String> {
        map.as_object().expect("map").keys().cloned().collect()
    }

    #[test]
    fn models_past_16_roles_keep_the_first_in_sorted_order_and_count_the_rest() {
        let mut roles = numbered_keys("role", 17);
        roles.reverse();

        let what_ran = collect(&plan_with_entries(&roles, &[]));

        assert_eq!(map_keys(&what_ran["models"]), numbered_keys("role", 16));
        assert_eq!(
            what_ran["models_truncated"],
            json!({"reason": "entry_cap", "dropped": 1})
        );
        assert!(what_ran["mcp"].get("servers_truncated").is_none());
        assert!(what_ran_validator().is_valid(&what_ran), "{what_ran}");
    }

    #[test]
    fn mcp_servers_past_32_keep_the_first_in_sorted_order_and_count_the_rest() {
        let what_ran = collect(&plan_with_entries(&[], &numbered_keys("server", 35)));

        assert_eq!(
            map_keys(&what_ran["mcp"]["servers"]),
            numbered_keys("server", 32)
        );
        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "entry_cap", "dropped": 3})
        );
        assert!(what_ran.get("models_truncated").is_none());
        assert!(what_ran_validator().is_valid(&what_ran), "{what_ran}");
    }

    #[test]
    fn model_roles_past_the_cap_keep_the_code_point_order_the_helpers_share() {
        let vectors: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/what_ran/role-order.json"
        ))
        .expect("role order JSON");
        let roles: Vec<String> = serde_json::from_value(vectors["roles"].clone()).expect("roles");
        let expected: Vec<String> = serde_json::from_value(vectors["kept"].clone()).expect("kept");

        let what_ran = collect(&plan_with_entries(&roles, &[]));

        assert_eq!(map_keys(&what_ran["models"]), expected);
        assert_eq!(
            what_ran["models_truncated"],
            json!({"reason": "entry_cap", "dropped": 1})
        );
    }

    #[test]
    fn keys_longer_than_128_characters_are_dropped_without_taking_a_place() {
        let longest = "é".repeat(128);
        let too_long = "é".repeat(129);
        let mut roles = numbered_keys("role", 15);
        roles.extend([longest.clone(), too_long.clone()]);
        let mut expected_roles = numbered_keys("role", 15);
        expected_roles.push(longest.clone());

        let what_ran = collect(&plan_with_entries(&roles, &[longest, too_long]));

        assert_eq!(map_keys(&what_ran["models"]), expected_roles);
        assert_eq!(
            what_ran["models_truncated"],
            json!({"reason": "key_too_long", "dropped": 1})
        );
        assert_eq!(map_keys(&what_ran["mcp"]["servers"]), ["é".repeat(128)]);
        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "key_too_long", "dropped": 1})
        );
        assert!(what_ran_validator().is_valid(&what_ran), "{what_ran}");
    }

    #[test]
    fn a_long_key_and_the_cap_together_count_every_dropped_entry_as_entry_cap() {
        let mut roles = numbered_keys("role", 17);
        roles.push("r".repeat(129));

        let what_ran = collect(&plan_with_entries(&roles, &[]));

        assert_eq!(map_keys(&what_ran["models"]), numbered_keys("role", 16));
        assert_eq!(
            what_ran["models_truncated"],
            json!({"reason": "entry_cap", "dropped": 2})
        );
    }

    #[test]
    fn retained_entries_do_not_depend_on_the_order_the_plan_lists_them() {
        let roles = numbered_keys("role", 20);
        let servers = numbered_keys("server", 40);
        let mut reversed_roles = roles.clone();
        reversed_roles.reverse();
        let mut reversed_servers = servers.clone();
        reversed_servers.reverse();

        assert_eq!(
            collect(&plan_with_entries(&roles, &servers)),
            collect(&plan_with_entries(&reversed_roles, &reversed_servers))
        );
    }

    /// A plan whose 16 model providers take most of the 128 KiB budget.
    fn plan_near_the_byte_budget(server_names: &[String]) -> RunPlan {
        let mut plan = plan_with_entries(&numbered_keys("role", 16), server_names);
        for model in plan.config.models.values_mut() {
            model.provider = "p".repeat(7_000);
        }
        plan
    }

    fn json_len(value: &Value) -> usize {
        serde_json::to_vec(value).expect("JSON").len()
    }

    #[test]
    fn mcp_servers_past_the_byte_budget_are_dropped_from_the_end_of_sorted_order() {
        let servers = numbered_keys("server", 32);

        let what_ran = collect(&plan_near_the_byte_budget(&servers));

        assert!(json_len(&what_ran) <= 128 * 1024, "{}", json_len(&what_ran));
        let kept = map_keys(&what_ran["mcp"]["servers"]);
        assert!(!kept.is_empty() && kept.len() < servers.len(), "{kept:?}");
        assert_eq!(kept, servers[..kept.len()]);
        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "byte_budget", "dropped": servers.len() - kept.len()})
        );
        assert_eq!(map_keys(&what_ran["models"]).len(), 16);
        assert!(what_ran_validator().is_valid(&what_ran));

        // Keeping the next server in sorted order would exceed the budget.
        let full = collect(&plan_with_entries(&[], &servers));
        let mut one_more = what_ran.clone();
        let next = &servers[kept.len()];
        one_more["mcp"]["servers"][next] = full["mcp"]["servers"][next].clone();
        assert!(json_len(&one_more) > 128 * 1024);
    }

    #[test]
    fn the_byte_budget_counts_servers_the_entry_cap_already_dropped() {
        let within_cap = collect(&plan_near_the_byte_budget(&numbered_keys("server", 32)));
        let budget_dropped = within_cap["mcp"]["servers_truncated"]["dropped"]
            .as_u64()
            .expect("dropped count");

        let what_ran = collect(&plan_near_the_byte_budget(&numbered_keys("server", 35)));

        assert_eq!(what_ran["mcp"]["servers"], within_cap["mcp"]["servers"]);
        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "byte_budget", "dropped": budget_dropped + 3})
        );
    }

    /// A plan with the given model roles and MCP servers whose providers
    /// together exceed the 128 KiB budget.
    fn plan_over_the_byte_budget(roles: &[String], server_names: &[String]) -> RunPlan {
        let mut plan = plan_with_entries(roles, server_names);
        for model in plan.config.models.values_mut() {
            model.provider = "p".repeat(9_000);
        }
        plan
    }

    #[test]
    fn model_roles_are_dropped_for_the_byte_budget_after_every_mcp_server() {
        let roles = numbered_keys("role", 18);
        let servers = numbered_keys("server", 3);

        let what_ran = collect(&plan_over_the_byte_budget(&roles, &servers));

        assert!(json_len(&what_ran) <= 128 * 1024, "{}", json_len(&what_ran));
        assert_eq!(what_ran["mcp"]["servers"], json!({}));
        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "byte_budget", "dropped": 3})
        );
        let kept = map_keys(&what_ran["models"]);
        assert!(!kept.is_empty() && kept.len() < 16, "{kept:?}");
        assert_eq!(kept, roles[..kept.len()]);
        // Two roles past the cap, then every role the budget dropped.
        assert_eq!(
            what_ran["models_truncated"],
            json!({"reason": "byte_budget", "dropped": roles.len() - kept.len()})
        );
        assert!(what_ran_validator().is_valid(&what_ran));

        // Keeping the next role in sorted order would exceed the budget.
        let full = collect(&plan_with_entries(&roles, &[]));
        let mut one_more = what_ran.clone();
        let next = &roles[kept.len()];
        one_more["models"][next] = full["models"][next].clone();
        one_more["models"][next]["provider"] = json!("p".repeat(9_000));
        assert!(json_len(&one_more) > 128 * 1024);
    }

    #[test]
    fn an_object_over_the_byte_budget_without_maps_to_drop_becomes_minimal() {
        let plan = plan_with_entries(&numbered_keys("role", 2), &numbered_keys("server", 2));
        let adapter_id = "a".repeat(140_000);

        let what_ran =
            serde_json::to_value(collect_before_request(&plan, Some(adapter_id), "local"))
                .expect("serialize what_ran");

        assert_eq!(
            what_ran,
            json!({
                "schema": "nemo.fabric.what_ran",
                "version": "v1alpha1",
                "what_ran_truncated": {"reason": "byte_budget", "dropped": 1}
            })
        );
        assert!(what_ran_validator().is_valid(&what_ran));
        let decoded = run_result_with(what_ran)
            .what_ran
            .expect("a minimal what_ran reads as present");
        assert_eq!(decoded.fabric, None);
    }

    #[test]
    fn plan_without_models_or_mcp_servers_reads_empty_maps() {
        let what_ran = collect(&resolve_plan(config_for("nvidia.fabric.claude")));

        assert_eq!(what_ran["models"], json!({}));
        assert_eq!(what_ran["mcp"], json!({"servers": {}}));
    }

    /// Every adapter-reported fact in `what_ran`, by its JSON path.
    fn reported_facts(value: &Value, path: &str) -> Vec<(String, Value)> {
        match value {
            Value::Object(map) if map.get("source") == Some(&json!("reported")) => {
                vec![(path.to_string(), value.clone())]
            }
            Value::Object(map) => map
                .iter()
                .flat_map(|(key, value)| reported_facts(value, &format!("{path}.{key}")))
                .collect(),
            _ => Vec::new(),
        }
    }

    #[test]
    fn a_plan_from_each_bundled_descriptor_collects_plan_facts_and_undeclared_adapter_facts() {
        let catalog_path = repository_root().join(
            "sdk/python/nemo-fabric-adapter-catalog/src/nemo_fabric_adapter_catalog/catalog.json",
        );
        let catalog: Value =
            serde_json::from_str(&std::fs::read_to_string(catalog_path).expect("read catalog"))
                .expect("catalog JSON");
        let adapters = catalog["adapters"].as_object().expect("catalog adapters");
        assert!(!adapters.is_empty());
        let validator = what_ran_validator();
        for (adapter_id, entry) in adapters {
            let descriptor = &entry["descriptor"];
            let mut config = config_for(adapter_id);
            for setting in descriptor["settings_schema"]["required"]
                .as_array()
                .into_iter()
                .flatten()
            {
                let setting = setting.as_str().expect("setting name");
                config["harness"]["settings"][setting] = json!("https://agent.example.com");
            }
            let accepts = |field: &str| {
                descriptor["config"]["accepts"]
                    .as_array()
                    .is_some_and(|accepts| accepts.contains(&json!(field)))
            };
            // A model each adapter's model schema accepts: its fixed provider
            // if it has one, and every connection field the schema names.
            let model_fields = &descriptor["model_schema"]["properties"];
            let provider = model_fields["provider"]["const"]
                .as_str()
                .unwrap_or("nvidia");
            config["models"] =
                json!({"default": {"provider": provider, "model": "nvidia/nemotron"}});
            for (field, value) in [
                ("base_url", "https://inference.example.com/v1"),
                ("api_key_env", "MODEL_API_KEY"),
            ] {
                if model_fields.get(field).is_some() {
                    config["models"]["default"][field] = model_fields[field]
                        .get("const")
                        .cloned()
                        .unwrap_or_else(|| json!(value));
                }
            }
            let accepts_mcp = accepts("mcp");
            if accepts_mcp {
                config["mcp"] = json!({"servers": {
                    "events": {"transport": "sse", "url": "http://localhost:8931/sse",
                               "exposure": "harness_native"},
                    "filesystem": {"transport": "stdio", "url": "npx",
                                   "args": ["-y", "@scope/server-filesystem"],
                                   "exposure": "harness_native"},
                    "search": {"transport": "streamable-http", "url": "https://mcp.example.com/mcp",
                               "exposure": "harness_native"}
                }});
            }

            // Core discovers the repository's adapters as its bundled set.
            let config: FabricConfig = serde_json::from_value(config).expect("typed config");
            let plan = crate::config::resolve_run_plan_from_config(
                config,
                ResolveContext::new(repository_root()),
            )
            .unwrap_or_else(|error| panic!("{adapter_id}: {error}"));
            let what_ran = collect(&plan);

            assert!(validator.is_valid(&what_ran), "{adapter_id}: {what_ran}");
            assert_eq!(what_ran["adapter"]["id"], json!(adapter_id));
            assert_eq!(
                what_ran["fabric"]["core_version"]["value"],
                crate::version()
            );
            assert_eq!(
                what_ran["models"]["default"]["requested"],
                configured("nvidia/nemotron")
            );
            let servers = if accepts_mcp { 3 } else { 0 };
            assert_eq!(
                map_keys(&what_ran["mcp"]["servers"]).len(),
                servers,
                "{adapter_id}"
            );
            // package version, harness name and version, served, 5 per server.
            let reported = reported_facts(&what_ran, "what_ran");
            assert_eq!(reported.len(), 4 + 5 * servers, "{adapter_id}");
            for (path, fact) in reported {
                assert_eq!(fact, not_declared(), "{adapter_id} {path}");
            }
            for marker in [
                &what_ran["models_truncated"],
                &what_ran["mcp"]["servers_truncated"],
                &what_ran["what_ran_truncated"],
            ] {
                assert!(marker.is_null(), "{adapter_id}: {what_ran}");
            }
            assert!(json_len(&what_ran) <= 128 * 1024, "{adapter_id}");
        }
    }

    #[test]
    fn both_entry_caps_with_max_length_ascii_keys_and_models_fit_the_byte_budget() {
        let longest_key = |prefix: &str, index: usize| {
            let key = format!("{prefix}-{index:02}-");
            format!("{key}{}", "k".repeat(128 - key.len()))
        };
        let roles: Vec<String> = (0..16).map(|index| longest_key("role", index)).collect();
        let servers: Vec<String> = (0..32).map(|index| longest_key("server", index)).collect();
        let mut plan = plan_with_entries(&roles, &servers);
        for model in plan.config.models.values_mut() {
            model.model = "m".repeat(256);
        }

        let what_ran = collect(&plan);

        assert_eq!(map_keys(&what_ran["models"]), roles);
        assert_eq!(map_keys(&what_ran["mcp"]["servers"]), servers);
        assert!(what_ran.get("models_truncated").is_none(), "{what_ran}");
        assert!(
            what_ran["mcp"].get("servers_truncated").is_none(),
            "{what_ran}"
        );
        assert!(json_len(&what_ran) <= 128 * 1024, "{}", json_len(&what_ran));
    }

    #[test]
    fn stdio_executable_with_whitespace_reads_unparsed_and_keeps_no_secret() {
        let mut config = config_for("nvidia.fabric.claude");
        config["mcp"] = json!({"servers": {
            "arguments": {
                "transport": "stdio",
                "url": "npx --token a/SECRET",
                "exposure": "harness_native"
            },
            "spaced": {
                "transport": "stdio",
                "url": "/opt/My Tools/db-server",
                "exposure": "harness_native"
            }
        }});

        let what_ran = collect(&resolve_plan(config));

        let unparsed = json!({"state": "collection_failed", "reason": "executable_unparsed",
            "source": "configured", "collected_by": "host_core", "method": "resolved_plan"});
        for name in ["arguments", "spaced"] {
            assert_eq!(
                what_ran["mcp"]["servers"][name]["endpoint"], unparsed,
                "{name}"
            );
        }
        let text = serde_json::to_string(&what_ran).expect("JSON");
        assert!(!text.contains("SECRET"), "{text}");
    }

    #[test]
    fn unparsed_or_empty_endpoint_reads_a_state_without_the_raw_value() {
        for (transport, url, state, reason) in [
            (
                McpTransport::StreamableHttp,
                "https://u:p/SECRET@host/x",
                "collection_failed",
                "endpoint_unparsed",
            ),
            (
                McpTransport::Stdio,
                "TOKEN=prefix/SECRET",
                "collection_failed",
                "executable_unparsed",
            ),
            // Placeholders a harness expands later, not a file name.
            (
                McpTransport::Stdio,
                "$MCP_COMMAND",
                "collection_failed",
                "executable_unparsed",
            ),
            (
                McpTransport::Stdio,
                "/opt/mcp/server-{env:MCP_VARIANT}",
                "collection_failed",
                "executable_unparsed",
            ),
            (McpTransport::Sse, " ", "unavailable", "not_configured"),
            (McpTransport::Stdio, "", "unavailable", "not_configured"),
        ] {
            let endpoint = serde_json::to_value(configured_endpoint(transport, url))
                .expect("serialize endpoint");

            assert_eq!(
                endpoint,
                json!({"state": state, "reason": reason, "source": "configured",
                       "collected_by": "host_core", "method": "resolved_plan"}),
                "{url}"
            );
        }
    }

    #[test]
    fn string_values_longer_than_256_characters_end_with_a_digest_prefix() {
        // The vectors the Python and TypeScript helpers also read. Expected
        // digests are the first 15 hex digits of `sha256sum` over the whole
        // value's UTF-8 bytes.
        let vectors: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/what_ran/string-length.json"
        ))
        .expect("string length JSON");
        let cases = vectors["cases"].as_array().expect("cases");
        assert_eq!(cases.len(), 3);
        for case in cases {
            let text = case["text"].as_str().expect("text");
            let bounded = bounded_text(text);

            assert_eq!(bounded, case["expected"].as_str().expect("expected"));
            assert_eq!(bounded.chars().count(), text.chars().count().min(256));
        }
    }

    fn run_result_with(what_ran: Value) -> crate::RunResult {
        serde_json::from_value(json!({
            "agent_name": "agent", "harness": "nvidia.fabric.codex", "adapter_kind": "python",
            "runtime_id": "runtime-1", "invocation_id": "invocation-1",
            "request_id": "request-1", "status": "succeeded", "what_ran": what_ran
        }))
        .expect("the result decodes whatever what_ran holds")
    }

    /// The reader vectors the Python SDK tests share, with each case's
    /// patch applied to the base object.
    fn reader_vectors(group: &str) -> Vec<(String, Value)> {
        let vectors: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/what_ran/reader-vectors.json"
        ))
        .expect("reader vectors JSON");
        vectors[group]
            .as_array()
            .expect("vector group")
            .iter()
            .map(|case| {
                let mut what_ran = vectors["base"].clone();
                for operation in case["patch"].as_array().expect("patch") {
                    apply_patch(&mut what_ran, operation);
                }
                (case["name"].as_str().expect("name").to_string(), what_ran)
            })
            .collect()
    }

    fn apply_patch(target: &mut Value, operation: &Value) {
        let path: Vec<&str> = operation["path"]
            .as_array()
            .expect("path")
            .iter()
            .map(|key| key.as_str().expect("key"))
            .collect();
        let Some((last, parents)) = path.split_last() else {
            *target = operation["value"].clone();
            return;
        };
        let parent = parents
            .iter()
            .fold(target, |value, key| &mut value[*key])
            .as_object_mut()
            .expect("patch parent is an object");
        if operation.get("remove").is_some() {
            parent.remove(*last);
        } else {
            parent.insert((*last).to_string(), operation["value"].clone());
        }
    }

    fn what_ran_validator() -> jsonschema::Validator {
        let schema = crate::schema::generate_schema(crate::schema::SchemaName::RunResult)
            .expect("run result schema");
        let mut what_ran_schema = schema["properties"]["what_ran"].clone();
        what_ran_schema["$defs"] = schema["$defs"].clone();
        jsonschema::validator_for(&what_ran_schema).expect("valid what_ran schema")
    }

    #[test]
    fn reader_and_schema_keep_every_valid_vector() {
        let validator = what_ran_validator();
        for (name, what_ran) in reader_vectors("valid") {
            assert!(validator.is_valid(&what_ran), "{name}");
            let decoded = run_result_with(what_ran.clone()).what_ran;
            assert!(decoded.is_some(), "{name}");
        }
    }

    #[test]
    fn reader_reads_every_invalid_vector_as_absent_and_schema_rejects_it() {
        let validator = what_ran_validator();
        for (name, what_ran) in reader_vectors("invalid") {
            assert!(!validator.is_valid(&what_ran), "{name}");
            assert_eq!(run_result_with(what_ran).what_ran, None, "{name}");
        }
        assert_eq!(run_result_with(Value::Null).what_ran, None);
    }

    #[test]
    fn collected_facts_match_the_schema() {
        let what_ran = collect(&plan_with_secrets());

        assert!(what_ran_validator().is_valid(&what_ran), "{what_ran}");
    }

    #[test]
    fn run_result_accepts_unknown_properties_and_open_codes() {
        let (_, newer) = reader_vectors("valid")
            .into_iter()
            .find(|(name, _)| name == "unknown properties and vendor codes")
            .expect("vector");

        let what_ran = run_result_with(newer)
            .what_ran
            .expect("unknown properties never fail what_ran");

        assert_eq!(
            what_ran
                .fabric
                .expect("fabric section")
                .core_version
                .origin()
                .collected_by,
            "acme.collector"
        );
    }

    /// The secrets plan with a descriptor that declares `keys` in its
    /// `run_result` extension schema.
    fn plan_declaring(keys: &[&str]) -> RunPlan {
        let mut plan = plan_with_secrets();
        let properties: serde_json::Map<String, Value> = keys
            .iter()
            .map(|key| (key.to_string(), json!({"type": "object"})))
            .collect();
        let schema = json!({"type": "object", "properties": properties});
        plan.adapter_descriptor
            .as_mut()
            .expect("adapter descriptor")
            .descriptor
            .extension_schemas
            .insert(
                AdapterExtensionPoint::RunResult,
                schema.as_object().expect("schema object").clone(),
            );
        plan
    }

    fn reported(value: Value, method: &str) -> Value {
        json!({"value": value, "source": "reported", "collected_by": "adapter",
               "method": method})
    }

    fn report_state(state: &str, reason: &str) -> Value {
        json!({"state": state, "reason": reason, "source": "reported",
               "collected_by": "adapter", "method": "adapter_report"})
    }

    /// Ingest `extensions` into the facts collected for `plan`, and return
    /// `what_ran` with the extensions core leaves on the result.
    fn ingest(plan: &RunPlan, extensions: Value) -> (Value, Value) {
        let adapter_id = plan
            .adapter_descriptor
            .as_ref()
            .map(|adapter| adapter.descriptor.adapter_id.clone());
        let mut what_ran = collect_before_request(plan, adapter_id, "local");
        let mut extensions: BTreeMap<String, Value> =
            serde_json::from_value(extensions).expect("extensions map");
        ingest_adapter_reports(&mut what_ran, plan, &mut extensions);
        let what_ran = serde_json::to_value(what_ran).expect("serialize what_ran");
        assert!(what_ran_validator().is_valid(&what_ran), "{what_ran}");
        (
            what_ran,
            serde_json::to_value(extensions).expect("extensions JSON"),
        )
    }

    fn versions_report() -> Value {
        json!({
            "adapter": {"package_version": reported(json!("0.6.0"), "package_metadata")},
            "harness": {
                "name": reported(json!("claude-code"), "harness_init_message"),
                "version": reported(json!("2.1.294"), "harness_init_message")
            },
            "models": {"default": {"served": reported(json!("claude-sonnet-4-5-20250929"),
                                                      "harness_api")}}
        })
    }

    #[test]
    fn a_declared_versions_report_fills_the_adapter_facts_and_leaves_the_extensions() {
        let plan = plan_declaring(&[VERSIONS_REPORT_KEY]);

        let (what_ran, extensions) = ingest(
            &plan,
            json!({VERSIONS_REPORT_KEY: versions_report(), "trace_id": "trace-1"}),
        );

        assert_eq!(
            what_ran["adapter"]["package_version"],
            reported(json!("0.6.0"), "package_metadata")
        );
        assert_eq!(
            what_ran["harness"],
            json!({"name": reported(json!("claude-code"), "harness_init_message"),
                   "version": reported(json!("2.1.294"), "harness_init_message")})
        );
        assert_eq!(
            what_ran["models"]["default"]["served"],
            reported(json!("claude-sonnet-4-5-20250929"), "harness_api")
        );
        assert_eq!(extensions, json!({"trace_id": "trace-1"}));
        // The MCP key is not declared, so its facts keep `key_not_declared`.
        assert_eq!(
            what_ran["mcp"]["servers"]["github"]["runtime"]["connected"],
            not_declared()
        );
    }

    #[test]
    fn an_undeclared_report_key_stays_in_the_extensions_and_reads_key_not_declared() {
        let plan = plan_declaring(&[]);
        let extensions = json!({VERSIONS_REPORT_KEY: versions_report()});

        let (what_ran, kept) = ingest(&plan, extensions.clone());

        assert_eq!(kept, extensions);
        for fact in [
            &what_ran["adapter"]["package_version"],
            &what_ran["harness"]["name"],
            &what_ran["harness"]["version"],
            &what_ran["models"]["default"]["served"],
        ] {
            assert_eq!(fact, &not_declared());
        }
    }

    #[test]
    fn a_report_key_declared_only_through_a_reference_is_not_declared() {
        let mut plan = plan_declaring(&[]);
        let schema = json!({"type": "object", "allOf": [{"properties": {
            VERSIONS_REPORT_KEY: {"type": "object"}}}]});
        plan.adapter_descriptor
            .as_mut()
            .expect("adapter descriptor")
            .descriptor
            .extension_schemas
            .insert(
                AdapterExtensionPoint::RunResult,
                schema.as_object().expect("schema").clone(),
            );

        let (what_ran, kept) = ingest(&plan, json!({VERSIONS_REPORT_KEY: versions_report()}));

        assert_eq!(what_ran["harness"]["version"], not_declared());
        assert_eq!(kept, json!({VERSIONS_REPORT_KEY: versions_report()}));
    }

    #[test]
    fn an_omitted_non_object_or_oversized_report_key_marks_every_fact_of_that_key() {
        let plan = plan_declaring(&[VERSIONS_REPORT_KEY, MCP_REPORT_KEY]);
        let oversized = json!({"harness": {"version": reported(
            json!("v".repeat(MAX_REPORT_BYTES)), "harness_api")}});
        for (extensions, state, reason, kept) in [
            (json!({}), "unavailable", "key_omitted", json!({})),
            (
                json!({VERSIONS_REPORT_KEY: "0.6.0", MCP_REPORT_KEY: []}),
                "collection_failed",
                "invalid_report",
                json!({VERSIONS_REPORT_KEY: "0.6.0", MCP_REPORT_KEY: []}),
            ),
            (
                json!({VERSIONS_REPORT_KEY: oversized, MCP_REPORT_KEY: {"servers": {
                    "github": {"runtime": {"server_name": reported(
                        json!("s".repeat(MAX_REPORT_BYTES)), "harness_status")}}}}}),
                "collection_failed",
                "too_large",
                json!({}),
            ),
        ] {
            let (what_ran, extensions) = ingest(&plan, extensions);

            let expected = report_state(state, reason);
            assert_eq!(extensions, kept, "{reason}");
            for fact in [
                &what_ran["adapter"]["package_version"],
                &what_ran["harness"]["name"],
                &what_ran["harness"]["version"],
                &what_ran["models"]["default"]["served"],
            ] {
                assert_eq!(fact, &expected, "{reason}");
            }
            for server in what_ran["mcp"]["servers"]
                .as_object()
                .expect("servers")
                .values()
            {
                for fact in server["runtime"].as_object().expect("runtime").values() {
                    assert_eq!(fact, &expected, "{reason}");
                }
            }
        }
    }

    #[test]
    fn a_report_at_64_kib_is_read_and_one_byte_more_is_too_large() {
        let plan = plan_declaring(&[VERSIONS_REPORT_KEY]);
        let with_padding = |padding: usize| {
            let mut report = versions_report();
            report["padding"] = json!("");
            let base = serde_json::to_vec(&report).expect("JSON").len();
            report["padding"] = json!("p".repeat(padding - base));
            report
        };

        let (at_limit, _) = ingest(
            &plan,
            json!({VERSIONS_REPORT_KEY: with_padding(MAX_REPORT_BYTES)}),
        );
        let (past_limit, _) = ingest(
            &plan,
            json!({VERSIONS_REPORT_KEY: with_padding(MAX_REPORT_BYTES + 1)}),
        );

        assert_eq!(
            at_limit["harness"]["version"],
            reported(json!("2.1.294"), "harness_init_message")
        );
        assert_eq!(
            past_limit["harness"]["version"],
            report_state("collection_failed", "too_large")
        );
    }

    #[test]
    fn the_origin_rule_rejects_forged_or_malformed_facts_and_keeps_the_rest_of_the_report() {
        let plan = plan_declaring(&[VERSIONS_REPORT_KEY]);
        let forged = json!({"value": "9.9.9", "source": "observed", "collected_by": "adapter",
                            "method": "harness_api"});
        let other_collector = json!({"value": "9.9.9", "source": "reported",
                                     "collected_by": "task_core", "method": "harness_api"});
        let report = json!({
            "fabric": {"core_version": reported(json!("0.0.0-forged"), "crate_version")},
            "adapter": {"id": "forged.adapter",
                        "package_version": {"value": "0.6.0", "state": "unavailable",
                            "reason": "both", "source": "reported", "collected_by": "adapter",
                            "method": "package_metadata"}},
            "harness": {"name": forged, "version": other_collector},
            "models": {
                "default": {"served": reported(json!(42), "harness_api"),
                            "requested": reported(json!("forged"), "harness_api"),
                            "provider": "forged"},
                "unplanned": {"served": reported(json!("other"), "harness_api")}
            }
        });

        let (what_ran, _) = ingest(&plan, json!({VERSIONS_REPORT_KEY: report}));

        let invalid = report_state("collection_failed", "invalid_report");
        assert_eq!(what_ran["adapter"]["package_version"], invalid);
        assert_eq!(what_ran["harness"]["name"], invalid);
        assert_eq!(what_ran["harness"]["version"], invalid);
        assert_eq!(what_ran["models"]["default"]["served"], invalid);
        // Facts core owns keep core's values, and roles stay the plan's.
        assert_eq!(
            what_ran["fabric"]["core_version"]["value"],
            crate::version()
        );
        assert_eq!(what_ran["adapter"]["id"], "nvidia.fabric.claude");
        assert_eq!(
            what_ran["models"]["default"]["requested"],
            configured("claude-sonnet-4-5")
        );
        assert_eq!(what_ran["models"]["default"]["provider"], "anthropic");
        assert_eq!(map_keys(&what_ran["models"]), ["default"]);
    }

    #[test]
    fn omitted_facts_read_fact_omitted_and_a_reported_state_is_kept() {
        let plan = plan_declaring(&[VERSIONS_REPORT_KEY]);
        let no_source = json!({"state": "unsupported", "reason": "no_source",
            "source": "reported", "collected_by": "adapter", "method": "adapter_report"});
        let report = json!({
            "harness": {"name": no_source, "version": no_source},
            "models": "not an object"
        });

        let (what_ran, _) = ingest(&plan, json!({VERSIONS_REPORT_KEY: report}));

        assert_eq!(
            what_ran["adapter"]["package_version"],
            report_state("unavailable", "fact_omitted")
        );
        assert_eq!(what_ran["harness"]["name"], no_source);
        assert_eq!(
            what_ran["models"]["default"]["served"],
            report_state("collection_failed", "invalid_report")
        );
    }

    #[test]
    fn reported_strings_follow_the_length_rule() {
        let plan = plan_declaring(&[VERSIONS_REPORT_KEY]);
        let long = format!("{}a", "m".repeat(300));
        let mut report = versions_report();
        report["harness"]["version"] = reported(json!(long), "harness_api");

        let (what_ran, _) = ingest(&plan, json!({VERSIONS_REPORT_KEY: report}));

        assert_eq!(
            what_ran["harness"]["version"]["value"],
            format!("{}~96858eedeb3c8e6", "m".repeat(240))
        );
    }

    fn tool_names(count: usize) -> Vec<String> {
        // Reverse order, so core's sort decides which names stay.
        (0..count)
            .rev()
            .map(|index| format!("tool-{index:02}"))
            .collect()
    }

    #[test]
    fn mcp_runtime_facts_fill_plan_servers_and_cap_tool_names_at_64() {
        let plan = plan_declaring(&[MCP_REPORT_KEY]);
        let runtime = |names: Vec<String>, marker: Option<Value>| {
            let mut runtime = json!({
                "connected": reported(json!(true), "harness_status"),
                "protocol_version": reported(json!("2025-06-18"), "mcp_initialize"),
                "server_name": reported(json!("github-mcp"), "harness_status"),
                "server_version": reported(json!("1.2.3"), "harness_status"),
                "tool_names": reported(json!(names), "harness_status")
            });
            if let Some(marker) = marker {
                runtime["tool_names_truncated"] = marker;
            }
            runtime
        };
        let report = json!({"servers": {
            "github": {"transport": "stdio", "endpoint": configured("forged"),
                       "runtime": runtime(tool_names(65), None)},
            "events": {"runtime": runtime(tool_names(70),
                       Some(json!({"reason": "byte_budget", "dropped": 3})))},
            "files": {"runtime": runtime(tool_names(64), None)}
        }});

        let (what_ran, extensions) = ingest(&plan, json!({MCP_REPORT_KEY: report}));

        assert_eq!(extensions, json!({}));
        let servers = &what_ran["mcp"]["servers"];
        let github = &servers["github"];
        let kept: Vec<String> = tool_names(64).into_iter().rev().collect();
        assert_eq!(github["runtime"]["tool_names"]["value"], json!(kept));
        assert_eq!(
            github["runtime"]["tool_names_truncated"],
            json!({"reason": "entry_cap", "dropped": 1})
        );
        assert_eq!(
            servers["events"]["runtime"]["tool_names_truncated"],
            json!({"reason": "entry_cap", "dropped": 9})
        );
        assert!(servers["files"]["runtime"]["tool_names_truncated"].is_null());
        // A plan entry keeps its own origin, transport and endpoint.
        assert_eq!(github["origin"], "plan");
        assert_eq!(github["transport"], "streamable-http");
        assert_eq!(
            github["endpoint"],
            configured("https://api.githubcopilot.com:443")
        );
        assert_eq!(
            github["runtime"]["connected"],
            reported(json!(true), "harness_status")
        );
    }

    #[test]
    fn mcp_report_adds_harness_entries_and_marks_omitted_plan_servers() {
        let plan = plan_declaring(&[MCP_REPORT_KEY]);
        let not_collected = json!({"state": "unavailable", "reason": "not_collected",
            "source": "reported", "collected_by": "adapter",
            "method": "harness_startup_status"});
        let long_name = "h".repeat(129);
        let report = json!({"servers": {
            "plugin-docs": {"transport": "STDIO", "runtime": {
                "connected": reported(json!(true), "harness_startup_status"),
                "protocol_version": not_collected,
                "server_name": not_collected,
                "server_version": not_collected,
                "tool_names": not_collected,
                "tool_names_truncated": {"reason": "entry_cap", "dropped": 2}
            }},
            "broken": "not an object",
            long_name: {"runtime": {}}
        }});

        let (what_ran, _) = ingest(&plan, json!({MCP_REPORT_KEY: report}));

        let servers = &what_ran["mcp"]["servers"];
        assert_eq!(
            map_keys(servers),
            ["broken", "events", "files", "github", "plugin-docs"]
        );
        let harness_entry = &servers["plugin-docs"];
        assert_eq!(harness_entry["origin"], "harness");
        assert_eq!(harness_entry["transport"], "stdio");
        assert_eq!(
            harness_entry["endpoint"],
            json!({"state": "not_applicable", "reason": "not_configured",
                   "source": "configured", "collected_by": "host_core",
                   "method": "resolved_plan"})
        );
        assert_eq!(harness_entry["runtime"]["server_name"], not_collected);
        // A marker beside a list without a value is dropped.
        assert!(harness_entry["runtime"]["tool_names_truncated"].is_null());
        let invalid = report_state("collection_failed", "invalid_report");
        assert_eq!(servers["broken"]["runtime"]["connected"], invalid);
        assert!(servers["broken"].get("transport").is_none());
        assert_eq!(
            servers["github"]["runtime"]["connected"],
            report_state("unavailable", "fact_omitted")
        );
        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "key_too_long", "dropped": 1})
        );
    }

    #[test]
    fn a_malformed_helper_marker_invalidates_tool_names() {
        let plan = plan_declaring(&[MCP_REPORT_KEY]);
        let report = json!({"servers": {"github": {"runtime": {
            "tool_names": reported(json!(["a"]), "harness_status"),
            "tool_names_truncated": {"reason": "entry_cap", "dropped": 0}
        }}}});

        let (what_ran, _) = ingest(&plan, json!({MCP_REPORT_KEY: report}));

        let runtime = &what_ran["mcp"]["servers"]["github"]["runtime"];
        assert_eq!(
            runtime["tool_names"],
            report_state("collection_failed", "invalid_report")
        );
        assert!(runtime["tool_names_truncated"].is_null());
    }

    #[test]
    fn a_plan_server_dropped_by_the_entry_cap_never_returns_as_a_harness_entry() {
        let names = numbered_keys("server", 33);
        let mut plan = plan_declaring(&[MCP_REPORT_KEY]);
        let template = plan.capability_plan.mcp_servers["events"].clone();
        plan.capability_plan.mcp_servers = names
            .iter()
            .map(|name| (name.clone(), template.clone()))
            .collect();
        let servers: serde_json::Map<String, Value> = names
            .iter()
            .map(|name| {
                (
                    name.clone(),
                    json!({"runtime": {
                    "connected": reported(json!(true), "harness_status")}}),
                )
            })
            .collect();

        let (what_ran, _) = ingest(&plan, json!({MCP_REPORT_KEY: {"servers": servers}}));

        let servers = what_ran["mcp"]["servers"].as_object().expect("servers");
        assert_eq!(servers.len(), 32);
        assert!(servers.values().all(|server| server["origin"] == "plan"));
        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "entry_cap", "dropped": 1})
        );
    }

    /// A plan declaring the MCP report key, with MCP servers under the given
    /// names and, when `heavy_roles` is set, 16 model roles whose providers
    /// take most of the 128 KiB budget.
    fn mcp_plan(server_names: &[String], heavy_roles: bool) -> RunPlan {
        let mut plan = plan_declaring(&[MCP_REPORT_KEY]);
        let server = plan.capability_plan.mcp_servers["files"].clone();
        plan.capability_plan.mcp_servers = server_names
            .iter()
            .map(|name| (name.clone(), server.clone()))
            .collect();
        if heavy_roles {
            let model = plan.config.models["default"].clone();
            plan.config.models = numbered_keys("role", 16)
                .into_iter()
                .map(|role| (role, model.clone()))
                .collect();
            for model in plan.config.models.values_mut() {
                model.provider = "p".repeat(7_000);
            }
        }
        plan
    }

    /// An MCP report naming each server in `names` with a `runtime` block
    /// that lists `tool_count` long tool names.
    fn mcp_report(names: &[String], tool_count: usize) -> Value {
        let tools: Vec<String> = (0..tool_count)
            .map(|index| format!("{index:03}{}", "t".repeat(250)))
            .collect();
        let servers: serde_json::Map<String, Value> = names
            .iter()
            .map(|name| {
                (
                    name.clone(),
                    json!({"transport": "stdio", "runtime": {
                        "connected": reported(json!(true), "harness_status"),
                        "tool_names": reported(json!(tools), "harness_api")}}),
                )
            })
            .collect();
        json!({MCP_REPORT_KEY: {"servers": servers}})
    }

    fn origins(what_ran: &Value) -> Vec<(String, String)> {
        what_ran["mcp"]["servers"]
            .as_object()
            .expect("servers")
            .iter()
            .map(|(name, server)| {
                (
                    name.clone(),
                    server["origin"].as_str().expect("origin").to_string(),
                )
            })
            .collect()
    }

    #[test]
    fn harness_entries_fill_only_the_places_plan_servers_leave_under_the_cap() {
        let plan_names = numbered_keys("server", 30);
        let harness_names = numbered_keys("aa", 5);

        let (what_ran, _) = ingest(&mcp_plan(&plan_names, false), mcp_report(&harness_names, 0));

        let kept = origins(&what_ran);
        let plan_kept: Vec<_> = kept
            .iter()
            .filter(|(_, origin)| origin == "plan")
            .map(|(name, _)| name.clone())
            .collect();
        let harness_kept: Vec<_> = kept
            .iter()
            .filter(|(_, origin)| origin == "harness")
            .map(|(name, _)| name.clone())
            .collect();
        assert_eq!(plan_kept, plan_names);
        assert_eq!(harness_kept, harness_names[..2]);
        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "entry_cap", "dropped": 3})
        );
        // Every kept plan server keeps the endpoint core configured.
        for name in &plan_names {
            assert_eq!(
                what_ran["mcp"]["servers"][name]["endpoint"]["source"],
                "configured"
            );
        }
    }

    #[test]
    fn the_byte_budget_drops_harness_entries_before_plan_servers() {
        // The plan servers that fit beside 16 heavy model roles.
        let collected = collect(&plan_near_the_byte_budget(&numbered_keys("server", 32)));
        let plan_names = map_keys(&collected["mcp"]["servers"]);
        assert!(!plan_names.is_empty());
        // Harness names sort before every plan server. Each entry is about
        // 10 KiB, so the five cannot all fit, and the report stays under its
        // own 64 KiB limit.
        let harness_names = numbered_keys("aa", 5);

        let (what_ran, _) = ingest(&mcp_plan(&plan_names, true), mcp_report(&harness_names, 40));

        assert!(json_len(&what_ran) <= 128 * 1024, "{}", json_len(&what_ran));
        let kept = origins(&what_ran);
        let plan_kept: Vec<_> = kept
            .iter()
            .filter(|(_, origin)| origin == "plan")
            .map(|(name, _)| name.clone())
            .collect();
        let harness_kept = kept.len() - plan_kept.len();
        assert_eq!(plan_kept, plan_names);
        assert!(harness_kept < harness_names.len(), "{kept:?}");
        assert!(
            json_len(&mcp_report(&harness_names, 40)[MCP_REPORT_KEY]) < 64 * 1024,
            "the report must be ingested, not marked too_large"
        );
        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "byte_budget", "dropped": harness_names.len() - harness_kept})
        );
        assert_eq!(map_keys(&what_ran["models"]).len(), 16);
    }

    fn connected_runtime() -> Value {
        json!({"runtime": {"connected": reported(json!(true), "harness_status")}})
    }

    #[test]
    fn a_helper_servers_marker_adds_its_harness_drops_to_core_marker() {
        let plan = plan_declaring(&[MCP_REPORT_KEY]);
        // Every plan server is reported, so all three helper drops were
        // harness entries.
        let report = json!({
            "servers": {"events": connected_runtime(), "files": connected_runtime(),
                        "github": connected_runtime(), "plugin-docs": connected_runtime()},
            "servers_truncated": {"reason": "byte_budget", "dropped": 3}
        });

        let (what_ran, _) = ingest(&plan, json!({MCP_REPORT_KEY: report}));

        assert_eq!(
            map_keys(&what_ran["mcp"]["servers"]),
            ["events", "files", "github", "plugin-docs"]
        );
        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "byte_budget", "dropped": 3})
        );
    }

    #[test]
    fn a_helper_servers_marker_counts_harness_drops_whatever_plan_servers_the_report_omits() {
        let plan = plan_declaring(&[MCP_REPORT_KEY]);
        // `github` was never reported, and the helper dropped the harness-only
        // `plugin-docs` for its byte budget.
        let report = json!({
            "servers": {"events": connected_runtime(), "files": connected_runtime()},
            "servers_truncated": {"reason": "byte_budget", "dropped": 1}
        });

        let (what_ran, _) = ingest(&plan, json!({MCP_REPORT_KEY: report}));

        assert_eq!(
            map_keys(&what_ran["mcp"]["servers"]),
            ["events", "files", "github"]
        );
        assert_eq!(
            what_ran["mcp"]["servers"]["github"]["runtime"]["connected"],
            report_state("unavailable", "fact_omitted")
        );
        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "byte_budget", "dropped": 1})
        );
    }

    #[test]
    fn a_helper_servers_marker_adds_to_the_marker_core_already_holds() {
        let mut plan = plan_declaring(&[MCP_REPORT_KEY]);
        // Core drops this plan server for its long key when it collects.
        let server = plan.capability_plan.mcp_servers["files"].clone();
        plan.capability_plan
            .mcp_servers
            .insert("s".repeat(129), server);
        let report = json!({
            "servers": {"events": connected_runtime(), "files": connected_runtime(),
                        "github": connected_runtime()},
            "servers_truncated": {"reason": "byte_budget", "dropped": 2}
        });

        let (what_ran, _) = ingest(&plan, json!({MCP_REPORT_KEY: report}));

        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "byte_budget", "dropped": 3})
        );
    }

    #[test]
    fn core_entry_cap_and_byte_budget_drops_follow_a_helper_marker() {
        // Entry cap: 30 plan servers leave two places for five harness entries.
        let plan_names = numbered_keys("server", 30);
        let mut report = mcp_report(&numbered_keys("aa", 5), 0);
        report[MCP_REPORT_KEY]["servers_truncated"] =
            json!({"reason": "byte_budget", "dropped": 2});

        let (what_ran, _) = ingest(&mcp_plan(&plan_names, false), report);

        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "entry_cap", "dropped": 5})
        );

        // Byte budget: harness entries that do not fit add to the helper's
        // count.
        let collected = collect(&plan_near_the_byte_budget(&numbered_keys("server", 32)));
        let plan_names = map_keys(&collected["mcp"]["servers"]);
        let harness_names = numbered_keys("aa", 5);
        let mut report = mcp_report(&harness_names, 40);
        report[MCP_REPORT_KEY]["servers_truncated"] =
            json!({"reason": "byte_budget", "dropped": 4});

        let (what_ran, _) = ingest(&mcp_plan(&plan_names, true), report);

        let harness_kept = origins(&what_ran)
            .iter()
            .filter(|(_, origin)| origin == "harness")
            .count();
        assert!(harness_kept < harness_names.len());
        assert_eq!(
            what_ran["mcp"]["servers_truncated"],
            json!({"reason": "byte_budget",
                   "dropped": 4 + harness_names.len() - harness_kept})
        );
    }

    #[test]
    fn a_malformed_helper_servers_marker_invalidates_the_servers_report() {
        let plan = plan_declaring(&[MCP_REPORT_KEY]);
        for marker in [
            json!({"reason": "byte_budget", "dropped": 0}),
            json!({"reason": "byte_budget"}),
            json!("three"),
            Value::Null,
        ] {
            let report = json!({
                "servers": {"events": connected_runtime(), "plugin-docs": connected_runtime()},
                "servers_truncated": marker
            });

            let (what_ran, _) = ingest(&plan, json!({MCP_REPORT_KEY: report}));

            let servers = &what_ran["mcp"]["servers"];
            assert_eq!(map_keys(servers), ["events", "files", "github"], "{marker}");
            assert_eq!(
                servers["events"]["runtime"]["connected"],
                report_state("collection_failed", "invalid_report"),
                "{marker}"
            );
            assert!(
                what_ran["mcp"].get("servers_truncated").is_none(),
                "{marker}"
            );
        }
    }
}

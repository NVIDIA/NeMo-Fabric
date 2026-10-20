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
//! sorted key order. An entry whose key is longer than 128 characters is
//! dropped. While the compact JSON of the whole object is larger than
//! 128 KiB, MCP server entries are dropped from the end of the sorted order,
//! then model roles. Dropped entries are counted in a [`TruncatedEntries`]
//! marker next to the map: `models_truncated` and `mcp.servers_truncated`.
//! If the object is still too large with no MCP server and no model role,
//! it is replaced by a minimal one that holds only the schema name, the
//! version and the `what_ran_truncated` marker.
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
    /// rule.
    pub tool_names: Fact<Vec<String>>,
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
        Self::missing(
            FactState::Unsupported,
            REASON_KEY_NOT_DECLARED,
            FactSource::Reported,
            ADAPTER,
            METHOD_ADAPTER_REPORT,
        )
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
    /// is larger, drop the last MCP server in sorted key order, then the
    /// last model role, counting each in its map's marker with reason
    /// `byte_budget`. With neither left and the object still too large,
    /// replace it with [`WhatRan::minimal`].
    fn apply_byte_budget(&mut self) {
        while self.serialized_len() > MAX_WHAT_RAN_BYTES {
            if let Some(mcp) = self.mcp.as_mut()
                && mcp.servers.pop_last().is_some()
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
            let (kept, dropped) =
                bounded_entries(&mcp.servers, MAX_MCP_SERVERS, McpServerFacts::clone);
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
/// Core reads no adapter report key yet, so every adapter-reported fact
/// reads `unsupported` with reason `key_not_declared`. The collecting core
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
        assert_eq!(adapters.len(), 16);
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
                    config["models"]["default"][field] = json!(value);
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
    fn both_entry_caps_with_the_longest_keys_and_values_fit_the_byte_budget() {
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
        let exact = "é".repeat(256);
        assert_eq!(bounded_text(&exact), exact);

        // Expected digests: the first 15 hex digits of `sha256sum` over the
        // whole value's UTF-8 bytes.
        for (long, prefix, digest) in [
            (format!("{}a", "m".repeat(300)), "m", "96858eedeb3c8e6"),
            ("é".repeat(300), "é", "7250b66610f8b7d"),
        ] {
            let bounded = bounded_text(&long);

            assert_eq!(bounded, format!("{}~{digest}", prefix.repeat(240)));
            assert_eq!(bounded.chars().count(), 256);
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
}

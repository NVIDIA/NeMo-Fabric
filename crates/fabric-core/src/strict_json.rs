// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validate JSON object keys before conversion to lossy map representations.

use std::collections::BTreeSet;
use std::fmt;

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

pub(crate) fn validate(encoded: &[u8]) -> serde_json::Result<()> {
    serde_json::from_slice::<UniqueKeys>(encoded).map(|_| ())
}

struct UniqueKeys;

impl<'de> Deserialize<'de> for UniqueKeys {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(Self)
    }
}

impl<'de> Visitor<'de> for UniqueKeys {
    type Value = Self;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON without duplicate object fields")
    }

    fn visit_bool<E: de::Error>(self, _: bool) -> Result<Self, E> {
        Ok(self)
    }
    fn visit_i64<E: de::Error>(self, _: i64) -> Result<Self, E> {
        Ok(self)
    }
    fn visit_u64<E: de::Error>(self, _: u64) -> Result<Self, E> {
        Ok(self)
    }
    fn visit_f64<E: de::Error>(self, _: f64) -> Result<Self, E> {
        Ok(self)
    }
    fn visit_str<E: de::Error>(self, _: &str) -> Result<Self, E> {
        Ok(self)
    }
    fn visit_unit<E: de::Error>(self) -> Result<Self, E> {
        Ok(self)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self, A::Error> {
        while sequence.next_element::<Self>()?.is_some() {}
        Ok(self)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut mapping: A) -> Result<Self, A::Error> {
        let mut keys = BTreeSet::new();
        while let Some(key) = mapping.next_key::<String>()? {
            if !keys.insert(key) {
                return Err(de::Error::custom("duplicate JSON field"));
            }
            mapping.next_value::<Self>()?;
        }
        Ok(self)
    }
}

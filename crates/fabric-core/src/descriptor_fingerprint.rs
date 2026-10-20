// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adapter descriptor fingerprint `nemo.fabric.adapter_descriptor/v1`.
//!
//! The fingerprint is the SHA-256 of the RFC 8785 (JSON Canonicalization
//! Scheme) form of the descriptor as authored: the parsed descriptor file,
//! before core fills in defaults. Hashing the authored form keeps the value
//! stable when core starts serializing a new default field.
//!
//! Core computes the fingerprint once, when plan resolution selects the
//! descriptor, and `what_ran` copies it from the resolved plan. It is never
//! recomputed or checked at invocation, so it identifies the selected
//! descriptor, not what executed.
//!
//! The bundled adapter catalog records its own `sha256`, computed by the
//! catalog generator from Python's `json.dumps(sort_keys=True)`, which is
//! not RFC 8785. The two produce the same bytes for a descriptor without
//! floating-point numbers, integers beyond 2^53 or keys outside the Basic
//! Multilingual Plane. That condition is sufficient, not necessary: a float
//! such as `1.5` or a non-BMP key can still give the same bytes. All bundled
//! descriptors match today, and a golden test fails as soon as one doesn't.
//!
//! Parsing must round floats correctly, so core enables serde_json's
//! `float_roundtrip` feature.

use std::fmt::Write as _;

use serde_json::Value;
use sha2::{Digest, Sha256};

/// Name of the fingerprint spec, recorded as the fact's method.
pub(crate) const METHOD_ADAPTER_DESCRIPTOR: &str = "nemo.fabric.adapter_descriptor/v1";

/// The fingerprint of an authored descriptor as `sha256:<hex>`.
pub(crate) fn fingerprint_authored_descriptor(authored: &Value) -> String {
    format!("sha256:{}", authored_descriptor_sha256(authored))
}

/// Fingerprint a descriptor as authored, as lowercase hex.
///
/// `descriptor` must be the JSON parsed from the authored file, never the
/// descriptor core normalized. Three properties of that input matter:
///
/// - Parsing into a JSON value keeps the last of duplicate object keys.
///   RFC 8785 forbids duplicates. Core's typed descriptor loader rejects a
///   duplicate known field, so a descriptor that loads can repeat a key
///   only in a map member or a flattened additive field. There the loader
///   and the catalog generator's `json.loads` keep the last value too, so
///   all three read the same value.
/// - serde_json's `arbitrary_precision` feature is not supported. Core does
///   not enable it, and every number must convert to an `f64`.
/// - Integers beyond 2^53 follow RFC 8785's double semantics, so two
///   descriptors that differ only in such an integer's low digits hash the
///   same.
fn authored_descriptor_sha256(descriptor: &Value) -> String {
    format!(
        "{:x}",
        Sha256::digest(canonical_json(descriptor).as_bytes())
    )
}

/// Serialize a JSON value in the RFC 8785 canonical form.
fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_value(value, &mut out);
    out
}

fn write_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => write_number(
            number
                .as_f64()
                .expect("serde_json numbers are finite f64 values"),
            out,
        ),
        Value::String(text) => write_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_value(item, out);
            }
            out.push(']');
        }
        Value::Object(members) => {
            // RFC 8785 section 3.2.3 orders keys by their UTF-16 code units.
            let mut members: Vec<_> = members.iter().collect();
            members.sort_by(|(left, _), (right, _)| left.encode_utf16().cmp(right.encode_utf16()));
            out.push('{');
            for (index, (key, member)) in members.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                write_value(member, out);
            }
            out.push('}');
        }
    }
}

/// RFC 8785 section 3.2.2.2: escape only control characters, quote and backslash.
fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            control if control < ' ' => {
                write!(out, "\\u{:04x}", u32::from(control))
                    .expect("writing to a String cannot fail");
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

/// RFC 8785 section 3.2.2.3: the ECMAScript `Number.prototype.toString` form.
fn write_number(value: f64, out: &mut String) {
    // Covers minus zero, which ECMAScript writes as "0".
    if value == 0.0 {
        out.push('0');
        return;
    }
    if value < 0.0 {
        out.push('-');
    }
    let (digits, n) = shortest_digits(value.abs());
    let k = digits.len() as i32;
    if k <= n && n <= 21 {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', (n - k) as usize));
    } else if 0 < n && n <= 21 {
        let (integer, fraction) = digits.split_at(n as usize);
        write!(out, "{integer}.{fraction}").expect("writing to a String cannot fail");
    } else if -6 < n && n <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-n) as usize));
        out.push_str(&digits);
    } else {
        let (first, rest) = digits.split_at(1);
        out.push_str(first);
        if !rest.is_empty() {
            out.push('.');
            out.push_str(rest);
        }
        let exponent = n - 1;
        let sign = if exponent < 0 { '-' } else { '+' };
        write!(out, "e{sign}{}", exponent.abs()).expect("writing to a String cannot fail");
    }
}

/// Return the shortest round-trip digits `d1..dk` of a positive finite value and
/// the exponent `n` with `value = 0.d1..dk * 10^n`, as ECMAScript defines them.
///
/// serde_json already writes the shortest digits, breaking ties to even as
/// ECMAScript does (RFC 8785 Appendix B note 4); only the layout differs.
/// Do not switch to `format!("{:e}")`: std does not break ties to even (it
/// writes 1658206780088562.3 where ECMAScript writes 1658206780088562.2).
fn shortest_digits(value: f64) -> (String, i32) {
    let text = serde_json::Number::from_f64(value)
        .expect("JSON numbers are finite")
        .to_string();
    let (mantissa, exponent) = text.split_once('e').unwrap_or((&text, "0"));
    let exponent: i32 = exponent
        .parse()
        .expect("serde_json writes an integer exponent");
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let all_digits = format!("{integer}{fraction}");
    let significant = all_digits.trim_start_matches('0');
    let leading_zeros = (all_digits.len() - significant.len()) as i32;
    let digits = significant.trim_end_matches('0').to_owned();
    (digits, integer.len() as i32 + exponent - leading_zeros)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use serde_json::{Value, json};

    use super::{authored_descriptor_sha256, canonical_json};
    use crate::config::{FabricConfig, ResolveContext, resolve_run_plan_from_config};

    fn repository_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn catalog() -> Value {
        let catalog_path = repository_root().join(
            "sdk/python/nemo-fabric-adapter-catalog/src/nemo_fabric_adapter_catalog/catalog.json",
        );
        serde_json::from_slice(&fs::read(catalog_path).unwrap()).unwrap()
    }

    /// Plan resolution, the production path, hashes the descriptor file as
    /// authored: the value equals the catalog's, and differs from the hash
    /// of the descriptor core normalized.
    #[test]
    fn plan_resolution_fingerprints_the_authored_descriptor_file() {
        let root = repository_root();
        let config: FabricConfig = serde_json::from_value(json!({
            "schema_version": "fabric.agent/v1alpha1",
            "metadata": {"name": "fingerprint-agent"},
            "harness": {"adapter_id": "nvidia.fabric.claude"},
            "runtime": {}
        }))
        .unwrap();
        let plan = resolve_run_plan_from_config(config, ResolveContext::new(&root)).unwrap();
        let adapter = plan.adapter_descriptor.expect("resolved adapter");
        let catalog_sha256 = catalog()["adapters"]["nvidia.fabric.claude"]["sha256"]
            .as_str()
            .expect("catalog sha256")
            .to_string();
        let normalized = serde_json::to_value(&adapter.descriptor).unwrap();

        assert_eq!(
            adapter.descriptor_fingerprint,
            Some(format!("sha256:{catalog_sha256}"))
        );
        assert_ne!(authored_descriptor_sha256(&normalized), catalog_sha256);
    }

    #[test]
    fn fingerprint_matches_the_catalog_for_every_bundled_adapter() {
        let root = repository_root();
        let catalog = catalog();
        let adapters = catalog["adapters"].as_object().unwrap();
        assert!(!adapters.is_empty());
        for (adapter_id, record) in adapters {
            let source = root.join(record["source"].as_str().unwrap());
            let authored: Value = serde_json::from_slice(&fs::read(source).unwrap()).unwrap();
            assert_eq!(
                Some(authored_descriptor_sha256(&authored).as_str()),
                record["sha256"].as_str(),
                "{adapter_id}: the catalog generator's json.dumps(sort_keys=True) and \
                 RFC 8785 produced different bytes; floats, integers beyond 2^53 and \
                 non-BMP keys can cause that"
            );
        }
    }

    #[test]
    fn fingerprint_is_lowercase_hex_sha256_of_empty_object() {
        assert_eq!(
            authored_descriptor_sha256(&json!({})),
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        );
    }

    #[test]
    fn canonical_json_matches_the_rfc_8785_example() {
        // RFC 8785 section 3.2.2 input and section 3.2.3 output.
        let input: Value = serde_json::from_str(
            r#"{
                "numbers": [333333333.33333329, 1E30, 4.50, 2e-3, 0.000000000000000000000000001],
                "string": "\u20ac$\u000F\u000aA'\u0042\u0022\u005c\\\"\/",
                "literals": [null, true, false]
            }"#,
        )
        .unwrap();
        assert_eq!(
            canonical_json(&input),
            r#"{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"string":"€$\u000f\nA'B\"\\\\\"/"}"#
        );
    }

    #[test]
    fn canonical_json_sorts_keys_by_utf16_code_units() {
        // RFC 8785 section 3.2.3 sorting test data.
        let input: Value = serde_json::from_str(
            r#"{
                "\u20ac": "Euro Sign",
                "\r": "Carriage Return",
                "\ufb33": "Hebrew Letter Dalet With Dagesh",
                "1": "One",
                "\ud83d\ude00": "Emoji: Grinning Face",
                "\u0080": "Control",
                "\u00f6": "Latin Small Letter O With Diaeresis"
            }"#,
        )
        .unwrap();
        let canonical = canonical_json(&input);
        let order: Vec<usize> = [
            "Carriage Return",
            "One",
            "Control",
            "Latin Small Letter O With Diaeresis",
            "Euro Sign",
            "Emoji: Grinning Face",
            "Hebrew Letter Dalet With Dagesh",
        ]
        .iter()
        .map(|name| canonical.find(name).unwrap())
        .collect();
        assert!(order.is_sorted(), "{canonical}");
    }

    #[test]
    fn canonical_json_sorts_nested_objects_and_keeps_array_order() {
        let input = json!({"b": [3, {"z": 1, "a": 2}], "a": {"d": null, "c": true}});
        assert_eq!(
            canonical_json(&input),
            r#"{"a":{"c":true,"d":null},"b":[3,{"a":2,"z":1}]}"#
        );
    }

    #[test]
    fn canonical_json_escapes_only_controls_quote_and_backslash() {
        let input = json!("\u{0}\u{8}\t\n\u{c}\r\u{1f}\"\\/\u{7f}é\u{2028}😀");
        assert_eq!(
            canonical_json(&input),
            "\"\\u0000\\b\\t\\n\\f\\r\\u001f\\\"\\\\/\u{7f}é\u{2028}😀\""
        );
    }

    #[test]
    fn canonical_json_formats_numbers_as_ecmascript() {
        // RFC 8785 Appendix B samples, by IEEE 754 bit pattern.
        let samples = [
            (0x0000000000000000, "0"),
            (0x8000000000000000, "0"),
            (0x0000000000000001, "5e-324"),
            (0x8000000000000001, "-5e-324"),
            (0x7fefffffffffffff, "1.7976931348623157e+308"),
            (0xffefffffffffffff, "-1.7976931348623157e+308"),
            (0x4340000000000000, "9007199254740992"),
            (0xc340000000000000, "-9007199254740992"),
            (0x4430000000000000, "295147905179352830000"),
            (0x44b52d02c7e14af5, "9.999999999999997e+22"),
            (0x44b52d02c7e14af6, "1e+23"),
            (0x44b52d02c7e14af7, "1.0000000000000001e+23"),
            (0x444b1ae4d6e2ef4e, "999999999999999700000"),
            (0x444b1ae4d6e2ef4f, "999999999999999900000"),
            (0x444b1ae4d6e2ef50, "1e+21"),
            (0x3eb0c6f7a0b5ed8c, "9.999999999999997e-7"),
            (0x3eb0c6f7a0b5ed8d, "0.000001"),
            (0x41b3de4355555553, "333333333.3333332"),
            (0x41b3de4355555554, "333333333.33333325"),
            (0x41b3de4355555555, "333333333.3333333"),
            (0x41b3de4355555556, "333333333.3333334"),
            (0x41b3de4355555557, "333333333.33333343"),
            (0xbecbf647612f3696, "-0.0000033333333333333333"),
            (0x43143ff3c1cb0959, "1424953923781206.2"),
        ];
        for (bits, expected) in samples {
            let value = json!(f64::from_bits(bits));
            assert_eq!(canonical_json(&value), expected, "{bits:016x}");
        }
    }

    #[test]
    fn canonical_json_writes_integers_as_ecmascript_doubles() {
        let input: Value =
            serde_json::from_str("[0, 1, -1, 30, 9007199254740993, -9007199254740993, 18446744073709551615, 1000000000000000000000]")
                .unwrap();
        assert_eq!(
            canonical_json(&input),
            "[0,1,-1,30,9007199254740992,-9007199254740992,18446744073709552000,1e+21]"
        );
    }
}

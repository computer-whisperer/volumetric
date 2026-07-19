//! Generic string-to-f64 data shared through the project DAG.
//!
//! The wire representation is a CBOR map with text keys and finite numeric
//! values. Operators and hosts use these helpers so maps produced in one
//! place have deterministic key ordering and consistent validation.

use std::collections::{BTreeMap, BTreeSet};

use ciborium::value::Value;

/// A decoded flat numeric map. `BTreeMap` makes encoding and presentation
/// deterministic without assigning parameter-specific meaning to the keys.
pub type F64Map = BTreeMap<String, f64>;

/// Encode a validated map as CBOR. This encoder always writes values as CBOR
/// floats even though the decoder accepts integer-valued CBOR numbers too.
pub fn encode(values: &F64Map) -> Result<Vec<u8>, String> {
    let mut entries = Vec::with_capacity(values.len());
    for (key, value) in values {
        validate_entry(key, *value)?;
        entries.push((Value::Text(key.clone()), Value::Float(*value)));
    }

    let mut out = Vec::new();
    ciborium::ser::into_writer(&Value::Map(entries), &mut out)
        .map_err(|error| format!("failed to encode F64Map CBOR: {error}"))?;
    Ok(out)
}

/// Decode and validate an F64Map. Integer-valued CBOR numbers are accepted
/// for interoperability with JSON-to-CBOR tools; callers always receive f64.
pub fn decode(bytes: &[u8]) -> Result<F64Map, String> {
    if bytes.is_empty() {
        return Ok(F64Map::new());
    }

    let value: Value = ciborium::de::from_reader(bytes)
        .map_err(|error| format!("failed to decode F64Map CBOR: {error}"))?;
    let Value::Map(entries) = value else {
        return Err("F64Map payload must be a CBOR map".to_string());
    };

    let mut seen = BTreeSet::new();
    let mut out = F64Map::new();
    for (key, value) in entries {
        let Value::Text(key) = key else {
            return Err("F64Map keys must be CBOR text strings".to_string());
        };
        if !seen.insert(key.clone()) {
            return Err(format!("F64Map contains duplicate key `{key}`"));
        }
        let number = match value {
            Value::Float(value) => value,
            Value::Integer(value) => {
                if let Ok(signed) = i64::try_from(value) {
                    signed as f64
                } else if let Ok(unsigned) = u64::try_from(value) {
                    unsigned as f64
                } else {
                    return Err(format!("F64Map value for `{key}` is outside numeric range"));
                }
            }
            _ => return Err(format!("F64Map value for `{key}` must be numeric")),
        };
        validate_entry(&key, number)?;
        out.insert(key, number);
    }
    Ok(out)
}

fn validate_entry(key: &str, value: f64) -> Result<(), String> {
    if key.is_empty() {
        return Err("F64Map keys must not be empty".to_string());
    }
    if !value.is_finite() {
        return Err(format!("F64Map value for `{key}` must be finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_sorted_and_exact() {
        let values = F64Map::from([
            ("spinner.pitch".to_string(), 0.035),
            ("bearing.od".to_string(), 0.022),
        ]);
        let bytes = encode(&values).unwrap();
        assert_eq!(decode(&bytes).unwrap(), values);

        let Value::Map(entries) = ciborium::de::from_reader(bytes.as_slice()).unwrap() else {
            panic!("encoded F64Map must be a map");
        };
        assert_eq!(entries[0].0, Value::Text("bearing.od".to_string()));
        assert_eq!(entries[1].0, Value::Text("spinner.pitch".to_string()));
    }

    #[test]
    fn integer_values_are_accepted_but_invalid_shapes_are_not() {
        let mut integer_map = Vec::new();
        ciborium::ser::into_writer(
            &Value::Map(vec![(
                Value::Text("count".to_string()),
                Value::Integer(3.into()),
            )]),
            &mut integer_map,
        )
        .unwrap();
        assert_eq!(decode(&integer_map).unwrap()["count"], 3.0);

        let mut invalid = Vec::new();
        ciborium::ser::into_writer(&Value::Array(Vec::new()), &mut invalid).unwrap();
        assert!(decode(&invalid).unwrap_err().contains("must be a CBOR map"));
    }

    #[test]
    fn rejects_duplicate_empty_and_non_finite_entries() {
        let mut duplicate = Vec::new();
        ciborium::ser::into_writer(
            &Value::Map(vec![
                (Value::Text("x".to_string()), Value::Float(1.0)),
                (Value::Text("x".to_string()), Value::Float(2.0)),
            ]),
            &mut duplicate,
        )
        .unwrap();
        assert!(decode(&duplicate).unwrap_err().contains("duplicate key"));

        assert!(encode(&F64Map::from([(String::new(), 1.0)])).is_err());
        assert!(encode(&F64Map::from([("x".to_string(), f64::NAN)])).is_err());
    }
}

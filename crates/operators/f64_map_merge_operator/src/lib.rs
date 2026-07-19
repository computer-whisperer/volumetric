//! Generic composition operator for [`volumetric_abi::f64_map::F64Map`]
//! project data.
//!
//! Inputs 0 through 4 are optional F64Maps. They are merged from left to
//! right, so a value in a later slot replaces the same key from every earlier
//! slot. Missing and empty inputs are the identity. This right-biased union is
//! associative, allowing larger compositions to be chained without changing
//! their meaning.
//!
//! Output 0 is the deterministically encoded merged F64Map.

use volumetric_abi::f64_map::{F64Map, decode, encode};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// A fixed arity keeps operator metadata and project steps simple. Merge is
/// associative, so projects needing more sources can chain nodes.
const MAP_SLOTS: usize = 5;

fn merge_encoded_inputs(inputs: &[Vec<u8>]) -> Result<Vec<u8>, String> {
    let mut merged = F64Map::new();
    for (index, bytes) in inputs.iter().enumerate() {
        if bytes.is_empty() {
            continue;
        }
        let values = decode(bytes)
            .map_err(|error| format!("input {} is not a usable F64Map: {error}", index + 1))?;
        merged.extend(values);
    }
    encode(&merged)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let inputs: Vec<Vec<u8>> = (0..MAP_SLOTS)
        .map(|index| read_input(index as i32))
        .collect();
    match merge_encoded_inputs(&inputs) {
        Ok(output) => post_output(0, &output),
        Err(error) => report_error(&format!("F64Map merge failed: {error}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata {
        name: "f64_map_merge_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "Merge F64 Maps".to_string(),
        description:
            "Composes numeric project data left-to-right; later maps override duplicate keys."
                .to_string(),
        category: "Data".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<rect x="3" y="3" width="13" height="13" rx="2"/>"##,
            r##"<rect x="8" y="8" width="13" height="13" rx="2"/>"##,
            r##"<path d="M11 12h7M11 16h7"/>"##,
        )
        .to_string(),
        inputs: vec![OperatorMetadataInput::F64Map; MAP_SLOTS],
        input_names: vec![
            "Base".to_string(),
            "Override 1".to_string(),
            "Override 2".to_string(),
            "Override 3".to_string(),
            "Override 4".to_string(),
        ],
        outputs: vec![OperatorMetadataOutput::F64Map],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(entries: &[(&str, f64)]) -> Vec<u8> {
        let values = entries
            .iter()
            .map(|(key, value)| ((*key).to_string(), *value))
            .collect();
        encode(&values).unwrap()
    }

    #[test]
    fn later_inputs_override_and_empty_slots_are_identity() {
        let output = merge_encoded_inputs(&[
            encoded(&[("shared.scale", 1.0), ("spinner.pitch", 0.035)]),
            Vec::new(),
            encoded(&[("spinner.pitch", 0.04), ("spinner.clearance", 0.0002)]),
        ])
        .unwrap();
        assert_eq!(
            decode(&output).unwrap(),
            F64Map::from([
                ("shared.scale".to_string(), 1.0),
                ("spinner.clearance".to_string(), 0.0002),
                ("spinner.pitch".to_string(), 0.04),
            ])
        );
    }

    #[test]
    fn invalid_input_reports_its_one_based_slot() {
        let error = merge_encoded_inputs(&[encoded(&[("x", 1.0)]), vec![0xff]])
            .expect_err("invalid CBOR must fail");
        assert!(error.contains("input 2"), "unexpected error: {error}");
    }

    #[test]
    fn chained_merges_preserve_direct_merge_semantics() {
        let base = encoded(&[("a", 1.0), ("shared", 1.0)]);
        let middle = encoded(&[("b", 2.0), ("shared", 2.0)]);
        let last = encoded(&[("c", 3.0), ("shared", 3.0)]);
        let direct = merge_encoded_inputs(&[base.clone(), middle.clone(), last.clone()]).unwrap();
        let first_stage = merge_encoded_inputs(&[base, middle]).unwrap();
        let chained = merge_encoded_inputs(&[first_stage, last]).unwrap();
        assert_eq!(direct, chained);
    }
}

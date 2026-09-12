//! Assemble operator: a mechanism, its part models and a state make an
//! [`Assembly`], and the parts posed and unioned make one model.
//!
//! Host/operator ABI: see the `volumetric_abi` crate; the value:
//! `volumetric_abi::mechanism`; posing: `assembly_core`; design:
//! `ASSEMBLY_PLAN.md`.
//!
//! Inputs:
//! - 0: Mechanism.
//! - 1..n: the part models (the variadic block), one per part in the
//!   mechanism's part order, each authored in the world frame at rest.
//! - last: F64Map state keyed by joint name; empty for the rest state.
//!
//! Outputs:
//! - 0 `assembly`: the Assembly (parts, mechanism, the completed state).
//! - 1 `model`: the union of the parts at that state, a ModelWASM.

use volumetric_abi::f64_map::decode as decode_state;
use volumetric_abi::host::{input_count, post_output, read_input, report_error};
use volumetric_abi::mechanism::{Assembly, AssemblyPart, decode_mechanism, encode_assembly};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

fn assemble(mechanism: &[u8], parts: Vec<Vec<u8>>, state: &[u8]) -> Result<Assembly, String> {
    let mechanism = decode_mechanism(mechanism)?;
    if parts.len() != mechanism.parts.len() {
        return Err(format!(
            "the mechanism names {} parts ({}) but {} model inputs are wired",
            mechanism.parts.len(),
            mechanism.parts.join(", "),
            parts.len()
        ));
    }
    let parts = mechanism
        .parts
        .iter()
        .zip(parts)
        .map(|(name, model)| {
            if model.is_empty() {
                return Err(format!("part `{name}`: its model input is not wired"));
            }
            Ok(AssemblyPart {
                name: name.clone(),
                model,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let state = decode_state(state)?;
    Assembly::new(mechanism, parts, &state)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let count = input_count();
    if count < 3 {
        report_error("assemble needs a mechanism, at least one part model and the state");
        return;
    }
    let mechanism = read_input(0);
    if mechanism.is_empty() {
        report_error("the mechanism input is not wired");
        return;
    }
    let parts: Vec<Vec<u8>> = (1..count - 1).map(|idx| read_input(idx as i32)).collect();
    let state = read_input(count as i32 - 1);
    let assembly = match assemble(&mechanism, parts, &state) {
        Ok(assembly) => assembly,
        Err(e) => {
            report_error(&format!("assemble failed: {e}"));
            return;
        }
    };
    match assembly_core::posed_union(&assembly, None) {
        Ok(model) => {
            post_output(0, &encode_assembly(&assembly));
            post_output(1, &model);
        }
        Err(e) => report_error(&format!("assemble failed to pose the parts: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        OperatorMetadata {
        name: "assemble_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        docs: include_str!("../README.md").to_string(),
        display_name: "Assemble".to_string(),
        description: "A mechanism, its part models and a state: the assembly, and the parts posed as one model.".to_string(),
        category: "Assembly".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<rect x="3" y="12" width="8" height="8" rx="1"/>"##,
            r##"<rect x="13" y="4" width="8" height="8" rx="1"/>"##,
            r##"<path d="M11 16h4a2 2 0 0 0 2-2v-2"/>"##,
        )
        .to_string(),
        inputs: vec![
            OperatorMetadataInput::Mechanism,
            OperatorMetadataInput::ModelWASM,
            OperatorMetadataInput::F64Map,
        ],
        variadic_input: Some(1),
        input_names: vec![
            "Mechanism".to_string(),
            "Parts".to_string(),
            "State".to_string(),
        ],
        outputs: vec![
            OperatorMetadataOutput::Assembly,
            OperatorMetadataOutput::ModelWASM,
        ],
        output_names: vec!["assembly".to_string(), "model".to_string()],
    }
    })
}

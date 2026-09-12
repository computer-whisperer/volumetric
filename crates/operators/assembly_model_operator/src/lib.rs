//! Assembly Model operator: one part of an assembly, or all of them, posed
//! at the assembly's state or another, as a ModelWASM.
//!
//! Host/operator ABI: see the `volumetric_abi` crate; posing:
//! `assembly_core`; design: `ASSEMBLY_PLAN.md`.
//!
//! Inputs:
//! - 0: Assembly.
//! - 1: CBOR config `{ part: tstr .default "all" }`: a part by name, or
//!   `all` for the union.
//! - 2: optional F64Map state override; empty keeps the assembly's state.
//!
//! Output 0: the posed model.

use volumetric_abi::f64_map::decode as decode_state;
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::mechanism::decode_assembly;
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct ModelConfig {
    part: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            part: "all".to_string(),
        }
    }
}

fn model(assembly: &[u8], cfg: &ModelConfig, state: &[u8]) -> Result<Vec<u8>, String> {
    let assembly = decode_assembly(assembly)?;
    let state = if state.is_empty() {
        None
    } else {
        Some(decode_state(state)?)
    };
    if cfg.part == "all" {
        assembly_core::posed_union(&assembly, state.as_ref())
    } else {
        assembly_core::posed_part(&assembly, &cfg.part, state.as_ref())
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let assembly = read_input(0);
    if assembly.is_empty() {
        report_error("the assembly input is not wired");
        return;
    }
    let cfg_buf = read_input(1);
    let cfg = if cfg_buf.is_empty() {
        ModelConfig::default()
    } else {
        match ciborium::de::from_reader(std::io::Cursor::new(&cfg_buf)) {
            Ok(cfg) => cfg,
            Err(e) => {
                report_error(&format!("invalid configuration: {e}"));
                return;
            }
        }
    };
    let state = read_input(2);
    match model(&assembly, &cfg, &state) {
        Ok(model) => post_output(0, &model),
        Err(e) => report_error(&format!("assembly model failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata {
        name: "assembly_model_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        docs: include_str!("../README.md").to_string(),
        display_name: "Assembly Model".to_string(),
        description: "One part of an assembly, or all of them, posed at a state, as a model."
            .to_string(),
        category: "Assembly".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<rect x="4" y="4" width="16" height="16" rx="2"/>"##,
            r##"<path d="M4 12h16"/>"##,
            r##"<path d="m12 4 0 16"/>"##,
        )
        .to_string(),
        inputs: vec![
            OperatorMetadataInput::Assembly,
            OperatorMetadataInput::CBORConfiguration(
                r#"{ part: tstr .default "all" }"#.to_string(),
            ),
            OperatorMetadataInput::F64Map,
        ],
        variadic_input: None,
        input_names: vec![
            "Assembly".to_string(),
            "Config".to_string(),
            "State".to_string(),
        ],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
        output_names: vec!["model".to_string()],
    })
}

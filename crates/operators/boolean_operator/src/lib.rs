#![doc = include_str!("../README.md")]
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! Inputs: one or more `ModelWASM` slots (the variadic block, slot 0)
//! followed by the CBOR configuration (schema declared in metadata). The
//! step's input count comes from `host::input_count`; the config is the
//! last input and every earlier non-empty input is a model.
//!
//! The merged module's ABI is documented on `model_merge_core::combine`,
//! which holds the glue (the assembly operators union posed parts through
//! the same code).

use model_merge_core::{Combine, combine_models};
use volumetric_abi::host::{input_count, post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum BooleanOp {
    Union,
    Subtract,
    Intersect,
}

// deny_unknown_fields: a misspelled key (e.g. the old `operation`) must
// error rather than silently fall back to union.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BooleanConfig {
    op: Option<BooleanOp>,
}

impl Default for BooleanConfig {
    fn default() -> Self {
        Self {
            op: Some(BooleanOp::Union),
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let count = input_count();
    if count < 2 {
        report_error("boolean needs at least one model input followed by the config");
        return;
    }

    let cfg = {
        let cfg_buf = read_input(count as i32 - 1);
        if cfg_buf.is_empty() {
            BooleanConfig::default()
        } else {
            match ciborium::de::from_reader::<BooleanConfig, _>(std::io::Cursor::new(&cfg_buf)) {
                Ok(cfg) => cfg,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };
    let op = cfg.op.unwrap_or(BooleanOp::Union);

    // Unwired entries in the model block read back empty and are skipped.
    let models: Vec<Vec<u8>> = (0..count - 1)
        .map(|idx| read_input(idx as i32))
        .filter(|bytes| !bytes.is_empty())
        .collect();
    if models.is_empty() {
        report_error("no models wired (connect at least one model input)");
        return;
    }

    let op = match op {
        BooleanOp::Union => Combine::Union,
        BooleanOp::Subtract => Combine::Subtract,
        BooleanOp::Intersect => Combine::Intersect,
    };
    match combine_models(&models, op) {
        Ok(output) => post_output(0, &output),
        Err(e) => report_error(&format!("model merge failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema =
            "{ op: \"union\" / \"subtract\" / \"intersect\" .default \"union\" }".to_string();
        OperatorMetadata {
            name: "boolean_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Boolean".to_string(),
            description: "Union, intersection, or subtraction of one or more models (subtract: the first minus the rest)."
                .to_string(),
            category: "Combine".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<rect x="3" y="3" width="12" height="12" rx="2"/>"##,
                r##"<rect x="9" y="9" width="12" height="12" rx="2"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: Some(0),
            input_names: vec!["Model".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
            output_names: vec![],
        }
    })
}

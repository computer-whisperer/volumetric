//! Step outputs carry the operator-declared output types.
//!
//! The run loop reads each operator's `get_metadata()` and stamps every
//! output asset with the declared type; operators without decodable
//! metadata fall back to the historical Model assumption. These tests build
//! tiny WAT operators inline (wasmtime compiles text modules directly), so
//! they don't depend on the wasm32 operator artifacts.

#![cfg(feature = "native")]

use volumetric::{
    AssetTypeHint, Environment, ExecutionStep, ImportedAsset, OperatorMetadata,
    OperatorMetadataOutput, Project,
};

/// A minimal operator that declares `outputs` in its metadata and posts one
/// 4-byte blob per entry when run.
fn operator_with_declared_outputs(outputs: Vec<OperatorMetadataOutput>) -> Vec<u8> {
    let metadata = volumetric_abi::encode_metadata(&OperatorMetadata {
        name: "typed_outputs_test".to_string(),
        version: "0.0.0".to_string(),
        inputs: vec![],
        input_names: vec![],
        outputs: outputs.clone(),
    });

    let data: String = metadata.iter().map(|b| format!("\\{b:02x}")).collect();
    let packed = 1024_i64 | ((metadata.len() as i64) << 32);
    let posts: String = (0..outputs.len())
        .map(|idx| format!("(call $post_output (i32.const {idx}) (i32.const 0) (i32.const 4))"))
        .collect();

    format!(
        r#"(module
            (import "host" "post_output" (func $post_output (param i32 i32 i32)))
            (memory (export "memory") 1)
            (data (i32.const 1024) "{data}")
            (func (export "get_metadata") (result i64) (i64.const {packed}))
            (func (export "run") {posts}))"#
    )
    .into_bytes()
}

/// An operator with no `get_metadata` export at all.
fn operator_without_metadata() -> Vec<u8> {
    r#"(module
        (import "host" "post_output" (func $post_output (param i32 i32 i32)))
        (memory (export "memory") 1)
        (func (export "run")
            (call $post_output (i32.const 0) (i32.const 0) (i32.const 4))))"#
        .as_bytes()
        .to_vec()
}

fn run_project(operator: Vec<u8>, output_ids: &[&str]) -> Vec<volumetric::LoadedAsset> {
    let project = Project {
        version: 2,
        imports: vec![ImportedAsset::operator("op".to_string(), operator)],
        timeline: vec![ExecutionStep {
            operator_id: "op".to_string(),
            inputs: vec![],
            outputs: output_ids.iter().map(|id| id.to_string()).collect(),
        }],
        exports: output_ids.iter().map(|id| id.to_string()).collect(),
    };

    let mut env = Environment::new();
    project.run(&mut env).expect("project run failed")
}

fn hint_of(exports: &[volumetric::LoadedAsset], id: &str) -> Option<AssetTypeHint> {
    exports
        .iter()
        .find(|asset| asset.id() == id)
        .unwrap_or_else(|| panic!("export {id} missing"))
        .type_hint()
}

#[test]
fn outputs_are_stamped_with_declared_types() {
    let operator = operator_with_declared_outputs(vec![
        OperatorMetadataOutput::ModelWASM,
        OperatorMetadataOutput::FeaMesh,
    ]);
    let exports = run_project(operator, &["out_model", "out_mesh"]);

    assert_eq!(hint_of(&exports, "out_model"), Some(AssetTypeHint::Model));
    assert_eq!(hint_of(&exports, "out_mesh"), Some(AssetTypeHint::FeaMesh));
}

#[test]
fn extra_undeclared_outputs_fall_back_to_model() {
    // The step maps two outputs but the operator only declares (and posts)
    // one... so post the second anyway: declare one output kind, post two.
    let metadata_outputs = vec![OperatorMetadataOutput::FeaMesh];
    let operator = {
        // Same module as operator_with_declared_outputs but posting one more
        // output than declared, to exercise the index-out-of-range fallback.
        let mut wat = String::from_utf8(operator_with_declared_outputs(metadata_outputs)).unwrap();
        wat = wat.replace(
            "(func (export \"run\") ",
            "(func (export \"run\") (call $post_output (i32.const 1) (i32.const 0) (i32.const 4)) ",
        );
        wat.into_bytes()
    };
    let exports = run_project(operator, &["out_mesh", "out_extra"]);

    assert_eq!(hint_of(&exports, "out_mesh"), Some(AssetTypeHint::FeaMesh));
    assert_eq!(hint_of(&exports, "out_extra"), Some(AssetTypeHint::Model));
}

#[test]
fn operator_without_metadata_falls_back_to_model() {
    let exports = run_project(operator_without_metadata(), &["out"]);
    assert_eq!(hint_of(&exports, "out"), Some(AssetTypeHint::Model));
}

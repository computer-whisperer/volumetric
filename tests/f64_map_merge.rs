//! End-to-end coverage for generic F64Map composition. The project routes
//! two imported maps through the merge operator, then routes its typed output
//! into the Lua operator as shared model parameters.
//!
//! Requires:
//!   cargo build --release --target wasm32-unknown-unknown \
//!     -p f64_map_merge_operator -p lua_script_operator

#![cfg(feature = "native")]

use volumetric::wasm::{ModelExecutor, create_model_executor};
use volumetric::{
    AssetTypeHint, Environment, ExecutionInput, ExecutionStep, ImportedAsset,
    OperatorMetadataInput, OperatorMetadataOutput, Project,
};
use volumetric_abi::f64_map::{F64Map, decode, encode};

fn wasm_artifact(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-unknown-unknown/release")
        .join(format!("{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing wasm artifact {} ({error}); build it with the command in this test's module docs",
            path.display()
        )
    })
}

fn map(entries: &[(&str, f64)]) -> Vec<u8> {
    let values = entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), *value))
        .collect();
    encode(&values).unwrap()
}

#[test]
fn metadata_declares_five_maps_and_a_typed_map_output() {
    let metadata =
        volumetric::operator_metadata_from_wasm_bytes(&wasm_artifact("f64_map_merge_operator"))
            .expect("merge metadata");
    assert_eq!(metadata.name, "f64_map_merge_operator");
    assert_eq!(metadata.inputs.len(), 5);
    assert!(
        metadata
            .inputs
            .iter()
            .all(|input| matches!(input, OperatorMetadataInput::F64Map))
    );
    assert_eq!(metadata.input_name(0), Some("Base"));
    assert_eq!(metadata.input_name(4), Some("Override 4"));
    assert_eq!(metadata.outputs, vec![OperatorMetadataOutput::F64Map]);
}

#[test]
fn merged_project_data_routes_into_a_lua_model() {
    const SOURCE: &str = r#"
local radius = 1.0 -- @param key=shared.radius min=0.25 max=4.0
local margin = 0.5 -- @param key=shared.margin min=0.0 max=1.0
local bound = radius + margin
function is_inside(x, y) return x*x + y*y <= radius*radius end
function get_bounds_min_x() return -bound end
function get_bounds_max_x() return bound end
function get_bounds_min_y() return -bound end
function get_bounds_max_y() return bound end
"#;

    let project = Project {
        version: 2,
        imports: vec![
            ImportedAsset::operator("merge".to_string(), wasm_artifact("f64_map_merge_operator")),
            ImportedAsset::operator("lua".to_string(), wasm_artifact("lua_script_operator")),
            ImportedAsset::new(
                "global_dimensions".to_string(),
                map(&[("shared.radius", 1.5), ("shared.margin", 0.25)]),
                Some(AssetTypeHint::F64Map),
            ),
            ImportedAsset::new(
                "part_overrides".to_string(),
                map(&[("shared.radius", 2.0), ("unrelated.value", 17.0)]),
                Some(AssetTypeHint::F64Map),
            ),
        ],
        timeline: vec![
            ExecutionStep {
                operator_id: "merge".to_string(),
                inputs: vec![
                    ExecutionInput::AssetRef("global_dimensions".to_string()),
                    ExecutionInput::AssetRef("part_overrides".to_string()),
                ],
                outputs: vec!["composed_parameters".to_string()],
            },
            ExecutionStep {
                operator_id: "lua".to_string(),
                inputs: vec![
                    ExecutionInput::Inline(SOURCE.as_bytes().to_vec()),
                    ExecutionInput::AssetRef("composed_parameters".to_string()),
                ],
                outputs: vec!["parameterized_model".to_string()],
            },
        ],
        exports: vec![
            "composed_parameters".to_string(),
            "parameterized_model".to_string(),
        ],
        baked: None,
    };
    assert!(project.validate().is_empty());

    let mut environment = Environment::new();
    let exports = project.run(&mut environment).expect("project run");
    let parameters = exports
        .iter()
        .find(|asset| asset.id() == "composed_parameters")
        .expect("merged map export");
    assert_eq!(parameters.type_hint(), Some(AssetTypeHint::F64Map));
    assert_eq!(
        decode(parameters.data()).unwrap(),
        F64Map::from([
            ("shared.margin".to_string(), 0.25),
            ("shared.radius".to_string(), 2.0),
            ("unrelated.value".to_string(), 17.0),
        ])
    );

    let model = exports
        .iter()
        .find(|asset| asset.id() == "parameterized_model")
        .expect("Lua model export");
    assert_eq!(model.type_hint(), Some(AssetTypeHint::Model));
    let mut executor = create_model_executor(model.data()).expect("model executor");
    let bounds = executor.get_bounds_nd().expect("model bounds");
    assert_eq!((bounds.min(0), bounds.max(0)), (-2.25, 2.25));
    assert_eq!((bounds.min(1), bounds.max(1)), (-2.25, 2.25));
    assert_eq!(executor.sample_nd(&[1.75, 0.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[2.1, 0.0]).unwrap(), 0.0);
}

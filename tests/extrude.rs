//! Integration tests for the sketch + extrude pipeline: a 2D Lua sketch
//! compiled by lua_script_operator, lifted to 3D by extrude_operator.
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

use volumetric::wasm::{
    NativeModelExecutor, OperatorExecutor, OperatorIo, create_operator_executor,
    create_parallel_sampler,
};

fn wasm_artifact(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-unknown-unknown/release")
        .join(format!("{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing wasm artifact {} ({e}); build it with `cargo build-wasm`",
            path.display()
        )
    })
}

fn run_operator(operator: &str, inputs: Vec<Vec<u8>>) -> Result<Vec<u8>, String> {
    let operator_wasm = wasm_artifact(operator);
    let mut executor = create_operator_executor(&operator_wasm).expect("create operator executor");
    let result = executor
        .run(OperatorIo::new(inputs))
        .map_err(|e| e.to_string())?;
    result
        .outputs
        .get(&0)
        .cloned()
        .ok_or_else(|| "operator posted no output".to_string())
}

const CIRCLE_SKETCH: &str = r#"
function is_inside(x, y)
    if x*x + y*y <= 1.0 then
        return 1.0
    else
        return 0.0
    end
end

function get_bounds_min_x() return -1.5 end
function get_bounds_max_x() return 1.5 end
function get_bounds_min_y() return -1.5 end
function get_bounds_max_y() return 1.5 end
"#;

fn circle_sketch_wasm() -> Vec<u8> {
    run_operator(
        "lua_script_operator",
        vec![CIRCLE_SKETCH.as_bytes().to_vec()],
    )
    .expect("compile 2D lua sketch")
}

fn height_config(height: f64) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(vec![(
            ciborium::value::Value::Text("height".into()),
            ciborium::value::Value::Float(height),
        )]),
        &mut out,
    )
    .unwrap();
    out
}

#[test]
fn lua_compiles_2d_sketches() {
    let mut executor = NativeModelExecutor::new(&circle_sketch_wasm()).unwrap();
    assert_eq!(executor.dimensions(), 2);

    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!(bounds.dimensions(), 2);
    assert_eq!(bounds.min(0), -1.5);
    assert_eq!(bounds.max(1), 1.5);

    assert_eq!(executor.sample_nd(&[0.0, 0.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.9, 0.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[1.2, 0.0]).unwrap(), 0.0);
}

#[test]
fn sketches_refuse_3d_sampling_gracefully() {
    use volumetric::wasm::ModelExecutor;
    let sketch = circle_sketch_wasm();

    let mut executor = NativeModelExecutor::new(&sketch).unwrap();
    let err = executor
        .is_inside(0.0, 0.0, 0.0)
        .expect_err("3D-sampling a 2D sketch must error, not panic");
    assert!(err.to_string().contains("3D sampling"), "{err}");

    let err = create_parallel_sampler(&sketch)
        .err()
        .expect("parallel sampler on a 2D sketch must be rejected");
    assert!(err.to_string().contains("3D sampling"), "{err}");
}

#[test]
fn extrude_lifts_sketch_to_3d() {
    let extruded = run_operator(
        "extrude_operator",
        vec![circle_sketch_wasm(), height_config(2.0)],
    )
    .expect("extrude sketch");

    let mut executor = NativeModelExecutor::new(&extruded).unwrap();
    assert_eq!(executor.dimensions(), 3);

    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!(bounds.dimensions(), 3);
    assert_eq!((bounds.min(0), bounds.max(0)), (-1.5, 1.5));
    assert_eq!((bounds.min(1), bounds.max(1)), (-1.5, 1.5));
    assert_eq!((bounds.min(2), bounds.max(2)), (0.0, 2.0));

    // Inside the cylinder
    assert_eq!(executor.sample_nd(&[0.0, 0.0, 1.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.9, 0.0, 0.1]).unwrap(), 1.0);
    // Outside radially
    assert_eq!(executor.sample_nd(&[1.2, 0.0, 1.0]).unwrap(), 0.0);
    // Outside axially (below and above the extrusion)
    assert_eq!(executor.sample_nd(&[0.0, 0.0, -0.1]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[0.0, 0.0, 2.5]).unwrap(), 0.0);
}

#[test]
fn extrude_default_height_is_one() {
    let extruded =
        run_operator("extrude_operator", vec![circle_sketch_wasm()]).expect("extrude sketch");
    let mut executor = NativeModelExecutor::new(&extruded).unwrap();
    assert_eq!(executor.sample_nd(&[0.0, 0.0, 0.5]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.0, 0.0, 1.5]).unwrap(), 0.0);
}

#[test]
fn extruded_model_composes_downstream() {
    // The extruded cylinder must be usable like any other 3D model — here
    // through the parallel sampler used by the meshing pipeline.
    let extruded = run_operator(
        "extrude_operator",
        vec![circle_sketch_wasm(), height_config(2.0)],
    )
    .expect("extrude sketch");

    use volumetric::wasm::ParallelModelSampler;
    let sampler = create_parallel_sampler(&extruded).expect("parallel sampler");
    assert!(volumetric_abi::is_occupied(sampler.sample(0.0, 0.0, 1.0)));
    assert!(!volumetric_abi::is_occupied(sampler.sample(0.0, 0.0, 2.5)));

    let bounds = sampler.get_bounds().expect("bounds");
    assert_eq!(bounds.min.2, 0.0);
    assert_eq!(bounds.max.2, 2.0);
}

#[test]
fn extruding_a_3d_model_is_rejected() {
    let err = run_operator(
        "extrude_operator",
        vec![wasm_artifact("simple_sphere_model"), height_config(1.0)],
    )
    .expect_err("extruding a 3D model must fail");
    assert!(err.contains("2D sketch"), "{err}");
}

#[test]
fn sketches_rasterize_for_preview() {
    let raster = volumetric::rasterize_sketch_from_bytes(&circle_sketch_wasm(), 64)
        .expect("rasterize sketch");
    assert_eq!((raster.width, raster.height), (64, 64));
    assert_eq!(raster.bounds_min, (-1.5, -1.5));
    assert_eq!(raster.bounds_max, (1.5, 1.5));
    // Center of the circle is occupied, the corner is not.
    assert!(raster.cell(32, 32));
    assert!(!raster.cell(0, 0));

    // 3D models are rejected.
    let err = volumetric::rasterize_sketch_from_bytes(&wasm_artifact("simple_sphere_model"), 8)
        .expect_err("rasterizing a 3D model must fail");
    assert!(err.to_string().contains("2D sketch"), "{err}");
}

#[test]
fn lua_3d_scripts_still_compile() {
    let src = r#"
function is_inside(x, y, z)
    if x*x + y*y + z*z <= 1.0 then
        return 1.0
    else
        return 0.0
    end
end
function get_bounds_min_x() return -1.0 end
function get_bounds_max_x() return 1.0 end
function get_bounds_min_y() return -1.0 end
function get_bounds_max_y() return 1.0 end
function get_bounds_min_z() return -1.0 end
function get_bounds_max_z() return 1.0 end
"#;
    let model = run_operator("lua_script_operator", vec![src.as_bytes().to_vec()])
        .expect("compile 3D lua model");
    let mut executor = NativeModelExecutor::new(&model).unwrap();
    assert_eq!(executor.dimensions(), 3);
    assert_eq!(executor.sample_nd(&[0.0, 0.0, 0.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[1.5, 0.0, 0.0]).unwrap(), 0.0);
}

//! Integration tests for the sketch + extrude pipeline: a 2D Lua sketch
//! compiled by lua_script_operator, lifted to 3D by extrude_operator.
//!
//! Requires the wasm32 artifacts (`cargo build-wasm`).

#![cfg(feature = "native")]

use volumetric::subspace::{Subspace, encode_subspace};
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

fn cbor_floats(fields: &[(&str, f64)]) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(
            fields
                .iter()
                .map(|(k, v)| {
                    (
                        ciborium::value::Value::Text((*k).into()),
                        ciborium::value::Value::Float(*v),
                    )
                })
                .collect(),
        ),
        &mut out,
    )
    .unwrap();
    out
}

fn height_config(height: f64) -> Vec<u8> {
    cbor_floats(&[("height", height)])
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
fn lua_module_constants_are_shared_by_helpers_sampling_and_bounds() {
    const PARAMETERIZED_SKETCH: &str = r#"
local diameter = 2.0
local radius = diameter / 2.0
local clearance = 0.125
local bound = radius + clearance

function radial_squared(x, y)
    return x*x + y*y
end

function is_inside(x, y)
    return radial_squared(x, y) <= radius*radius
end

function get_bounds_min_x() return -bound end
function get_bounds_max_x() return bound end
function get_bounds_min_y() return -bound end
function get_bounds_max_y() return bound end
"#;

    let model = run_operator(
        "lua_script_operator",
        vec![PARAMETERIZED_SKETCH.as_bytes().to_vec()],
    )
    .expect("compile parameterized sketch");
    let mut executor = NativeModelExecutor::new(&model).unwrap();

    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(0), bounds.max(0)), (-1.125, 1.125));
    assert_eq!((bounds.min(1), bounds.max(1)), (-1.125, 1.125));
    assert_eq!(executor.sample_nd(&[0.75, 0.5]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[1.01, 0.0]).unwrap(), 0.0);
}

#[test]
fn lua_f64_map_routes_shared_values_into_geometry_and_bounds() {
    const ROUTED_SKETCH: &str = r#"
local radius = 1.0 -- @param key="shared.radius" min=0.25 max=4.0
local margin = 0.5
local bound = radius + margin
function is_inside(x, y) return x*x + y*y <= radius*radius end
function get_bounds_min_x() return -bound end
function get_bounds_max_x() return bound end
function get_bounds_min_y() return -bound end
function get_bounds_max_y() return bound end
"#;
    let values = volumetric_abi::f64_map::F64Map::from([
        ("shared.radius".to_string(), 2.0),
        ("shared.unused".to_string(), 123.0),
    ]);
    let model = run_operator(
        "lua_script_operator",
        vec![
            ROUTED_SKETCH.as_bytes().to_vec(),
            volumetric_abi::f64_map::encode(&values).unwrap(),
        ],
    )
    .expect("compile sketch with routed F64Map");
    let mut executor = NativeModelExecutor::new(&model).unwrap();

    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(0), bounds.max(0)), (-2.5, 2.5));
    assert_eq!((bounds.min(1), bounds.max(1)), (-2.5, 2.5));
    assert_eq!(executor.sample_nd(&[1.5, 0.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[2.1, 0.0]).unwrap(), 0.0);
}

#[test]
fn lua_fidget_spinner_reference_has_expected_geometry() {
    let source = include_str!("../examples/fidget_spinner.lua");
    let model = run_operator("lua_script_operator", vec![source.as_bytes().to_vec()])
        .expect("compile reference spinner");
    let mut executor = NativeModelExecutor::new(&model).unwrap();

    assert_eq!(executor.dimensions(), 3);
    let bounds = executor.get_bounds_nd().unwrap();
    assert!((bounds.min(0) + 0.0515).abs() < 1.0e-12);
    assert!((bounds.max(0) - 0.0515).abs() < 1.0e-12);
    assert!((bounds.max(1) - 0.046810_889_132_455_35).abs() < 1.0e-12);
    assert_eq!(bounds.min(1), -bounds.max(1));
    assert!((bounds.min(2) + 0.0036).abs() < 1.0e-12);
    assert!((bounds.max(2) - 0.0036).abs() < 1.0e-12);

    // Four bearing seats are empty, while the surrounding rings and the
    // connecting web remain solid.
    assert_eq!(executor.sample_nd(&[0.0, 0.0, 0.0]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[0.035, 0.0, 0.0]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[0.014, 0.0, 0.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.0175, 0.0, 0.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.048, 0.0, 0.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.0175, 0.0, 0.004]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[0.06, 0.0, 0.0]).unwrap(), 0.0);
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

/// Rectangle profile in the r/z plane: r in [0.5, 1.0], z in [0.0, 2.0].
/// Revolved, it becomes a tube (hollow cylinder) around the z axis.
const RING_PROFILE: &str = r#"
function is_inside(x, y)
    if x >= 0.5 and x <= 1.0 and y >= 0.0 and y <= 2.0 then
        return 1.0
    else
        return 0.0
    end
end

function get_bounds_min_x() return 0.0 end
function get_bounds_max_x() return 1.25 end
function get_bounds_min_y() return -0.25 end
function get_bounds_max_y() return 2.25 end
"#;

#[test]
fn revolve_turns_profile_into_tube() {
    let sketch = run_operator(
        "lua_script_operator",
        vec![RING_PROFILE.as_bytes().to_vec()],
    )
    .expect("compile ring profile");
    let revolved = run_operator("revolve_operator", vec![sketch]).expect("revolve profile");

    let mut executor = NativeModelExecutor::new(&revolved).unwrap();
    assert_eq!(executor.dimensions(), 3);

    // Bounds: R = max(sketch max_x, 0) = 1.25; z takes the sketch's y bounds.
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(0), bounds.max(0)), (-1.25, 1.25));
    assert_eq!((bounds.min(1), bounds.max(1)), (-1.25, 1.25));
    assert_eq!((bounds.min(2), bounds.max(2)), (-0.25, 2.25));

    // In the tube wall, in several directions (rotational symmetry).
    assert_eq!(executor.sample_nd(&[0.75, 0.0, 1.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.0, 0.75, 1.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[-0.6, -0.45, 1.0]).unwrap(), 1.0); // r = 0.75
    // In the hole (r < 0.5).
    assert_eq!(executor.sample_nd(&[0.2, 0.0, 1.0]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[0.0, 0.0, 1.0]).unwrap(), 0.0);
    // Outside radially (r > 1.0).
    assert_eq!(executor.sample_nd(&[1.1, 0.0, 1.0]).unwrap(), 0.0);
    // Outside axially.
    assert_eq!(executor.sample_nd(&[0.75, 0.0, -0.1]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[0.75, 0.0, 2.1]).unwrap(), 0.0);
}

#[test]
fn revolving_a_3d_model_is_rejected() {
    let err = run_operator(
        "revolve_operator",
        vec![wasm_artifact("simple_sphere_model")],
    )
    .expect_err("revolving a 3D model must fail");
    assert!(err.contains("2D sketch"), "{err}");
}

#[test]
fn revolved_model_extrude_style_composes() {
    // Revolve output must behave like any 3D model downstream: translate it
    // up by 1 and check the tube moved.
    let sketch = run_operator(
        "lua_script_operator",
        vec![RING_PROFILE.as_bytes().to_vec()],
    )
    .expect("compile ring profile");
    let revolved = run_operator("revolve_operator", vec![sketch]).expect("revolve profile");

    let mut cfg = Vec::new();
    ciborium::ser::into_writer(
        &ciborium::value::Value::Map(vec![
            (
                ciborium::value::Value::Text("dx".into()),
                ciborium::value::Value::Float(0.0),
            ),
            (
                ciborium::value::Value::Text("dy".into()),
                ciborium::value::Value::Float(0.0),
            ),
            (
                ciborium::value::Value::Text("dz".into()),
                ciborium::value::Value::Float(1.0),
            ),
        ]),
        &mut cfg,
    )
    .unwrap();
    let moved = run_operator("translate_operator", vec![revolved, cfg]).expect("translate tube");

    let mut executor = NativeModelExecutor::new(&moved).unwrap();
    assert_eq!(executor.sample_nd(&[0.75, 0.0, 2.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.75, 0.0, 0.5]).unwrap(), 0.0);
}

#[test]
fn translate_adapts_to_2d_sketches() {
    // dz would overrun a 2D model's bounds buffer if the wrapper weren't
    // dimension-adaptive — pass a nonzero one to prove it's ignored safely.
    let moved = run_operator(
        "translate_operator",
        vec![
            circle_sketch_wasm(),
            cbor_floats(&[("dx", 1.0), ("dy", 0.5), ("dz", 99.0)]),
        ],
    )
    .expect("translate sketch");

    let mut executor = NativeModelExecutor::new(&moved).unwrap();
    assert_eq!(executor.dimensions(), 2);

    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!(bounds.dimensions(), 2);
    assert_eq!((bounds.min(0), bounds.max(0)), (-0.5, 2.5));
    assert_eq!((bounds.min(1), bounds.max(1)), (-1.0, 2.0));

    // The old circle center moved to (1.0, 0.5); the old origin is now on
    // the outside (distance to new center > 1).
    assert_eq!(executor.sample_nd(&[1.0, 0.5]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[2.4, 0.5]).unwrap(), 0.0);
    // Sample again after get_bounds to catch buffer-overrun corruption.
    assert_eq!(executor.sample_nd(&[1.0, 0.5]).unwrap(), 1.0);
}

/// Off-axis rectangle: 1 <= x <= 2, -0.5 <= y <= 0.5.
const OFFSET_RECT_SKETCH: &str = r#"
function is_inside(x, y)
    if x >= 1.0 and x <= 2.0 and y >= -0.5 and y <= 0.5 then
        return 1.0
    else
        return 0.0
    end
end
function get_bounds_min_x() return 0.75 end
function get_bounds_max_x() return 2.25 end
function get_bounds_min_y() return -0.75 end
function get_bounds_max_y() return 0.75 end
"#;

#[test]
fn rotation_rotates_sketches_in_plane() {
    let sketch = run_operator(
        "lua_script_operator",
        vec![OFFSET_RECT_SKETCH.as_bytes().to_vec()],
    )
    .expect("compile offset rect");
    let rotated = run_operator(
        "rotation_operator",
        vec![sketch.clone(), cbor_floats(&[("rz_deg", 90.0)])],
    )
    .expect("rotate sketch");

    let mut executor = NativeModelExecutor::new(&rotated).unwrap();
    assert_eq!(executor.dimensions(), 2);

    // The rectangle rotated from +x onto +y.
    assert_eq!(executor.sample_nd(&[0.0, 1.5]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[1.5, 0.0]).unwrap(), 0.0);

    // Bounds: center (1.5, 0) -> (0, 1.5); half-extents (0.75, 0.75) stay.
    let bounds = executor.get_bounds_nd().unwrap();
    assert!((bounds.min(0) - -0.75).abs() < 1e-9, "{}", bounds.min(0));
    assert!((bounds.max(0) - 0.75).abs() < 1e-9, "{}", bounds.max(0));
    assert!((bounds.min(1) - 0.75).abs() < 1e-9, "{}", bounds.min(1));
    assert!((bounds.max(1) - 2.25).abs() < 1e-9, "{}", bounds.max(1));

    // Out-of-plane rotation of a 2D model errors loudly instead of being
    // silently dropped.
    let err = run_operator(
        "rotation_operator",
        vec![sketch, cbor_floats(&[("rx_deg", 10.0)])],
    )
    .expect_err("rx on a 2D model must fail");
    assert!(err.contains("in-plane"), "{err}");
}

#[test]
fn scale_adapts_to_2d_sketches() {
    let scaled = run_operator(
        "scale_operator",
        vec![
            circle_sketch_wasm(),
            cbor_floats(&[("sx", 2.0), ("sy", 1.0), ("sz", 99.0)]),
        ],
    )
    .expect("scale sketch");

    let mut executor = NativeModelExecutor::new(&scaled).unwrap();
    assert_eq!(executor.dimensions(), 2);

    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(0), bounds.max(0)), (-3.0, 3.0));
    assert_eq!((bounds.min(1), bounds.max(1)), (-1.5, 1.5));

    // Ellipse: (1.8, 0) maps back to sketch (0.9, 0) — inside; (0, 1.2) is
    // outside (y unscaled).
    assert_eq!(executor.sample_nd(&[1.8, 0.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.0, 1.2]).unwrap(), 0.0);
}

#[test]
fn transformed_sketch_extrudes() {
    // The point of 2D transforms: position a sketch before lifting it.
    let moved = run_operator(
        "translate_operator",
        vec![circle_sketch_wasm(), cbor_floats(&[("dx", 1.0)])],
    )
    .expect("translate sketch");
    let extruded =
        run_operator("extrude_operator", vec![moved, height_config(1.0)]).expect("extrude");

    let mut executor = NativeModelExecutor::new(&extruded).unwrap();
    assert_eq!(executor.sample_nd(&[1.0, 0.0, 0.5]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[-0.9, 0.0, 0.5]).unwrap(), 0.0);
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

/// The plane input relocates the extrusion: the circle extruded from an
/// offset xy plane spans z in [2, 4] instead of [0, 2].
#[test]
fn extrude_onto_an_offset_plane() {
    let plane = Subspace::axis_aligned(vec![0.0, 0.0, 2.0], &[0, 1]).unwrap();
    let extruded = run_operator(
        "extrude_operator",
        vec![
            circle_sketch_wasm(),
            height_config(2.0),
            encode_subspace(&plane),
        ],
    )
    .expect("extrude onto offset plane");

    let mut executor = NativeModelExecutor::new(&extruded).unwrap();
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(2), bounds.max(2)), (2.0, 4.0));
    assert_eq!(executor.sample_nd(&[0.0, 0.0, 3.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.0, 0.0, 1.0]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[0.0, 0.0, 4.5]).unwrap(), 0.0);
}

/// A tilted plane extrudes along its oriented normal. Basis {x, (y+z)/s2}
/// has normal (0, -1, 1)/s2; the circle sweeps that way for one unit.
#[test]
fn extrude_along_a_tilted_normal() {
    let s = 0.5f64.sqrt();
    let plane = Subspace {
        dimensions: 3,
        origin: vec![0.0, 0.0, 0.0],
        basis: vec![1.0, 0.0, 0.0, 0.0, s, s],
    };
    plane.validate().unwrap();
    let extruded = run_operator(
        "extrude_operator",
        vec![
            circle_sketch_wasm(),
            height_config(1.0),
            encode_subspace(&plane),
        ],
    )
    .expect("extrude along tilted normal");

    let mut executor = NativeModelExecutor::new(&extruded).unwrap();
    // Mid-sweep (w = 0.5) at the chart origin and near the circle's rim.
    assert_eq!(executor.sample_nd(&[0.0, -0.5 * s, 0.5 * s]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.9, -0.5 * s, 0.5 * s]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[1.2, -0.5 * s, 0.5 * s]).unwrap(), 0.0);
    // Past the sweep (w = 1.5) and behind the plane (w = -0.5).
    assert_eq!(executor.sample_nd(&[0.0, -1.5 * s, 1.5 * s]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[0.0, 0.5 * s, -0.5 * s]).unwrap(), 0.0);

    // Interval bounds: chart box [-1.5, 1.5]^2 swept by (0, -s, s).
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(0), bounds.max(0)), (-1.5, 1.5));
    assert!(
        (bounds.min(1) - -2.5 * s).abs() < 1e-12,
        "{}",
        bounds.min(1)
    );
    assert!((bounds.max(1) - 1.5 * s).abs() < 1e-12, "{}", bounds.max(1));
    assert!(
        (bounds.min(2) - -1.5 * s).abs() < 1e-12,
        "{}",
        bounds.min(2)
    );
    assert!((bounds.max(2) - 2.5 * s).abs() < 1e-12, "{}", bounds.max(2));
}

/// With a plane wired, the profile is no longer restricted to 2D: a 1D
/// profile (a line slice of the sphere) extrudes to a 2D ribbon.
#[test]
fn extrude_lifts_a_1d_profile_with_a_wired_plane() {
    let line = Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[0]).unwrap();
    let profile = run_operator(
        "slice_operator",
        vec![wasm_artifact("simple_sphere_model"), encode_subspace(&line)],
    )
    .expect("slice sphere to a 1D profile");

    let plane = Subspace::axis_aligned(vec![0.0, 0.0], &[0]).unwrap();
    let extruded = run_operator(
        "extrude_operator",
        vec![profile, Vec::new(), encode_subspace(&plane)],
    )
    .expect("extrude 1D profile to 2D");

    let mut executor = NativeModelExecutor::new(&extruded).unwrap();
    assert_eq!(executor.dimensions(), 2);
    assert_eq!(executor.sample_nd(&[0.0, 0.5]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[0.9, 0.5]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[1.1, 0.5]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[0.0, 1.5]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[0.0, -0.1]).unwrap(), 0.0);
}

/// The plane's chart must match the profile: rank k in (k+1)-space.
#[test]
fn extrude_rejects_a_mismatched_plane() {
    let line = Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[2]).unwrap();
    let err = run_operator(
        "extrude_operator",
        vec![
            circle_sketch_wasm(),
            height_config(1.0),
            encode_subspace(&line),
        ],
    )
    .expect_err("a line cannot carry a 2D profile");
    assert!(err.contains("hyperplane"), "{err}");

    let plane = Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[0, 1]).unwrap();
    let err = run_operator(
        "extrude_operator",
        vec![
            wasm_artifact("simple_sphere_model"),
            height_config(1.0),
            encode_subspace(&plane),
        ],
    )
    .expect_err("a 3-space plane cannot carry a 3D profile");
    assert!(err.contains("hyperplane"), "{err}");
}

/// Typed channels survive the lift: a planar slice of the density-gradient
/// cube keeps its "infill" channel when extruded back to 3D. Outside the
/// slab the occupancy slot reads 0.
#[test]
fn extrude_preserves_channels() {
    let plane = Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[0, 1]).unwrap();
    let profile = run_operator(
        "slice_operator",
        vec![
            wasm_artifact("density_gradient_model"),
            encode_subspace(&plane),
        ],
    )
    .expect("slice the gradient cube");
    let extruded = run_operator("extrude_operator", vec![profile]).expect("extrude the slice");

    let mut model = NativeModelExecutor::new(&extruded).unwrap();
    assert_eq!(model.sample_format().channels.len(), 2);
    let row = model.sample_channels_nd(&[0.5, 0.0, 0.5]).unwrap();
    assert_eq!(row[0], 1.0);
    assert!((row[1] - 0.75).abs() < 1e-6, "infill was {}", row[1]);
    // Above the slab: occupancy forced to 0.
    let row = model.sample_channels_nd(&[0.5, 0.0, 1.5]).unwrap();
    assert_eq!(row[0], 0.0);
}

/// The axis input relocates the revolution: the ring profile around a line
/// through (2, 0, 0) along z becomes a tube centered on x = 2.
#[test]
fn revolve_around_a_wired_offset_axis() {
    let sketch = run_operator(
        "lua_script_operator",
        vec![RING_PROFILE.as_bytes().to_vec()],
    )
    .expect("compile ring profile");
    let axis = Subspace::axis_aligned(vec![2.0, 0.0, 0.0], &[2]).unwrap();
    let revolved = run_operator("revolve_operator", vec![sketch, encode_subspace(&axis)])
        .expect("revolve around offset axis");

    let mut executor = NativeModelExecutor::new(&revolved).unwrap();
    assert_eq!(executor.dimensions(), 3);
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(0), bounds.max(0)), (0.75, 3.25));
    assert_eq!((bounds.min(1), bounds.max(1)), (-1.25, 1.25));
    assert_eq!((bounds.min(2), bounds.max(2)), (-0.25, 2.25));

    assert_eq!(executor.sample_nd(&[2.75, 0.0, 1.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[2.0, 0.75, 1.0]).unwrap(), 1.0);
    // The hole moved with the axis: the world origin's old tube is gone.
    assert_eq!(executor.sample_nd(&[2.0, 0.0, 1.0]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[0.75, 0.0, 1.0]).unwrap(), 0.0);
}

/// Revolving around the x axis: the profile's second coordinate runs along
/// world x and the radius spans the yz plane.
#[test]
fn revolve_around_the_x_axis() {
    let sketch = run_operator(
        "lua_script_operator",
        vec![RING_PROFILE.as_bytes().to_vec()],
    )
    .expect("compile ring profile");
    let axis = Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[0]).unwrap();
    let revolved = run_operator("revolve_operator", vec![sketch, encode_subspace(&axis)])
        .expect("revolve around x");

    let mut executor = NativeModelExecutor::new(&revolved).unwrap();
    let bounds = executor.get_bounds_nd().unwrap();
    assert_eq!((bounds.min(0), bounds.max(0)), (-0.25, 2.25));
    assert_eq!((bounds.min(1), bounds.max(1)), (-1.25, 1.25));
    assert_eq!((bounds.min(2), bounds.max(2)), (-1.25, 1.25));

    assert_eq!(executor.sample_nd(&[1.0, 0.75, 0.0]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[1.0, 0.0, -0.75]).unwrap(), 1.0);
    assert_eq!(executor.sample_nd(&[1.0, 0.2, 0.0]).unwrap(), 0.0);
    assert_eq!(executor.sample_nd(&[-0.5, 0.75, 0.0]).unwrap(), 0.0);
}

/// The axis must have rank k-1 in (k+1)-space.
#[test]
fn revolve_rejects_a_mismatched_axis() {
    let plane = Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[0, 1]).unwrap();
    let err = run_operator(
        "revolve_operator",
        vec![circle_sketch_wasm(), encode_subspace(&plane)],
    )
    .expect_err("a plane is not a revolution axis for a 2D profile");
    assert!(err.contains("rank 1"), "{err}");
}

/// Typed channels survive the revolution: the sliced gradient's "infill"
/// channel reads at (r, a), so it becomes radially symmetric.
#[test]
fn revolve_preserves_channels() {
    let plane = Subspace::axis_aligned(vec![0.0, 0.0, 0.0], &[0, 1]).unwrap();
    let profile = run_operator(
        "slice_operator",
        vec![
            wasm_artifact("density_gradient_model"),
            encode_subspace(&plane),
        ],
    )
    .expect("slice the gradient cube");
    let revolved = run_operator("revolve_operator", vec![profile]).expect("revolve the slice");

    let mut model = NativeModelExecutor::new(&revolved).unwrap();
    assert_eq!(model.sample_format().channels.len(), 2);
    // r = 0.5 in any direction reads the profile at (0.5, 0): infill 0.75.
    for p in [[0.5, 0.0, 0.0], [0.0, -0.5, 0.0], [0.3, 0.4, 0.0]] {
        let row = model.sample_channels_nd(&p).unwrap();
        assert_eq!(row[0], 1.0, "at {p:?}");
        assert!(
            (row[1] - 0.75).abs() < 1e-6,
            "infill at {p:?} was {}",
            row[1]
        );
    }
}

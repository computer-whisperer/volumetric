//! Demand-driven integration coverage for the parametric Raspberry Pi 4 tray.
//!
//! Requires `cargo build-wasm`.

#![cfg(feature = "native")]

use volumetric::wasm::{
    NativeModelExecutor, OperatorExecutor, OperatorIo, create_operator_executor,
};
use volumetric_abi::f64_map::{F64Map, encode};

fn wasm_artifact(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-unknown-unknown/release")
        .join(format!("{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing wasm artifact {} ({error}); build it with `cargo build-wasm`",
            path.display()
        )
    })
}

fn compile_tray(parameters: F64Map) -> NativeModelExecutor {
    let operator_wasm = wasm_artifact("lua_script_operator");
    let mut operator = create_operator_executor(&operator_wasm).expect("create Lua operator");
    let source = include_bytes!("../examples/raspberry_pi_4_tray.lua").to_vec();
    let parameter_bytes = encode(&parameters).expect("encode parameters");
    let result = operator
        .run(OperatorIo::new(vec![source, parameter_bytes]))
        .expect("compile Pi tray");
    let model = result.outputs.get(&0).expect("Lua model output");
    NativeModelExecutor::new(model).expect("instantiate Pi tray")
}

#[test]
fn tray_has_shell_standoffs_ports_and_vents() {
    let mut tray = compile_tray(F64Map::new());
    assert_eq!(tray.dimensions(), 3);

    let bounds = tray.get_bounds_nd().unwrap();
    assert!((bounds.min(0) + 0.0025).abs() < 1.0e-12);
    assert!((bounds.max(0) - 0.0875).abs() < 1.0e-12);
    assert!((bounds.min(1) + 0.0025).abs() < 1.0e-12);
    assert!((bounds.max(1) - 0.0585).abs() < 1.0e-12);
    assert_eq!((bounds.min(2), bounds.max(2)), (0.0, 0.023));

    // Open cavity and retained left wall.
    assert_eq!(tray.sample_nd(&[0.0425, 0.028, 0.01]).unwrap(), 0.0);
    assert_eq!(tray.sample_nd(&[-0.0015, 0.028, 0.01]).unwrap(), 1.0);

    // Through-bored lower-left standoff.
    assert_eq!(tray.sample_nd(&[0.0035, 0.0035, 0.003]).unwrap(), 0.0);
    assert_eq!(tray.sample_nd(&[0.0057, 0.0035, 0.003]).unwrap(), 1.0);

    // A floor vent is open while material between vent rows remains.
    assert_eq!(tray.sample_nd(&[0.0425, 0.012, 0.001]).unwrap(), 0.0);
    assert_eq!(tray.sample_nd(&[0.0425, 0.016, 0.001]).unwrap(), 1.0);

    // USB-C and lower USB connector apertures cut two different walls.
    assert_eq!(tray.sample_nd(&[0.010, -0.0015, 0.007]).unwrap(), 0.0);
    assert_eq!(tray.sample_nd(&[0.020, -0.0015, 0.007]).unwrap(), 1.0);
    assert_eq!(tray.sample_nd(&[0.0865, 0.010, 0.010]).unwrap(), 0.0);
}

#[test]
fn routed_wall_thickness_updates_geometry_and_bounds_together() {
    let mut tray = compile_tray(F64Map::from([("case.wall_thickness".to_string(), 0.003)]));
    let bounds = tray.get_bounds_nd().unwrap();
    assert!((bounds.min(0) + 0.0035).abs() < 1.0e-12);
    assert!((bounds.max(0) - 0.0885).abs() < 1.0e-12);
    assert_eq!(tray.sample_nd(&[-0.0025, 0.028, 0.01]).unwrap(), 1.0);
}

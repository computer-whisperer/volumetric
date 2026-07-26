//! Sampler failure accounting: traps inside a model are counted (and still
//! read as "outside" per the ABI), and the mesh pipeline surfaces them
//! without failing. Instantiation failures — the environment-level failure
//! mode that fabricates "outside" without consulting the model — hard-fail
//! the mesh; that path needs induced resource exhaustion (verified manually
//! under `ulimit -v`), so here we only pin the healthy-run counters to zero.

#![cfg(feature = "native")]

use volumetric::adaptive_surface_nets_2::AdaptiveMeshConfig2;
use volumetric::wasm::{ParallelModelSampler, create_parallel_sampler};

/// A 3D model over bounds [-1, 1]³ that occupies the x <= 0 half and TRAPS
/// (unreachable) for any sample with x > 0.
const TRAP_HALF_MODEL: &str = r#"(module
    (memory (export "memory") 1)
    (func (export "get_dimensions") (result i32) (i32.const 3))
    (func (export "get_io_ptr") (result i32) (i32.const 1024))
    (func (export "get_bounds") (param $p i32)
        (f64.store (local.get $p) (f64.const -1))
        (f64.store offset=8 (local.get $p) (f64.const 1))
        (f64.store offset=16 (local.get $p) (f64.const -1))
        (f64.store offset=24 (local.get $p) (f64.const 1))
        (f64.store offset=32 (local.get $p) (f64.const -1))
        (f64.store offset=40 (local.get $p) (f64.const 1)))
    (func (export "sample") (param $p i32) (result f32)
        (if (f64.gt (f64.load (local.get $p)) (f64.const 0))
            (then unreachable))
        (f32.const 1)))"#;

#[test]
fn traps_are_counted_and_read_as_outside() {
    let wasm = wat::parse_str(TRAP_HALF_MODEL).unwrap();
    let sampler = create_parallel_sampler(&wasm).unwrap();

    assert_eq!(sampler.sample(-0.5, 0.0, 0.0), 1.0);
    assert_eq!(sampler.sample_traps(), 0);
    assert_eq!(sampler.instantiation_failures(), 0);

    assert_eq!(sampler.sample(0.5, 0.0, 0.0), 0.0, "trap reads as outside");
    assert_eq!(sampler.sample_traps(), 1);

    // The store survives a trap: the same thread keeps sampling correctly.
    assert_eq!(sampler.sample(-0.5, 0.0, 0.0), 1.0);
    assert_eq!(sampler.sample(0.25, 0.25, 0.25), 0.0);
    assert_eq!(sampler.sample_traps(), 2);
    assert_eq!(sampler.instantiation_failures(), 0);
}

#[test]
fn meshing_a_trapping_model_succeeds_and_meshes_the_healthy_half() {
    let config = AdaptiveMeshConfig2 {
        base_resolution: 8,
        max_depth: 2,
        ..Default::default()
    };
    let wasm = wat::parse_str(TRAP_HALF_MODEL).unwrap();
    let result = volumetric::generate_adaptive_mesh_v2_from_bytes(&wasm, &config)
        .expect("model traps read as outside; meshing must not fail");

    assert!(!result.indices.is_empty(), "the x <= 0 half must mesh");
    let max_x = result
        .vertices
        .iter()
        .map(|v| v.0)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_x = result
        .vertices
        .iter()
        .map(|v| v.0)
        .fold(f32::INFINITY, f32::min);
    // The occupied half spans x in [-1, 0]; allow one padded-grid cell of
    // meshing slack on either side.
    assert!(
        min_x < -0.9 && max_x < 0.15,
        "mesh should cover [-1, 0], got [{min_x}, {max_x}]"
    );
}

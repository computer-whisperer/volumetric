//! `model_dimensions_static`: dimensionality from the wasm binary alone.

use volumetric::model_dimensions_static;

fn module(dimensions: &str) -> Vec<u8> {
    wat::parse_str(format!(
        r#"(module
            (memory (export "memory") 1)
            (func (export "get_dimensions") (result i32) {dimensions})
            (func (export "get_io_ptr") (result i32) (i32.const 1024))
        )"#
    ))
    .expect("test module assembles")
}

#[test]
fn reads_constant_dimensions() {
    assert_eq!(model_dimensions_static(&module("(i32.const 2)")), Some(2));
    assert_eq!(model_dimensions_static(&module("(i32.const 3)")), Some(3));
}

#[test]
fn non_constant_bodies_and_garbage_are_none() {
    // Computed dimensionality: not a bare constant.
    assert_eq!(
        model_dimensions_static(&module("(i32.add (i32.const 1) (i32.const 1))")),
        None
    );
    // Negative constant is nonsense.
    assert_eq!(model_dimensions_static(&module("(i32.const -1)")), None);
    // Missing export.
    let no_export = wat::parse_str(r#"(module (memory (export "memory") 1))"#).unwrap();
    assert_eq!(model_dimensions_static(&no_export), None);
    // Not wasm at all.
    assert_eq!(model_dimensions_static(b"not wasm"), None);
}

#[test]
fn imported_functions_do_not_shift_the_index() {
    // An imported function precedes local ones in the index space; the
    // exported get_dimensions must still resolve to the right body.
    let bytes = wat::parse_str(
        r#"(module
            (import "env" "host_fn" (func (result i32)))
            (memory (export "memory") 1)
            (func (export "get_dimensions") (result i32) (i32.const 2))
        )"#,
    )
    .unwrap();
    assert_eq!(model_dimensions_static(&bytes), Some(2));
}

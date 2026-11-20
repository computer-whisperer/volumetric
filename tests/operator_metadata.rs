//! Verify every bundled operator's get_metadata() decodes against the host's
//! metadata types (the CBOR contract is name-based, so type drift between an
//! operator and the host shows up here).
//!
//! Requires the wasm32 artifacts:
//!   cargo build --target wasm32-unknown-unknown --release -p <operator>

#![cfg(feature = "native")]

use volumetric::operator_metadata_from_wasm_bytes;

/// Every operator crate under `crates/operators/` — the directory is the
/// source of truth for what is bundled, so a new operator is covered
/// without editing a list here (a missing artifact fails the test).
fn operator_names() -> Vec<String> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/operators");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("listing {}: {e}", dir.display()))
        .map(|entry| {
            entry
                .expect("directory entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name.ends_with("_operator"))
        .collect();
    names.sort();
    assert!(names.len() > 40, "unexpectedly few operators: {names:?}");
    names
}

#[test]
fn every_operator_metadata_decodes() {
    for name in &operator_names() {
        let metadata = metadata_of(name);

        assert_eq!(&metadata.name, name, "{name}: unexpected metadata name");
        assert!(
            !metadata.inputs.is_empty(),
            "{name}: metadata declares no inputs"
        );
        assert!(
            !metadata.outputs.is_empty(),
            "{name}: metadata declares no outputs"
        );

        // Every declared config schema must parse with the host's CDDL
        // reader — an unparseable schema means no config form in the UI.
        for (idx, input) in metadata.inputs.iter().enumerate() {
            if let volumetric::OperatorMetadataInput::CBORConfiguration(schema) = input {
                volumetric::operator_config::parse_schema(schema).unwrap_or_else(|e| {
                    panic!("{name}: config schema (input {idx}) failed to parse: {e:?}")
                });
            }
        }
    }
}

fn metadata_of(name: &str) -> volumetric::OperatorMetadata {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-unknown-unknown/release")
        .join(format!("{name}.wasm"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing wasm artifact {} ({e}); build it with \
             `cargo build --target wasm32-unknown-unknown --release -p {name}`",
            path.display()
        )
    });
    operator_metadata_from_wasm_bytes(&bytes)
        .unwrap_or_else(|e| panic!("{name}: metadata failed to decode: {e}"))
}

/// `input_constraints` is parallel to `inputs` by position, so an entry
/// put against the wrong slot is silently ignored: every declared shape
/// must sit on a Subspace slot, and the list must not outrun the slots.
/// A declared shape must also be one some subspace has: ranks in order
/// and within the ambient dimension.
#[test]
fn every_declared_subspace_shape_sits_on_a_subspace_slot() {
    for name in &operator_names() {
        let metadata = metadata_of(name);
        assert!(
            metadata.input_constraints.len() <= metadata.inputs.len(),
            "{name}: {} constraints for {} inputs",
            metadata.input_constraints.len(),
            metadata.inputs.len()
        );
        for (slot, constraint) in metadata.input_constraints.iter().enumerate() {
            assert!(
                constraint.subspace.is_any()
                    || matches!(
                        metadata.inputs[slot],
                        volumetric::OperatorMetadataInput::Subspace
                    ),
                "{name}: input {slot} declares a subspace shape but is {:?}",
                metadata.inputs[slot]
            );
            let shape = constraint.subspace;
            let lowest = shape.min_rank.unwrap_or(0);
            let highest = shape
                .max_rank
                .unwrap_or(u32::MAX)
                .min(shape.ambient.unwrap_or(u32::MAX));
            assert!(
                lowest <= highest,
                "{name}: input {slot} declares a shape nothing fits: {shape:?}"
            );
        }
    }
}

/// What each bundled operator with a Subspace input declares for it, as
/// a host labels the slot. The ones that declare nothing take a shape
/// that depends on another input, or any shape at all. An operator that
/// gains a Subspace input fails here until it is given a line.
#[test]
fn bundled_operators_declare_their_subspace_inputs() {
    let labels = |name: &str| -> Vec<String> {
        let metadata = metadata_of(name);
        (0..metadata.inputs.len())
            .filter(|&slot| {
                matches!(
                    metadata.inputs[slot],
                    volumetric::OperatorMetadataInput::Subspace
                )
            })
            .map(|slot| volumetric::project_edit::slot_type_label(&metadata, slot))
            .collect()
    };
    let with_a_subspace_input: Vec<String> = operator_names()
        .into_iter()
        .filter(|name| !labels(name).is_empty())
        .collect();
    assert_eq!(
        with_a_subspace_input,
        [
            "brim_operator",
            "cloud_fit_operator",
            "extrude_operator",
            "intersect_operator",
            "mechanism_operator",
            "revolve_operator",
            "slice_operator",
            "span_operator",
        ]
    );
    assert_eq!(labels("brim_operator"), ["Subspace (plane in 3-space)"]);
    assert_eq!(
        labels("mechanism_operator"),
        ["Subspace (rank 1 or more in 3-space)"]
    );
    assert_eq!(
        labels("cloud_fit_operator"),
        ["Subspace (rank 0 to 2 in 3-space)"]
    );
    assert_eq!(labels("slice_operator"), ["Subspace (rank 1 or more)"]);
    for open in [
        "extrude_operator",
        "revolve_operator",
        "span_operator",
        "intersect_operator",
    ] {
        assert_eq!(labels(open), ["Subspace"], "{open}");
    }
}

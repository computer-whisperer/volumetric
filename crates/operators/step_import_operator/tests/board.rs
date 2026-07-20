//! Integration test against the representative KiCad board export at
//! the repository root. Skips (with a note) when the fixture is absent
//! so CI without the file stays green.

use step_import_operator::{StepConfig, import};

fn board_text() -> Option<String> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../board.step");
    std::fs::read_to_string(path).ok()
}

#[test]
fn imports_the_representative_board() {
    let Some(text) = board_text() else {
        eprintln!("board.step fixture not present; skipping");
        return;
    };
    let model = import(&text, &StepConfig::default()).expect("import board.step");

    // 121 manifold solids exist in the file; each is converted once and
    // placed at least once.
    assert!(
        model.solids.len() >= 100,
        "expected ~121 distinct solids, got {}",
        model.solids.len()
    );
    assert!(
        model.instances.len() >= model.solids.len(),
        "instances {} < solids {}",
        model.instances.len(),
        model.solids.len()
    );

    let payload = brep_core::payload::build_payload(&model).expect("build payload");
    let view = brep_core::payload::PayloadView::new(&payload).unwrap();

    // OCCT (DRAWEXE, `bounding`) reports the assembled model, in the
    // file's millimetres, as x [-17.02, 17.02], y [-21.02, 12.84],
    // z [-0.02, 5.28] — padded by OCCT's own box tolerance (~0.021 on
    // every side: the PCB bottom is exactly z = 0). We import to metres
    // (the engine's canonical unit). Both boxes are conservative, so
    // agreement within 0.05mm per component is the meaningful check.
    // Ours are conservative (curved-face AABBs inflate by a sampling
    // margin), so: contain OCCT's un-padded box, exceed it by < 0.6mm.
    const M_PER_MM: f64 = 1e-3;
    let b = view.bounds();
    let occt_mm = [-17.02, 17.02, -21.02, 12.84, -0.02, 5.28];
    for axis in 0..3 {
        let o_lo = occt_mm[axis * 2] * M_PER_MM;
        let o_hi = occt_mm[axis * 2 + 1] * M_PER_MM;
        let (lo, hi) = (b[axis * 2], b[axis * 2 + 1]);
        assert!(
            lo <= o_lo + 0.05 * M_PER_MM && lo >= o_lo - 0.6 * M_PER_MM,
            "axis {axis} min: ours {lo} vs OCCT {o_lo}"
        );
        assert!(
            hi >= o_hi - 0.05 * M_PER_MM && hi <= o_hi + 0.6 * M_PER_MM,
            "axis {axis} max: ours {hi} vs OCCT {o_hi}"
        );
    }

    // Spot classification: the board substrate around the origin is
    // solid (PCB is 1.6mm thick starting at z=0); far away is empty.
    assert!(view.is_inside([0.0, 0.0, 0.0008]), "PCB interior");
    assert!(!view.is_inside([0.0, 0.0, 0.1]), "far above");
    assert!(!view.is_inside([0.05, 0.0, 0.0008]), "off the board");

    // Label filtering by reference designator.
    let filtered = import(
        &text,
        &StepConfig {
            include_labels: vec!["J7".to_string()],
            ..StepConfig::default()
        },
    )
    .expect("filtered import");
    assert!(
        !filtered.instances.is_empty() && filtered.instances.len() < model.instances.len(),
        "J7 filter: {} instances",
        filtered.instances.len()
    );
    for inst in &filtered.instances {
        assert_eq!(inst.label, "J7");
    }
}

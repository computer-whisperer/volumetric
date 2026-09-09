//! Reads real slicer/CAD output. The FreeCAD button is checked in under
//! tests/fixtures; the SuperSlicer calibration cube and the rest of the
//! thermo_scope housing set are local-only and skip when absent.

use std::collections::HashMap;

use threemf_core::{TriMesh, Unit, read_3mf, write_3mf};

fn repo_fixture(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {}: {e}", path.display()))
}

fn local_fixture(path: &str) -> Option<Vec<u8>> {
    let home = std::env::var("HOME").ok()?;
    std::fs::read(format!("{home}/{path}")).ok()
}

/// Every edge shared by exactly two triangles, traversed in opposite
/// directions: the closed, consistently wound surface a solid needs.
/// Checked after welding bit-identical vertices — the reader keeps the
/// file's vertex list, and CAD exporters write seam vertices twice.
fn is_closed_and_consistently_wound(mesh: &TriMesh) -> bool {
    let mesh = TriMesh::from_soup(
        (0..mesh.triangle_count()).map(|t| mesh.triangle(t).map(|v| mesh.position(v as usize))),
    );
    let mut directed: HashMap<(u32, u32), usize> = HashMap::new();
    for t in 0..mesh.triangle_count() {
        let [a, b, c] = mesh.triangle(t);
        for (from, to) in [(a, b), (b, c), (c, a)] {
            *directed.entry((from, to)).or_default() += 1;
        }
    }
    directed
        .iter()
        .all(|(&(from, to), &count)| count == 1 && directed.get(&(to, from)) == Some(&1))
}

#[test]
fn freecad_button_is_a_closed_millimetre_solid() {
    let file = read_3mf(&repo_fixture("freecad_button.3mf")).unwrap();
    assert_eq!(file.unit, Unit::Millimeter);
    assert_eq!(file.items.len(), 1);
    let mesh = &file.items[0];
    assert_eq!(mesh.triangle_count(), 108);
    assert_eq!(mesh.vertex_count(), 94, "the file's vertices, unwelded");
    assert!(
        is_closed_and_consistently_wound(mesh),
        "closed solid expected"
    );
    // The build item translates the object by (34.55, 65, 10.5): a small
    // button lands within a few centimetres of that.
    let b = mesh.bounds().unwrap();
    assert!(b[0] > 20.0 && b[1] < 50.0, "x bounds {:?}", &b[0..2]);
    assert!(b[2] > 50.0 && b[3] < 80.0, "y bounds {:?}", &b[2..4]);
    assert!(b[4] > 5.0 && b[5] < 20.0, "z bounds {:?}", &b[4..6]);

    // Writing it back preserves connectivity and, to f32 precision (what
    // the writer emits), the coordinates.
    let bytes = write_3mf(mesh, file.unit, "button").unwrap();
    let again = &read_3mf(&bytes).unwrap().items[0];
    assert_eq!(again.indices, mesh.indices);
    for (a, b) in again.positions.iter().zip(&mesh.positions) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

#[test]
fn superslicer_calibration_cube_reads() {
    let Some(bytes) = local_fixture("Calibration cube.3mf") else {
        eprintln!("Calibration cube.3mf not present; skipping");
        return;
    };
    let file = read_3mf(&bytes).unwrap();
    assert_eq!(file.unit, Unit::Millimeter);
    assert_eq!(file.items.len(), 1);
    let mesh = &file.items[0];
    assert!(mesh.triangle_count() > 100);
    assert!(is_closed_and_consistently_wound(mesh));
    // A 20 mm calibration cube, placed on the bed at z = 0.
    let b = mesh.bounds().unwrap();
    assert!(
        (b[1] - b[0] - 20.0).abs() < 1e-3,
        "x extent {}",
        b[1] - b[0]
    );
    assert!(
        (b[3] - b[2] - 20.0).abs() < 1e-3,
        "y extent {}",
        b[3] - b[2]
    );
    assert!(
        (b[5] - b[4] - 20.0).abs() < 1e-3,
        "z extent {}",
        b[5] - b[4]
    );
    assert!(b[4].abs() < 1e-3, "sits on the bed, z min {}", b[4]);
}

#[test]
fn thermo_scope_housing_set_reads() {
    let dir = match std::env::var("HOME") {
        Ok(home) => format!("{home}/workspace/thermo_scope/hardware"),
        Err(_) => return,
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("{dir} not present; skipping");
        return;
    };
    let mut seen = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("3mf") {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        if bytes.is_empty() {
            continue; // one zero-byte placeholder exists in that directory
        }
        let file = read_3mf(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(file.items.len(), 1, "{}", path.display());
        assert!(
            file.items[0].triangle_count() > 0,
            "{}: empty",
            path.display()
        );
        seen += 1;
    }
    eprintln!("read {seen} housing files");
}

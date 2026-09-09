//! TriMesh → 3MF package.

use std::fmt::Write as _;

use crate::zip::Writer;
use crate::{TriMesh, Unit};

const CORE_NS: &str = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02";

const CONTENT_TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"#;

const RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"#;

fn escape_xml(text: &str, out: &mut String) {
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
}

/// The model part's XML: one mesh object, one build item placing it.
/// Coordinates are written as f32 (the precision every consumer keeps).
/// Triangles that repeat a vertex index are dropped — the spec forbids
/// them and strict readers reject the whole file over one.
fn model_xml(mesh: &TriMesh, unit: Unit, title: &str) -> String {
    let mut xml = String::with_capacity(64 + mesh.vertex_count() * 40 + mesh.triangle_count() * 36);
    let _ = writeln!(xml, r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    let _ = writeln!(
        xml,
        r#"<model unit="{}" xml:lang="en-US" xmlns="{CORE_NS}">"#,
        unit.name()
    );
    xml.push_str(r#" <metadata name="Title">"#);
    escape_xml(title, &mut xml);
    xml.push_str("</metadata>\n");
    xml.push_str(" <metadata name=\"Application\">volumetric</metadata>\n");
    xml.push_str(" <resources>\n");
    xml.push_str(r#"  <object id="1" name=""#);
    escape_xml(title, &mut xml);
    xml.push_str("\" type=\"model\">\n   <mesh>\n    <vertices>\n");
    for v in 0..mesh.vertex_count() {
        let [x, y, z] = mesh.position(v);
        let _ = writeln!(
            xml,
            r#"     <vertex x="{}" y="{}" z="{}"/>"#,
            x as f32, y as f32, z as f32
        );
    }
    xml.push_str("    </vertices>\n    <triangles>\n");
    for t in 0..mesh.triangle_count() {
        let [a, b, c] = mesh.triangle(t);
        if a == b || b == c || a == c {
            continue;
        }
        let _ = writeln!(xml, r#"     <triangle v1="{a}" v2="{b}" v3="{c}"/>"#);
    }
    xml.push_str("    </triangles>\n   </mesh>\n  </object>\n </resources>\n <build>\n");
    xml.push_str("  <item objectid=\"1\"/>\n </build>\n</model>\n");
    xml
}

/// Encodes a mesh as a 3MF package with coordinates in `unit` (the caller
/// scales them beforehand; this only labels them). `title` names the
/// object. Fails for a structurally invalid mesh or one with no triangles.
pub fn write_3mf(mesh: &TriMesh, unit: Unit, title: &str) -> Result<Vec<u8>, String> {
    mesh.validate()?;
    if mesh.triangle_count() == 0 {
        return Err("mesh has no triangles".to_string());
    }
    let mut writer = Writer::new();
    writer.add("[Content_Types].xml", CONTENT_TYPES.as_bytes());
    writer.add("_rels/.rels", RELS.as_bytes());
    writer.add("3D/3dmodel.model", model_xml(mesh, unit, title).as_bytes());
    Ok(writer.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read_3mf;

    fn tetrahedron() -> TriMesh {
        TriMesh {
            positions: vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            ],
            indices: vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
            vertex_fields: vec![],
            face_fields: vec![],
        }
    }

    #[test]
    fn round_trips_through_the_reader() {
        let mesh = tetrahedron();
        let bytes = write_3mf(&mesh, Unit::Millimeter, "tet <\"&\">").unwrap();
        let file = read_3mf(&bytes).unwrap();
        assert_eq!(file.unit, Unit::Millimeter);
        assert_eq!(file.items.len(), 1);
        assert_eq!(file.items[0], mesh);

        let bytes = write_3mf(&mesh, Unit::Meter, "tet").unwrap();
        assert_eq!(read_3mf(&bytes).unwrap().unit, Unit::Meter);
    }

    #[test]
    fn drops_degenerate_triangles_and_refuses_empty_meshes() {
        let mut mesh = tetrahedron();
        mesh.indices.extend([0, 0, 1]);
        let bytes = write_3mf(&mesh, Unit::Millimeter, "tet").unwrap();
        assert_eq!(read_3mf(&bytes).unwrap().items[0].triangle_count(), 4);

        mesh.indices.clear();
        assert!(write_3mf(&mesh, Unit::Millimeter, "tet").is_err());
    }

    #[test]
    fn package_has_the_conventional_parts() {
        let bytes = write_3mf(&tetrahedron(), Unit::Millimeter, "tet").unwrap();
        let archive = crate::zip::Archive::parse(&bytes).unwrap();
        assert_eq!(
            archive.names().collect::<Vec<_>>(),
            ["[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"]
        );
        let model = String::from_utf8(archive.read("3D/3dmodel.model").unwrap()).unwrap();
        assert!(model.contains(r#"<model unit="millimeter""#));
        assert!(model.contains(r#"<vertex x="1" y="0" z="0"/>"#));
        assert!(model.contains(r#"<item objectid="1"/>"#));
    }
}

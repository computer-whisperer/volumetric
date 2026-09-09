//! 3MF package → flattened build items.

use std::collections::HashMap;

use crate::zip::Archive;
use crate::{TriMesh, Unit};

const REL_TYPE_MODEL: &str = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel";
const PRODUCTION_NS: &str = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06";
const DEFAULT_MODEL_PART: &str = "3D/3dmodel.model";
/// Component nesting deeper than this is treated as a reference cycle.
const MAX_COMPONENT_DEPTH: usize = 64;

/// A read 3MF file: the model's unit and one flattened mesh per build
/// item, in that unit, with the item's transform applied.
#[derive(Clone, Debug, PartialEq)]
pub struct ThreeMf {
    pub unit: Unit,
    pub items: Vec<TriMesh>,
}

/// A 3MF affine transform: the spec's 4x3 row-vector matrix, so a point
/// maps as `p' = p * M` with `m[9..12]` the translation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform(pub [f64; 12]);

impl Transform {
    pub const IDENTITY: Transform =
        Transform([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

    fn parse(text: &str) -> Result<Transform, String> {
        let mut m = [0.0f64; 12];
        let mut parts = text.split_whitespace();
        for (i, slot) in m.iter_mut().enumerate() {
            let part = parts
                .next()
                .ok_or_else(|| format!("transform {text:?} has fewer than 12 numbers"))?;
            *slot = part
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| format!("transform {text:?}: bad number {part:?} at {i}"))?;
        }
        if parts.next().is_some() {
            return Err(format!("transform {text:?} has more than 12 numbers"));
        }
        Ok(Transform(m))
    }

    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let m = &self.0;
        [
            p[0] * m[0] + p[1] * m[3] + p[2] * m[6] + m[9],
            p[0] * m[1] + p[1] * m[4] + p[2] * m[7] + m[10],
            p[0] * m[2] + p[1] * m[5] + p[2] * m[8] + m[11],
        ]
    }

    /// `self` followed by `next`: `p * self * next`.
    pub fn then(&self, next: &Transform) -> Transform {
        let a = &self.0;
        let b = &next.0;
        let mut out = [0.0f64; 12];
        for row in 0..3 {
            for col in 0..3 {
                out[row * 3 + col] =
                    a[row * 3] * b[col] + a[row * 3 + 1] * b[3 + col] + a[row * 3 + 2] * b[6 + col];
            }
        }
        for col in 0..3 {
            out[9 + col] = a[9] * b[col] + a[10] * b[3 + col] + a[11] * b[6 + col] + b[9 + col];
        }
        Transform(out)
    }

    /// Whether the linear part mirrors (negative determinant): such a
    /// placement reverses triangle winding.
    pub fn mirrors(&self) -> bool {
        let m = &self.0;
        let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6]);
        det < 0.0
    }
}

struct Mesh {
    positions: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
}

struct Component {
    /// Package path of the part holding the referenced object (production
    /// extension `p:path`); `None` for the same part.
    path: Option<String>,
    object_id: u32,
    transform: Transform,
}

struct Object {
    mesh: Option<Mesh>,
    components: Vec<Component>,
}

/// One parsed model part.
struct Part {
    unit: Option<Unit>,
    objects: HashMap<u32, Object>,
    build: Vec<(u32, Transform)>,
}

/// Normalises a package path: absolute within the package, no leading
/// slash (ZIP entry names have none). Relative references resolve against
/// the referencing part's directory.
fn resolve_path(reference: &str, from_part: &str) -> String {
    if let Some(absolute) = reference.strip_prefix('/') {
        absolute.to_string()
    } else if let Some((dir, _)) = from_part.rsplit_once('/') {
        format!("{dir}/{reference}")
    } else {
        reference.to_string()
    }
}

/// The package's start part: the `.rels` relationship of the 3D-model type,
/// falling back to the conventional `3D/3dmodel.model`.
fn start_part(archive: &Archive) -> Result<String, String> {
    if let Ok(rels) = archive.read("_rels/.rels") {
        let text = String::from_utf8_lossy(&rels);
        if let Ok(doc) = roxmltree::Document::parse(&text) {
            for node in doc.descendants().filter(|n| n.is_element()) {
                if node.tag_name().name() == "Relationship"
                    && node.attribute("Type") == Some(REL_TYPE_MODEL)
                    && let Some(target) = node.attribute("Target")
                {
                    return Ok(resolve_path(target, ""));
                }
            }
        }
    }
    if archive.contains(DEFAULT_MODEL_PART) {
        return Ok(DEFAULT_MODEL_PART.to_string());
    }
    Err("package has no 3D model part (no 3dmodel relationship, no 3D/3dmodel.model)".to_string())
}

fn attr_number(node: roxmltree::Node, name: &str) -> Result<f64, String> {
    let text = node
        .attribute(name)
        .ok_or_else(|| format!("<{}> lacks attribute {name}", node.tag_name().name()))?;
    text.trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .ok_or_else(|| {
            format!(
                "<{}> {name}={text:?} is not a number",
                node.tag_name().name()
            )
        })
}

fn attr_index(node: roxmltree::Node, name: &str) -> Result<u32, String> {
    let text = node
        .attribute(name)
        .ok_or_else(|| format!("<{}> lacks attribute {name}", node.tag_name().name()))?;
    text.trim().parse::<u32>().map_err(|_| {
        format!(
            "<{}> {name}={text:?} is not an index",
            node.tag_name().name()
        )
    })
}

fn parse_mesh(node: roxmltree::Node) -> Result<Mesh, String> {
    let mut positions = Vec::new();
    let mut triangles = Vec::new();
    for child in node.children().filter(|n| n.is_element()) {
        match child.tag_name().name() {
            "vertices" => {
                for vertex in child.children().filter(|n| n.is_element()) {
                    if vertex.tag_name().name() != "vertex" {
                        continue;
                    }
                    positions.push([
                        attr_number(vertex, "x")?,
                        attr_number(vertex, "y")?,
                        attr_number(vertex, "z")?,
                    ]);
                }
            }
            "triangles" => {
                for triangle in child.children().filter(|n| n.is_element()) {
                    if triangle.tag_name().name() != "triangle" {
                        continue;
                    }
                    triangles.push([
                        attr_index(triangle, "v1")?,
                        attr_index(triangle, "v2")?,
                        attr_index(triangle, "v3")?,
                    ]);
                }
            }
            _ => {}
        }
    }
    let vertex_count = positions.len() as u32;
    for (i, tri) in triangles.iter().enumerate() {
        if let Some(bad) = tri.iter().find(|&&v| v >= vertex_count) {
            return Err(format!(
                "triangle {i} references vertex {bad} but the mesh has {vertex_count} vertices"
            ));
        }
    }
    Ok(Mesh {
        positions,
        triangles,
    })
}

fn parse_transform(node: roxmltree::Node) -> Result<Transform, String> {
    match node.attribute("transform") {
        Some(text) => Transform::parse(text),
        None => Ok(Transform::IDENTITY),
    }
}

fn parse_part(text: &str, part_path: &str) -> Result<Part, String> {
    let doc = roxmltree::Document::parse(text)
        .map_err(|e| format!("model part {part_path:?} is not well-formed XML: {e}"))?;
    let model = doc.root_element();
    if model.tag_name().name() != "model" {
        return Err(format!(
            "model part {part_path:?}: root element is <{}>, expected <model>",
            model.tag_name().name()
        ));
    }
    let unit = match model.attribute("unit") {
        Some(name) => Some(
            Unit::parse(name)
                .ok_or_else(|| format!("model part {part_path:?}: unknown unit {name:?}"))?,
        ),
        None => None,
    };

    let mut objects = HashMap::new();
    let mut build = Vec::new();
    for section in model.children().filter(|n| n.is_element()) {
        match section.tag_name().name() {
            "resources" => {
                for object in section.children().filter(|n| n.is_element()) {
                    if object.tag_name().name() != "object" {
                        continue;
                    }
                    let id = attr_index(object, "id")?;
                    let mut mesh = None;
                    let mut components = Vec::new();
                    for child in object.children().filter(|n| n.is_element()) {
                        match child.tag_name().name() {
                            "mesh" => {
                                mesh = Some(
                                    parse_mesh(child).map_err(|e| format!("object {id}: {e}"))?,
                                )
                            }
                            "components" => {
                                for component in child.children().filter(|n| n.is_element()) {
                                    if component.tag_name().name() != "component" {
                                        continue;
                                    }
                                    let path = component
                                        .attribute((PRODUCTION_NS, "path"))
                                        .or_else(|| {
                                            component
                                                .attributes()
                                                .find(|a| a.name() == "path")
                                                .map(|a| a.value())
                                        })
                                        .map(|reference| resolve_path(reference, part_path));
                                    components.push(Component {
                                        path,
                                        object_id: attr_index(component, "objectid")?,
                                        transform: parse_transform(component)?,
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                    if objects.insert(id, Object { mesh, components }).is_some() {
                        return Err(format!(
                            "model part {part_path:?}: duplicate object id {id}"
                        ));
                    }
                }
            }
            "build" => {
                for item in section.children().filter(|n| n.is_element()) {
                    if item.tag_name().name() != "item" {
                        continue;
                    }
                    build.push((attr_index(item, "objectid")?, parse_transform(item)?));
                }
            }
            _ => {}
        }
    }
    Ok(Part {
        unit,
        objects,
        build,
    })
}

struct Loader<'a> {
    archive: &'a Archive<'a>,
    parts: HashMap<String, Part>,
}

impl Loader<'_> {
    fn load(&mut self, path: &str) -> Result<&Part, String> {
        if !self.parts.contains_key(path) {
            let bytes = self.archive.read(path)?;
            let text = String::from_utf8(bytes)
                .map_err(|_| format!("model part {path:?} is not UTF-8"))?;
            let part = parse_part(&text, path)?;
            self.parts.insert(path.to_string(), part);
        }
        Ok(&self.parts[path])
    }

    /// Appends object `object_id` of part `path`, placed by `transform`, to
    /// `out`, recursing through components.
    fn flatten(
        &mut self,
        path: &str,
        object_id: u32,
        transform: Transform,
        depth: usize,
        out: &mut TriMesh,
    ) -> Result<(), String> {
        if depth > MAX_COMPONENT_DEPTH {
            return Err(format!(
                "components nest deeper than {MAX_COMPONENT_DEPTH} levels (reference cycle?)"
            ));
        }
        let part = self.load(path)?;
        let object = part
            .objects
            .get(&object_id)
            .ok_or_else(|| format!("model part {path:?} has no object {object_id}"))?;
        if let Some(mesh) = &object.mesh {
            let base = (out.positions.len() / 3) as u32;
            let flip = transform.mirrors();
            for &p in &mesh.positions {
                out.positions.extend(transform.apply(p));
            }
            for &[a, b, c] in &mesh.triangles {
                let (b, c) = if flip { (c, b) } else { (b, c) };
                out.indices.extend([base + a, base + b, base + c]);
            }
        }
        let components: Vec<(String, u32, Transform)> = object
            .components
            .iter()
            .map(|component| {
                (
                    component.path.clone().unwrap_or_else(|| path.to_string()),
                    component.object_id,
                    component.transform.then(&transform),
                )
            })
            .collect();
        for (component_path, component_id, placed) in components {
            self.flatten(&component_path, component_id, placed, depth + 1, out)?;
        }
        Ok(())
    }
}

/// Reads a 3MF package. See the crate docs for what is and isn't honoured.
pub fn read_3mf(bytes: &[u8]) -> Result<ThreeMf, String> {
    let archive = Archive::parse(bytes)?;
    let root_path = start_part(&archive)?;
    let mut loader = Loader {
        archive: &archive,
        parts: HashMap::new(),
    };
    let root = loader.load(&root_path)?;
    // The spec's default when the attribute is absent.
    let unit = root.unit.unwrap_or(Unit::Millimeter);
    let build = root.build.clone();

    let mut items = Vec::with_capacity(build.len());
    for (object_id, transform) in build {
        let mut mesh = TriMesh {
            positions: Vec::new(),
            indices: Vec::new(),
            vertex_fields: vec![],
            face_fields: vec![],
        };
        loader.flatten(&root_path, object_id, transform, 0, &mut mesh)?;
        mesh.validate()?;
        items.push(mesh);
    }
    Ok(ThreeMf { unit, items })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zip::Writer;

    const CORE_NS: &str = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02";

    fn package(parts: &[(&str, &str)]) -> Vec<u8> {
        let mut writer = Writer::new();
        writer.add(
            "[Content_Types].xml",
            br#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>"#,
        );
        for (name, text) in parts {
            writer.add(name, text.as_bytes());
        }
        writer.finish()
    }

    /// A single triangle in the xy plane.
    fn triangle_model(extra_objects: &str, build: &str, unit_attr: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<model {unit_attr} xml:lang="en-US" xmlns="{CORE_NS}" xmlns:p="{PRODUCTION_NS}">
 <resources>
  <object id="1" type="model">
   <mesh>
    <vertices>
     <vertex x="0" y="0" z="0"/>
     <vertex x="1" y="0" z="0"/>
     <vertex x="0" y="1" z="0"/>
    </vertices>
    <triangles>
     <triangle v1="0" v2="1" v3="2"/>
    </triangles>
   </mesh>
  </object>
  {extra_objects}
 </resources>
 <build>
  {build}
 </build>
</model>"#
        )
    }

    #[test]
    fn transform_is_row_vector_and_composes() {
        let translate = Transform([
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 10.0, 20.0, 30.0,
        ]);
        assert_eq!(translate.apply([1.0, 2.0, 3.0]), [11.0, 22.0, 33.0]);
        // 90° about z (row-vector convention: x axis maps to +y).
        let rotate = Transform([0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(rotate.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
        // Rotate then translate.
        let both = rotate.then(&translate);
        assert_eq!(both.apply([1.0, 0.0, 0.0]), [10.0, 21.0, 30.0]);
        // Translate then rotate: translation is rotated too.
        let other = translate.then(&rotate);
        assert_eq!(other.apply([1.0, 0.0, 0.0]), [-20.0, 11.0, 30.0]);
        assert!(!both.mirrors());
        let mirror = Transform([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        assert!(mirror.mirrors());
    }

    #[test]
    fn reads_the_start_part_via_rels_and_defaults_the_unit() {
        let bytes = package(&[
            (
                "_rels/.rels",
                r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/Parts/odd.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>"#,
            ),
            (
                "Parts/odd.model",
                &triangle_model("", r#"<item objectid="1"/>"#, ""),
            ),
        ]);
        let file = read_3mf(&bytes).unwrap();
        assert_eq!(file.unit, Unit::Millimeter);
        assert_eq!(file.items.len(), 1);
        assert_eq!(file.items[0].indices, vec![0, 1, 2]);
        assert_eq!(
            file.items[0].positions,
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn falls_back_to_the_conventional_part_and_reads_the_unit() {
        let bytes = package(&[(
            "3D/3dmodel.model",
            &triangle_model("", r#"<item objectid="1"/>"#, r#"unit="inch""#),
        )]);
        let file = read_3mf(&bytes).unwrap();
        assert_eq!(file.unit, Unit::Inch);
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn build_items_place_objects_and_components_compose() {
        let extra = r#"<object id="2" type="model">
   <components>
    <component objectid="1" transform="1 0 0 0 1 0 0 0 1 100 0 0"/>
    <component objectid="1" transform="-1 0 0 0 1 0 0 0 1 0 0 0"/>
   </components>
  </object>"#;
        let build = r#"<item objectid="2" transform="1 0 0 0 1 0 0 0 1 0 0 5"/>
  <item objectid="1"/>"#;
        let bytes = package(&[("3D/3dmodel.model", &triangle_model(extra, build, ""))]);
        let file = read_3mf(&bytes).unwrap();
        assert_eq!(file.items.len(), 2);

        // Item 0: two placed copies of the triangle, lifted by 5.
        let assembly = &file.items[0];
        assert_eq!(assembly.triangle_count(), 2);
        assert_eq!(assembly.vertex_count(), 6);
        assert_eq!(assembly.position(0), [100.0, 0.0, 5.0]);
        assert_eq!(assembly.position(1), [101.0, 0.0, 5.0]);
        // The mirrored copy has its winding flipped to stay outward.
        assert_eq!(assembly.position(3), [0.0, 0.0, 5.0]);
        assert_eq!(assembly.position(4), [-1.0, 0.0, 5.0]);
        assert_eq!(assembly.triangle(0), [0, 1, 2]);
        assert_eq!(assembly.triangle(1), [3, 5, 4]);

        // Item 1: the bare triangle.
        assert_eq!(file.items[1].triangle_count(), 1);
        assert_eq!(file.items[1].position(1), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn production_extension_paths_reach_other_parts() {
        let root = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="{CORE_NS}" xmlns:p="{PRODUCTION_NS}">
 <resources>
  <object id="7" type="model">
   <components>
    <component p:path="/3D/Objects/object_1.model" objectid="1" transform="1 0 0 0 1 0 0 0 1 0 0 2"/>
   </components>
  </object>
 </resources>
 <build>
  <item objectid="7" transform="1 0 0 0 1 0 0 0 1 0 3 0"/>
 </build>
</model>"#
        );
        // The referenced part carries its own (ignored) build section.
        let object_part = triangle_model("", "", r#"unit="millimeter""#);
        let bytes = package(&[
            ("3D/3dmodel.model", &root),
            ("3D/Objects/object_1.model", &object_part),
        ]);
        let file = read_3mf(&bytes).unwrap();
        assert_eq!(file.items.len(), 1);
        assert_eq!(file.items[0].position(0), [0.0, 3.0, 2.0]);
        assert_eq!(file.items[0].position(2), [0.0, 4.0, 2.0]);
    }

    #[test]
    fn rejects_cycles_bad_indices_and_missing_objects() {
        let cycle = r#"<object id="2" type="model"><components><component objectid="3"/></components></object>
  <object id="3" type="model"><components><component objectid="2"/></components></object>"#;
        let bytes = package(&[(
            "3D/3dmodel.model",
            &triangle_model(cycle, r#"<item objectid="2"/>"#, ""),
        )]);
        assert!(read_3mf(&bytes).unwrap_err().contains("cycle"));

        let bytes = package(&[(
            "3D/3dmodel.model",
            &triangle_model("", r#"<item objectid="9"/>"#, ""),
        )]);
        assert!(read_3mf(&bytes).unwrap_err().contains("no object 9"));

        let bad = r#"<object id="2" type="model"><mesh><vertices><vertex x="0" y="0" z="0"/></vertices><triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh></object>"#;
        let bytes = package(&[(
            "3D/3dmodel.model",
            &triangle_model(bad, r#"<item objectid="2"/>"#, ""),
        )]);
        assert!(
            read_3mf(&bytes)
                .unwrap_err()
                .contains("references vertex 1")
        );

        assert!(read_3mf(b"definitely not a package").is_err());
        let bytes = package(&[("readme.txt", "nothing here")]);
        assert!(read_3mf(&bytes).unwrap_err().contains("no 3D model part"));
    }
}

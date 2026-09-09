//! 3MF (3D Manufacturing Format) reading and writing.
//!
//! A 3MF file is an OPC package — a ZIP archive — whose start part is an
//! XML model: `<resources>` of mesh objects (and component objects that
//! place other objects), and a `<build>` of items placing objects into the
//! scene with 4x3 affine transforms. The `unit` attribute on `<model>`
//! makes the geometry's scale explicit, which is the format's main
//! advantage over STL for printing.
//!
//! [`read_3mf`] flattens every build item into a [`TriMesh`] in the file's
//! unit: components are resolved recursively (including the production
//! extension's cross-part `p:path` references), transforms composed, and
//! mirroring transforms flip the winding so triangles stay outward-facing.
//! Objects of every type are imported (FreeCAD, for one, tags closed
//! solids `type="surface"`); materials, textures, and slicer metadata are
//! ignored.
//!
//! [`write_3mf`] emits one mesh object placed by one build item — the shape
//! every slicer reads — with coordinates in the given unit.

pub mod read;
pub mod write;
pub mod zip;

pub use read::{ThreeMf, read_3mf};
pub use write::write_3mf;

pub use volumetric_abi::trimesh::TriMesh;

/// The `unit` attribute of a 3MF `<model>`: the length unit of every
/// coordinate in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Micron,
    Millimeter,
    Centimeter,
    Inch,
    Foot,
    Meter,
}

impl Unit {
    /// The attribute value as the spec spells it.
    pub fn name(self) -> &'static str {
        match self {
            Unit::Micron => "micron",
            Unit::Millimeter => "millimeter",
            Unit::Centimeter => "centimeter",
            Unit::Inch => "inch",
            Unit::Foot => "foot",
            Unit::Meter => "meter",
        }
    }

    pub fn parse(name: &str) -> Option<Unit> {
        Some(match name {
            "micron" => Unit::Micron,
            "millimeter" => Unit::Millimeter,
            "centimeter" => Unit::Centimeter,
            "inch" => Unit::Inch,
            "foot" => Unit::Foot,
            "meter" => Unit::Meter,
            _ => return None,
        })
    }

    /// Metres per one of this unit.
    pub fn metres(self) -> f64 {
        match self {
            Unit::Micron => 1e-6,
            Unit::Millimeter => 1e-3,
            Unit::Centimeter => 1e-2,
            Unit::Inch => 0.0254,
            Unit::Foot => 0.3048,
            Unit::Meter => 1.0,
        }
    }
}

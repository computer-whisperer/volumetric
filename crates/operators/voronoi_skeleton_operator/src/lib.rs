//! Voronoi Skeleton Operator.
//!
//! Builds the Voronoi edge skeleton of a point cloud: every input point
//! becomes a cell seed, and the output Bar2 [`FeaMesh`] is the strut
//! network along the boundaries where three or more cells meet — the
//! same construction behind the built-in foam lattice, but with the site
//! set under user control (see `lattice_model_core::voronoi`). A
//! `point_fill_operator` bcc cloud at seed 0 reproduces the foam family's
//! skeleton exactly; edited clouds (clipped, merged, transformed) make
//! foams the built-in family can't.
//!
//! Hull cells are infinite; `boundary` picks what happens to their
//! outward edges. The default `"trim"` keeps only edges supported by
//! genuine Voronoi vertices on both ends AND within `max_reach` local
//! spacings of their cell's site — infinite hull edges vanish and the
//! hull shell's ballooning vertices (near-coplanar site slivers with
//! huge empty circumspheres) are cut, so the skeleton simply ends at the
//! cloud. `"box"` instead truncates hull edges at the cloud's bounding
//! box plus `padding` (endpoints on the box, no reach cap), for when a
//! downstream `mesh_clip_operator` should cut the rays to a real domain
//! surface as skin-contact stubs. There is deliberately no domain input:
//! clip the output against a model with `mesh_clip_operator` — before or
//! after editing — which also welds the boundary stubs clipping creates.
//!
//! Removing sites from a cloud makes the *neighboring cells grow* into
//! the vacated space (locally coarser foam); it does not cut holes. For
//! holes, clip the Bar2 output instead.
//!
//! Inputs:
//! - Input 0: FeaMesh (must be Point1) — the cell seed sites
//! - Input 1: CBOR configuration:
//!   `{ boundary: "trim" / "box" .default "trim", max_reach: float
//!   .default 1.5 (trim: cap on edge reach in units of each cell's
//!   nearest-neighbor distance; 0 disables), radius: float .default 0.0
//!   (0 = typical spacing / 10), weld_factor: float .default 1.0 (welds
//!   struts shorter than weld_factor * radius; 0 disables), padding:
//!   float .default 0.0 (0 = two typical spacings) }`
//!
//! Output 0: CBOR-encoded Bar2 `FeaMesh` with a uniform scalar `radius`
//! element field.

use lattice_model_core::voronoi::{Boundary, VoronoiOptions, voronoi_skeleton};
use volumetric_abi::fea::{FeaElementKind, FeaField, FeaMesh, decode_fea_mesh, encode_fea_mesh};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

/// Refuse site sets whose skeleton would exceed this many struts (a
/// Voronoi skeleton carries ~7 edges per site; matches the strut
/// operator's cap).
const MAX_STRUTS: u64 = 2_000_000;

/// Edges per site, for the size estimate.
const STRUTS_PER_SITE: u64 = 7;

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum BoundaryConfig {
    /// Keep only edges between genuine Voronoi vertices: the skeleton
    /// ends at the cloud.
    Trim,
    /// Truncate infinite hull edges at the padded bounding box.
    Box,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct VoronoiConfig {
    /// What happens to infinite hull-cell edges.
    boundary: BoundaryConfig,
    /// Trim only: edges reaching past this many local spacings from
    /// their cell's site are dropped; 0 disables.
    max_reach: f64,
    /// Strut cross-section radius; 0 = typical site spacing / 10.
    radius: f64,
    /// Struts shorter than `weld_factor * radius` weld into a single
    /// joint node; 0 disables.
    weld_factor: f64,
    /// How far past the sites' bounding box the diagram extends before
    /// hull-cell edges truncate; 0 = two typical spacings.
    padding: f64,
}

impl Default for VoronoiConfig {
    fn default() -> Self {
        Self {
            boundary: BoundaryConfig::Trim,
            max_reach: 1.5,
            radius: 0.0,
            weld_factor: 1.0,
            padding: 0.0,
        }
    }
}

fn build_skeleton(config: &VoronoiConfig) -> Result<FeaMesh, String> {
    if !(config.radius.is_finite() && config.radius >= 0.0) {
        return Err(format!(
            "radius must be non-negative, got {}",
            config.radius
        ));
    }
    if !(config.weld_factor.is_finite() && config.weld_factor >= 0.0) {
        return Err(format!(
            "weld_factor must be non-negative, got {}",
            config.weld_factor
        ));
    }
    if !(config.padding.is_finite() && config.padding >= 0.0) {
        return Err(format!(
            "padding must be non-negative, got {}",
            config.padding
        ));
    }
    if !(config.max_reach.is_finite() && config.max_reach >= 0.0) {
        return Err(format!(
            "max_reach must be non-negative, got {}",
            config.max_reach
        ));
    }

    let cloud = decode_fea_mesh(&read_input(0))?;
    if cloud.element_kind != FeaElementKind::Point1 {
        return Err(format!(
            "Voronoi sites must be a Point1 point cloud, got {:?} elements",
            cloud.element_kind
        ));
    }
    let sites: Vec<[f64; 3]> = (0..cloud.element_count())
        .map(|e| cloud.node_position(cloud.element(e)[0] as usize))
        .collect();
    let estimate = sites.len() as u64 * STRUTS_PER_SITE;
    if estimate > MAX_STRUTS {
        return Err(format!(
            "{} sites would build ~{estimate} struts (cap {MAX_STRUTS}); \
             thin the cloud first",
            sites.len()
        ));
    }

    let options = VoronoiOptions {
        padding: config.padding,
        boundary: match config.boundary {
            BoundaryConfig::Trim => Boundary::Trim,
            BoundaryConfig::Box => Boundary::Box,
        },
        max_reach: config.max_reach,
    };
    let result = voronoi_skeleton(&sites, &options)?;
    let skeleton = result.skeleton;
    if skeleton.edges.is_empty() {
        return Err(format!(
            "no Voronoi edges from {} sites (a skeleton needs at least ~4 \
             non-degenerate sites)",
            result.site_count
        ));
    }

    let radius = if config.radius == 0.0 {
        result.spacing / 10.0
    } else {
        config.radius
    };
    let strut_count = skeleton.edges.len();
    let mut mesh = FeaMesh {
        element_kind: FeaElementKind::Bar2,
        node_positions: skeleton.nodes.iter().flatten().copied().collect(),
        connectivity: skeleton.edges.iter().flatten().copied().collect(),
        node_fields: vec![],
        element_fields: vec![FeaField {
            name: "radius".to_string(),
            components: 1,
            data: vec![radius; strut_count],
        }],
    };
    mesh.validate()?;

    let weld_length = config.weld_factor * radius;
    if weld_length > 0.0 {
        mesh = mesh_edit_core::weld_short_bars(&mesh, &|_| weld_length)?;
        if mesh.element_count() == 0 {
            return Err(format!(
                "welding at length {weld_length} collapsed every strut \
                 (is weld_factor * radius larger than the cells?)"
            ));
        }
    }
    Ok(mesh)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(1);
        if buf.is_empty() {
            VoronoiConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&buf)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    match build_skeleton(&config) {
        Ok(mesh) => post_output(0, &encode_fea_mesh(&mesh)),
        Err(e) => report_error(&format!("Voronoi skeleton failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ boundary: "trim" / "box" .default "trim", max_reach: float .default 1.5, radius: float .default 0.0, weld_factor: float .default 1.0, padding: float .default 0.0 }"#
            .to_string();
        OperatorMetadata {
            name: "voronoi_skeleton_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Voronoi Skeleton".to_string(),
            description: "Build the Voronoi cell-edge strut network (Bar2 mesh) of a \
                          point cloud of cell seeds."
                .to_string(),
            category: "Lattice".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M3 9.5 8 3h5l3.5 5-3 5.5H8z"/>"##,
                r##"<path d="M16.5 8H21"/>"##,
                r##"<path d="m13.5 13.5 3 4"/>"##,
                r##"<path d="M8 13.5 5.5 19"/>"##,
                r##"<path d="M3 9.5v0"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::FeaMesh,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["Sites".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::FeaMesh],
        }
    })
}

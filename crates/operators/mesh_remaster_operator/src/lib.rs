//! Mesh Remaster Operator.
//!
//! Reworks a strut lattice against composable *requirements*, solved
//! jointly in one operator so their fixes cannot silently undo each
//! other. Each requirement is an optional config block; at least one
//! must be present:
//!
//! - `surface` — fold the parts of the network that protrude outside a
//!   model down onto that model's surface (the drape pass). The outside
//!   segments become a surface-conforming net (the "skin"), so a trimmed
//!   Voronoi lattice ends in a smooth printable face instead of open
//!   cell rims — without adding material the way a separate conforming
//!   surface lattice would. Draping relocates the outer cells' own
//!   struts and the net inherits the skeleton's degree-3 vertices: a
//!   polygonal, sub-isostatic net that deforms by strut bending, far
//!   softer in-plane than a triangulated skin. `skin_radius_factor`
//!   tunes what stiffness remains (bending scales as radius^4) and the
//!   `skin` element flag hands the net to downstream optimization.
//! - `connectivity` — no floating fragments (they make FEA singular and
//!   fall off prints). Fix `"reconnect"` re-drapes the cheapest arcs the
//!   surface pass dropped until everything is one component (a spanning
//!   forest over components, shortest arcs first), then synthesizes
//!   direct ties for pieces with no dropped arc to reuse (up to
//!   `max_new_strut` x the median strut length); whatever nothing can
//!   reach is pruned. Fix `"prune"` keeps the largest component only.
//!   Reconnection struts are flagged in a `tie` element field.
//! - `support` — printable along the build axis on a resin printer:
//!   overhangs are fine, hooks toward the bed are not. A piece fails
//!   exactly where it appears in a slice unattached to already-cured
//!   material; on the strut graph that is sub-level-set connectivity —
//!   ascend the build axis and every node must connect to the bed
//!   through nodes at or below its own height (`max_descent` degrees of
//!   per-strut slack models in-slice cohesion; the slack deliberately
//!   does not compound across struts). Fix `"raise"` projects the
//!   descent out of invalid struts — unsupported nodes rise straight up
//!   (x/y preserved) by the minimum that makes every support path
//!   monotone, a hanging hook flattening into a fan; the raise distance
//!   lands in a `raise` node field for inspection. Fix `"drop"` removes
//!   unsupported struts instead — transitively, like the voxel island
//!   remover, but exact and graph-aware.
//!
//! Pass order is chosen so later passes only violate earlier
//! requirements when physically forced: surface first (it decides which
//! outside arcs survive and hands the dropped ones to reconnection),
//! connectivity second, support last. Raising only moves nodes, so it
//! can break neither of the first two — where it pulls skin off the
//! surface, the printer is overruling cosmetics. The one destructive
//! interaction — `support.fix: "drop"` can split a component by
//! removing a bridge — is closed by re-running connectivity once; a tie
//! between two supported nodes can never create new unsupported
//! geometry, so no further rounds are needed.
//!
//! The model is a binary occupancy oracle (the operator ABI contract),
//! so surface projection estimates a direction from a signed stencil of
//! occupancy samples, marches along it to bracket the surface, and
//! bisects — all in lock-step batched rounds, one host call per round.
//! Connectivity and support are pure graph passes and never touch the
//! model; with no `surface` block the Surface input may stay unwired.
//!
//! Point1 clouds accept the `surface` requirement only (outside points
//! land or drop); Hex8 meshes are rejected — volume elements cannot
//! fold. Node positions are always 3D.
//!
//! Inputs:
//! - Input 0: FeaMesh (Bar2, or Point1 for surface-only) — the mesh
//! - Input 1: ModelWASM (must be 3D) — the surface to drape onto; only
//!   required when the `surface` block is present
//! - Input 2: CBOR configuration (every block optional, at least one;
//!   the UI seeds new steps with `surface` enabled):
//!   `{ ? surface: { outside: "project" / "drop" .default "project",
//!   skin_radius_factor: float .default 1.0, chord_tolerance: float
//!   .default 0.0 (0 = each strut's own radius), inset_factor: float
//!   .default 0.0 (sink skin nodes below the surface by this many strut
//!   radii — the largest radius at the node, bulk stubs included),
//!   max_distance: float .default 0.0 (0 = 4 x the median strut length;
//!   for Point1, an eighth of the cloud's bounding diagonal),
//!   weld_factor: float .default 1.0 (welds struts shorter than
//!   weld_factor * radius; 0 disables) } .default true,
//!   ? connectivity: { fix: "reconnect" / "prune" .default "reconnect",
//!   max_new_strut: float .default 1.5 (x the median strut length; 0
//!   never synthesizes) },
//!   ? support: { axis: "auto" / "x" / "y" / "z" .default "auto" (auto =
//!   z), extreme: "min" / "max" .default "min" (which end of the axis
//!   the bed is on), max_descent: float .default 0.0 (degrees below
//!   horizontal a strut may descend from its supported end),
//!   bed_tolerance: float .default 0.0 (0 = 1e-4 x the axis extent;
//!   nodes this close to the extreme seed as bed-supported),
//!   fix: "raise" / "drop" .default "raise" } }`
//!
//! Output 0: CBOR-encoded `FeaMesh`. The surface pass adds a scalar
//! `skin` element field (1.0 on draped elements), connectivity adds
//! `tie` (1.0 on reconnection struts), and support raising adds a
//! `raise` node field (the distance each node rose).

mod connect;
mod drape;
mod support;

use volumetric_abi::fea::{FeaElementKind, FeaMesh, decode_fea_mesh, encode_fea_mesh};
use volumetric_abi::host::{
    input_model_dimensions, input_model_sample, post_output, read_input, report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

/// Occupancy samples per batched host call.
const SAMPLE_CHUNK: usize = 8192;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum OutsideConfig {
    /// Drape struts with both nodes outside onto the surface.
    Project,
    /// Remove them; only the outside halves of crossing struts fold.
    /// The removed arcs stay available to `connectivity: "reconnect"`.
    Drop,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct SurfaceConfig {
    /// What happens to struts entirely outside the model.
    pub outside: OutsideConfig,
    /// Multiplies the `radius` element field on skin struts.
    pub skin_radius_factor: f64,
    /// Draped chords subdivide until they sag off the surface by less
    /// than this; 0 = each strut's own radius.
    pub chord_tolerance: f64,
    /// Sink skin nodes this many strut radii below the surface (the
    /// largest radius at the node, bulk stubs included).
    pub inset_factor: f64,
    /// Nodes farther than this from the surface drop with their struts;
    /// 0 = 4 x the median strut length.
    pub max_distance: f64,
    /// Welds struts shorter than `weld_factor * radius`; 0 disables.
    pub weld_factor: f64,
}

impl Default for SurfaceConfig {
    fn default() -> Self {
        Self {
            outside: OutsideConfig::Project,
            skin_radius_factor: 1.0,
            chord_tolerance: 0.0,
            inset_factor: 0.0,
            max_distance: 0.0,
            weld_factor: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ConnectivityFix {
    /// Re-drape dropped arcs, then synthesize ties, then prune the rest.
    Reconnect,
    /// Keep the largest connected component only.
    Prune,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ConnectivityConfig {
    pub fix: ConnectivityFix,
    /// Synthesized ties may span up to this many median strut lengths;
    /// 0 never synthesizes.
    pub max_new_strut: f64,
}

impl Default for ConnectivityConfig {
    fn default() -> Self {
        Self {
            fix: ConnectivityFix::Reconnect,
            max_new_strut: 1.5,
        }
    }
}

/// The build axis. `Auto` is z — compose with the transform operator for
/// a tilted build direction. (A named value rather than an optional
/// field: the UI's schema editor cannot omit fields.)
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum BuildAxis {
    Auto,
    X,
    Y,
    Z,
}

impl BuildAxis {
    pub(crate) fn index(self) -> usize {
        match self {
            BuildAxis::X => 0,
            BuildAxis::Y => 1,
            BuildAxis::Auto | BuildAxis::Z => 2,
        }
    }
}

/// Which end of the build axis the bed is on.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Extreme {
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum SupportFix {
    /// Raise unsupported nodes along the build axis until every support
    /// path is monotone (x/y preserved).
    Raise,
    /// Remove unsupported struts, transitively.
    Drop,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct SupportConfig {
    pub axis: BuildAxis,
    pub extreme: Extreme,
    /// Degrees below horizontal a strut may descend from its supported
    /// end and still count as attached (in-slice cohesion). Per strut;
    /// the slack does not compound along chains.
    pub max_descent: f64,
    /// Nodes this close to the bed extreme seed as bed-supported;
    /// 0 = 1e-4 x the axis extent.
    pub bed_tolerance: f64,
    pub fix: SupportFix,
}

impl Default for SupportConfig {
    fn default() -> Self {
        Self {
            axis: BuildAxis::Auto,
            extreme: Extreme::Min,
            max_descent: 0.0,
            bed_tolerance: 0.0,
            fix: SupportFix::Raise,
        }
    }
}

/// The remaster configuration: named requirement blocks, each optional.
/// A block's absence means that requirement is not enforced. Unknown
/// fields are rejected everywhere — this operator exists to enforce
/// guarantees, and a typo (`max_descend`) silently reverting to a
/// default would defeat exactly that.
#[derive(Clone, Copy, Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct RemasterConfig {
    pub surface: Option<SurfaceConfig>,
    pub connectivity: Option<ConnectivityConfig>,
    pub support: Option<SupportConfig>,
}

pub(crate) fn validate_config(config: &RemasterConfig) -> Result<(), String> {
    if config.surface.is_none() && config.connectivity.is_none() && config.support.is_none() {
        return Err(
            "configure at least one requirement block: surface, connectivity, or support"
                .to_string(),
        );
    }
    if let Some(surface) = &config.surface {
        if !(surface.skin_radius_factor.is_finite() && surface.skin_radius_factor > 0.0) {
            return Err(format!(
                "surface.skin_radius_factor must be positive, got {}",
                surface.skin_radius_factor
            ));
        }
        for (name, v) in [
            ("chord_tolerance", surface.chord_tolerance),
            ("inset_factor", surface.inset_factor),
            ("max_distance", surface.max_distance),
            ("weld_factor", surface.weld_factor),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!("surface.{name} must be non-negative, got {v}"));
            }
        }
    }
    if let Some(connectivity) = &config.connectivity
        && !(connectivity.max_new_strut.is_finite() && connectivity.max_new_strut >= 0.0)
    {
        return Err(format!(
            "connectivity.max_new_strut must be non-negative, got {}",
            connectivity.max_new_strut
        ));
    }
    if let Some(support) = &config.support {
        if !(support.max_descent.is_finite() && (0.0..90.0).contains(&support.max_descent)) {
            return Err(format!(
                "support.max_descent must be in [0, 90), got {}",
                support.max_descent
            ));
        }
        if !(support.bed_tolerance.is_finite() && support.bed_tolerance >= 0.0) {
            return Err(format!(
                "support.bed_tolerance must be non-negative, got {}",
                support.bed_tolerance
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared geometry and graph helpers
// ---------------------------------------------------------------------------

/// A batched occupancy oracle: for each position, whether it lies inside
/// the model. The operator backs this with chunked host sampling; tests
/// with analytic domains.
pub(crate) type OccupiedBatch<'a> = dyn FnMut(&[[f64; 3]]) -> Result<Vec<bool>, String> + 'a;

pub(crate) fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|c| a[c] + b[c])
}

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|c| a[c] - b[c])
}

pub(crate) fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    core::array::from_fn(|c| a[c] * s)
}

pub(crate) fn norm(a: [f64; 3]) -> f64 {
    a.iter().map(|c| c * c).sum::<f64>().sqrt()
}

pub(crate) fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

pub(crate) fn lerp3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    core::array::from_fn(|c| a[c] + t * (b[c] - a[c]))
}

pub(crate) fn normalize(a: [f64; 3]) -> Option<[f64; 3]> {
    let len = norm(a);
    (len > 1e-12).then(|| scale(a, 1.0 / len))
}

/// Union-find lookup with path halving.
pub(crate) fn uf_find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Union toward the smaller root; returns false if already joined.
pub(crate) fn uf_union(parent: &mut [u32], a: u32, b: u32) -> bool {
    let (ra, rb) = (uf_find(parent, a), uf_find(parent, b));
    if ra == rb {
        return false;
    }
    let (lo, hi) = (ra.min(rb), ra.max(rb));
    parent[hi as usize] = lo;
    true
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

/// Shared entry for host and tests: run the requested requirement passes
/// in order. `occupied` may be `None` when no `surface` block is present.
pub(crate) fn remaster(
    mesh: &FeaMesh,
    occupied: Option<&mut OccupiedBatch>,
    config: &RemasterConfig,
) -> Result<FeaMesh, String> {
    validate_config(config)?;
    if mesh.element_count() == 0 {
        return Err("the input mesh has no elements".to_string());
    }
    if mesh.element_kind == FeaElementKind::Point1
        && (config.connectivity.is_some() || config.support.is_some())
    {
        return Err(
            "connectivity and support requirements need a Bar2 strut mesh; point clouds \
             take the surface requirement only"
                .to_string(),
        );
    }

    let mut out = match &config.surface {
        Some(surface) => {
            let occupied = occupied.ok_or_else(|| {
                "the surface requirement needs a model wired to the Surface input".to_string()
            })?;
            drape::drape(mesh, occupied, surface, config.connectivity.as_ref())?
        }
        None => mesh.clone(),
    };

    if let Some(connectivity) = &config.connectivity {
        out = connect::enforce(&out, connectivity)?;
    }

    if let Some(support) = &config.support {
        let (fixed, dropped_any) = support::enforce(&out, support, config.connectivity.is_some())?;
        out = fixed;
        // Dropping unsupported struts can split a component whose bridge
        // died; one reconnection round closes it. A new tie joins two
        // supported nodes, so at the default max_descent 0 it is always
        // support-valid itself (one end is the lower one); within a
        // nonzero slack window the tie is accepted on its endpoints'
        // attachment rather than re-checked, so support needs no second
        // pass either way.
        if dropped_any && let Some(connectivity) = &config.connectivity {
            out = connect::enforce(&out, connectivity)?;
        }
    }
    Ok(out)
}

fn build_remastered(config: &RemasterConfig) -> Result<FeaMesh, String> {
    let mesh = decode_fea_mesh(&read_input(0))?;
    if config.surface.is_none() {
        return remaster(&mesh, None, config);
    }
    let dims = input_model_dimensions(1).ok_or_else(|| {
        "the surface requirement needs a model wired to the Surface input".to_string()
    })?;
    if dims != 3 {
        return Err(format!(
            "the surface requirement needs a 3D model; input has {dims} dimensions"
        ));
    }
    let mut occupied = |points: &[[f64; 3]]| -> Result<Vec<bool>, String> {
        let mut out = Vec::with_capacity(points.len());
        for chunk in points.chunks(SAMPLE_CHUNK) {
            let positions: Vec<f64> = chunk.iter().flatten().copied().collect();
            let samples = input_model_sample(1, &positions, 3)
                .ok_or_else(|| "sampling the model failed".to_string())?;
            out.extend(samples.iter().map(|&s| is_occupied(s)));
        }
        Ok(out)
    };
    remaster(&mesh, Some(&mut occupied), config)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let buf = read_input(2);
    let config: RemasterConfig = if buf.is_empty() {
        RemasterConfig::default() // fails validation with the schema hint
    } else {
        match ciborium::de::from_reader(std::io::Cursor::new(&buf)) {
            Ok(config) => config,
            Err(e) => {
                report_error(&format!("invalid configuration: {e}"));
                return;
            }
        }
    };

    match build_remastered(&config) {
        Ok(mesh) => post_output(0, &encode_fea_mesh(&mesh)),
        Err(e) => report_error(&format!("mesh remaster failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        // Optional blocks use the CDDL `?` marker; `.default true` after a
        // group's brace seeds fresh steps with that block enabled, so a new
        // step drapes with defaults out of the box.
        let schema = r#"{ ? surface: { outside: "project" / "drop" .default "project", skin_radius_factor: float .default 1.0, chord_tolerance: float .default 0.0, inset_factor: float .default 0.0, max_distance: float .default 0.0, weld_factor: float .default 1.0 } .default true, ? connectivity: { fix: "reconnect" / "prune" .default "reconnect", max_new_strut: float .default 1.5 }, ? support: { axis: "auto" / "x" / "y" / "z" .default "auto", extreme: "min" / "max" .default "min", max_descent: float .default 0.0, bed_tolerance: float .default 0.0, fix: "raise" / "drop" .default "raise" } }"#
            .to_string();
        OperatorMetadata {
            name: "mesh_remaster_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Mesh Remaster".to_string(),
            description: "Rework a strut lattice against composable requirements: drape \
                          protruding struts onto a model surface as a compliance-tunable \
                          skin, reconnect or prune floating islands, and raise or drop \
                          struts unsupported along the build direction."
                .to_string(),
            category: "Mesh".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M3 18c3.5-8.5 14.5-8.5 18 0"/>"##,
                r##"<path d="M13.5 4.5 11.7 9.9"/>"##,
                r##"<path d="M11.7 9.9 15.5 11.3"/>"##,
                r##"<path d="M8 20l2.7-5.3"/>"##,
                r##"<circle cx="11.7" cy="9.9" r=".4"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::FeaMesh,
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec![
                "Mesh".to_string(),
                "Surface".to_string(),
                "Config".to_string(),
            ],
            outputs: vec![OperatorMetadataOutput::FeaMesh],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::fea::FeaField;

    /// Occupancy oracle for a sphere of radius `r` at the origin.
    pub(crate) fn sphere(r: f64) -> impl FnMut(&[[f64; 3]]) -> Result<Vec<bool>, String> {
        move |points| {
            Ok(points
                .iter()
                .map(|p| p.iter().map(|c| c * c).sum::<f64>() < r * r)
                .collect())
        }
    }

    pub(crate) fn bar_mesh(nodes: &[[f64; 3]], bars: &[[u32; 2]], radius: f64) -> FeaMesh {
        FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions: nodes.iter().flatten().copied().collect(),
            connectivity: bars.iter().flatten().copied().collect(),
            node_fields: vec![],
            element_fields: vec![FeaField {
                name: "radius".to_string(),
                components: 1,
                data: vec![radius; bars.len()],
            }],
        }
    }

    pub(crate) fn component_count(mesh: &FeaMesh) -> usize {
        let mut parent: Vec<u32> = (0..mesh.node_count() as u32).collect();
        for pair in mesh.connectivity.chunks_exact(2) {
            uf_union(&mut parent, pair[0], pair[1]);
        }
        let mut roots: Vec<u32> = mesh
            .connectivity
            .chunks_exact(2)
            .map(|pair| uf_find(&mut parent, pair[0]))
            .collect();
        roots.sort_unstable();
        roots.dedup();
        roots.len()
    }

    #[test]
    fn optional_blocks_decode_independently() {
        use ciborium::value::Value;
        let v = Value::Map(vec![(
            Value::Text("support".into()),
            Value::Map(vec![(
                Value::Text("max_descent".into()),
                Value::Float(30.0),
            )]),
        )]);
        let mut buf = Vec::new();
        ciborium::ser::into_writer(&v, &mut buf).unwrap();
        let config: RemasterConfig = ciborium::de::from_reader(&buf[..]).unwrap();
        assert!(config.surface.is_none(), "absent blocks must stay absent");
        assert!(config.connectivity.is_none());
        let support = config.support.expect("support block present");
        assert_eq!(support.max_descent, 30.0);
        assert_eq!(support.fix, SupportFix::Raise);
        assert_eq!(support.axis, BuildAxis::Auto);
    }

    #[test]
    fn config_validation_rejects_bad_blocks() {
        assert!(
            validate_config(&RemasterConfig::default()).is_err(),
            "no blocks at all must fail with the schema hint"
        );
        let bad = RemasterConfig {
            surface: Some(SurfaceConfig {
                skin_radius_factor: 0.0,
                ..SurfaceConfig::default()
            }),
            ..RemasterConfig::default()
        };
        assert!(validate_config(&bad).is_err());
        let bad = RemasterConfig {
            support: Some(SupportConfig {
                max_descent: 90.0,
                ..SupportConfig::default()
            }),
            ..RemasterConfig::default()
        };
        assert!(validate_config(&bad).is_err());
        let bad = RemasterConfig {
            connectivity: Some(ConnectivityConfig {
                max_new_strut: f64::NAN,
                ..ConnectivityConfig::default()
            }),
            ..RemasterConfig::default()
        };
        assert!(validate_config(&bad).is_err());
    }

    #[test]
    fn unknown_config_fields_are_rejected() {
        use ciborium::value::Value;
        // A typo'd field inside a block must error, not silently revert
        // to the default in a guarantee-enforcing operator.
        let v = Value::Map(vec![(
            Value::Text("support".into()),
            Value::Map(vec![(
                Value::Text("max_descend".into()),
                Value::Float(30.0),
            )]),
        )]);
        let mut buf = Vec::new();
        ciborium::ser::into_writer(&v, &mut buf).unwrap();
        let result: Result<RemasterConfig, _> = ciborium::de::from_reader(&buf[..]);
        assert!(result.is_err(), "misspelled max_descent must not decode");
        // Same for a misspelled block name.
        let v = Value::Map(vec![(Value::Text("suport".into()), Value::Map(vec![]))]);
        let mut buf = Vec::new();
        ciborium::ser::into_writer(&v, &mut buf).unwrap();
        let result: Result<RemasterConfig, _> = ciborium::de::from_reader(&buf[..]);
        assert!(result.is_err(), "misspelled block name must not decode");
    }

    #[test]
    fn point_clouds_reject_graph_requirements() {
        let cloud = FeaMesh {
            element_kind: FeaElementKind::Point1,
            node_positions: vec![0.0, 0.0, 0.5],
            connectivity: vec![0],
            node_fields: vec![],
            element_fields: vec![],
        };
        let config = RemasterConfig {
            support: Some(SupportConfig::default()),
            ..RemasterConfig::default()
        };
        let err = remaster(&cloud, None, &config).unwrap_err();
        assert!(err.contains("point clouds"), "unexpected error: {err}");
    }

    /// The destructive interaction end to end: support "drop" removes a
    /// hook bridging two towers, splitting the mesh; the connectivity
    /// re-run synthesizes a tie so the output is whole again.
    #[test]
    fn support_drop_then_reconnect_closes_the_split() {
        let nodes = [
            [0.0, 0.0, 0.0],  // bed of tower 1
            [0.0, 0.0, 1.0],  // top of tower 1
            [1.5, 0.0, 0.0],  // bed of tower 2
            [1.5, 0.0, 1.0],  // top of tower 2
            [0.75, 0.0, 0.2], // the hook: hangs between the tower tops
        ];
        let bars = [[0, 1], [2, 3], [1, 4], [4, 3]];
        let mesh = bar_mesh(&nodes, &bars, 0.05);
        let config = RemasterConfig {
            connectivity: Some(ConnectivityConfig::default()),
            support: Some(SupportConfig {
                fix: SupportFix::Drop,
                ..SupportConfig::default()
            }),
            ..RemasterConfig::default()
        };
        let out = remaster(&mesh, None, &config).unwrap();

        assert_eq!(component_count(&out), 1, "the split must be reconnected");
        assert_eq!(out.element_count(), 3, "two towers + one synthesized tie");
        assert_eq!(out.node_count(), 4, "the hook node is gone");
        let tie = out
            .element_fields
            .iter()
            .find(|f| f.name == "tie")
            .expect("tie field");
        assert_eq!(tie.data.iter().filter(|&&t| t == 1.0).count(), 1);
    }

    /// Surface + support composed: drape onto the sphere, then raising
    /// pulls a hook up without disturbing supported geometry.
    #[test]
    fn surface_and_support_compose() {
        // A bulk chain inside the sphere: its bottom node is bed-seeded;
        // node 3 hangs as a hook off node 2 (all inside, no draping
        // needed for them), and one strut pokes out the top to exercise
        // the drape alongside.
        let nodes = [
            [0.0, 0.0, -0.9], // bed (lowest node)
            [0.0, 0.0, 0.0],  // mid column
            [0.3, 0.0, 0.5],  // upper
            [0.5, 0.0, 0.2],  // hook: below its only neighbor
            [0.3, 0.0, 1.4],  // outside: the strut 2-4 crosses and folds
        ];
        let bars = [[0, 1], [1, 2], [2, 3], [2, 4]];
        let mesh = bar_mesh(&nodes, &bars, 0.02);
        let config = RemasterConfig {
            surface: Some(SurfaceConfig::default()),
            support: Some(SupportConfig::default()),
            ..RemasterConfig::default()
        };
        let mut oracle = sphere(1.0);
        let out = remaster(&mesh, Some(&mut oracle), &config).unwrap();

        let raise = out
            .node_fields
            .iter()
            .find(|f| f.name == "raise")
            .expect("raise node field");
        let raised: Vec<usize> = (0..out.node_count())
            .filter(|&n| raise.data[n] > 1e-9)
            .collect();
        assert_eq!(raised.len(), 1, "exactly the hook rises: {:?}", raise.data);
        let p = out.node_position(raised[0]);
        assert!((p[0] - 0.5).abs() < 1e-9, "the hook keeps its x: {p:?}");
        assert!(
            (p[2] - 0.5).abs() < 1e-9,
            "the hook rises to its neighbor's height: {p:?}"
        );
        assert_eq!(component_count(&out), 1);
        let skin = out
            .element_fields
            .iter()
            .find(|f| f.name == "skin")
            .unwrap();
        assert!(skin.data.contains(&1.0), "the fold is skin");
    }
}

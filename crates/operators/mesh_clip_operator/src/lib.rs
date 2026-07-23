//! Mesh Clip Operator.
//!
//! The spatial boolean for explicit meshes: clips a [`FeaMesh`] (point
//! cloud, strut network, or volume mesh) against a model's occupancy,
//! keeping the inside or the outside. With the point-fill and Voronoi
//! operators this is the "boolean clusters of cells away" step — placed
//! before Voronoi generation it locally coarsens the foam (neighboring
//! cells grow into the vacated space); placed after, it cuts actual voids
//! in the strut network.
//!
//! Element rules:
//! - Point1: a point survives iff its node is kept.
//! - Bar2: struts with both endpoints kept survive whole; crossings are
//!   `crossing: "clip"` shortened to the surface (bisection, one fresh
//!   node per crossing, node fields interpolated) or `crossing: "drop"`
//!   dropped whole. Endpoint classification alone misses struts that
//!   *tunnel* through a removed region thinner than a strut (both
//!   endpoints kept, midsection not); `interior_samples: n` additionally
//!   classifies n evenly spaced points along each strut and drops whole
//!   any strut whose interior leaves the kept region — choose n so the
//!   sample spacing (strut length / (n+1)) undercuts the thinnest gap
//!   being cut. Struts with no endpoint kept are dropped. After
//!   clipping, struts shorter than `weld_factor * radius` weld into
//!   joints (needs a scalar `radius` element field; without one welding
//!   is skipped) — see `mesh_edit_core::weld_short_bars` for why. With
//!   `prune_islands` only the largest connected component survives
//!   (default off: editing workflows often *want* disconnected pieces).
//! - Hex8: an element survives iff all eight nodes are kept (volume
//!   elements are never cut — that would be remeshing).
//!
//! Named node/element fields follow the surviving entries throughout.
//!
//! Inputs:
//! - Input 0: FeaMesh — the mesh to clip
//! - Input 1: ModelWASM (must be 3D) — the clip region
//! - Input 2: CBOR configuration:
//!   `{ keep: "inside" / "outside" .default "inside", crossing: "clip" /
//!   "drop" .default "clip", interior_samples: int .default 0 (0
//!   disables tunnel detection), weld_factor: float .default 1.0 (0
//!   disables), prune_islands: bool .default false }`
//!
//! Output 0: the clipped CBOR-encoded `FeaMesh` (same element kind).

use volumetric_abi::fea::{FeaElementKind, FeaMesh, decode_fea_mesh, encode_fea_mesh};
use volumetric_abi::host::{
    input_model_dimensions, input_model_sample, post_output, read_input, report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

/// Occupancy samples per batched host call.
const SAMPLE_CHUNK: usize = 8192;

/// Bisection rounds for a boundary crossing. Every active crossing
/// advances one round per batched sample call, so the whole clip costs
/// `CLIP_BISECTIONS` host calls rather than per-strut loops.
const CLIP_BISECTIONS: usize = 24;

/// Clipped struts shorter than this fraction of the strut's full length
/// are dropped: near-zero frame elements are stiffness spikes, and the
/// kept endpoint stays connected through its other struts.
const MIN_CLIP_FRACTION: f64 = 1e-3;

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum KeepConfig {
    Inside,
    Outside,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum CrossingConfig {
    /// Shorten boundary-crossing struts to the surface (new node on the
    /// skin, node fields interpolated).
    Clip,
    /// Drop boundary-crossing struts whole.
    Drop,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct ClipConfig {
    keep: KeepConfig,
    /// Bar2 only; other kinds ignore it.
    crossing: CrossingConfig,
    /// Bar2 only: interior points classified per strut to catch tunnels
    /// through removed regions thinner than a strut; 0 disables.
    interior_samples: u32,
    /// Bar2 only: weld struts shorter than `weld_factor * radius` (their
    /// own `radius` element field); 0 disables, no radius field skips.
    weld_factor: f64,
    /// Bar2 only: keep just the largest connected component.
    prune_islands: bool,
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            keep: KeepConfig::Inside,
            crossing: CrossingConfig::Clip,
            interior_samples: 0,
            weld_factor: 1.0,
            prune_islands: false,
        }
    }
}

/// A batched keep-classifier: for each position, whether it lies in the
/// kept region. The operator backs this with chunked host sampling;
/// tests with analytic domains.
type KeptBatch<'a> = dyn FnMut(&[[f64; 3]]) -> Result<Vec<bool>, String> + 'a;

/// Clip `mesh` to the kept region. `node_kept` is the per-node
/// classification; `kept` answers arbitrary points (crossing bisection).
/// Pure so tests can drive it without a host.
fn clip_mesh(
    mesh: &FeaMesh,
    node_kept: &[bool],
    kept: &mut KeptBatch,
    config: &ClipConfig,
) -> Result<FeaMesh, String> {
    // Tunnel detection: classify interior points of every strut in one
    // batch; a strut whose interior leaves the kept region is dropped
    // whole in both crossing modes (a tunnel cannot be shortened into a
    // single kept segment).
    let tunnel: Vec<bool> =
        if mesh.element_kind == FeaElementKind::Bar2 && config.interior_samples > 0 {
            let n = config.interior_samples as usize;
            let mut points = Vec::with_capacity(mesh.element_count() * n);
            for e in 0..mesh.element_count() {
                let pair = mesh.element(e);
                let (a, b) = (
                    mesh.node_position(pair[0] as usize),
                    mesh.node_position(pair[1] as usize),
                );
                for i in 1..=n {
                    let t = i as f64 / (n + 1) as f64;
                    points.push(core::array::from_fn(|c| a[c] + t * (b[c] - a[c])));
                }
            }
            let verdicts = kept(&points)?;
            verdicts.chunks(n).map(|c| c.iter().any(|&k| !k)).collect()
        } else {
            vec![false; mesh.element_count()]
        };

    let mut clipped = match mesh.element_kind {
        FeaElementKind::Point1 => {
            let keep: Vec<bool> = mesh
                .connectivity
                .iter()
                .map(|&n| node_kept[n as usize])
                .collect();
            mesh_edit_core::filter_elements(mesh, &keep)?
        }
        FeaElementKind::Hex8 => {
            let keep: Vec<bool> = (0..mesh.element_count())
                .map(|e| mesh.element(e).iter().all(|&n| node_kept[n as usize]))
                .collect();
            mesh_edit_core::filter_elements(mesh, &keep)?
        }
        FeaElementKind::Bar2 if config.crossing == CrossingConfig::Drop => {
            let keep: Vec<bool> = (0..mesh.element_count())
                .map(|e| !tunnel[e] && mesh.element(e).iter().all(|&n| node_kept[n as usize]))
                .collect();
            mesh_edit_core::filter_elements(mesh, &keep)?
        }
        FeaElementKind::Bar2 => clip_bars(mesh, node_kept, kept, &tunnel)?,
    };

    if clipped.element_count() == 0 {
        return Err(format!(
            "clipping kept no elements (of {}) — is the clip region {} the mesh?",
            mesh.element_count(),
            match config.keep {
                KeepConfig::Inside => "disjoint from",
                KeepConfig::Outside => "covering",
            }
        ));
    }

    if mesh.element_kind == FeaElementKind::Bar2 {
        if config.weld_factor > 0.0
            && let Some(radius) = clipped
                .element_fields
                .iter()
                .find(|f| f.name == "radius" && f.components == 1)
        {
            let thresholds: Vec<f64> = radius.data.iter().map(|r| r * config.weld_factor).collect();
            clipped = mesh_edit_core::weld_short_bars(&clipped, &|e| thresholds[e])?;
        }
        if config.prune_islands && clipped.element_count() > 0 {
            clipped = mesh_edit_core::largest_bar_component(&clipped)?;
        }
        if clipped.element_count() == 0 {
            return Err("welding collapsed every clipped strut".to_string());
        }
    }
    Ok(clipped)
}

/// Bar2 clip-to-surface: kept struts intern their nodes, crossings gain a
/// fresh node bisected onto the boundary (node fields interpolated at the
/// crossing parameter), fully-dropped struts vanish. Element fields
/// follow surviving struts.
fn clip_bars(
    mesh: &FeaMesh,
    node_kept: &[bool],
    kept: &mut KeptBatch,
    tunnel: &[bool],
) -> Result<FeaMesh, String> {
    // Crossing struts: (element, kept node, lost node), bisected in
    // lock-step so each round is one batched classification.
    struct Crossing {
        element: usize,
        keep_node: u32,
        lose_node: u32,
        t_in: f64,
        t_out: f64,
    }
    let mut crossings: Vec<Crossing> = Vec::new();
    let mut whole: Vec<usize> = Vec::new();
    for (e, &tunneling) in tunnel.iter().enumerate() {
        if tunneling {
            continue; // interior leaves the kept region: dropped whole
        }
        let pair = mesh.element(e);
        match (node_kept[pair[0] as usize], node_kept[pair[1] as usize]) {
            (true, true) => whole.push(e),
            (false, false) => {}
            (a_kept, _) => {
                let (keep_node, lose_node) = if a_kept {
                    (pair[0], pair[1])
                } else {
                    (pair[1], pair[0])
                };
                crossings.push(Crossing {
                    element: e,
                    keep_node,
                    lose_node,
                    t_in: 0.0,
                    t_out: 1.0,
                });
            }
        }
    }

    let lerp = |a: u32, b: u32, t: f64| -> [f64; 3] {
        let (pa, pb) = (
            mesh.node_position(a as usize),
            mesh.node_position(b as usize),
        );
        core::array::from_fn(|c| pa[c] + t * (pb[c] - pa[c]))
    };
    for _ in 0..CLIP_BISECTIONS {
        if crossings.is_empty() {
            break;
        }
        let midpoints: Vec<[f64; 3]> = crossings
            .iter()
            .map(|x| lerp(x.keep_node, x.lose_node, 0.5 * (x.t_in + x.t_out)))
            .collect();
        let verdicts = kept(&midpoints)?;
        for (x, in_kept) in crossings.iter_mut().zip(verdicts) {
            let mid = 0.5 * (x.t_in + x.t_out);
            if in_kept {
                x.t_in = mid;
            } else {
                x.t_out = mid;
            }
        }
    }

    // Assemble: original kept nodes intern on first use; each surviving
    // crossing appends one fresh node. `sources` records, in output-node
    // order, where every node's field values come from — originals copy,
    // fresh nodes interpolate — so field assembly cannot fall out of step
    // with the (interleaved) position order.
    enum Source {
        Original(u32),
        Lerp { a: u32, b: u32, t: f64 },
    }
    let mut node_remap = vec![u32::MAX; mesh.node_count()];
    let mut sources: Vec<Source> = Vec::new();
    let mut positions: Vec<f64> = Vec::new();
    let mut connectivity: Vec<u32> = Vec::new();
    let mut origins: Vec<u32> = Vec::new();
    let intern =
        |node: u32, positions: &mut Vec<f64>, sources: &mut Vec<Source>, node_remap: &mut [u32]| {
            if node_remap[node as usize] == u32::MAX {
                node_remap[node as usize] = (positions.len() / 3) as u32;
                sources.push(Source::Original(node));
                positions.extend(mesh.node_position(node as usize));
            }
            node_remap[node as usize]
        };
    for &e in &whole {
        let pair = mesh.element(e);
        let ia = intern(pair[0], &mut positions, &mut sources, &mut node_remap);
        let ib = intern(pair[1], &mut positions, &mut sources, &mut node_remap);
        connectivity.extend([ia, ib]);
        origins.push(e as u32);
    }
    for x in &crossings {
        let t = 0.5 * (x.t_in + x.t_out);
        if t < MIN_CLIP_FRACTION {
            continue; // stub too short to be a sane element
        }
        let ia = intern(x.keep_node, &mut positions, &mut sources, &mut node_remap);
        let ib = (positions.len() / 3) as u32;
        positions.extend(lerp(x.keep_node, x.lose_node, t));
        sources.push(Source::Lerp {
            a: x.keep_node,
            b: x.lose_node,
            t,
        });
        connectivity.extend([ia, ib]);
        origins.push(x.element as u32);
    }

    let node_fields = mesh
        .node_fields
        .iter()
        .map(|f| {
            let mut data = Vec::with_capacity(sources.len() * f.components);
            for source in &sources {
                match *source {
                    Source::Original(n) => {
                        let base = n as usize * f.components;
                        data.extend_from_slice(&f.data[base..base + f.components]);
                    }
                    Source::Lerp { a, b, t } => {
                        let (a, b) = (a as usize * f.components, b as usize * f.components);
                        for c in 0..f.components {
                            data.push(f.data[a + c] + t * (f.data[b + c] - f.data[a + c]));
                        }
                    }
                }
            }
            volumetric_abi::fea::FeaField {
                name: f.name.clone(),
                components: f.components,
                data,
            }
        })
        .collect();

    let mesh = FeaMesh {
        element_kind: FeaElementKind::Bar2,
        node_positions: positions,
        connectivity,
        node_fields,
        element_fields: mesh
            .element_fields
            .iter()
            .map(|f| volumetric_abi::fea::FeaField {
                name: f.name.clone(),
                components: f.components,
                data: origins
                    .iter()
                    .flat_map(|&e| {
                        let base = e as usize * f.components;
                        f.data[base..base + f.components].iter().copied()
                    })
                    .collect(),
            })
            .collect(),
    };
    mesh.validate()?;
    Ok(mesh)
}

fn build_clipped(config: &ClipConfig) -> Result<FeaMesh, String> {
    if !(config.weld_factor.is_finite() && config.weld_factor >= 0.0) {
        return Err(format!(
            "weld_factor must be non-negative, got {}",
            config.weld_factor
        ));
    }
    if config.interior_samples > 256 {
        return Err(format!(
            "interior_samples capped at 256, got {}",
            config.interior_samples
        ));
    }
    let mesh = decode_fea_mesh(&read_input(0))?;
    let dims =
        input_model_dimensions(1).ok_or_else(|| "input 1 is not a usable model".to_string())?;
    if dims != 3 {
        return Err(format!(
            "mesh clip needs a 3D clip model; input has {dims} dimensions"
        ));
    }

    let keep_inside = config.keep == KeepConfig::Inside;
    let mut kept = |points: &[[f64; 3]]| -> Result<Vec<bool>, String> {
        let mut out = Vec::with_capacity(points.len());
        for chunk in points.chunks(SAMPLE_CHUNK) {
            let positions: Vec<f64> = chunk.iter().flatten().copied().collect();
            let samples = input_model_sample(1, &positions, 3)
                .ok_or_else(|| "sampling the clip model failed".to_string())?;
            out.extend(samples.iter().map(|&s| is_occupied(s) == keep_inside));
        }
        Ok(out)
    };

    let node_points: Vec<[f64; 3]> = (0..mesh.node_count())
        .map(|n| mesh.node_position(n))
        .collect();
    let node_kept = kept(&node_points)?;

    clip_mesh(&mesh, &node_kept, &mut kept, config)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(2);
        if buf.is_empty() {
            ClipConfig::default()
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

    match build_clipped(&config) {
        Ok(mesh) => post_output(0, &encode_fea_mesh(&mesh)),
        Err(e) => report_error(&format!("mesh clip failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ keep: "inside" / "outside" .default "inside", crossing: "clip" / "drop" .default "clip", interior_samples: int .default 0, weld_factor: float .default 1.0, prune_islands: bool .default false }"#
            .to_string();
        OperatorMetadata {
            name: "mesh_clip_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Mesh Clip".to_string(),
            description: "Clip a mesh (points, struts, volumes) against a model: \
                          keep the inside or the outside."
                .to_string(),
            category: "Mesh".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<circle cx="6" cy="6" r="3"/>"##,
                r##"<path d="M8.12 8.12 12 12"/>"##,
                r##"<path d="M20 4 8.12 15.88"/>"##,
                r##"<circle cx="6" cy="18" r="3"/>"##,
                r##"<path d="M14.8 14.8 20 20"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::FeaMesh,
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec![
                "Mesh".to_string(),
                "Clip model".to_string(),
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

    /// Kept-region classifier for a sphere of radius `r` at the origin.
    fn sphere_kept(
        r: f64,
        keep_inside: bool,
    ) -> impl FnMut(&[[f64; 3]]) -> Result<Vec<bool>, String> {
        move |points| {
            Ok(points
                .iter()
                .map(|p| (p.iter().map(|c| c * c).sum::<f64>() < r * r) == keep_inside)
                .collect())
        }
    }

    fn classify(
        mesh: &FeaMesh,
        kept: &mut impl FnMut(&[[f64; 3]]) -> Result<Vec<bool>, String>,
    ) -> Vec<bool> {
        let points: Vec<[f64; 3]> = (0..mesh.node_count())
            .map(|n| mesh.node_position(n))
            .collect();
        kept(&points).unwrap()
    }

    fn point_cloud() -> FeaMesh {
        // Five points along +x at 0.1 .. 0.9.
        let xs = [0.1, 0.3, 0.5, 0.7, 0.9];
        FeaMesh {
            element_kind: FeaElementKind::Point1,
            node_positions: xs.iter().flat_map(|&x| [x, 0.0, 0.0]).collect(),
            connectivity: (0..5).collect(),
            node_fields: vec![FeaField {
                name: "weight".to_string(),
                components: 1,
                data: xs.to_vec(),
            }],
            element_fields: vec![],
        }
    }

    #[test]
    fn point_clouds_partition_between_inside_and_outside() {
        let cloud = point_cloud();
        let config = ClipConfig::default();

        let mut inside = sphere_kept(0.6, true);
        let node_kept = classify(&cloud, &mut inside);
        let kept = clip_mesh(&cloud, &node_kept, &mut inside, &config).unwrap();
        assert_eq!(kept.element_count(), 3);
        assert_eq!(kept.node_fields[0].data, vec![0.1, 0.3, 0.5]);

        let mut outside = sphere_kept(0.6, false);
        let node_kept = classify(&cloud, &mut outside);
        let kept = clip_mesh(&cloud, &node_kept, &mut outside, &config).unwrap();
        assert_eq!(kept.element_count(), 2);
        assert_eq!(kept.node_fields[0].data, vec![0.7, 0.9]);
    }

    fn strut_path() -> FeaMesh {
        // A 3-strut path along +x crossing the r=0.5 sphere skin between
        // nodes 1 and 2: 0.0 -- 0.4 -- 0.8 -- 1.2.
        FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions: vec![
                0.0, 0.0, 0.0, //
                0.4, 0.0, 0.0, //
                0.8, 0.0, 0.0, //
                1.2, 0.0, 0.0,
            ],
            connectivity: vec![0, 1, 1, 2, 2, 3],
            node_fields: vec![FeaField {
                name: "score".to_string(),
                components: 1,
                data: vec![0.0, 0.4, 0.8, 1.2],
            }],
            element_fields: vec![FeaField {
                name: "radius".to_string(),
                components: 1,
                data: vec![0.01, 0.02, 0.03],
            }],
        }
    }

    #[test]
    fn bar_crossings_clip_to_the_surface_with_interpolated_fields() {
        let mesh = strut_path();
        let config = ClipConfig::default();
        let mut inside = sphere_kept(0.5, true);
        let node_kept = classify(&mesh, &mut inside);
        let clipped = clip_mesh(&mesh, &node_kept, &mut inside, &config).unwrap();

        // Strut 0 survives whole; strut 1 clips at x = 0.5; strut 2 dies.
        assert_eq!(clipped.element_count(), 2);
        let skin = (0..clipped.node_count())
            .map(|n| clipped.node_position(n))
            .find(|p| (p[0] - 0.5).abs() < 1e-6)
            .expect("crossing node on the skin");
        assert_eq!(&skin[1..], &[0.0, 0.0]);
        // The node field interpolates to the crossing (score == x here).
        let score = &clipped.node_fields[0];
        for n in 0..clipped.node_count() {
            let x = clipped.node_position(n)[0];
            assert!(
                (score.data[n] - x).abs() < 1e-6,
                "node {n}: score {} != x {x}",
                score.data[n]
            );
        }
        // Element fields follow their struts.
        assert_eq!(clipped.element_fields[0].data, vec![0.01, 0.02]);

        // Drop mode: the crossing strut vanishes instead.
        let config = ClipConfig {
            crossing: CrossingConfig::Drop,
            ..config
        };
        let mut inside = sphere_kept(0.5, true);
        let dropped = clip_mesh(&mesh, &node_kept, &mut inside, &config).unwrap();
        assert_eq!(dropped.element_count(), 1);
        assert_eq!(dropped.element_fields[0].data, vec![0.01]);
    }

    #[test]
    fn bar_clip_welds_boundary_stubs_by_their_own_radius() {
        // A strut whose kept sliver (0.48 .. crossing at 0.5) is shorter
        // than weld_factor * radius: the stub welds away, leaving the
        // interior strut intact.
        let mesh = FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions: vec![
                0.0, 0.0, 0.0, //
                0.48, 0.0, 0.0, //
                0.9, 0.0, 0.0,
            ],
            connectivity: vec![0, 1, 1, 2],
            node_fields: vec![],
            element_fields: vec![FeaField {
                name: "radius".to_string(),
                components: 1,
                data: vec![0.01, 0.05],
            }],
        };
        let config = ClipConfig::default(); // weld_factor 1.0
        let mut inside = sphere_kept(0.5, true);
        let node_kept = classify(&mesh, &mut inside);
        let clipped = clip_mesh(&mesh, &node_kept, &mut inside, &config).unwrap();
        assert_eq!(clipped.element_count(), 1, "the 0.02 stub should weld");
        assert_eq!(clipped.element_fields[0].data, vec![0.01]);
    }

    #[test]
    fn hex_elements_need_every_node_kept() {
        // Two unit hexes sharing the x = 1 face; a half-space keeps only
        // the first hex's nodes.
        let mut node_positions = Vec::new();
        for x in 0..3 {
            for y in 0..2 {
                for z in 0..2 {
                    node_positions.extend([x as f64, y as f64, z as f64]);
                }
            }
        }
        let hex_at = |x0: u32| {
            let n = |dx: u32, dy: u32, dz: u32| (x0 + dx) * 4 + dy * 2 + dz;
            [
                n(0, 0, 0),
                n(1, 0, 0),
                n(1, 1, 0),
                n(0, 1, 0),
                n(0, 0, 1),
                n(1, 0, 1),
                n(1, 1, 1),
                n(0, 1, 1),
            ]
        };
        let mesh = FeaMesh {
            element_kind: FeaElementKind::Hex8,
            node_positions,
            connectivity: [hex_at(0), hex_at(1)].concat(),
            node_fields: vec![],
            element_fields: vec![],
        };
        let mut kept_fn = |points: &[[f64; 3]]| -> Result<Vec<bool>, String> {
            Ok(points.iter().map(|p| p[0] < 1.5).collect())
        };
        let node_kept = classify(&mesh, &mut kept_fn);
        let clipped = clip_mesh(&mesh, &node_kept, &mut kept_fn, &ClipConfig::default()).unwrap();
        assert_eq!(clipped.element_count(), 1);
        assert_eq!(clipped.node_count(), 8);
    }

    #[test]
    fn clipping_everything_away_is_an_error() {
        let cloud = point_cloud();
        let mut nothing =
            |points: &[[f64; 3]]| -> Result<Vec<bool>, String> { Ok(vec![false; points.len()]) };
        let node_kept = classify(&cloud, &mut nothing);
        let err = clip_mesh(&cloud, &node_kept, &mut nothing, &ClipConfig::default()).unwrap_err();
        assert!(err.contains("kept no elements"), "unexpected error: {err}");
    }

    #[test]
    fn interior_samples_drop_struts_tunneling_a_thin_gap() {
        // Kept region: everything except a thin slab 0.55 < x < 0.65.
        // The 2-strut path 0.0 -- 0.4 -- 1.2 has all endpoints kept, but
        // its second strut tunnels the gap.
        let mut slab_gap = |points: &[[f64; 3]]| -> Result<Vec<bool>, String> {
            Ok(points
                .iter()
                .map(|p| !(0.55 < p[0] && p[0] < 0.65))
                .collect())
        };
        let mesh = FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions: vec![0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 1.2, 0.0, 0.0],
            connectivity: vec![0, 1, 1, 2],
            node_fields: vec![],
            element_fields: vec![],
        };
        let node_kept = classify(&mesh, &mut slab_gap);
        assert!(node_kept.iter().all(|&k| k), "all endpoints sit in bands");

        // Endpoint-only classification misses the tunnel in both modes.
        for crossing in [CrossingConfig::Clip, CrossingConfig::Drop] {
            let config = ClipConfig {
                crossing,
                ..ClipConfig::default()
            };
            let kept = clip_mesh(&mesh, &node_kept, &mut slab_gap, &config).unwrap();
            assert_eq!(kept.element_count(), 2, "{crossing:?} without samples");

            // 15 interior samples (spacing 0.05 < gap 0.1) catch it.
            let config = ClipConfig {
                interior_samples: 15,
                ..config
            };
            let kept = clip_mesh(&mesh, &node_kept, &mut slab_gap, &config).unwrap();
            assert_eq!(kept.element_count(), 1, "{crossing:?} with samples");
            let p = kept.node_position(1);
            assert!(p[0] <= 0.4 + 1e-12, "the surviving strut is 0.0 -- 0.4");
        }
    }
}

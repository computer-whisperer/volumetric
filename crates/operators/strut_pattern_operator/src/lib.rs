//! Strut Pattern Operator.
//!
//! Generates an explicit strut lattice — a Bar2 [`FeaMesh`] — filling a 3D
//! domain model: the chosen family's skeleton (see
//! `lattice_model_core::skeleton`; the same networks the implicit
//! `lattice_operator` thickens, in the same coordinates) is enumerated
//! over the domain's bounds and clipped against its occupancy. The output
//! feeds `fea_solve_operator` / `fea_inverse_operator` directly (frame
//! elements), and later the strut-model operator for geometry realization.
//!
//! Clipping:
//! - Struts with both endpoints inside the domain are kept whole.
//! - Struts crossing the boundary are shortened to the surface (bisection
//!   along the strut), creating a node ON the domain skin — that's what
//!   makes the lattice contactable by a rigid body and glueable at a
//!   fixed face. Crossing nodes are per-strut (no welding of distinct
//!   struts at the skin).
//! - Struts with both endpoints outside are dropped.
//! - Struts shorter than `weld_factor * radius` are welded away: their
//!   endpoints merge at the cluster centroid and any parallel duplicates
//!   collapse. A strut shorter than its own radius is a joint blob, not a
//!   beam — near-degenerate Voronoi edges (and boundary-clip stubs) carry
//!   bending stiffness ~1/L^3, and leaving them in makes the solver's
//!   conditioning explode (measured: an unwelded foam fails CG at 3e8
//!   stiffness contrast; welded at 1 radius it converges in ~10k
//!   iterations).
//! - Unless `prune_islands` is off, only the largest connected component
//!   survives: floating fragments (domain concavities can cut them loose)
//!   would make the FEA solve singular.
//!
//! Inputs:
//! - Input 0: ModelWASM (must be 3D) — the domain to fill
//! - Input 1: CBOR configuration:
//!   `{ family: "cubic" / "tetra" / "foam" .default "tetra", cell_size:
//!   float .default 0.05, radius: float .default 0.0 (0 = cell_size / 10),
//!   prune_islands: bool .default true, irregularity: float .default 0.3
//!   (foam only), weld_factor: float .default 1.0 (0 disables welding) }`
//!
//! Output 0: CBOR-encoded Bar2 `FeaMesh` with a uniform scalar `radius`
//! element field.

use lattice_model_core::skeleton::{
    Skeleton, SkeletonFamily, enumerate_skeleton, estimate_strut_count,
};
use volumetric_abi::fea::{FeaElementKind, FeaField, FeaMesh, encode_fea_mesh};
use volumetric_abi::host::{
    input_model_bounds, input_model_dimensions, input_model_sample, post_output, read_input,
    report_error,
};
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, is_occupied,
};

/// Refuse enumerations past this many candidate struts (a 128-cells-per-
/// axis cubic domain is ~6.3M; solves want far fewer).
const MAX_STRUTS: u64 = 2_000_000;

/// Occupancy samples per batched host call.
const SAMPLE_CHUNK: usize = 8192;

/// Bisection steps for a boundary crossing (resolves the surface to
/// `~cell / 2^24` along the strut).
const CLIP_BISECTIONS: usize = 24;

/// Clipped struts shorter than this fraction of the strut's full length
/// are dropped: near-zero frame elements are stiffness spikes, and the
/// interior endpoint stays connected through its other struts.
const MIN_CLIP_FRACTION: f64 = 1e-3;

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum FamilyConfig {
    Cubic,
    Tetra,
    /// Voronoi foam (the production cushion pattern): `irregularity`
    /// picks the cell shape, 0 = periodic Kelvin cells.
    Foam,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct PatternConfig {
    family: FamilyConfig,
    cell_size: f64,
    /// Strut cross-section radius; 0 = `cell_size / 10`.
    radius: f64,
    prune_islands: bool,
    /// Foam cell-shape jitter, 0 (Kelvin) ..= 1 (fully organic); the
    /// other families ignore it. Matches `lattice_operator`'s knob.
    irregularity: f64,
    /// Struts shorter than `weld_factor * radius` are welded into a
    /// single joint node; 0 disables.
    weld_factor: f64,
}

impl PatternConfig {
    fn as_family(&self) -> SkeletonFamily {
        match self.family {
            FamilyConfig::Cubic => SkeletonFamily::Cubic,
            FamilyConfig::Tetra => SkeletonFamily::Tetra,
            FamilyConfig::Foam => SkeletonFamily::Foam {
                irregularity: self.irregularity,
            },
        }
    }
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            family: FamilyConfig::Tetra,
            cell_size: 0.05,
            radius: 0.0,
            prune_islands: true,
            irregularity: 0.3,
            weld_factor: 1.0,
        }
    }
}

/// Clip the skeleton against a domain and assemble the Bar2 mesh.
/// `node_occupied` is per skeleton node; `occupied` answers arbitrary
/// points (boundary bisection). Pure so tests can drive it with analytic
/// domains.
fn clip_skeleton(
    skeleton: &Skeleton,
    node_occupied: &[bool],
    occupied: &mut dyn FnMut([f64; 3]) -> bool,
    radius: f64,
    prune_islands: bool,
    weld_length: f64,
) -> Result<FeaMesh, String> {
    // Output nodes: skeleton nodes on first use (compacted), plus one
    // fresh node per clipped crossing.
    let mut out_id = vec![u32::MAX; skeleton.nodes.len()];
    let mut positions: Vec<f64> = Vec::new();
    let mut connectivity: Vec<u32> = Vec::new();

    let intern = |node: usize, out_id: &mut [u32], positions: &mut Vec<f64>| -> u32 {
        if out_id[node] == u32::MAX {
            out_id[node] = (positions.len() / 3) as u32;
            positions.extend(skeleton.nodes[node]);
        }
        out_id[node]
    };

    for edge in &skeleton.edges {
        let (a, b) = (edge[0] as usize, edge[1] as usize);
        match (node_occupied[a], node_occupied[b]) {
            (true, true) => {
                let ia = intern(a, &mut out_id, &mut positions);
                let ib = intern(b, &mut out_id, &mut positions);
                connectivity.extend([ia, ib]);
            }
            (true, false) | (false, true) => {
                let (inside, outside) = if node_occupied[a] { (a, b) } else { (b, a) };
                let p_in = skeleton.nodes[inside];
                let p_out = skeleton.nodes[outside];
                // Bisect for the surface crossing on the strut.
                let (mut t_in, mut t_out) = (0.0f64, 1.0f64);
                for _ in 0..CLIP_BISECTIONS {
                    let mid = 0.5 * (t_in + t_out);
                    let p = [
                        p_in[0] + mid * (p_out[0] - p_in[0]),
                        p_in[1] + mid * (p_out[1] - p_in[1]),
                        p_in[2] + mid * (p_out[2] - p_in[2]),
                    ];
                    if occupied(p) {
                        t_in = mid;
                    } else {
                        t_out = mid;
                    }
                }
                let t = 0.5 * (t_in + t_out);
                if t < MIN_CLIP_FRACTION {
                    continue; // stub too short to be a sane frame element
                }
                let ia = intern(inside, &mut out_id, &mut positions);
                let ib = (positions.len() / 3) as u32;
                positions.extend([
                    p_in[0] + t * (p_out[0] - p_in[0]),
                    p_in[1] + t * (p_out[1] - p_in[1]),
                    p_in[2] + t * (p_out[2] - p_in[2]),
                ]);
                connectivity.extend([ia, ib]);
            }
            (false, false) => {}
        }
    }

    if connectivity.is_empty() {
        return Err(
            "no struts inside the domain (is cell_size much larger than the model, \
             or the model empty?)"
                .to_string(),
        );
    }

    let strut_count = connectivity.len() / 2;
    let mut mesh = FeaMesh {
        element_kind: FeaElementKind::Bar2,
        node_positions: positions,
        connectivity,
        node_fields: vec![],
        element_fields: vec![FeaField {
            name: "radius".to_string(),
            components: 1,
            data: vec![radius; strut_count],
        }],
    };
    mesh.validate()?;

    if weld_length > 0.0 {
        mesh = mesh_edit_core::weld_short_bars(&mesh, &|_| weld_length)?;
        if mesh.element_count() == 0 {
            return Err(format!(
                "welding at length {weld_length} collapsed every strut \
                 (is weld_factor * radius larger than the struts?)"
            ));
        }
    }

    if prune_islands {
        mesh = mesh_edit_core::largest_bar_component(&mesh)?;
    }
    Ok(mesh)
}

fn build_pattern(config: &PatternConfig) -> Result<FeaMesh, String> {
    if !(config.cell_size.is_finite() && config.cell_size > 0.0) {
        return Err(format!(
            "cell_size must be positive, got {}",
            config.cell_size
        ));
    }
    let radius = if config.radius == 0.0 {
        config.cell_size / 10.0
    } else {
        config.radius
    };
    if !(radius.is_finite() && radius > 0.0) {
        return Err(format!("radius must be positive, got {}", config.radius));
    }
    if !(config.irregularity.is_finite() && (0.0..=1.0).contains(&config.irregularity)) {
        return Err(format!(
            "irregularity must be in 0..=1, got {}",
            config.irregularity
        ));
    }
    if !(config.weld_factor.is_finite() && config.weld_factor >= 0.0) {
        return Err(format!(
            "weld_factor must be non-negative, got {}",
            config.weld_factor
        ));
    }

    let dims =
        input_model_dimensions(0).ok_or_else(|| "input 0 is not a usable model".to_string())?;
    if dims != 3 {
        return Err(format!(
            "strut patterns need a 3D domain model; input has {dims} dimensions"
        ));
    }
    let bounds =
        input_model_bounds(0, 3).ok_or_else(|| "failed to read model bounds".to_string())?;
    let lo = [bounds[0], bounds[2], bounds[4]];
    let hi = [bounds[1], bounds[3], bounds[5]];

    let family = config.as_family();
    let estimate = estimate_strut_count(family, lo, hi, config.cell_size);
    if estimate > MAX_STRUTS {
        return Err(format!(
            "cell_size {} would enumerate ~{estimate} struts (cap {MAX_STRUTS}); \
             raise cell_size",
            config.cell_size
        ));
    }

    let skeleton = enumerate_skeleton(family, lo, hi, config.cell_size);

    // Batched occupancy of every skeleton node.
    let mut node_occupied = vec![false; skeleton.nodes.len()];
    for (chunk_index, chunk) in skeleton.nodes.chunks(SAMPLE_CHUNK).enumerate() {
        let positions: Vec<f64> = chunk.iter().flatten().copied().collect();
        let samples = input_model_sample(0, &positions, 3)
            .ok_or_else(|| "sampling the domain model failed".to_string())?;
        for (i, sample) in samples.iter().enumerate() {
            node_occupied[chunk_index * SAMPLE_CHUNK + i] = is_occupied(*sample);
        }
    }

    let mut occupied = |p: [f64; 3]| {
        input_model_sample(0, &p, 3)
            .map(|samples| is_occupied(samples[0]))
            .unwrap_or(false)
    };
    clip_skeleton(
        &skeleton,
        &node_occupied,
        &mut occupied,
        radius,
        config.prune_islands,
        config.weld_factor * radius,
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(1);
        if buf.is_empty() {
            PatternConfig::default()
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

    match build_pattern(&config) {
        Ok(mesh) => post_output(0, &encode_fea_mesh(&mesh)),
        Err(e) => report_error(&format!("strut pattern generation failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ family: "cubic" / "tetra" / "foam" .default "tetra", cell_size: float .default 0.05, radius: float .default 0.0, prune_islands: bool .default true, irregularity: float .default 0.3, weld_factor: float .default 1.0 }"#
            .to_string();
        OperatorMetadata {
            name: "strut_pattern_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            display_name: "Strut Pattern".to_string(),
            description:
                "Generate an explicit strut lattice (Bar2 mesh) filling a 3D domain model."
                    .to_string(),
            category: "Lattice".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<path d="M3 19h18"/>"##,
                r##"<path d="m3 19 4.5-8 4.5 8 4.5-8 4.5 8"/>"##,
                r##"<path d="M7.5 11h9"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            input_names: vec!["Domain model".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::FeaMesh],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_model_core::skeleton::enumerate_skeleton;

    /// Occupancy for a sphere of radius `r` at `c`.
    fn sphere(c: [f64; 3], r: f64) -> impl FnMut([f64; 3]) -> bool {
        move |p: [f64; 3]| (0..3).map(|i| (p[i] - c[i]).powi(2)).sum::<f64>() < r * r
    }

    fn clip_with(
        skeleton: &Skeleton,
        occupied: &mut dyn FnMut([f64; 3]) -> bool,
        prune: bool,
    ) -> Result<FeaMesh, String> {
        let node_occupied: Vec<bool> = skeleton.nodes.iter().map(|&p| occupied(p)).collect();
        clip_skeleton(skeleton, &node_occupied, occupied, 0.01, prune, 0.01)
    }

    #[test]
    fn sphere_clip_keeps_interior_and_reaches_the_skin() {
        let (center, radius) = ([0.5, 0.5, 0.5], 0.45);
        let skeleton = enumerate_skeleton(SkeletonFamily::Tetra, [0.0; 3], [1.0; 3], 0.2);
        let mesh = clip_with(&skeleton, &mut sphere(center, radius), true).unwrap();
        assert_eq!(mesh.element_kind, FeaElementKind::Bar2);
        assert!(
            mesh.element_count() > 50,
            "few struts: {}",
            mesh.element_count()
        );

        // Every node is inside or on the sphere (clip tolerance is tiny
        // relative to the cell).
        let mut on_skin = 0usize;
        for n in 0..mesh.node_count() {
            let p = mesh.node_position(n);
            let d = (0..3)
                .map(|i| (p[i] - center[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(d <= radius + 1e-6, "node {n} at distance {d} > {radius}");
            if (d - radius).abs() < 1e-6 {
                on_skin += 1;
            }
        }
        assert!(on_skin > 10, "no clipped nodes reached the skin: {on_skin}");

        // The radius field is uniform and sized to the struts.
        let field = &mesh.element_fields[0];
        assert_eq!(field.name, "radius");
        assert_eq!(field.data.len(), mesh.element_count());
        assert!(field.data.iter().all(|&r| r == 0.01));
    }

    #[test]
    fn pruning_drops_disconnected_islands() {
        // Two disjoint spheres: without pruning both lattices appear; with
        // pruning only the larger survives.
        let mut two_spheres = |p: [f64; 3]| {
            let big = (0..3).map(|i| (p[i] - 0.5).powi(2)).sum::<f64>() < 0.45 * 0.45;
            let small_center = [2.5, 0.5, 0.5];
            let small = (0..3)
                .map(|i| (p[i] - small_center[i]).powi(2))
                .sum::<f64>()
                < 0.25 * 0.25;
            big || small
        };
        let skeleton = enumerate_skeleton(SkeletonFamily::Cubic, [0.0; 3], [3.0, 1.0, 1.0], 0.15);

        let unpruned = clip_with(&skeleton, &mut two_spheres, false).unwrap();
        let pruned = clip_with(&skeleton, &mut two_spheres, true).unwrap();
        assert!(pruned.element_count() < unpruned.element_count());
        // Everything pruned sits in the big sphere.
        for n in 0..pruned.node_count() {
            let p = pruned.node_position(n);
            assert!(p[0] < 1.5, "island node {p:?} survived pruning");
        }
        // Pruned output is one connected component.
        let mut parent: Vec<usize> = (0..pruned.node_count()).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for e in 0..pruned.element_count() {
            let pair = pruned.element(e).to_vec();
            let (ra, rb) = (
                find(&mut parent, pair[0] as usize),
                find(&mut parent, pair[1] as usize),
            );
            if ra != rb {
                parent[ra] = rb;
            }
        }
        let root = find(&mut parent, 0);
        for n in 0..pruned.node_count() {
            assert_eq!(find(&mut parent, n), root, "node {n} disconnected");
        }
    }

    #[test]
    fn empty_domains_are_an_error() {
        let skeleton = enumerate_skeleton(SkeletonFamily::Tetra, [0.0; 3], [1.0; 3], 0.2);
        let err = clip_with(&skeleton, &mut |_| false, true).unwrap_err();
        assert!(err.contains("no struts"), "unexpected error: {err}");
    }

    #[test]
    fn clipped_meshes_have_no_short_struts() {
        // The property the FEA solver needs: no strut shorter than the
        // weld length survives clipping (foam's near-degenerate Voronoi
        // edges and boundary stubs are otherwise 1/L^3 stiffness spikes).
        let skeleton = enumerate_skeleton(
            SkeletonFamily::Foam { irregularity: 0.3 },
            [0.0; 3],
            [1.0; 3],
            0.2,
        );
        let mesh = clip_with(&skeleton, &mut sphere([0.5; 3], 0.45), true).unwrap();
        for e in 0..mesh.element_count() {
            let pair = mesh.element(e);
            let a = mesh.node_position(pair[0] as usize);
            let b = mesh.node_position(pair[1] as usize);
            let len = (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>().sqrt();
            assert!(
                len >= 0.01,
                "strut {e} survived below the weld length: {len}"
            );
        }
    }

    #[test]
    fn interior_struts_keep_lattice_lengths() {
        // Cubic struts fully inside the sphere are exactly one cell long.
        let cell = 0.2;
        let skeleton = enumerate_skeleton(SkeletonFamily::Cubic, [0.0; 3], [1.0; 3], cell);
        let mesh = clip_with(&skeleton, &mut sphere([0.5; 3], 0.45), true).unwrap();
        let mut full = 0usize;
        for e in 0..mesh.element_count() {
            let pair = mesh.element(e);
            let a = mesh.node_position(pair[0] as usize);
            let b = mesh.node_position(pair[1] as usize);
            let len = (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>().sqrt();
            assert!(len <= cell + 1e-9, "strut {e} longer than a cell: {len}");
            if (len - cell).abs() < 1e-9 {
                full += 1;
            }
        }
        assert!(full > 20, "no full-length interior struts: {full}");
    }
}

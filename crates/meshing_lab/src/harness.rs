//! Shared pipeline: oracle shape -> real mesher output -> per-vertex patch fits.
//!
//! Every benchmark in this crate goes through here so they all measure the
//! same data the production mesher would actually have.

use glam::DVec3;
use volumetric::adaptive_surface_nets_2::{AdaptiveMeshConfig2, adaptive_surface_nets_2};

use crate::adjacency::MeshAdjacency;
use crate::fit::{VertexFit, ring_fits};
use crate::oracle::{OracleShape, SurfaceTruth};

/// A shape meshed by the real adaptive surface nets pipeline, with oracle
/// truth evaluated at every vertex.
pub struct MeshedShape<'a> {
    pub shape: &'a dyn OracleShape,
    pub positions: Vec<DVec3>,
    /// Accumulated (topology) normals from the mesher; outward-oriented.
    pub mesh_normals: Vec<DVec3>,
    pub indices: Vec<u32>,
    pub adjacency: MeshAdjacency,
    /// Finest cell size (max over axes).
    pub cell: f64,
    pub truths: Vec<SurfaceTruth>,
    pub total_samples: u64,
    pub refine_misses: u64,
}

/// Mesh `shape` at `base_resolution * 2^max_depth` effective cells with a 12%
/// bounds margin, refined vertices, and accumulated normals (no normal
/// probing).
pub fn mesh_shape(shape: &dyn OracleShape, max_depth: usize) -> MeshedShape<'_> {
    mesh_shape_with_margin(shape, max_depth, 0.12)
}

/// [`mesh_shape`] with an explicit bounds margin fraction. The production
/// path pads by ~2 finest cells (fraction `2 / effective_resolution`), which
/// aligns features with the grid very differently than a generous margin --
/// benchmark both.
pub fn mesh_shape_with_margin(
    shape: &dyn OracleShape,
    max_depth: usize,
    margin_frac: f64,
) -> MeshedShape<'_> {
    let (lo, hi) = shape.world_bounds();
    let margin = (hi - lo).max_element() * margin_frac;
    let lo = lo - DVec3::splat(margin);
    let hi = hi + DVec3::splat(margin);

    let config = AdaptiveMeshConfig2 {
        base_resolution: 8,
        discovery_probes: 8,
        max_depth,
        vertex_refinement_iterations: 12,
        normal_sample_iterations: 0,
        normal_epsilon_frac: 0.1,
        num_threads: 0,
        sharp_features: None,
        edge_constrained_refinement: false,
        decimation: None,
    };
    let finest_cells = config.base_resolution * (1 << config.max_depth);
    let cell = (hi - lo).max_element() / finest_cells as f64;

    let result = adaptive_surface_nets_2(
        |x, y, z| {
            if shape.is_inside(DVec3::new(x, y, z)) {
                1.0
            } else {
                0.0
            }
        },
        (lo.x as f32, lo.y as f32, lo.z as f32),
        (hi.x as f32, hi.y as f32, hi.z as f32),
        &config,
    );

    let positions: Vec<DVec3> = result
        .mesh
        .vertices
        .iter()
        .map(|&(x, y, z)| DVec3::new(x as f64, y as f64, z as f64))
        .collect();
    let mesh_normals: Vec<DVec3> = result
        .mesh
        .normals
        .iter()
        .map(|&(x, y, z)| DVec3::new(x as f64, y as f64, z as f64))
        .collect();
    let adjacency = MeshAdjacency::build(positions.len(), &result.mesh.indices);
    let truths: Vec<SurfaceTruth> = positions.iter().map(|&p| shape.truth(p)).collect();

    MeshedShape {
        shape,
        positions,
        mesh_normals,
        indices: result.mesh.indices,
        adjacency,
        cell,
        truths,
        total_samples: result.stats.total_samples,
        refine_misses: result.stats.stage4_refine_miss,
    }
}

impl MeshedShape<'_> {
    pub fn ring_fits(&self, k: usize) -> Vec<Option<VertexFit>> {
        ring_fits(
            &self.positions,
            &self.adjacency,
            &self.mesh_normals,
            self.cell,
            k,
        )
    }

    /// Feature-distance bucket of a vertex: 0 = interior (>=2 cells),
    /// 1 = near (1-2 cells), 2 = at-feature (<1 cell).
    pub fn bucket(&self, v: usize) -> usize {
        let cells = self.truths[v].dist_to_sharp / self.cell;
        if cells >= 2.0 {
            0
        } else if cells >= 1.0 {
            1
        } else {
            2
        }
    }
}

pub const BUCKET_NAMES: [&str; 3] = ["interior (>=2c)", "near (1-2c)", "feature (<1c)"];

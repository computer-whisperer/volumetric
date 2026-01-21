//! Stage 4: STUBBED Vertex Refinement
//!
//! This module provides a passthrough implementation that preserves the
//! Stage 3 output without any refinement. The original Phase 4 implementation
//! (4a-4f) has been removed as part of the research branch cleanup.
//!
//! # What Was Removed
//!
//! The original Stage 4 implementation included:
//!
//! - **Phase 4a**: Binary search vertex refinement - moved vertices to actual
//!   surface crossings by searching along the accumulated normal direction
//!   with fallback to cardinal axes.
//!
//! - **Phase 4b**: Normal recomputation - recalculated accumulated face normals
//!   from the refined positions for better accuracy.
//!
//! - **Phase 4c**: Bayesian normal probing - refined normals by probing the
//!   surface in tangent directions, fitting planes to discovered points, and
//!   blending with the topology normal using inverse-variance weighting.
//!   Also detected "Case 1" sharp edges where a vertex straddles two faces.
//!
//! - **Phase 4d**: Case 2 edge crossing detection - identified mesh edges that
//!   cross geometric sharp edges by comparing endpoint normals.
//!
//! - **Phase 4e**: Re-triangulation with crossings - split triangles that
//!   contained Case 2 edge crossings.
//!
//! - **Phase 4f**: Sharp vertex duplication - duplicated vertices at sharp edges
//!   so each face could have its own normal direction.
//!
//! # Why It Was Removed
//!
//! The sharp edge detection algorithms (4c-4f) were experimental research code
//! that didn't achieve production-quality results. See PHASE4_ARCHIVE.md and
//! EDGE_REFINEMENT_RESEARCH.md for detailed analysis of why these approaches
//! were problematic and potential future directions.
//!
//! The basic refinement (4a-4b) could be re-enabled if needed, but for now
//! the Stage 3 output (edge midpoints with accumulated normals) is sufficient
//! for most use cases.
//!
//! # Current Behavior
//!
//! This stub simply:
//! 1. Converts f64 positions to f32
//! 2. Normalizes the accumulated normals
//! 3. Returns the mesh unchanged
//!
//! Sharp edge configuration (`AdaptiveMeshConfig2::sharp_edge_config`) is
//! preserved for API compatibility but has no effect.

use crate::adaptive_surface_nets_2::stage3::normalize_or_default;
use crate::adaptive_surface_nets_2::types::{IndexedMesh2, Stage3Result};

/// Result of Stage 4 processing.
///
/// In the stub implementation, this is just the Stage 3 data converted
/// to the appropriate types for final output.
pub struct Stage4Result {
    /// Vertex positions in f64 (same as Stage 3 output)
    pub positions_f64: Vec<(f64, f64, f64)>,
    /// Normals in f64 (same as Stage 3 accumulated normals)
    pub normals_f64: Vec<(f64, f64, f64)>,
    /// Triangle indices (unchanged from Stage 3)
    pub indices: Vec<u32>,
}

/// Stage 4: Vertex Refinement (STUBBED)
///
/// This is a passthrough implementation that does no actual refinement.
/// The Stage 3 positions (edge midpoints) and accumulated normals are
/// passed through unchanged.
///
/// # Arguments
/// * `stage3_result` - The result from Stage 3 topology finalization
/// * `_sampler` - Unused (would be used for surface sampling in full implementation)
/// * `_cell_size` - Unused (would determine search distances)
/// * `_config` - Unused (vertex_refinement_iterations and normal_sample_iterations ignored)
/// * `_stats` - Unused (no samples taken)
pub fn stage4_vertex_refinement_stub<F>(
    stage3_result: Stage3Result,
    _sampler: &F,
    _cell_size: (f64, f64, f64),
    _config: &crate::adaptive_surface_nets_2::types::AdaptiveMeshConfig2,
    _stats: &crate::adaptive_surface_nets_2::types::SamplingStats,
) -> Stage4Result
where
    F: crate::adaptive_surface_nets_2::types::SamplerFn,
{
    // Passthrough: no refinement, just return Stage 3 data
    Stage4Result {
        positions_f64: stage3_result.vertices,
        normals_f64: stage3_result.accumulated_normals,
        indices: stage3_result.indices,
    }
}

/// Convert Stage4Result to IndexedMesh2.
///
/// Converts f64 positions to f32 and normalizes the accumulated normals.
pub fn stage4_result_to_mesh(stage4: Stage4Result) -> IndexedMesh2 {
    let vertices: Vec<(f32, f32, f32)> = stage4
        .positions_f64
        .iter()
        .map(|p| (p.0 as f32, p.1 as f32, p.2 as f32))
        .collect();

    let normals: Vec<(f32, f32, f32)> = stage4
        .normals_f64
        .iter()
        .map(|n| normalize_or_default(*n))
        .collect();

    IndexedMesh2 {
        vertices,
        normals,
        indices: stage4.indices,
    }
}

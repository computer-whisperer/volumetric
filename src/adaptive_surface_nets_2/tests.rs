//! Unit tests for the adaptive surface nets algorithm.

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use crate::adaptive_surface_nets_2::{
        adaptive_surface_nets_2,
        stage1::stage1_coarse_discovery,
        stage2::{
            corner_world_position, shared_corners_with_neighbor, shared_face_is_mixed,
            stage2_subdivision_and_emission, subdivide_and_filter_mixed,
        },
        stage3::stage3_topology_finalization,
        types::{
            AdaptiveMeshConfig2, CornerMask, CuboidId, SamplingStats, WorkQueueEntry,
        },
    };

    #[test]
    fn test_corner_mask() {
        let mask = CornerMask::from_bools([true, false, true, false, false, false, false, false]);
        assert!(mask.is_inside(0));
        assert!(!mask.is_inside(1));
        assert!(mask.is_inside(2));
        assert!(mask.is_mixed());

        let all_inside = CornerMask(0xFF);
        assert!(!all_inside.is_mixed());

        let all_outside = CornerMask(0x00);
        assert!(!all_outside.is_mixed());
    }

    #[test]
    fn test_config_default() {
        let config = AdaptiveMeshConfig2::default();
        assert_eq!(config.base_resolution, 8);
        assert_eq!(config.max_depth, 4);
    }

    // =========================================================================
    // Stage 1 Tests
    // =========================================================================

    /// Helper: Unit sphere centered at origin (radius 1)
    fn sphere_sampler(x: f64, y: f64, z: f64) -> f32 {
        let r2 = x * x + y * y + z * z;
        if r2 < 1.0 { 1.0 } else { 0.0 }
    }

    #[test]
    fn test_stage1_sample_count() {
        // With base_resolution=4, we should sample 5³ = 125 corner points
        let config = AdaptiveMeshConfig2 {
            base_resolution: 4,
            ..Default::default()
        };
        let stats = SamplingStats::default();

        let _ = stage1_coarse_discovery(
            &sphere_sampler,
            (-2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0),
            &config,
            &stats,
        );

        // With expanded grid: (base_resolution + 3)³ corners
        // base_resolution=4 -> (4+3)³ = 7³ = 343 corners
        let expected_samples = 7 * 7 * 7;
        assert_eq!(
            stats.total_samples.load(Ordering::Relaxed),
            expected_samples,
            "Should sample exactly (base_resolution + 3)³ corner points (expanded grid)"
        );
    }

    #[test]
    fn test_stage1_finds_mixed_cells() {
        // Sphere of radius 1 in a [-2, 2]³ box with resolution 4
        // Cell size = 4/4 = 1, so cells at the surface should be detected
        let config = AdaptiveMeshConfig2 {
            base_resolution: 4,
            ..Default::default()
        };
        let stats = SamplingStats::default();

        let work_queue = stage1_coarse_discovery(
            &sphere_sampler,
            (-2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0),
            &config,
            &stats,
        );

        // Should find some mixed cells (sphere surface crosses the grid)
        assert!(
            !work_queue.is_empty(),
            "Should find mixed cells where sphere surface crosses grid"
        );

        // All returned entries should have all corners known
        for entry in &work_queue {
            assert!(
                entry.all_corners_known(),
                "All work queue entries should have all corners pre-sampled"
            );

            // The entry should be mixed
            let mask = entry.to_corner_mask();
            assert!(mask.is_mixed(), "All entries should be mixed cells");
        }
    }

    #[test]
    fn test_stage1_empty_for_fully_inside() {
        // Tiny sphere that fits entirely within a single cell
        // Box is [-1, 1]³, resolution 2, so cell size = 1
        // Sphere radius 0.1 centered at (0.5, 0.5, 0.5) fits in one cell
        let tiny_sphere = |x: f64, y: f64, z: f64| -> f32 {
            let dx = x - 0.5;
            let dy = y - 0.5;
            let dz = z - 0.5;
            if dx * dx + dy * dy + dz * dz < 0.01 { 1.0 } else { 0.0 }
        };

        let config = AdaptiveMeshConfig2 {
            base_resolution: 2,
            ..Default::default()
        };
        let stats = SamplingStats::default();

        let work_queue = stage1_coarse_discovery(
            &tiny_sphere,
            (-1.0, -1.0, -1.0),
            (1.0, 1.0, 1.0),
            &config,
            &stats,
        );

        // The sphere is so small that no coarse grid corner will hit it
        // All cells should be fully outside, hence no mixed cells
        assert!(
            work_queue.is_empty(),
            "Tiny sphere should not create mixed cells at coarse resolution"
        );
    }

    #[test]
    fn test_stage1_corner_ordering() {
        // Verify that corner ordering matches CORNER_OFFSETS
        // Create a sampler that returns different values based on position
        // to verify the corners are correctly indexed
        let config = AdaptiveMeshConfig2 {
            base_resolution: 1, // Single cell (with expanded grid: cells from -1 to 1)
            ..Default::default()
        };
        let stats = SamplingStats::default();

        // Sampler: inside only if x > 0.5 (so corners 1, 3, 5, 7 are inside for cell at origin)
        let half_space = |x: f64, _y: f64, _z: f64| -> f32 {
            if x > 0.5 { 1.0 } else { 0.0 }
        };

        let work_queue = stage1_coarse_discovery(
            &half_space,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            &config,
            &stats,
        );

        // With expanded grid, multiple cells will be mixed where x=0.5 plane crosses
        // Find the cell at position (0, 0, 0) which is the "original" cell
        let entry = work_queue
            .iter()
            .find(|e| e.cuboid.x == 0 && e.cuboid.y == 0 && e.cuboid.z == 0)
            .expect("Should find the cell at (0,0,0)");

        let corners = entry.known_corners;

        // Corners with x=1 should be inside (corners 1, 3, 5, 7)
        assert_eq!(corners[0], Some(false), "Corner 0 (0,0,0) should be outside");
        assert_eq!(corners[1], Some(true), "Corner 1 (1,0,0) should be inside");
        assert_eq!(corners[2], Some(false), "Corner 2 (0,1,0) should be outside");
        assert_eq!(corners[3], Some(true), "Corner 3 (1,1,0) should be inside");
        assert_eq!(corners[4], Some(false), "Corner 4 (0,0,1) should be outside");
        assert_eq!(corners[5], Some(true), "Corner 5 (1,0,1) should be inside");
        assert_eq!(corners[6], Some(false), "Corner 6 (0,1,1) should be outside");
        assert_eq!(corners[7], Some(true), "Corner 7 (1,1,1) should be inside");
    }

    #[test]
    fn test_cuboid_id_to_finest_level() {
        // At depth 0 with max_depth 3, scale = 2^3 = 8
        let cuboid = CuboidId::new(1, 2, 3, 0);
        let (fx, fy, fz) = cuboid.to_finest_level(3);
        assert_eq!((fx, fy, fz), (8, 16, 24));

        // At depth 2 with max_depth 3, scale = 2^1 = 2
        let cuboid = CuboidId::new(5, 6, 7, 2);
        let (fx, fy, fz) = cuboid.to_finest_level(3);
        assert_eq!((fx, fy, fz), (10, 12, 14));

        // At max depth, scale = 1
        let cuboid = CuboidId::new(5, 6, 7, 3);
        let (fx, fy, fz) = cuboid.to_finest_level(3);
        assert_eq!((fx, fy, fz), (5, 6, 7));
    }

    #[test]
    fn test_cuboid_subdivide() {
        let parent = CuboidId::new(1, 2, 3, 0);
        let children = parent.subdivide();

        // Check that children are at depth 1
        for child in &children {
            assert_eq!(child.depth, 1);
        }

        // Check child positions follow canonical corner ordering
        assert_eq!((children[0].x, children[0].y, children[0].z), (2, 4, 6)); // (0,0,0)
        assert_eq!((children[1].x, children[1].y, children[1].z), (3, 4, 6)); // (1,0,0)
        assert_eq!((children[2].x, children[2].y, children[2].z), (2, 5, 6)); // (0,1,0)
        assert_eq!((children[3].x, children[3].y, children[3].z), (3, 5, 6)); // (1,1,0)
        assert_eq!((children[4].x, children[4].y, children[4].z), (2, 4, 7)); // (0,0,1)
        assert_eq!((children[5].x, children[5].y, children[5].z), (3, 4, 7)); // (1,0,1)
        assert_eq!((children[6].x, children[6].y, children[6].z), (2, 5, 7)); // (0,1,1)
        assert_eq!((children[7].x, children[7].y, children[7].z), (3, 5, 7)); // (1,1,1)
    }

    #[test]
    fn test_edge_id_computation() {
        // Test that edge IDs are computed correctly from cell coordinates
        let cell = CuboidId::new(1, 2, 3, 2); // At depth 2
        let max_depth = 4;

        // At depth 2 with max_depth 4, scale = 2^2 = 4
        // So finest-level coords are (4, 8, 12)

        // Edge 0: X-axis at (0,0,0) offset -> (4, 8, 12, axis=0)
        let e0 = cell.edge_id(0, max_depth);
        assert_eq!((e0.x, e0.y, e0.z, e0.axis), (4, 8, 12, 0));

        // Edge 5: Y-axis at (1,0,0) offset -> (5, 8, 12, axis=1)
        let e5 = cell.edge_id(5, max_depth);
        assert_eq!((e5.x, e5.y, e5.z, e5.axis), (5, 8, 12, 1));

        // Edge 11: Z-axis at (1,1,0) offset -> (5, 9, 12, axis=2)
        let e11 = cell.edge_id(11, max_depth);
        assert_eq!((e11.x, e11.y, e11.z, e11.axis), (5, 9, 12, 2));
    }

    // =========================================================================
    // Stage 2 Tests
    // =========================================================================

    #[test]
    fn test_stage2_emits_triangles_for_sphere() {
        // Unit sphere in [-2, 2]³ box
        let config = AdaptiveMeshConfig2 {
            base_resolution: 4,
            max_depth: 2,
            ..Default::default()
        };
        let stats = SamplingStats::default();
        let bounds_min = (-2.0, -2.0, -2.0);
        let bounds_max = (2.0, 2.0, 2.0);

        // Calculate cell size at finest level (per-axis)
        let finest_cells = config.base_resolution * (1 << config.max_depth);
        let cell_size = (
            (bounds_max.0 - bounds_min.0) / finest_cells as f64,
            (bounds_max.1 - bounds_min.1) / finest_cells as f64,
            (bounds_max.2 - bounds_min.2) / finest_cells as f64,
        );

        let initial_queue = stage1_coarse_discovery(
            &sphere_sampler,
            bounds_min,
            bounds_max,
            &config,
            &stats,
        );

        let triangles = stage2_subdivision_and_emission(
            initial_queue,
            &sphere_sampler,
            bounds_min,
            cell_size,
            &config,
            &stats,
        );

        // Should produce triangles for a sphere
        assert!(
            !triangles.is_empty(),
            "Stage 2 should emit triangles for sphere surface"
        );

        // A sphere should produce a reasonable number of triangles
        // At resolution 4 * 2^2 = 16 cells per axis, we expect ~hundreds of triangles
        assert!(
            triangles.len() > 10,
            "Should have more than 10 triangles, got {}",
            triangles.len()
        );
    }

    #[test]
    fn test_stage2_shared_corners_neighbor_propagation() {
        // Test that shared_corners_with_neighbor returns correct mappings
        // +X neighbor: our corners 1,3,5,7 -> their corners 0,2,4,6
        let mapping = shared_corners_with_neighbor(1, 0, 0);
        assert_eq!(mapping, [(1, 0), (3, 2), (5, 4), (7, 6)]);

        // -X neighbor: our corners 0,2,4,6 -> their corners 1,3,5,7
        let mapping = shared_corners_with_neighbor(-1, 0, 0);
        assert_eq!(mapping, [(0, 1), (2, 3), (4, 5), (6, 7)]);

        // +Y neighbor: our corners 2,3,6,7 -> their corners 0,1,4,5
        let mapping = shared_corners_with_neighbor(0, 1, 0);
        assert_eq!(mapping, [(2, 0), (3, 1), (6, 4), (7, 5)]);

        // +Z neighbor: our corners 4,5,6,7 -> their corners 0,1,2,3
        let mapping = shared_corners_with_neighbor(0, 0, 1);
        assert_eq!(mapping, [(4, 0), (5, 1), (6, 2), (7, 3)]);
    }

    #[test]
    fn test_shared_face_is_mixed() {
        // All corners known, +X face is mixed (corners 1,3,5,7 have different states)
        let corners_mixed_x = [
            Some(true),  // 0
            Some(true),  // 1 - +X face
            Some(true),  // 2
            Some(false), // 3 - +X face (different!)
            Some(true),  // 4
            Some(true),  // 5 - +X face
            Some(true),  // 6
            Some(true),  // 7 - +X face
        ];
        assert!(
            shared_face_is_mixed(&corners_mixed_x, 1, 0, 0),
            "+X face should be mixed (corner 3 differs)"
        );

        // All corners same state on +X face
        let corners_uniform_x = [
            Some(false), // 0
            Some(true),  // 1 - +X face
            Some(false), // 2
            Some(true),  // 3 - +X face
            Some(false), // 4
            Some(true),  // 5 - +X face
            Some(false), // 6
            Some(true),  // 7 - +X face
        ];
        assert!(
            !shared_face_is_mixed(&corners_uniform_x, 1, 0, 0),
            "+X face should NOT be mixed (all +X corners are true)"
        );

        // -Z face: corners 0,1,2,3
        let corners_mixed_neg_z = [
            Some(true),  // 0 - -Z face
            Some(false), // 1 - -Z face (different!)
            Some(true),  // 2 - -Z face
            Some(true),  // 3 - -Z face
            Some(true),  // 4
            Some(true),  // 5
            Some(true),  // 6
            Some(true),  // 7
        ];
        assert!(
            shared_face_is_mixed(&corners_mixed_neg_z, 0, 0, -1),
            "-Z face should be mixed"
        );

        // Unknown corner should be conservative (return true)
        let corners_with_unknown = [
            Some(true),
            None, // unknown
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            Some(true),
            Some(true),
        ];
        assert!(
            shared_face_is_mixed(&corners_with_unknown, 1, 0, 0),
            "Unknown corner should trigger conservative exploration"
        );
    }

    #[test]
    fn test_subdivide_and_filter_mixed() {
        // Test that subdivide_and_filter_mixed correctly:
        // 1. Samples all 27 child corner positions
        // 2. Returns only mixed children
        // 3. All returned children have all corners filled in

        // Create a parent with mixed corners (half inside, half outside based on X)
        let parent_cuboid = CuboidId::new(0, 0, 0, 0);
        let parent_corners = [
            Some(false), // corner 0 (0,0,0) - x=0, outside
            Some(true),  // corner 1 (1,0,0) - x=1, inside
            Some(false), // corner 2 (0,1,0) - x=0, outside
            Some(true),  // corner 3 (1,1,0) - x=1, inside
            Some(false), // corner 4 (0,0,1) - x=0, outside
            Some(true),  // corner 5 (1,0,1) - x=1, inside
            Some(false), // corner 6 (0,1,1) - x=0, outside
            Some(true),  // corner 7 (1,1,1) - x=1, inside
        ];
        let parent = WorkQueueEntry::with_corners(parent_cuboid, parent_corners);

        // Sampler: inside if x > 0.25 (in [0,1] bounds)
        // This means the -X children (cx=0) will have surface crossing them
        // and the +X children (cx=1) will be fully inside
        let half_space = |x: f64, _y: f64, _z: f64| -> f32 {
            if x > 0.25 { 1.0 } else { 0.0 }
        };

        let stats = SamplingStats::default();
        let bounds_min = (0.0, 0.0, 0.0);
        let cell_size = (0.5, 0.5, 0.5); // Finest cell size (at max_depth=1, parent spans 2 finest cells)
        let max_depth = 1;

        let mixed_children = subdivide_and_filter_mixed(
            &parent,
            &half_space,
            bounds_min,
            cell_size,
            max_depth,
            &stats,
        );

        // All returned children should have all corners known
        for child in &mixed_children {
            assert!(
                child.all_corners_known(),
                "All returned children should have all corners filled in"
            );

            // And they should be mixed
            let mask = child.to_corner_mask();
            assert!(mask.is_mixed(), "All returned children should be mixed");
        }

        // We should have sampled exactly 19 new positions (27 - 8 parent corners)
        assert_eq!(
            stats.total_samples.load(Ordering::Relaxed),
            19,
            "Should sample exactly 19 new positions (27 total - 8 reused parent corners)"
        );
    }

    #[test]
    fn test_stage2_edge_welding() {
        // Two adjacent cells should produce the same EdgeId for their shared edge
        let cell_a = CuboidId::new(0, 0, 0, 2);
        let cell_b = CuboidId::new(1, 0, 0, 2); // +X neighbor
        let max_depth = 2;

        // Cell A's edge 5 (Y-axis at +X face) should equal cell B's edge 4 (Y-axis at origin)
        // Edge 5: corners 1-3, offset (1, 0, 0)
        // Edge 4: corners 0-2, offset (0, 0, 0)
        // For cell A at (0,0,0): edge 5 is at (0+1, 0+0, 0+0) = (1, 0, 0)
        // For cell B at (1,0,0): edge 4 is at (1+0, 0+0, 0+0) = (1, 0, 0)

        let edge_a = cell_a.edge_id(5, max_depth);
        let edge_b = cell_b.edge_id(4, max_depth);

        assert_eq!(
            edge_a, edge_b,
            "Adjacent cells should produce identical EdgeIds for shared edges"
        );
    }

    #[test]
    fn test_stage2_triangle_count_consistency() {
        // Run twice with same input, should get same triangle count
        let config = AdaptiveMeshConfig2 {
            base_resolution: 2,
            max_depth: 2,
            ..Default::default()
        };

        let bounds_min = (-1.5, -1.5, -1.5);
        let bounds_max = (1.5, 1.5, 1.5);
        let finest_cells = config.base_resolution * (1 << config.max_depth);
        let cell_size = (
            (bounds_max.0 - bounds_min.0) / finest_cells as f64,
            (bounds_max.1 - bounds_min.1) / finest_cells as f64,
            (bounds_max.2 - bounds_min.2) / finest_cells as f64,
        );

        let stats1 = SamplingStats::default();
        let initial_queue1 = stage1_coarse_discovery(
            &sphere_sampler,
            bounds_min,
            bounds_max,
            &config,
            &stats1,
        );
        let triangles1 = stage2_subdivision_and_emission(
            initial_queue1,
            &sphere_sampler,
            bounds_min,
            cell_size,
            &config,
            &stats1,
        );

        let stats2 = SamplingStats::default();
        let initial_queue2 = stage1_coarse_discovery(
            &sphere_sampler,
            bounds_min,
            bounds_max,
            &config,
            &stats2,
        );
        let triangles2 = stage2_subdivision_and_emission(
            initial_queue2,
            &sphere_sampler,
            bounds_min,
            cell_size,
            &config,
            &stats2,
        );

        assert_eq!(
            triangles1.len(),
            triangles2.len(),
            "Repeated runs should produce same triangle count"
        );
    }

    #[test]
    fn test_stage2_no_triangles_for_empty_space() {
        // Sampler that returns nothing inside
        let empty_sampler = |_x: f64, _y: f64, _z: f64| -> f32 { 0.0 };

        let config = AdaptiveMeshConfig2 {
            base_resolution: 4,
            max_depth: 2,
            ..Default::default()
        };
        let stats = SamplingStats::default();
        let bounds_min = (-1.0, -1.0, -1.0);
        let bounds_max = (1.0, 1.0, 1.0);
        let finest_cells = config.base_resolution * (1 << config.max_depth);
        let cell_size = (
            (bounds_max.0 - bounds_min.0) / finest_cells as f64,
            (bounds_max.1 - bounds_min.1) / finest_cells as f64,
            (bounds_max.2 - bounds_min.2) / finest_cells as f64,
        );

        let initial_queue = stage1_coarse_discovery(
            &empty_sampler,
            bounds_min,
            bounds_max,
            &config,
            &stats,
        );

        // Empty space should produce no mixed cells
        assert!(initial_queue.is_empty(), "Empty space should have no mixed cells");

        let triangles = stage2_subdivision_and_emission(
            initial_queue,
            &empty_sampler,
            bounds_min,
            cell_size,
            &config,
            &stats,
        );

        assert!(triangles.is_empty(), "Empty space should produce no triangles");
    }

    #[test]
    fn test_corner_world_position() {
        let cell = CuboidId::new(0, 0, 0, 0);
        let bounds_min = (0.0, 0.0, 0.0);
        let cell_size = (1.0, 1.0, 1.0);
        let max_depth = 2;

        // At depth 0, max_depth 2: cell spans 4 finest-level cells
        // Corner 0 (0,0,0) should be at (0, 0, 0)
        let c0 = corner_world_position(&cell, 0, bounds_min, cell_size, max_depth);
        assert_eq!(c0, (0.0, 0.0, 0.0));

        // Corner 7 (1,1,1) should be at (4, 4, 4) since scale = 2^2 = 4
        let c7 = corner_world_position(&cell, 7, bounds_min, cell_size, max_depth);
        assert_eq!(c7, (4.0, 4.0, 4.0));

        // A cell at depth 2 (finest level) should have corners 1 unit apart
        let fine_cell = CuboidId::new(2, 3, 4, 2);
        let c0_fine = corner_world_position(&fine_cell, 0, bounds_min, cell_size, max_depth);
        let c7_fine = corner_world_position(&fine_cell, 7, bounds_min, cell_size, max_depth);
        assert_eq!(c0_fine, (2.0, 3.0, 4.0));
        assert_eq!(c7_fine, (3.0, 4.0, 5.0));
    }

    // =========================================================================
    // Stage 3 Tests
    // =========================================================================

    #[test]
    fn test_stage3_produces_indexed_mesh() {
        // Run stages 1-3 and verify output structure
        let config = AdaptiveMeshConfig2 {
            base_resolution: 4,
            max_depth: 2,
            ..Default::default()
        };
        let stats = SamplingStats::default();
        let bounds_min = (-1.5, -1.5, -1.5);
        let bounds_max = (1.5, 1.5, 1.5);
        let finest_cells = config.base_resolution * (1 << config.max_depth);
        let cell_size = (
            (bounds_max.0 - bounds_min.0) / finest_cells as f64,
            (bounds_max.1 - bounds_min.1) / finest_cells as f64,
            (bounds_max.2 - bounds_min.2) / finest_cells as f64,
        );

        let initial_queue = stage1_coarse_discovery(
            &sphere_sampler,
            bounds_min,
            bounds_max,
            &config,
            &stats,
        );

        let triangles = stage2_subdivision_and_emission(
            initial_queue,
            &sphere_sampler,
            bounds_min,
            cell_size,
            &config,
            &stats,
        );

        let stage3_result = stage3_topology_finalization(triangles, bounds_min, cell_size);

        // Verify structure
        assert!(!stage3_result.vertices.is_empty(), "Should have vertices");
        assert_eq!(
            stage3_result.vertices.len(),
            stage3_result.accumulated_normals.len(),
            "Vertices and normals should have same length"
        );
        assert!(
            stage3_result.indices.len() % 3 == 0,
            "Indices should be multiple of 3 (triangles)"
        );
        assert!(
            !stage3_result.edge_to_vertex.is_empty(),
            "Should have edge-to-vertex mapping"
        );
    }

    // =========================================================================
    // Full Pipeline Tests
    // =========================================================================

    #[test]
    fn test_full_pipeline() {
        // Run the full pipeline (with stubbed Stage 4)
        let config = AdaptiveMeshConfig2 {
            base_resolution: 4,
            max_depth: 2,
            vertex_refinement_iterations: 4,
            normal_sample_iterations: 1,
            normal_epsilon_frac: 0.1,
            ..Default::default()
        };

        let result = adaptive_surface_nets_2(
            sphere_sampler,
            (-1.5, -1.5, -1.5),
            (1.5, 1.5, 1.5),
            &config,
        );
        let mesh = &result.mesh;

        // Should produce a mesh
        assert!(!mesh.vertices.is_empty(), "Should have vertices");
        assert!(!mesh.normals.is_empty(), "Should have normals");
        assert!(!mesh.indices.is_empty(), "Should have indices");

        // Vertices and normals should have same count
        assert_eq!(
            mesh.vertices.len(),
            mesh.normals.len(),
            "Vertex and normal counts should match"
        );

        // All normals should be unit length (approximately)
        for (i, n) in mesh.normals.iter().enumerate() {
            let len = (n.0 * n.0 + n.1 * n.1 + n.2 * n.2).sqrt();
            assert!(
                (len - 1.0).abs() < 0.01,
                "Normal {} should be unit length, got {}",
                i,
                len
            );
        }
    }

    #[test]
    fn test_full_pipeline_without_refinement() {
        // Run with refinement disabled
        let config = AdaptiveMeshConfig2 {
            base_resolution: 4,
            max_depth: 2,
            vertex_refinement_iterations: 0, // Disabled
            normal_sample_iterations: 0,     // Disabled
            ..Default::default()
        };

        let result = adaptive_surface_nets_2(
            sphere_sampler,
            (-1.5, -1.5, -1.5),
            (1.5, 1.5, 1.5),
            &config,
        );
        let mesh = &result.mesh;

        // Should still produce a valid mesh
        assert!(!mesh.vertices.is_empty(), "Should have vertices");
        assert_eq!(
            mesh.vertices.len(),
            mesh.normals.len(),
            "Vertex and normal counts should match"
        );
    }
}

//! Research Infrastructure for Geometry Probing
//!
//! This module provides reference tooling for geometry probing that can be validated
//! against analytical ground truth. The infrastructure enables experimentation with
//! sample-efficient algorithms by providing "correct answers" to compare against.
//!
//! # Philosophy
//!
//! The reference routines are designed to be **obviously correct** even if slow.
//! They can use hundreds of samples - accuracy is the only goal. Once we have
//! provably correct reference implementations, we can develop and validate cheaper
//! methods.
//!
//! # Components
//!
//! - [`analytical_cube`]: Extended rotated cube with analytical geometry queries
//! - [`sample_cache`]: Shared sample cache for comparing algorithm efficiency
//! - [`reference_surface`]: Dense probing to find nearest flat surface
//! - [`reference_edge`]: Dense probing to find nearest sharp edge
//! - [`reference_corner`]: Dense probing to find nearest corner (3+ faces)
//! - [`validation`]: Framework for comparing results against ground truth
//!
//! # Usage
//!
//! ```ignore
//! use crate::adaptive_surface_nets_2::stage4::research::*;
//!
//! // Create analytical ground truth
//! let cube = AnalyticalRotatedCube::standard_test_cube();
//!
//! // Generate validation points
//! let points = validation::generate_validation_points(&cube);
//!
//! // Test reference routines against ground truth
//! for point in &points {
//!     let result = validation::validate_point(point, &cube, &sampler);
//!     assert!(result.meets_criteria());
//! }
//! ```

pub mod analytical_cube;
pub mod attempt_0;
pub mod attempt_runner;
pub mod reference_corner;
pub mod reference_edge;
pub mod reference_surface;
pub mod robust_corner;
pub mod robust_edge;
pub mod robust_surface;
pub mod sample_cache;
pub mod validation;

// Re-exports for convenience
pub use analytical_cube::{
    AnalyticalCorner, AnalyticalEdge, AnalyticalRotatedCube, ClosestPointResult,
    SurfaceClassification,
};
pub use reference_corner::{reference_find_nearest_corner, CornerFindingResult};
pub use reference_edge::{reference_find_nearest_edge, EdgeFindingResult};
pub use reference_surface::{reference_find_nearest_surface, SurfaceFindingResult};
pub use sample_cache::{SampleCache, SampleCacheStats};
pub use validation::{
    generate_validation_points, validate_corner_detection, validate_edge_detection,
    validate_surface_detection, ValidationPoint, ValidationResult,
};

// Attempt 0: Crossing Count Algorithm
pub use attempt_0::{
    classify_geometry, locate_surface, measure_corner, measure_edge, measure_face, process_vertex,
    CrossingCountConfig, CornerMeasurement, EdgeMeasurement, FaceMeasurement, GeometryClassification,
    GeometryType, SurfaceLocation, VertexGeometry,
};

pub use attempt_runner::{run_attempt_0_benchmark, run_attempt_benchmark};

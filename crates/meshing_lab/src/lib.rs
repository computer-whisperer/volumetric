//! Research harness for feature-aware meshing (sharp edges and corners).
//!
//! This crate holds the *measurement* side of the analytical edge-refinement
//! effort: exact ground-truth oracles, geometric fitting primitives, and mesh
//! connectivity utilities. Algorithms under test live in the `volumetric`
//! library; everything here exists to evaluate them honestly.
//!
//! Methodology rules, learned the hard way on the earlier
//! `edge-probing-experiments` branch:
//!
//! - Benchmarks run against **real mesher output** (actual surface-nets
//!   vertices at realistic resolution), never against synthetic validation
//!   points with hand-picked offsets.
//! - Oracle truth is closed-form geometry and is never visible to the
//!   algorithm under test; the sampler is the only shared interface.
//! - All thresholds and error budgets are expressed relative to the finest
//!   cell size, never in absolute world units.
//! - Every claim gets re-measured on at least one curved shape (sphere,
//!   cylinder) before it is believed; planar-only results overfit.

pub mod adjacency;
pub mod fit;
pub mod oracle;

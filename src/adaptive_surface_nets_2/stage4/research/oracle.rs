//! Oracle interfaces for benchmark ground truth.
//!
//! The oracle must be independent from any probing algorithm. It should rely
//! on analytical geometry or exact CSG evaluation, not sampling.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OracleClassification {
    Face,
    Edge,
    Corner,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct OracleHit {
    pub classification: OracleClassification,
    pub surface_position: (f64, f64, f64),
    pub normals: Vec<(f64, f64, f64)>,
    pub edge_direction: Option<(f64, f64, f64)>,
    pub corner_position: Option<(f64, f64, f64)>,
}

pub trait OracleShape {
    fn name(&self) -> &str;

    /// Return exact classification and geometric data for a query point.
    fn classify(&self, point: (f64, f64, f64)) -> OracleHit;

    /// Return a deterministic set of validation points for this shape.
    fn validation_points(&self, seed: u64) -> Vec<(f64, f64, f64)>;
}

pub struct OracleBenchmarkCase<'a> {
    pub shape: &'a dyn OracleShape,
    pub seed: u64,
}

impl<'a> OracleBenchmarkCase<'a> {
    pub fn points(&self) -> Vec<(f64, f64, f64)> {
        self.shape.validation_points(self.seed)
    }

    pub fn expected(&self, point: (f64, f64, f64)) -> OracleHit {
        self.shape.classify(point)
    }
}


//! Per-point normal estimation: the plane through each point's nearest
//! neighbours, by principal component analysis, with a sign rule.

use crate::{Grid, Vec3, centroid, dot, eigen_symmetric, mul, scatter, sub};

/// Cell rings searched before a sparse point gives up on neighbours.
const MAX_RINGS: i64 = 8;

/// Which way an estimated normal points; principal components have no
/// sign of their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Orient {
    /// Away from the cloud's centroid.
    Outward,
    /// Towards +y.
    Up,
    /// As the analysis produced it.
    None,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct CloudNormalsConfig {
    /// Points used per normal, the point itself included.
    pub neighbours: u32,
    pub orient: Orient,
}

impl Default for CloudNormalsConfig {
    fn default() -> Self {
        Self {
            neighbours: 16,
            orient: Orient::Outward,
        }
    }
}

/// A unit normal per point, plus how many came out zero for want of a
/// well-defined neighbourhood plane (fewer than three neighbours, or
/// collinear ones).
pub fn estimate(
    points: &[Vec3],
    config: &CloudNormalsConfig,
) -> Result<(Vec<Vec3>, usize), String> {
    if config.neighbours < 3 {
        return Err(format!(
            "neighbours must be at least 3, got {}",
            config.neighbours
        ));
    }
    if points.is_empty() {
        return Err("the cloud has no points".to_string());
    }
    let k = config.neighbours as usize;
    let grid = Grid::new(points, k);
    let centre = centroid(points);
    let mut degenerate = 0usize;
    let mut neighbourhood = Vec::with_capacity(k);
    let normals = points
        .iter()
        .map(|&p| {
            neighbourhood.clear();
            neighbourhood.extend(
                grid.nearest(p, k, MAX_RINGS)
                    .into_iter()
                    .map(|(i, _)| points[i as usize]),
            );
            if neighbourhood.len() < 3 {
                degenerate += 1;
                return [0.0; 3];
            }
            let (_, m) = scatter(&neighbourhood);
            let (values, vectors) = eigen_symmetric(m);
            let threshold = 1e-12 * values[2].max(1e-300);
            if values[1].is_nan() || values[1] <= threshold {
                degenerate += 1;
                return [0.0; 3];
            }
            let n = vectors[0];
            let flip = match config.orient {
                Orient::Outward => dot(n, sub(p, centre)) < 0.0,
                Orient::Up => n[1] < 0.0,
                Orient::None => false,
            };
            if flip { mul(n, -1.0) } else { n }
        })
        .collect();
    Ok((normals, degenerate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{add, norm, normalized};

    /// A Fibonacci sphere of radius 0.5 about `centre`.
    pub fn sphere(centre: Vec3, count: usize) -> Vec<Vec3> {
        let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
        (0..count)
            .map(|i| {
                let y = 1.0 - 2.0 * (i as f64 + 0.5) / count as f64;
                let r = (1.0 - y * y).sqrt();
                let theta = golden * i as f64;
                add(centre, mul([r * theta.cos(), y, r * theta.sin()], 0.5))
            })
            .collect()
    }

    #[test]
    fn sphere_normals_are_radial_and_outward() {
        let centre = [0.3, -0.2, 0.1];
        let points = sphere(centre, 3000);
        let (normals, degenerate) = estimate(&points, &CloudNormalsConfig::default()).unwrap();
        assert_eq!(degenerate, 0);
        let mut worst = 0.0f64;
        for (p, n) in points.iter().zip(&normals) {
            assert!((norm(*n) - 1.0).abs() < 1e-9);
            let radial = normalized(sub(*p, centre)).unwrap();
            let cos = dot(*n, radial);
            assert!(cos > 0.0, "outward: {n:?} at {p:?}");
            worst = worst.max(cos.min(1.0).acos().to_degrees());
        }
        assert!(worst < 3.0, "worst angle {worst} deg");
    }

    #[test]
    fn up_orientation_and_degenerate_points() {
        // A flat grid in the x-z plane at y = 0.
        let mut points: Vec<Vec3> = Vec::new();
        for i in 0..30 {
            for j in 0..30 {
                points.push([i as f64 * 0.01, 0.0, j as f64 * 0.01]);
            }
        }
        let config = CloudNormalsConfig {
            neighbours: 8,
            orient: Orient::Up,
        };
        let (normals, degenerate) = estimate(&points, &config).unwrap();
        assert_eq!(degenerate, 0);
        assert!(normals.iter().all(|n| (n[1] - 1.0).abs() < 1e-9));

        let (normals, degenerate) =
            estimate(&[[0.0; 3], [1.0, 0.0, 0.0]], &CloudNormalsConfig::default()).unwrap();
        assert_eq!(degenerate, 2);
        assert_eq!(normals, vec![[0.0; 3]; 2]);
        let collinear: Vec<Vec3> = (0..10).map(|i| [i as f64, 0.0, 0.0]).collect();
        assert_eq!(estimate(&collinear, &config).unwrap().1, 10);
        assert!(
            estimate(
                &collinear,
                &CloudNormalsConfig {
                    neighbours: 2,
                    ..config
                }
            )
            .unwrap_err()
            .contains("neighbours")
        );
    }
}

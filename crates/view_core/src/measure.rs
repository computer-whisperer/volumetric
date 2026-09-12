//! Measuring in photographs: a pixel cast onto a plane, a world point
//! projected back, a feature triangulated from picks in several views,
//! and coordinates in a datum's chart. The CLI's `view-pick` and
//! `view-triangulate` and the Python bindings are the same calls.

use std::collections::BTreeMap;
use volumetric_abi::subspace::Subspace;

use volumetric_abi::viewset::{CameraModel, PickRole, View, ViewSet};

/// A plane by a point on it and its normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub point: [f64; 3],
    pub normal: [f64; 3],
}

impl Plane {
    /// The horizontal plane at height `z`.
    pub fn at_z(z: f64) -> Self {
        Self {
            point: [0.0, 0.0, z],
            normal: [0.0, 0.0, 1.0],
        }
    }
}

/// The plane a rank-2 subspace in 3-space is.
pub fn plane_of(subspace: &Subspace) -> Result<Plane, String> {
    if subspace.ambient() != 3 || subspace.rank() != 2 {
        return Err(format!(
            "a rank {} subspace in {}-space is not a plane",
            subspace.rank(),
            subspace.ambient()
        ));
    }
    let n = subspace.normal().ok_or("the plane has no normal")?;
    Ok(Plane {
        point: [subspace.origin[0], subspace.origin[1], subspace.origin[2]],
        normal: [n[0], n[1], n[2]],
    })
}

/// A world point's coordinates along a subspace's basis vectors, plus its
/// height above a plane (a rank-2 subspace gets a third coordinate).
pub fn chart(subspace: &Subspace, p: [f64; 3]) -> Vec<f64> {
    let d = [
        p[0] - subspace.origin[0],
        p[1] - subspace.origin[1],
        p[2] - subspace.origin[2],
    ];
    let mut out: Vec<f64> = (0..subspace.rank())
        .map(|i| {
            let b = subspace.basis_vector(i);
            b[0] * d[0] + b[1] * d[1] + b[2] * d[2]
        })
        .collect();
    if subspace.rank() == 2
        && let Some(n) = subspace.normal()
    {
        out.push(n[0] * d[0] + n[1] * d[1] + n[2] * d[2]);
    }
    out
}

/// A pixel cast onto a plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cast {
    pub world: [f64; 3],
    /// Distance along the camera's forward axis, metres.
    pub depth: f64,
}

/// Where the ray through `pixel` of a posed view meets `plane`.
pub fn cast(
    view: &View,
    camera: &CameraModel,
    pixel: [f64; 2],
    plane: &Plane,
) -> Result<Cast, String> {
    let eye = view
        .position()
        .ok_or_else(|| format!("view '{}' is not posed", view.id))?;
    let d = view
        .ray(camera, pixel)
        .ok_or_else(|| format!("view '{}' is not posed", view.id))?;
    let n = plane.normal;
    let p0 = plane.point;
    let denom = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
    if denom.abs() < 1e-9 {
        return Err(format!(
            "pixel ({}, {}) looks along the plane",
            pixel[0], pixel[1]
        ));
    }
    let t = (n[0] * (p0[0] - eye[0]) + n[1] * (p0[1] - eye[1]) + n[2] * (p0[2] - eye[2])) / denom;
    if t <= 0.0 {
        return Err(format!(
            "pixel ({}, {}) meets the plane behind the camera",
            pixel[0], pixel[1]
        ));
    }
    let world = [eye[0] + t * d[0], eye[1] + t * d[1], eye[2] + t * d[2]];
    let depth = view.to_camera(world).map(|c| c[2]).unwrap_or(t);
    Ok(Cast { world, depth })
}

/// The least-squares point nearest a set of rays (origin, unit direction),
/// with each ray's miss distance. None when the rays are parallel.
pub fn triangulate(rays: &[([f64; 3], [f64; 3])]) -> Option<([f64; 3], Vec<f64>)> {
    // Sum over rays of (I - d dᵀ)(x - o) = 0.
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    for (o, d) in rays {
        for i in 0..3 {
            for j in 0..3 {
                let p = if i == j { 1.0 } else { 0.0 } - d[i] * d[j];
                a[i][j] += p;
                b[i] += p * o[j];
            }
        }
    }
    let x = cloud_core::solve(a.iter().map(|r| r.to_vec()).collect(), b.to_vec())?;
    let x = [x[0], x[1], x[2]];
    let gaps = rays
        .iter()
        .map(|(o, d)| {
            let v = [x[0] - o[0], x[1] - o[1], x[2] - o[2]];
            let t = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
            let r = [v[0] - t * d[0], v[1] - t * d[1], v[2] - t * d[2]];
            (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt()
        })
        .collect();
    Some((x, gaps))
}

/// The ray (origin, unit direction) through a pixel of a named view.
pub fn ray_of(
    set: &ViewSet,
    view_id: &str,
    pixel: [f64; 2],
) -> Result<([f64; 3], [f64; 3]), String> {
    let (view, camera) = set
        .view(view_id)
        .ok_or_else(|| format!("no view '{view_id}' in the set"))?;
    let origin = view
        .position()
        .ok_or_else(|| format!("view '{view_id}' is not posed"))?;
    let direction = view
        .ray(camera, pixel)
        .ok_or_else(|| format!("view '{view_id}' is not posed"))?;
    Ok((origin, direction))
}

/// A feature picked in two or more views, triangulated: the point and
/// each pick's ray miss, in the picks' order.
pub fn triangulate_picks(
    set: &ViewSet,
    picks: &[(String, [f64; 2])],
) -> Result<([f64; 3], Vec<f64>), String> {
    if picks.len() < 2 {
        return Err("triangulation needs picks in at least two views".to_string());
    }
    let rays = picks
        .iter()
        .map(|(id, pixel)| ray_of(set, id, *pixel))
        .collect::<Result<Vec<_>, _>>()?;
    triangulate(&rays).ok_or_else(|| "the rays are parallel".to_string())
}

/// One pick's part in a feature fit.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct PickReport {
    pub view: String,
    pub pixel: [f64; 2],
    pub role: PickRole,
    /// For a fit pick: how far its ray passes from the point, metres.
    pub gap: Option<f64>,
    /// Where the fitted point projects in this view, if in front of it.
    pub projected: Option<[f64; 2]>,
    /// Distance from the pick to the projection, pixels.
    pub error_px: Option<f64>,
}

/// A named feature triangulated from its recorded picks.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct FeatureFit {
    pub name: String,
    pub world: [f64; 3],
    pub picks: Vec<PickReport>,
}

impl FeatureFit {
    /// The largest ray miss among the fit picks, metres.
    pub fn max_gap(&self) -> f64 {
        self.picks.iter().filter_map(|p| p.gap).fold(0.0, f64::max)
    }

    /// The largest reprojection error among the check picks, pixels.
    pub fn max_check_px(&self) -> Option<f64> {
        self.picks
            .iter()
            .filter(|p| p.role == PickRole::Check)
            .filter_map(|p| p.error_px)
            .reduce(f64::max)
    }
}

/// Triangulate a recorded feature from its fit picks and report every
/// pick: the fit picks' ray misses and every pick's reprojection error.
pub fn fit_feature(set: &ViewSet, name: &str) -> Result<FeatureFit, String> {
    let picks = set.picks();
    let picks = picks
        .get(name)
        .ok_or_else(|| format!("no picks recorded for feature {name:?}"))?;
    let fit: Vec<(String, [f64; 2])> = picks
        .iter()
        .filter(|(_, _, role)| *role == PickRole::Fit)
        .map(|(view, pixel, _)| (view.clone(), *pixel))
        .collect();
    if fit.len() < 2 {
        return Err(format!(
            "feature {name:?} has {} fit pick(s); triangulation needs two views",
            fit.len()
        ));
    }
    let (world, gaps) = triangulate_picks(set, &fit)?;
    let mut gap_of = fit
        .iter()
        .zip(gaps)
        .map(|((view, _), gap)| (view.clone(), gap));
    let reports = picks
        .iter()
        .map(|(view_id, pixel, role)| {
            let gap = if *role == PickRole::Fit {
                gap_of.next().map(|(_, g)| g)
            } else {
                None
            };
            let projected = set
                .view(view_id)
                .and_then(|(view, camera)| view.project(camera, world));
            let error_px =
                projected.map(|p| ((p[0] - pixel[0]).powi(2) + (p[1] - pixel[1]).powi(2)).sqrt());
            PickReport {
                view: view_id.clone(),
                pixel: *pixel,
                role: *role,
                gap,
                projected,
                error_px,
            }
        })
        .collect();
    Ok(FeatureFit {
        name: name.to_string(),
        world,
        picks: reports,
    })
}

/// Every recorded feature fitted, by name; a feature with fewer than two
/// fit picks is reported as an error under its name.
pub fn fit_features(set: &ViewSet) -> BTreeMap<String, Result<FeatureFit, String>> {
    set.picks()
        .keys()
        .map(|name| (name.clone(), fit_feature(set, name)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set() -> ViewSet {
        let mut set = ViewSet::default();
        set.cameras
            .push(CameraModel::pinhole(1000, 800, 900.0, 900.0, 500.0, 400.0));
        // Two cameras a metre up, looking straight down, half a metre apart.
        let down = |x: f64| [1.0, 0.0, 0.0, x, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0];
        set.views.push(View::posed("a", 0, down(-0.25)));
        set.views.push(View::posed("b", 0, down(0.25)));
        set
    }

    #[test]
    fn a_projected_point_triangulates_back() {
        let set = set();
        let target = [0.1, -0.05, 0.3];
        let picks: Vec<(String, [f64; 2])> = set
            .views
            .iter()
            .map(|v| (v.id.clone(), v.project(&set.cameras[0], target).unwrap()))
            .collect();
        let (point, gaps) = triangulate_picks(&set, &picks).unwrap();
        for i in 0..3 {
            assert!((point[i] - target[i]).abs() < 1e-9, "{point:?}");
        }
        assert!(gaps.iter().all(|g| *g < 1e-9));
        assert!(triangulate_picks(&set, &picks[..1]).is_err());
    }

    #[test]
    fn recorded_picks_fit_and_check() {
        let mut set = set();
        set.views.push(View::posed(
            "c",
            0,
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.3, 0.0, 0.0, -1.0, 1.0],
        ));
        let target = [0.1, -0.05, 0.3];
        for id in ["a", "b", "c"] {
            let (view, camera) = set.view(id).unwrap();
            let mut pixel = view.project(camera, target).unwrap();
            let role = if id == "c" {
                pixel[0] += 3.0; // a deliberately off check pick
                PickRole::Check
            } else {
                PickRole::Fit
            };
            set.record_pick(id, "hole", pixel, role).unwrap();
        }
        let fit = fit_feature(&set, "hole").unwrap();
        assert!((fit.world[0] - target[0]).abs() < 1e-9);
        assert!(fit.max_gap() < 1e-9);
        assert!((fit.max_check_px().unwrap() - 3.0).abs() < 1e-6);
        assert_eq!(fit.picks.len(), 3);
        assert_eq!(fit.picks[2].role, PickRole::Check);
        assert!(fit.picks[2].gap.is_none() && fit.picks[0].gap.is_some());
        assert!(fit_feature(&set, "missing").is_err());
        set.record_pick("a", "lone", [1.0, 1.0], PickRole::Fit)
            .unwrap();
        let all = fit_features(&set);
        assert!(all["hole"].is_ok() && all["lone"].as_ref().unwrap_err().contains("needs two"));
    }

    #[test]
    fn a_pixel_casts_onto_the_plane_it_came_from() {
        let set = set();
        let (view, camera) = set.view("a").unwrap();
        let target = [0.2, 0.1, 0.4];
        let pixel = view.project(camera, target).unwrap();
        let hit = cast(view, camera, pixel, &Plane::at_z(0.4)).unwrap();
        for i in 0..3 {
            assert!((hit.world[i] - target[i]).abs() < 1e-9, "{hit:?}");
        }
        assert!((hit.depth - 0.6).abs() < 1e-9);
        // The principal ray looks straight down: along a vertical plane it
        // never lands, and a plane above the camera is behind it.
        let principal = [camera.cx, camera.cy];
        assert!(
            cast(
                view,
                camera,
                principal,
                &Plane {
                    point: [0.0; 3],
                    normal: [1.0, 0.0, 0.0]
                }
            )
            .is_err()
        );
        assert!(cast(view, camera, principal, &Plane::at_z(2.0)).is_err());
    }

    #[test]
    fn chart_reads_plane_coordinates_and_height() {
        let plane = Subspace {
            dimensions: 3,
            origin: vec![1.0, 2.0, 3.0],
            basis: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        };
        let c = chart(&plane, [1.5, 2.0, 3.25]);
        assert_eq!(c.len(), 3);
        assert!((c[0] - 0.5).abs() < 1e-12 && c[1].abs() < 1e-12 && (c[2] - 0.25).abs() < 1e-12);
        assert_eq!(plane_of(&plane).unwrap().normal, [0.0, 0.0, 1.0]);
    }
}

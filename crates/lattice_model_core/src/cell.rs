//! A convex polytope under half-space clipping — the cell engine behind
//! both the foam skeleton (`skeleton.rs`) and the general Voronoi
//! skeleton (`voronoi.rs`).
//!
//! Plane ids 0..6 are the initial box; callers assign 6+ to their cut
//! planes (site bisectors). Every vertex keeps the set of planes it lies
//! on — that's what turns clipped geometry back into site
//! triples/quadruples. Generically that set has exactly 3 planes;
//! near-degenerate site configurations (several sites almost equidistant)
//! can grow it, and consumers choose their own policy for the resulting
//! edges (the foam skeleton keeps only cleanly-2-plane edges; the general
//! Voronoi accepts any all-bisector support, which is what makes
//! degenerate lattices like a perfect cubic grid come out clean).
//!
//! Callers are expected to work in coordinates where site spacing is
//! O(1) — cell units for the foam, spacing-normalized coordinates for
//! the general Voronoi — so the absolute [`CLIP_EPS`] tolerance means
//! the same thing everywhere.

/// On-plane tolerance for the cell clipping, in the caller's (unit
/// site-spacing) coordinates.
pub(crate) const CLIP_EPS: f64 = 1e-9;

/// A convex polytope under half-space clipping (see module docs).
pub(crate) struct ConvexCell {
    pub(crate) vertices: Vec<[f64; 3]>,
    pub(crate) generators: Vec<Vec<usize>>,
    /// (plane id, vertex loop). Loops stay consistently wound; edge
    /// extraction only needs incidence, not orientation.
    faces: Vec<(usize, Vec<u32>)>,
}

impl ConvexCell {
    /// The axis-aligned box `[lo, hi]`. Plane ids: 0/1 = -x/+x, 2/3 =
    /// -y/+y, 4/5 = -z/+z.
    pub(crate) fn from_aabb(lo: [f64; 3], hi: [f64; 3]) -> Self {
        let corner = |dx: usize, dy: usize, dz: usize| {
            [
                if dx == 0 { lo[0] } else { hi[0] },
                if dy == 0 { lo[1] } else { hi[1] },
                if dz == 0 { lo[2] } else { hi[2] },
            ]
        };
        // Vertex index = dx + 2 dy + 4 dz.
        let mut vertices = Vec::with_capacity(8);
        let mut generators = Vec::with_capacity(8);
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    vertices.push(corner(dx, dy, dz));
                    generators.push(vec![dx, 2 + dy, 4 + dz]);
                }
            }
        }
        let faces = vec![
            (0, vec![0u32, 4, 6, 2]),
            (1, vec![1, 3, 7, 5]),
            (2, vec![0, 1, 5, 4]),
            (3, vec![2, 6, 7, 3]),
            (4, vec![0, 2, 3, 1]),
            (5, vec![4, 5, 7, 6]),
        ];
        Self {
            vertices,
            generators,
            faces,
        }
    }

    /// Squared distance of the farthest vertex from `p` (the clip loop's
    /// early-out radius).
    pub(crate) fn max_dist2(&self, p: [f64; 3]) -> f64 {
        self.faces
            .iter()
            .flat_map(|(_, looped)| looped.iter())
            .map(|&v| {
                let q = self.vertices[v as usize];
                (0..3).map(|a| (q[a] - p[a]).powi(2)).sum::<f64>()
            })
            .fold(0.0, f64::max)
    }

    /// Clip to the half-space `n . x <= d` (plane `plane_id`). On-plane
    /// vertices (within CLIP_EPS) count as inside — a redundant plane
    /// doesn't cut — but are recorded as lying on the plane so later
    /// edge/node identification stays consistent.
    pub(crate) fn clip(&mut self, plane_id: usize, n: [f64; 3], d: f64) {
        let sd: Vec<f64> = self
            .vertices
            .iter()
            .map(|v| n[0] * v[0] + n[1] * v[1] + n[2] * v[2] - d)
            .collect();
        let inside = |v: u32| sd[v as usize] <= CLIP_EPS;
        let any_outside = self
            .faces
            .iter()
            .flat_map(|(_, looped)| looped.iter())
            .any(|&v| !inside(v));
        if !any_outside {
            // Still membership-tag on-plane vertices: the plane supports
            // them even though nothing was cut.
            for (_, looped) in &self.faces {
                for &v in looped {
                    if sd[v as usize].abs() <= CLIP_EPS
                        && !self.generators[v as usize].contains(&plane_id)
                    {
                        self.generators[v as usize].push(plane_id);
                        self.generators[v as usize].sort_unstable();
                    }
                }
            }
            return;
        }

        // One crossing vertex per cut edge, shared between the two faces
        // that walk it (watertight by construction).
        let mut crossings: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::new();
        let mut cross = |a: u32,
                         b: u32,
                         vertices: &mut Vec<[f64; 3]>,
                         generators: &mut Vec<Vec<usize>>|
         -> u32 {
            let key = (a.min(b), a.max(b));
            *crossings.entry(key).or_insert_with(|| {
                let (pa, pb) = (vertices[a as usize], vertices[b as usize]);
                let (da, db) = (sd[a as usize], sd[b as usize]);
                let t = da / (da - db);
                vertices.push([
                    pa[0] + t * (pb[0] - pa[0]),
                    pa[1] + t * (pb[1] - pa[1]),
                    pa[2] + t * (pb[2] - pa[2]),
                ]);
                // The cut edge's supporting planes are the ones both
                // endpoints lie on; the crossing adds the cutting plane.
                let (ga, gb) = (&generators[a as usize], &generators[b as usize]);
                let mut gens: Vec<usize> = ga.iter().filter(|g| gb.contains(g)).copied().collect();
                gens.push(plane_id);
                gens.sort_unstable();
                gens.dedup();
                generators.push(gens);
                (vertices.len() - 1) as u32
            })
        };

        let mut new_faces: Vec<(usize, Vec<u32>)> = Vec::with_capacity(self.faces.len() + 1);
        let mut cap: Vec<u32> = Vec::new();
        let mut on_plane: Vec<u32> = Vec::new();
        for (pid, looped) in &self.faces {
            let mut new_loop: Vec<u32> = Vec::with_capacity(looped.len() + 2);
            for i in 0..looped.len() {
                let (a, b) = (looped[i], looped[(i + 1) % looped.len()]);
                if inside(a) {
                    new_loop.push(a);
                    if sd[a as usize].abs() <= CLIP_EPS {
                        if !cap.contains(&a) {
                            cap.push(a);
                        }
                        if !on_plane.contains(&a) {
                            on_plane.push(a);
                        }
                    }
                }
                let strictly_in_a = sd[a as usize] < -CLIP_EPS;
                let strictly_out_a = sd[a as usize] > CLIP_EPS;
                let strictly_in_b = sd[b as usize] < -CLIP_EPS;
                let strictly_out_b = sd[b as usize] > CLIP_EPS;
                if (strictly_in_a && strictly_out_b) || (strictly_out_a && strictly_in_b) {
                    let v = cross(a, b, &mut self.vertices, &mut self.generators);
                    new_loop.push(v);
                    if !cap.contains(&v) {
                        cap.push(v);
                    }
                }
            }
            if new_loop.len() >= 3 {
                new_faces.push((*pid, new_loop));
            }
        }
        for &v in &on_plane {
            if !self.generators[v as usize].contains(&plane_id) {
                self.generators[v as usize].push(plane_id);
                self.generators[v as usize].sort_unstable();
            }
        }

        // The cap face: the cut cross-section, ordered by angle around its
        // centroid in the plane (convex, so this is a valid loop).
        if cap.len() >= 3 {
            let centroid = cap.iter().fold([0.0; 3], |acc, &v| {
                let p = self.vertices[v as usize];
                [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]]
            });
            let centroid = centroid.map(|c| c / cap.len() as f64);
            // An in-plane orthonormal basis.
            let axis = if n[0].abs() < n[1].abs() && n[0].abs() < n[2].abs() {
                [1.0, 0.0, 0.0]
            } else if n[1].abs() < n[2].abs() {
                [0.0, 1.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            let u = [
                n[1] * axis[2] - n[2] * axis[1],
                n[2] * axis[0] - n[0] * axis[2],
                n[0] * axis[1] - n[1] * axis[0],
            ];
            let w = [
                n[1] * u[2] - n[2] * u[1],
                n[2] * u[0] - n[0] * u[2],
                n[0] * u[1] - n[1] * u[0],
            ];
            let mut angles: Vec<(f64, u32)> = cap
                .iter()
                .map(|&v| {
                    let p = self.vertices[v as usize];
                    let r = [p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]];
                    let x = r[0] * u[0] + r[1] * u[1] + r[2] * u[2];
                    let y = r[0] * w[0] + r[1] * w[1] + r[2] * w[2];
                    (y.atan2(x), v)
                })
                .collect();
            angles.sort_by(|a, b| a.0.total_cmp(&b.0));
            new_faces.push((plane_id, angles.into_iter().map(|(_, v)| v).collect()));
        }

        self.faces = new_faces;
    }

    /// The polytope's edges as (vertex, vertex, shared plane ids) — the
    /// planes both endpoints lie on. Each edge is walked by exactly two
    /// faces; emitted once. Generic edges share exactly 2 planes;
    /// degenerate configurations share more, near-coincident vertices
    /// fewer — callers pick their policy.
    pub(crate) fn edges(&self) -> Vec<(u32, u32, Vec<usize>)> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for (_, looped) in &self.faces {
            for i in 0..looped.len() {
                let (a, b) = (looped[i], looped[(i + 1) % looped.len()]);
                let key = (a.min(b), a.max(b));
                if !seen.insert(key) {
                    continue;
                }
                let (ga, gb) = (&self.generators[a as usize], &self.generators[b as usize]);
                let shared: Vec<usize> = ga.iter().filter(|g| gb.contains(g)).copied().collect();
                out.push((a, b, shared));
            }
        }
        out
    }
}

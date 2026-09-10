//! A synthetic picture of markers with exact ground truth: cards placed in
//! the world, rendered through a view's camera by ray casting, so every
//! stage of detection and solving can be tested against known corners
//! and poses.

use volumetric_abi::viewset::{CameraModel, Marker, View};

use crate::dict::Dictionary;
use crate::gray::Gray;
use crate::linalg::{add, cross, dot, normalized, scale, sub};

/// A square marker with its canonical corners: top-left first, then
/// clockwise as printed, `right` along the top edge and `down` along the
/// left edge.
pub fn square_marker(
    id: u32,
    top_left: [f64; 3],
    size_m: f64,
    right: [f64; 3],
    down: [f64; 3],
) -> Marker {
    let right = scale(normalized(right), size_m);
    let down = scale(normalized(down), size_m);
    Marker {
        id,
        size_m,
        corners: [
            top_left,
            add(top_left, right),
            add(add(top_left, right), down),
            add(top_left, down),
        ],
    }
}

/// Rendering choices.
#[derive(Clone, Debug)]
pub struct Render {
    /// Background luma.
    pub background: u8,
    pub ink: u8,
    pub paper: u8,
    /// Samples per pixel side (anti-aliasing).
    pub supersample: u32,
    /// Gaussian blur after rendering, pixels (0 = none).
    pub blur_sigma: f64,
    /// White quiet zone around the marker's border, in cells.
    pub margin_cells: f64,
}

impl Default for Render {
    fn default() -> Self {
        Self {
            background: 128,
            ink: 20,
            paper: 235,
            supersample: 3,
            blur_sigma: 0.0,
            margin_cells: 1.0,
        }
    }
}

/// Renders `markers` (whose ids the dictionary must know) as seen by
/// `view` through `camera`.
pub fn render(
    camera: &CameraModel,
    view: &View,
    markers: &[Marker],
    dict: &Dictionary,
    options: &Render,
) -> Gray {
    let n = dict.size;
    let cells = f64::from(n + 2);
    let eye = view.position();
    let cards: Vec<Card> = markers
        .iter()
        .map(|m| Card::new(m, dict.code(m.id).expect("marker id in dictionary")))
        .collect();
    let mut out = Gray::new(camera.width, camera.height);
    let ss = options.supersample.max(1);
    let luma_at = |px: [f64; 2]| -> u8 {
        let dir = view.ray(camera, px);
        let mut nearest: Option<(f64, u8)> = None;
        for card in &cards {
            if let Some((t, luma)) = card.hit(eye, dir, n, cells, options)
                && nearest.is_none_or(|(best, _)| t < best)
            {
                nearest = Some((t, luma));
            }
        }
        nearest.map_or(options.background, |(_, l)| l)
    };
    // One ray per pixel first; only pixels whose neighbourhood is not
    // flat get the supersampled edge treatment.
    let (w, h) = (camera.width as usize, camera.height as usize);
    let mut coarse = vec![0u8; w * h];
    for y in 0..camera.height {
        for x in 0..camera.width {
            coarse[y as usize * w + x as usize] = luma_at([f64::from(x) + 0.5, f64::from(y) + 0.5]);
        }
    }
    for y in 0..camera.height {
        for x in 0..camera.width {
            let here = coarse[y as usize * w + x as usize];
            let mut flat = ss == 1;
            if !flat {
                flat = true;
                'scan: for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            continue;
                        }
                        if coarse[ny as usize * w + nx as usize] != here {
                            flat = false;
                            break 'scan;
                        }
                    }
                }
            }
            if flat {
                out.set(x, y, here);
                continue;
            }
            let mut total = 0.0;
            for sy in 0..ss {
                for sx in 0..ss {
                    total += f64::from(luma_at([
                        f64::from(x) + (f64::from(sx) + 0.5) / f64::from(ss),
                        f64::from(y) + (f64::from(sy) + 0.5) / f64::from(ss),
                    ]));
                }
            }
            out.set(x, y, (total / f64::from(ss * ss)).round() as u8);
        }
    }
    if options.blur_sigma > 0.0 {
        out.blurred(options.blur_sigma)
    } else {
        out
    }
}

/// A marker as a plane patch: origin at its top-left corner, `u` along
/// the top edge and `v` down the left edge, both the marker's side long.
struct Card {
    origin: [f64; 3],
    u: [f64; 3],
    v: [f64; 3],
    normal: [f64; 3],
    code: u32,
}

impl Card {
    fn new(marker: &Marker, code: u32) -> Self {
        let u = sub(marker.corners[1], marker.corners[0]);
        let v = sub(marker.corners[3], marker.corners[0]);
        Self {
            origin: marker.corners[0],
            u,
            v,
            normal: normalized(cross(u, v)),
            code,
        }
    }

    /// Where a ray meets the card (ray parameter and luma), if it does.
    fn hit(
        &self,
        eye: [f64; 3],
        dir: [f64; 3],
        n: u32,
        cells: f64,
        options: &Render,
    ) -> Option<(f64, u8)> {
        let denom = dot(dir, self.normal);
        if denom.abs() < 1e-9 {
            return None;
        }
        let t = dot(sub(self.origin, eye), self.normal) / denom;
        if t <= 0.0 {
            return None;
        }
        let p = sub(add(eye, scale(dir, t)), self.origin);
        // Plane coordinates in units of the marker side (u, v may be
        // slightly non-orthogonal for a triangulated map; solve the 2x2).
        let (uu, uv, vv) = (
            dot(self.u, self.u),
            dot(self.u, self.v),
            dot(self.v, self.v),
        );
        let (pu, pv) = (dot(p, self.u), dot(p, self.v));
        let det = uu * vv - uv * uv;
        let a = (pu * vv - pv * uv) / det;
        let b = (uu * pv - uv * pu) / det;
        let margin = options.margin_cells / cells;
        if a < -margin || a > 1.0 + margin || b < -margin || b > 1.0 + margin {
            return None;
        }
        if !(0.0..1.0).contains(&a) || !(0.0..1.0).contains(&b) {
            return Some((t, options.paper));
        }
        let col = (a * cells).floor() as u32;
        let row = (b * cells).floor() as u32;
        let dark = if col == 0 || row == 0 || col == n + 1 || row == n + 1 {
            true
        } else {
            let (r, c) = (row - 1, col - 1);
            (self.code >> (n * n - 1 - (r * n + c))) & 1 == 0
        };
        Some((t, if dark { options.ink } else { options.paper }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_card_renders_its_border_and_bits() {
        // Camera 1 m above a marker on the z = 0 plane looking straight
        // down, picture x along world x, picture y along world -y... the
        // marker's `down` is world -y so the picture reads upright.
        let camera = CameraModel::pinhole(200, 200, 400.0, 400.0, 100.0, 100.0);
        let view = View::posed(
            "top",
            0,
            [
                1.0, 0.0, 0.0, 0.0, //
                0.0, -1.0, 0.0, 0.0, //
                0.0, 0.0, -1.0, 1.0, //
            ],
        );
        let dict = Dictionary::aruco_5x5_100();
        let marker = square_marker(
            0,
            [-0.175, 0.175, 0.0],
            0.35,
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        );
        let picture = render(
            &camera,
            &view,
            std::slice::from_ref(&marker),
            &dict,
            &Render::default(),
        );
        // The marker spans 140 px centred; its border ring is ink, the
        // quiet zone paper, the corner of the picture background.
        assert_eq!(picture.get(2, 2), 128);
        assert_eq!(picture.get(100 - 70 - 5, 100), 235, "quiet zone");
        assert_eq!(picture.get(100 - 70 + 10, 100), 20, "border");
        // Inner cell (0, 0) of marker 0 is bit 1 (paper), cell (0, 1) is 0.
        let cell = 140.0 / 7.0;
        let inner = |r: f64, c: f64| {
            picture.get(
                (30.0 + cell * (c + 1.5)) as u32,
                (30.0 + cell * (r + 1.5)) as u32,
            )
        };
        assert_eq!(inner(0.0, 0.0), 235);
        assert_eq!(inner(0.0, 1.0), 20);
        // The projected corners match the picture's marker square.
        let tl = view.project(&camera, marker.corners[0]).unwrap();
        assert!(
            (tl[0] - 30.0).abs() < 1e-9 && (tl[1] - 30.0).abs() < 1e-9,
            "{tl:?}"
        );
        let br = view.project(&camera, marker.corners[2]).unwrap();
        assert!(
            (br[0] - 170.0).abs() < 1e-9 && (br[1] - 170.0).abs() < 1e-9,
            "{br:?}"
        );
    }
}

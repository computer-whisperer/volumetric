//! A synthetic picture of markers and boards with exact ground truth:
//! cards and a ChArUco board placed in the world, rendered through a
//! view's camera by ray casting, so every stage of detection and solving
//! can be tested against known corners and poses.

use volumetric_abi::viewset::{BoardSpec, CameraModel, Marker, View};

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

/// A ChArUco board placed in the world: its top-left outer corner at
/// `origin`, `right` along its columns and `down` along its rows (unit
/// vectors), the family's codes for its markers.
#[derive(Clone, Debug)]
pub struct PlacedBoard {
    pub spec: BoardSpec,
    pub origin: [f64; 3],
    pub right: [f64; 3],
    pub down: [f64; 3],
}

impl PlacedBoard {
    pub fn new(spec: BoardSpec, origin: [f64; 3], right: [f64; 3], down: [f64; 3]) -> Self {
        Self {
            spec,
            origin,
            right: normalized(right),
            down: normalized(down),
        }
    }

    /// A board point in the world.
    pub fn to_world(&self, board: [f64; 2]) -> [f64; 3] {
        add(
            add(self.origin, scale(self.right, board[0])),
            scale(self.down, board[1]),
        )
    }

    /// An interior corner in the world.
    pub fn corner_world(&self, id: u32) -> Option<[f64; 3]> {
        self.spec.corner(id).map(|c| self.to_world(c))
    }

    /// A marker as the map would carry it.
    pub fn marker(&self, id: u32) -> Option<Marker> {
        let corners = self.spec.marker_corners(id)?;
        Some(Marker {
            id,
            size_m: self.spec.marker_m,
            corners: corners.map(|c| self.to_world(c)),
        })
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
    render_scene(camera, view, markers, dict, &[], options)
}

/// Renders a board alone.
pub fn render_board(
    camera: &CameraModel,
    view: &View,
    board: &PlacedBoard,
    options: &Render,
) -> Gray {
    let dict = Dictionary::by_name(&board.spec.family).expect("known family");
    render_scene(
        camera,
        view,
        &[],
        &dict,
        std::slice::from_ref(board),
        options,
    )
}

/// Renders markers of one family and boards (each of its own family) as
/// seen by `view` through `camera`; the nearest surface wins.
pub fn render_scene(
    camera: &CameraModel,
    view: &View,
    markers: &[Marker],
    dict: &Dictionary,
    boards: &[PlacedBoard],
    options: &Render,
) -> Gray {
    let n = dict.size;
    let cells = f64::from(n + 2);
    let eye = view.position();
    let cards: Vec<Card> = markers
        .iter()
        .map(|m| Card::new(m, dict.code(m.id).expect("marker id in dictionary")))
        .collect();
    let patches: Vec<BoardPatch> = boards.iter().map(BoardPatch::new).collect();
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
        for patch in &patches {
            if let Some((t, luma)) = patch.hit(eye, dir, options)
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
    code: u64,
}

impl Card {
    fn new(marker: &Marker, code: u64) -> Self {
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

/// A board as a plane patch with its family's codes.
struct BoardPatch<'a> {
    board: &'a PlacedBoard,
    normal: [f64; 3],
    dict: Dictionary,
    size: [f64; 2],
}

impl<'a> BoardPatch<'a> {
    fn new(board: &'a PlacedBoard) -> Self {
        Self {
            board,
            normal: normalized(cross(board.right, board.down)),
            dict: Dictionary::by_name(&board.spec.family).expect("known family"),
            size: board.spec.size_m(),
        }
    }

    fn hit(&self, eye: [f64; 3], dir: [f64; 3], options: &Render) -> Option<(f64, u8)> {
        let denom = dot(dir, self.normal);
        if denom.abs() < 1e-9 {
            return None;
        }
        let t = dot(sub(self.board.origin, eye), self.normal) / denom;
        if t <= 0.0 {
            return None;
        }
        let p = sub(add(eye, scale(dir, t)), self.board.origin);
        let (u, v) = (dot(p, self.board.right), dot(p, self.board.down));
        let spec = &self.board.spec;
        // A paper margin of one square around the board.
        let (mx, my) = (
            options.margin_cells * spec.pitch_x_m,
            options.margin_cells * spec.pitch_y_m,
        );
        if u < -mx || u > self.size[0] + mx || v < -my || v > self.size[1] + my {
            return None;
        }
        if u < 0.0 || u >= self.size[0] || v < 0.0 || v >= self.size[1] {
            return Some((t, options.paper));
        }
        let col = (u / spec.pitch_x_m).floor() as u32;
        let row = (v / spec.pitch_y_m).floor() as u32;
        if (row + col) % 2 == 0 {
            return Some((t, options.ink));
        }
        let Some(id) = spec.marker_at(row, col) else {
            return Some((t, options.paper));
        };
        let corners = spec.marker_corners(id).expect("marker on the board");
        let (a, b) = (
            (u - corners[0][0]) / spec.marker_m,
            (v - corners[0][1]) / spec.marker_m,
        );
        if !(0.0..1.0).contains(&a) || !(0.0..1.0).contains(&b) {
            return Some((t, options.paper));
        }
        let n = self.dict.size;
        let cells = f64::from(n + 2);
        let (cc, cr) = ((a * cells).floor() as u32, (b * cells).floor() as u32);
        let dark = if cc == 0 || cr == 0 || cc == n + 1 || cr == n + 1 {
            true
        } else {
            let code = self.dict.code(id).expect("marker id in family");
            (code >> (n * n - 1 - ((cr - 1) * n + (cc - 1)))) & 1 == 0
        };
        Some((t, if dark { options.ink } else { options.paper }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_board_renders_squares_and_tags_where_the_spec_says() {
        // 12 px per nominal square, straight down: the card spans
        // 144 x 132 px from (20, 20).
        let camera = CameraModel::pinhole(200, 180, 1200.0, 1200.0, 100.0, 90.0);
        let view = View::posed(
            "top",
            0,
            [
                1.0, 0.0, 0.0, 0.0, //
                0.0, -1.0, 0.0, 0.0, //
                0.0, 0.0, -1.0, 1.0, //
            ],
        );
        let spec = BoardSpec {
            pitch_x_m: 0.01,
            pitch_y_m: 0.01,
            marker_m: 0.007,
            ..BoardSpec::survey_card()
        };
        let origin = view.unproject(&camera, [20.0, 20.0], 1.0);
        let board = PlacedBoard::new(spec.clone(), origin, [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]);
        let picture = render_board(&camera, &view, &board, &Render::default());
        assert_eq!(picture.get(2, 2), 128, "background");
        assert_eq!(picture.get(20 + 6, 20 + 6), 20, "square (0, 0) is ink");
        assert_eq!(picture.get(20 + 12, 20), 235, "square (0, 1) paper corner");
        assert_eq!(picture.get(20 + 12 + 6, 20 + 2), 20, "its tag's border");
        assert_eq!(picture.get(20 + 12 * 12 + 2, 20 + 2), 235, "paper margin");
        // The projected corner 0 is the square boundary at (32, 32).
        let c0 = view
            .project(&camera, board.corner_world(0).unwrap())
            .unwrap();
        assert!(
            (c0[0] - 32.0).abs() < 1e-9 && (c0[1] - 32.0).abs() < 1e-9,
            "{c0:?}"
        );
        let m = board.marker(100).unwrap();
        let tl = view.project(&camera, m.corners[0]).unwrap();
        assert!(
            (tl[0] - 33.8).abs() < 1e-9 && (tl[1] - 21.8).abs() < 1e-9,
            "{tl:?}"
        );
    }

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

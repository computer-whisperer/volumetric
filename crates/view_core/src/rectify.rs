//! A view's picture resampled onto a world plane: a metric image of the
//! plane, so a point that lies on it reads off in millimetres, and a plane
//! sweep across views finds a height (edges on the plane agree in every
//! view's rectification; edges off it slide apart with the view). The
//! CLI's `view-rectify` is this call.

use anyhow::{Result, bail};
use volumetric_abi::viewset::{CameraModel, View};

use crate::image::Rgb;

const GRID: [u8; 3] = [255, 170, 40];
const GRID_MAJOR: [u8; 3] = [255, 230, 120];
const WORLD_MARK: [u8; 3] = [255, 70, 200];

#[derive(Clone, Debug, PartialEq)]
pub struct RectifyOptions {
    /// The plane's chart: its origin and two orthonormal in-plane axes.
    pub origin: [f64; 3],
    pub u_axis: [f64; 3],
    pub v_axis: [f64; 3],
    /// The chart window, metres: u in `u_range`, v in `v_range`.
    pub u_range: [f64; 2],
    pub v_range: [f64; 2],
    /// Metres per output pixel.
    pub resolution: f64,
    /// Grid spacing in metres (0 = none); every fifth line brighter and
    /// labelled in millimetres.
    pub grid: f64,
    /// Crosses where these world points land (projected along the view's
    /// ray onto the plane, so an off-plane point shows its parallax).
    pub world_marks: Vec<[f64; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rectified {
    /// Columns run along +u, rows along -v (the chart reads like a map).
    pub image: Rgb,
    /// Chart coordinates (metres) of the grid lines drawn.
    pub u_lines: Vec<f64>,
    pub v_lines: Vec<f64>,
    /// Pixels of the output that the view does not see (behind the
    /// camera, off the picture).
    pub unseen: usize,
}

fn add(a: [f64; 3], b: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Bilinear sample of `picture` at `p`, in the OpenCV convention the
/// cameras use (pixel centres at integers).
fn sample(picture: &Rgb, p: [f64; 2]) -> Option<[u8; 3]> {
    let (x, y) = (p[0], p[1]);
    if !(x >= 0.0 && y >= 0.0) {
        return None;
    }
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    if x0 + 1 >= picture.width || y0 + 1 >= picture.height {
        return None;
    }
    let (fx, fy) = (x - f64::from(x0), y - f64::from(y0));
    let mut out = [0u8; 3];
    let [a, b, c, d] = [
        picture.get(x0, y0),
        picture.get(x0 + 1, y0),
        picture.get(x0, y0 + 1),
        picture.get(x0 + 1, y0 + 1),
    ];
    for i in 0..3 {
        let top = f64::from(a[i]) * (1.0 - fx) + f64::from(b[i]) * fx;
        let bottom = f64::from(c[i]) * (1.0 - fx) + f64::from(d[i]) * fx;
        out[i] = (top * (1.0 - fy) + bottom * fy).round() as u8;
    }
    Some(out)
}

/// Resample `picture` (the view's original) onto the options' plane.
pub fn rectify(
    view: &View,
    camera: &CameraModel,
    picture: &Rgb,
    options: &RectifyOptions,
) -> Result<Rectified> {
    if (picture.width, picture.height) != (camera.width, camera.height) {
        bail!(
            "the picture is {}x{} but the camera is {}x{}",
            picture.width,
            picture.height,
            camera.width,
            camera.height
        );
    }
    if view.pose().is_none() {
        bail!("view '{}' is not posed", view.id);
    }
    let (u, v) = (options.u_axis, options.v_axis);
    if (dot(u, u) - 1.0).abs() > 1e-6 || (dot(v, v) - 1.0).abs() > 1e-6 || dot(u, v).abs() > 1e-6 {
        bail!("the plane's axes must be orthonormal");
    }
    let r = options.resolution;
    let [u0, u1] = options.u_range;
    let [v0, v1] = options.v_range;
    if !(r > 0.0 && u1 > u0 && v1 > v0) {
        bail!("the window must be non-empty and the resolution positive");
    }
    let (w, h) = (((u1 - u0) / r).ceil() as u32, ((v1 - v0) / r).ceil() as u32);
    if u64::from(w) * u64::from(h) > 64_000_000 {
        bail!("the rectified image would be {w}x{h}: coarsen the resolution or shrink the window");
    }
    let mut out = Rgb::new(w, h);
    let mut unseen = 0;
    let chart = |x: u32, y: u32| {
        let cu = u0 + (f64::from(x) + 0.5) * r;
        let cv = v1 - (f64::from(y) + 0.5) * r;
        add(add(options.origin, u, cu), v, cv)
    };
    for y in 0..h {
        for x in 0..w {
            match view
                .project(camera, chart(x, y))
                .and_then(|p| sample(picture, p))
            {
                Some(rgb) => out.set(x, y, rgb),
                None => unseen += 1,
            }
        }
    }
    let mut u_lines = Vec::new();
    let mut v_lines = Vec::new();
    if options.grid > 0.0 {
        let g = options.grid;
        let first = |lo: f64| (lo / g).ceil() as i64;
        let mut k = first(u0);
        while (k as f64) * g < u1 {
            let cu = k as f64 * g;
            let x = ((cu - u0) / r).floor() as u32;
            if x < w {
                for y in 0..h {
                    out.set(x, y, if k % 5 == 0 { GRID_MAJOR } else { GRID });
                }
                u_lines.push(cu);
            }
            k += 1;
        }
        let mut k = first(v0);
        while (k as f64) * g < v1 {
            let cv = k as f64 * g;
            let y = ((v1 - cv) / r).floor() as u32;
            if y < h {
                for x in 0..w {
                    out.set(x, y, if k % 5 == 0 { GRID_MAJOR } else { GRID });
                }
                v_lines.push(cv);
            }
            k += 1;
        }
    }
    // A world mark lands where the view's ray through it meets the plane.
    let normal = cross(u, v);
    let eye = view.position().unwrap_or([0.0; 3]);
    for world in &options.world_marks {
        let d = [world[0] - eye[0], world[1] - eye[1], world[2] - eye[2]];
        let denom = dot(d, normal);
        if denom.abs() < 1e-12 {
            continue;
        }
        let t = dot(
            [
                options.origin[0] - eye[0],
                options.origin[1] - eye[1],
                options.origin[2] - eye[2],
            ],
            normal,
        ) / denom;
        let p = add(eye, d, t);
        let rel = [
            p[0] - options.origin[0],
            p[1] - options.origin[1],
            p[2] - options.origin[2],
        ];
        let x = ((dot(rel, u) - u0) / r).round() as i64;
        let y = ((v1 - dot(rel, v)) / r).round() as i64;
        for s in -8i64..=8 {
            for (px, py) in [(x + s, y), (x, y + s)] {
                if s.abs() > 1 && px >= 0 && py >= 0 && (px as u32) < w && (py as u32) < h {
                    out.set(px as u32, py as u32, WORLD_MARK);
                }
            }
        }
    }
    // Labels in millimetres on the major lines.
    let scale = 2;
    let label = |out: &mut Rgb, text: &str, x: i64, y: i64| {
        crate::text::draw_label(text, x, y, scale, |px, py, lit| {
            if px < out.width && py < out.height {
                out.set(px, py, if lit { GRID_MAJOR } else { [20, 20, 20] });
            }
        });
    };
    let g = options.grid;
    for &cu in &u_lines {
        if ((cu / g).round() as i64) % 5 == 0 {
            let x = ((cu - u0) / r).floor() as i64 + 2;
            label(&mut out, &format!("{:.0}", cu * 1000.0), x, 2);
        }
    }
    for &cv in &v_lines {
        if ((cv / g).round() as i64) % 5 == 0 {
            let y = ((v1 - cv) / r).floor() as i64 + 2;
            label(&mut out, &format!("{:.0}", cv * 1000.0), 2, y);
        }
    }
    Ok(Rectified {
        image: out,
        u_lines,
        v_lines,
        unseen,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plane_facing_the_camera_rectifies_to_the_picture() {
        // A camera 1 m above the origin looking straight down (-z): camera
        // x = world x, camera y = world -y, camera z = world -z.
        let camera = CameraModel::pinhole(40, 40, 20.0, 20.0, 20.0, 20.0);
        let view = View::posed(
            "v",
            0,
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
        );
        let mut picture = Rgb::new(40, 40);
        for y in 0..40 {
            for x in 0..40 {
                picture.set(x, y, [x as u8 * 5, y as u8 * 5, 0]);
            }
        }
        let options = RectifyOptions {
            origin: [0.0; 3],
            u_axis: [1.0, 0.0, 0.0],
            v_axis: [0.0, 1.0, 0.0],
            u_range: [-0.5, 0.5],
            v_range: [-0.5, 0.5],
            resolution: 0.05,
            grid: 0.0,
            world_marks: Vec::new(),
        };
        let out = rectify(&view, &camera, &picture, &options).unwrap();
        assert_eq!((out.image.width, out.image.height), (20, 20));
        // Output pixel (15, 4) is chart (0.275, 0.275): pixel (25.5, 14.5),
        // halfway between the samples.
        let [r, g, _] = out.image.get(15, 4);
        assert_eq!((r, g), (128, 73));
        // A plane the axes do not span is refused.
        let skew = RectifyOptions {
            v_axis: [1.0, 0.0, 0.0],
            ..options
        };
        assert!(rectify(&view, &camera, &picture, &skew).is_err());
    }
}

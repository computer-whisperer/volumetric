//! A magnified crop of a view's original picture with a labelled pixel
//! grid and marks: the picture an agent reads to pick a feature, and the
//! check that a modelled point lands on the real one. The CLI's
//! `view-crop` and the Python bindings are this call.

use anyhow::{Context, Result, bail};
use volumetric_abi::viewset::{CameraModel, View};

use crate::image::Rgb;

const GRID: [u8; 3] = [255, 170, 40];
const GRID_MAJOR: [u8; 3] = [255, 230, 120];
const MARK: [u8; 3] = [60, 220, 255];
const WORLD_MARK: [u8; 3] = [255, 70, 200];

#[derive(Clone, Debug, PartialEq)]
pub struct CropOptions {
    /// Centre of the crop, u,v in the original's pixels.
    pub centre: [f64; 2],
    /// Size of the crop in the original's pixels.
    pub size: (u32, u32),
    /// Integer magnification, nearest neighbour.
    pub scale: u32,
    /// Grid spacing in the original's pixels (0 = no grid); every fifth
    /// line is brighter and labelled.
    pub grid: u32,
    /// Crosses at these pixels of the original.
    pub marks: Vec<[f64; 2]>,
    /// Crosses where these world points land in the picture.
    pub world_marks: Vec<[f64; 3]>,
}

impl Default for CropOptions {
    fn default() -> Self {
        Self {
            centre: [0.0, 0.0],
            size: (600, 400),
            scale: 3,
            grid: 50,
            marks: Vec::new(),
            world_marks: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Crop {
    pub image: Rgb,
    /// The original's pixels the crop covers: `origin..end`.
    pub origin: (u32, u32),
    pub end: (u32, u32),
    pub scale: u32,
    /// The original's u coordinates of the vertical grid lines.
    pub verticals: Vec<u32>,
    /// The original's v coordinates of the horizontal grid lines.
    pub horizontals: Vec<u32>,
    /// Each world mark's pixel, None when behind the camera or unposed.
    pub projected: Vec<Option<[f64; 2]>>,
}

/// A ruler number: the shared bitmap font on a dark box. Coordinates
/// remain in the original image's pixels even for a magnified crop.
fn ruler_number(out: &mut Rgb, x: u32, y: u32, number: u32, scale: u32) {
    crate::text::draw_label(
        &number.to_string(),
        i64::from(x),
        i64::from(y),
        scale,
        |px, py, lit| {
            if px < out.width && py < out.height {
                out.set(px, py, if lit { GRID_MAJOR } else { [20, 20, 20] });
            }
        },
    );
}

/// Crop `picture` (the view's original, at the camera's size) around the
/// options' centre.
pub fn crop(
    view: &View,
    camera: &CameraModel,
    picture: &Rgb,
    options: &CropOptions,
) -> Result<Crop> {
    if (picture.width, picture.height) != (camera.width, camera.height) {
        bail!(
            "the picture is {}x{} but the camera is {}x{}",
            picture.width,
            picture.height,
            camera.width,
            camera.height
        );
    }
    let (w, h) = options.size;
    if w == 0 || h == 0 {
        bail!("the crop size must be positive");
    }
    if !options.centre.iter().all(|c| c.is_finite()) {
        bail!("the crop centre must be finite");
    }
    let scale = options.scale.max(1);
    let u0 = (options.centre[0] - f64::from(w) / 2.0)
        .round()
        .clamp(0.0, f64::from(picture.width - 1)) as u32;
    let v0 = (options.centre[1] - f64::from(h) / 2.0)
        .round()
        .clamp(0.0, f64::from(picture.height - 1)) as u32;
    let u1 = (u0 + w).min(picture.width);
    let v1 = (v0 + h).min(picture.height);
    let (cw, ch) = ((u1 - u0) * scale, (v1 - v0) * scale);
    let mut out = Rgb::new(cw, ch);
    for y in 0..ch {
        for x in 0..cw {
            out.set(x, y, picture.get(u0 + x / scale, v0 + y / scale));
        }
    }
    // The grid, on the original's multiples so coordinates read directly.
    let mut verticals = Vec::new();
    let mut horizontals = Vec::new();
    if options.grid > 0 {
        let g = options.grid;
        let mut u = u0.div_ceil(g) * g;
        while u < u1 {
            let major = (u / g) % 5 == 0;
            let x = (u - u0) * scale;
            for y in 0..ch {
                out.set(x, y, if major { GRID_MAJOR } else { GRID });
            }
            verticals.push(u);
            u += g;
        }
        let mut v = v0.div_ceil(g) * g;
        while v < v1 {
            let major = (v / g) % 5 == 0;
            let y = (v - v0) * scale;
            for x in 0..cw {
                out.set(x, y, if major { GRID_MAJOR } else { GRID });
            }
            horizontals.push(v);
            v += g;
        }
    }
    let cross = |out: &mut Rgb, p: [f64; 2], colour: [u8; 3]| {
        let x = ((p[0] - f64::from(u0)) * f64::from(scale)).round();
        let y = ((p[1] - f64::from(v0)) * f64::from(scale)).round();
        let arm = 6 * scale as i64;
        for d in -arm..=arm {
            for (px, py) in [(x as i64 + d, y as i64), (x as i64, y as i64 + d)] {
                if px >= 0 && py >= 0 && (px as u32) < cw && (py as u32) < ch && d.abs() > 1 {
                    out.set(px as u32, py as u32, colour);
                }
            }
        }
    };
    for mark in &options.marks {
        cross(&mut out, *mark, MARK);
    }
    let projected: Vec<Option<[f64; 2]>> = options
        .world_marks
        .iter()
        .map(|world| {
            let pixel = view.project(camera, *world);
            if let Some(pixel) = pixel {
                cross(&mut out, pixel, WORLD_MARK);
            }
            pixel
        })
        .collect();
    let label_scale = 3;
    // Keep labels readable when the requested grid is finer than a label.
    let mut next_x = 0;
    for &u in &verticals {
        let x = (u - u0) * scale + 2;
        let width = crate::text::text_size(&u.to_string(), label_scale).0 + 2 * label_scale;
        if x >= next_x && x + width <= out.width {
            ruler_number(&mut out, x, 2, u, label_scale);
            next_x = x + width;
        }
    }
    let mut next_y = 9 * label_scale;
    for &v in &horizontals {
        let y = (v - v0) * scale + 2;
        if y >= next_y && y + 9 * label_scale <= out.height {
            ruler_number(&mut out, 2, y, v, label_scale);
            next_y = y + 10 * label_scale;
        }
    }
    Ok(Crop {
        image: out,
        origin: (u0, v0),
        end: (u1, v1),
        scale,
        verticals,
        horizontals,
        projected,
    })
}

/// The view's original picture, checked against its camera's size.
pub fn picture_of(set: &volumetric_abi::viewset::ViewSet, view: &View) -> Result<Rgb> {
    crate::image::decode_rgb(&crate::stills::full_picture(set, view)?)
        .with_context(|| format!("view '{}': the picture does not decode", view.id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_crop_magnifies_grids_and_marks() {
        let camera = CameraModel::pinhole(40, 30, 20.0, 20.0, 20.0, 15.0);
        let view = View::posed(
            "v",
            0,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let mut picture = Rgb::new(40, 30);
        picture.set(33, 27, [200, 100, 50]);
        let options = CropOptions {
            centre: [10.0, 10.0],
            size: (60, 40),
            scale: 2,
            grid: 5,
            marks: vec![[10.0, 10.0]],
            world_marks: vec![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        };
        let out = crop(&view, &camera, &picture, &options).unwrap();
        // The crop clamps to the picture: a centre near the corner starts
        // at the origin and ends where the picture does.
        assert_eq!(out.origin, (0, 0));
        assert_eq!(out.end, (40, 30));
        assert_eq!((out.image.width, out.image.height), (80, 60));
        assert_eq!(out.verticals, (0..40).step_by(5).collect::<Vec<u32>>());
        assert_eq!(out.horizontals, (0..30).step_by(5).collect::<Vec<u32>>());
        // A source pixel off the grid lines and labels is magnified 2x.
        assert_eq!(out.image.get(66, 54), [200, 100, 50]);
        assert_eq!(out.image.get(67, 55), [200, 100, 50]);
        // The point on the optical axis lands at the principal point;
        // behind the camera is None.
        assert_eq!(out.projected, vec![Some([20.0, 15.0]), None]);
        assert!(crop(&view, &camera, &Rgb::new(41, 30), &options).is_err());
    }
}

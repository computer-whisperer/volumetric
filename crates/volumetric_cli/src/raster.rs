//! `sketch-raster`: flat occupancy preview of a 2D sketch model.
//!
//! The CLI counterpart of the GUI's sketch preview: rasterizes a 2D model
//! over its own bounds and writes a PNG, or prints ASCII art for a quick
//! look without leaving the terminal.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

use volumetric::{SketchRaster, rasterize_sketch_from_bytes};

use crate::load_wasm_bytes;

#[derive(Parser, Debug)]
pub struct SketchRasterArgs {
    /// Input file: a .wasm 2D sketch model or a .vproj project file
    #[arg(short, long)]
    pub input: PathBuf,

    /// For .vproj inputs with multiple exports: which exported asset to use
    #[arg(long)]
    pub asset: Option<String>,

    /// Output PNG file path; omit to print ASCII art to stdout
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Raster resolution in cells per axis (default: 512 for PNG, 48 for ASCII)
    #[arg(short, long)]
    pub resolution: Option<usize>,
}

pub fn run_sketch_raster(args: SketchRasterArgs) -> Result<()> {
    let wasm_bytes = load_wasm_bytes(&args.input, args.asset.as_deref())?;

    let resolution = args
        .resolution
        .unwrap_or(if args.output.is_some() { 512 } else { 48 })
        .clamp(2, 4096);

    let raster = rasterize_sketch_from_bytes(&wasm_bytes, resolution)
        .context("Failed to rasterize sketch")?;

    println!(
        "Rasterized {}x{} cells over bounds ({:.3}, {:.3}) to ({:.3}, {:.3})",
        raster.width,
        raster.height,
        raster.bounds_min.0,
        raster.bounds_min.1,
        raster.bounds_max.0,
        raster.bounds_max.1
    );

    match &args.output {
        Some(path) => write_png(&raster, path),
        None => {
            print_ascii(&raster);
            Ok(())
        }
    }
}

/// Occupied cells are black on white. Raster row 0 is min_y; image row 0 is
/// the top, so rows are flipped.
fn write_png(raster: &SketchRaster, path: &PathBuf) -> Result<()> {
    let mut img = image::GrayImage::new(raster.width as u32, raster.height as u32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let cell_y = raster.height - 1 - y as usize;
        *pixel = image::Luma([if raster.cell(x as usize, cell_y) {
            0u8
        } else {
            255u8
        }]);
    }
    img.save(path)
        .with_context(|| format!("Failed to write {}", path.display()))?;
    println!("Wrote {}", path.display());
    Ok(())
}

/// Two raster rows per character line (terminal cells are ~2:1) using half
/// blocks, top row of the sketch first.
fn print_ascii(raster: &SketchRaster) {
    let mut y = raster.height;
    while y > 0 {
        let top = y - 1;
        let bottom = y.checked_sub(2);
        let mut line = String::with_capacity(raster.width);
        for x in 0..raster.width {
            let t = raster.cell(x, top);
            let b = bottom.map(|b| raster.cell(x, b)).unwrap_or(false);
            line.push(match (t, b) {
                (true, true) => '█',
                (true, false) => '▀',
                (false, true) => '▄',
                (false, false) => ' ',
            });
        }
        println!("{}", line.trim_end());
        y = y.saturating_sub(2);
    }
}

//! Demo model: Rounded box centered at origin.
//!
//! This WASM module exports the N-dimensional model ABI:
//! - `get_dimensions() -> u32`: Returns 3
//! - `get_bounds(out_ptr: i32)`: Writes interleaved min/max bounds
//! - `sample(pos_ptr: i32) -> f32`: Reads position, returns density
//! - `memory`: Linear memory export
//!
//! Uses a standard signed-distance formula and treats SDF <= 0 as inside.

const B_X: f64 = 0.9;
const B_Y: f64 = 0.6;
const B_Z: f64 = 0.4;
const RADIUS: f64 = 0.2;

const MIN_X: f64 = -(B_X + RADIUS);
const MAX_X: f64 = B_X + RADIUS;
const MIN_Y: f64 = -(B_Y + RADIUS);
const MAX_Y: f64 = B_Y + RADIUS;
const MIN_Z: f64 = -(B_Z + RADIUS);
const MAX_Z: f64 = B_Z + RADIUS;

#[inline]
fn abs3(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    (x.abs(), y.abs(), z.abs())
}

#[inline]
fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

#[inline]
fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

#[inline]
fn length3(x: f64, y: f64, z: f64) -> f64 {
    libm::sqrt(x * x + y * y + z * z)
}

#[inline]
fn sdf_rounded_box(x: f64, y: f64, z: f64, bx: f64, by: f64, bz: f64, r: f64) -> f64 {
    // Quilez: sdRoundBox(p, b, r)
    let (ax, ay, az) = abs3(x, y, z);
    let qx = ax - bx;
    let qy = ay - by;
    let qz = az - bz;

    let mx = max_f64(qx, 0.0);
    let my = max_f64(qy, 0.0);
    let mz = max_f64(qz, 0.0);
    let outside = length3(mx, my, mz);

    let inside = min_f64(max_f64(qx, max_f64(qy, qz)), 0.0);
    outside + inside - r
}

/// Returns the number of dimensions (3 for this model).
#[unsafe(no_mangle)]
pub extern "C" fn get_dimensions() -> u32 {
    3
}

/// Writes the bounding box to memory at out_ptr.
/// Format: [min_x, max_x, min_y, max_y, min_z, max_z] as f64
#[unsafe(no_mangle)]
pub extern "C" fn get_bounds(out_ptr: i32) {
    let ptr = out_ptr as *mut f64;
    unsafe {
        *ptr.add(0) = MIN_X;
        *ptr.add(1) = MAX_X;
        *ptr.add(2) = MIN_Y;
        *ptr.add(3) = MAX_Y;
        *ptr.add(4) = MIN_Z;
        *ptr.add(5) = MAX_Z;
    }
}

/// Sample the density at the position read from pos_ptr.
#[unsafe(no_mangle)]
pub extern "C" fn sample(pos_ptr: i32) -> f32 {
    let ptr = pos_ptr as *const f64;
    let (x, y, z) = unsafe { (*ptr, *ptr.add(1), *ptr.add(2)) };

    if sdf_rounded_box(x, y, z, B_X, B_Y, B_Z, RADIUS) <= 0.0 {
        1.0
    } else {
        0.0
    }
}

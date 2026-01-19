#![cfg_attr(target_arch = "wasm32", no_std)]

//! Demo model: Torus (ring) centered at origin.
//!
//! This WASM module exports the N-dimensional model ABI:
//! - `get_dimensions() -> u32`: Returns 3
//! - `get_bounds(out_ptr: i32)`: Writes interleaved min/max bounds
//! - `sample(pos_ptr: i32) -> f32`: Reads position, returns density
//! - `memory`: Linear memory export

#[cfg(target_arch = "wasm32")]
use core::panic::PanicInfo;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    unsafe { core::arch::wasm32::unreachable() }
}

const R_MAJOR: f64 = 1.0;
const R_MINOR: f64 = 0.35;

const MIN_X: f64 = -(R_MAJOR + R_MINOR);
const MAX_X: f64 = R_MAJOR + R_MINOR;
const MIN_Y: f64 = -R_MINOR;
const MAX_Y: f64 = R_MINOR;
const MIN_Z: f64 = -(R_MAJOR + R_MINOR);
const MAX_Z: f64 = R_MAJOR + R_MINOR;

#[inline]
fn inside_torus(x: f64, y: f64, z: f64) -> bool {
    // Implicit torus: (sqrt(x^2+z^2) - R)^2 + y^2 <= r^2
    let q = libm::sqrt(x * x + z * z) - R_MAJOR;
    q * q + y * y <= R_MINOR * R_MINOR
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

    if inside_torus(x, y, z) { 1.0 } else { 0.0 }
}

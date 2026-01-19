#![cfg_attr(target_arch = "wasm32", no_std)]

//! Demo model: Gyroid lattice (finite chunk).
//!
//! This WASM module exports the N-dimensional model ABI:
//! - `get_dimensions() -> u32`: Returns 3
//! - `get_bounds(out_ptr: i32)`: Writes interleaved min/max bounds
//! - `sample(pos_ptr: i32) -> f32`: Reads position, returns density
//! - `memory`: Linear memory export
//!
//! The gyroid is an implicit surface:
//!   g(x,y,z) = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)
//! We create a "thickened" shell by taking |g| < thickness inside a bounded box.

#[cfg(target_arch = "wasm32")]
use core::panic::PanicInfo;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    unsafe { core::arch::wasm32::unreachable() }
}

const BOUND: f64 = 3.14159265; // ~pi
const THICKNESS: f64 = 0.28;

const MIN_X: f64 = -BOUND;
const MAX_X: f64 = BOUND;
const MIN_Y: f64 = -BOUND;
const MAX_Y: f64 = BOUND;
const MIN_Z: f64 = -BOUND;
const MAX_Z: f64 = BOUND;

#[inline]
fn gyroid(x: f64, y: f64, z: f64) -> f64 {
    libm::sin(x) * libm::cos(y) + libm::sin(y) * libm::cos(z) + libm::sin(z) * libm::cos(x)
}

#[inline]
fn in_bounds(x: f64, y: f64, z: f64, b: f64) -> bool {
    x >= -b && x <= b && y >= -b && y <= b && z >= -b && z <= b
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

    if !in_bounds(x, y, z, BOUND) {
        return 0.0;
    }

    if gyroid(x, y, z).abs() < THICKNESS { 1.0 } else { 0.0 }
}

#![cfg_attr(target_arch = "wasm32", no_std)]

//! Test model: A simple sphere centered at origin with radius 1.0
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

const MIN_X: f64 = -1.0;
const MAX_X: f64 = 1.0;
const MIN_Y: f64 = -1.0;
const MAX_Y: f64 = 1.0;
const MIN_Z: f64 = -1.0;
const MAX_Z: f64 = 1.0;

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
/// Returns 1.0 if inside the sphere, 0.0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn sample(pos_ptr: i32) -> f32 {
    let ptr = pos_ptr as *const f64;
    let (x, y, z) = unsafe { (*ptr, *ptr.add(1), *ptr.add(2)) };

    let distance_squared = x * x + y * y + z * z;
    if distance_squared <= 1.0 {
        1.0 // inside density
    } else {
        0.0 // outside density
    }
}

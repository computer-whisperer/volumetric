#![no_std]

//! Test model: A simple sphere centered at origin with radius 1.0
//!
//! This WASM module exports a function that determines whether a given
//! 3D point is inside the model or not.

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        core::arch::wasm32::unreachable();
    }

    #[cfg(not(target_arch = "wasm32"))]
    loop {}
}

/// Check if a point (x, y, z) is inside the model.
///
/// For this test model, we define a unit sphere centered at the origin.
/// A point is inside if x² + y² + z² <= 1.0
#[no_mangle]
pub extern "C" fn is_inside(x: f32, y: f32, z: f32) -> i32 {
    let distance_squared = x * x + y * y + z * z;
    if distance_squared <= 1.0 {
        1 // true - inside
    } else {
        0 // false - outside
    }
}

/// Get the bounding box of the model.
/// Returns the bounds as 6 floats: min_x, min_y, min_z, max_x, max_y, max_z
/// For a unit sphere, this is (-1, -1, -1) to (1, 1, 1)
#[no_mangle]
pub extern "C" fn get_bounds_min_x() -> f32 { -1.0 }
#[no_mangle]
pub extern "C" fn get_bounds_min_y() -> f32 { -1.0 }
#[no_mangle]
pub extern "C" fn get_bounds_min_z() -> f32 { -1.0 }
#[no_mangle]
pub extern "C" fn get_bounds_max_x() -> f32 { 1.0 }
#[no_mangle]
pub extern "C" fn get_bounds_max_y() -> f32 { 1.0 }
#[no_mangle]
pub extern "C" fn get_bounds_max_z() -> f32 { 1.0 }

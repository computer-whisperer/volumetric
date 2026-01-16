#![cfg_attr(target_arch = "wasm32", no_std)]

//! Demo model: Torus (ring) centered at origin.
//!
//! Exports `is_inside(x,y,z) -> f32` (density) and bounding-box getter functions.

#[cfg(target_arch = "wasm32")]
use core::panic::PanicInfo;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    unsafe { core::arch::wasm32::unreachable() }
}

#[inline]
fn inside_torus(x: f64, y: f64, z: f64) -> bool {
    // Implicit torus: (sqrt(x^2+z^2) - R)^2 + y^2 <= r^2
    // R: major radius, r: minor radius
    let r_major = 1.0;
    let r_minor = 0.35;
    let q = libm::sqrt(x * x + z * z) - r_major;
    q * q + y * y <= r_minor * r_minor
}

#[no_mangle]
pub extern "C" fn is_inside(x: f64, y: f64, z: f64) -> f32 {
    if inside_torus(x, y, z) { 1.0 } else { 0.0 }
}

// Bounds: x/z in [-R-r, R+r], y in [-r, r]
#[no_mangle]
pub extern "C" fn get_bounds_min_x() -> f64 {
    -(1.0 + 0.35)
}
#[no_mangle]
pub extern "C" fn get_bounds_min_y() -> f64 {
    -0.35
}
#[no_mangle]
pub extern "C" fn get_bounds_min_z() -> f64 {
    -(1.0 + 0.35)
}
#[no_mangle]
pub extern "C" fn get_bounds_max_x() -> f64 {
    1.0 + 0.35
}
#[no_mangle]
pub extern "C" fn get_bounds_max_y() -> f64 {
    0.35
}
#[no_mangle]
pub extern "C" fn get_bounds_max_z() -> f64 {
    1.0 + 0.35
}

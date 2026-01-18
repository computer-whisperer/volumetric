#![cfg_attr(target_arch = "wasm32", no_std)]

//! Demo model: Gyroid lattice (finite chunk).
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

#[inline]
fn gyroid(x: f64, y: f64, z: f64) -> f64 {
    libm::sin(x) * libm::cos(y) + libm::sin(y) * libm::cos(z) + libm::sin(z) * libm::cos(x)
}

#[inline]
fn in_bounds(x: f64, y: f64, z: f64, b: f64) -> bool {
    x >= -b && x <= b && y >= -b && y <= b && z >= -b && z <= b
}

#[unsafe(no_mangle)]
pub extern "C" fn is_inside(x: f64, y: f64, z: f64) -> f32 {
    let bound = 3.14159265f64; // ~pi
    if !in_bounds(x, y, z, bound) {
        return 0.0;
    }

    let thickness = 0.28f64;
    if gyroid(x, y, z).abs() < thickness { 1.0 } else { 0.0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_bounds_min_x() -> f64 {
    -3.14159265f64
}
#[unsafe(no_mangle)]
pub extern "C" fn get_bounds_min_y() -> f64 {
    -3.14159265f64
}
#[unsafe(no_mangle)]
pub extern "C" fn get_bounds_min_z() -> f64 {
    -3.14159265f64
}
#[unsafe(no_mangle)]
pub extern "C" fn get_bounds_max_x() -> f64 {
    3.14159265f64
}
#[unsafe(no_mangle)]
pub extern "C" fn get_bounds_max_y() -> f64 {
    3.14159265f64
}
#[unsafe(no_mangle)]
pub extern "C" fn get_bounds_max_z() -> f64 {
    3.14159265f64
}

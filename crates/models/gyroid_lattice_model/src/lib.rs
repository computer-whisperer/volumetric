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
fn gyroid(x: f32, y: f32, z: f32) -> f32 {
    libm::sinf(x) * libm::cosf(y) + libm::sinf(y) * libm::cosf(z) + libm::sinf(z) * libm::cosf(x)
}

#[inline]
fn in_bounds(x: f32, y: f32, z: f32, b: f32) -> bool {
    x >= -b && x <= b && y >= -b && y <= b && z >= -b && z <= b
}

#[no_mangle]
pub extern "C" fn is_inside(x: f32, y: f32, z: f32) -> i32 {
    let bound = 3.14159265; // ~pi
    if !in_bounds(x, y, z, bound) {
        return 0;
    }

    let thickness = 0.28;
    (gyroid(x, y, z).abs() < thickness) as i32
}

#[no_mangle]
pub extern "C" fn get_bounds_min_x() -> f32 {
    -3.14159265
}
#[no_mangle]
pub extern "C" fn get_bounds_min_y() -> f32 {
    -3.14159265
}
#[no_mangle]
pub extern "C" fn get_bounds_min_z() -> f32 {
    -3.14159265
}
#[no_mangle]
pub extern "C" fn get_bounds_max_x() -> f32 {
    3.14159265
}
#[no_mangle]
pub extern "C" fn get_bounds_max_y() -> f32 {
    3.14159265
}
#[no_mangle]
pub extern "C" fn get_bounds_max_z() -> f32 {
    3.14159265
}

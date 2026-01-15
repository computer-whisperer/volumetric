#![no_std]

//! Demo model: Mandelbulb fractal (3D Mandelbrot-like set).
//!
//! We treat a point `c` as inside if the iterative sequence
//!   z_{n+1} = z_n^p + c,  z_0 = 0
//! does not escape past a bailout radius within a fixed iteration count.

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

#[inline]
fn length3(x: f32, y: f32, z: f32) -> f32 {
    libm::sqrtf(x * x + y * y + z * z)
}

#[inline]
fn mandelbulb_inside(cx: f32, cy: f32, cz: f32) -> bool {
    let power: f32 = 8.0;
    let max_iter: u32 = 18;
    let bailout: f32 = 2.0;

    let mut x = 0.0f32;
    let mut y = 0.0f32;
    let mut z = 0.0f32;

    for _ in 0..max_iter {
        let r = length3(x, y, z);
        if r > bailout {
            return false;
        }
        if r < 1.0e-6 {
            x = cx;
            y = cy;
            z = cz;
            continue;
        }

        let theta = libm::acosf((z / r).clamp(-1.0, 1.0));
        let phi = libm::atan2f(y, x);

        let rp = libm::powf(r, power);
        let thetap = theta * power;
        let phip = phi * power;

        let sin_t = libm::sinf(thetap);
        let nx = rp * sin_t * libm::cosf(phip);
        let ny = rp * sin_t * libm::sinf(phip);
        let nz = rp * libm::cosf(thetap);

        x = nx + cx;
        y = ny + cy;
        z = nz + cz;
    }

    true
}

#[no_mangle]
pub extern "C" fn is_inside(x: f32, y: f32, z: f32) -> i32 {
    // Slight scale so the interesting structure fills the bounds better.
    let s = 0.9;
    mandelbulb_inside(x * s, y * s, z * s) as i32
}

#[no_mangle]
pub extern "C" fn get_bounds_min_x() -> f32 {
    -1.35
}
#[no_mangle]
pub extern "C" fn get_bounds_min_y() -> f32 {
    -1.35
}
#[no_mangle]
pub extern "C" fn get_bounds_min_z() -> f32 {
    -1.35
}
#[no_mangle]
pub extern "C" fn get_bounds_max_x() -> f32 {
    1.35
}
#[no_mangle]
pub extern "C" fn get_bounds_max_y() -> f32 {
    1.35
}
#[no_mangle]
pub extern "C" fn get_bounds_max_z() -> f32 {
    1.35
}

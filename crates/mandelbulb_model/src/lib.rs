//! Demo model: Mandelbulb fractal (3D Mandelbrot-like set).
//!
//! We treat a point `c` as inside if the iterative sequence
//!   z_{n+1} = z_n^p + c,  z_0 = 0
//! does not escape past a bailout radius within a fixed iteration count.

#[inline]
fn length3(x: f32, y: f32, z: f32) -> f32 {
    (x * x + y * y + z * z).sqrt()
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

        let theta = (z / r).clamp(-1.0, 1.0).acos();
        let phi = y.atan2(x);

        let rp = r.powf(power);
        let thetap = theta * power;
        let phip = phi * power;

        let sin_t = thetap.sin();
        let nx = rp * sin_t * phip.cos();
        let ny = rp * sin_t * phip.sin();
        let nz = rp * thetap.cos();

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

//!
//! Uses a standard signed-distance formula and treats SDF <= 0 as inside.

#[inline]
fn abs3(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    (x.abs(), y.abs(), z.abs())
}

#[inline]
fn max_f32(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

#[inline]
fn min_f32(a: f64, b: f64) -> f64 {
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

    let mx = max_f32(qx, 0.0);
    let my = max_f32(qy, 0.0);
    let mz = max_f32(qz, 0.0);
    let outside = length3(mx, my, mz);

    let inside = min_f32(max_f32(qx, max_f32(qy, qz)), 0.0);
    outside + inside - r
}

#[no_mangle]
pub extern "C" fn is_inside(x: f64, y: f64, z: f64) -> f32 {
    let b = (0.9, 0.6, 0.4);
    let r = 0.2;
    if sdf_rounded_box(x, y, z, b.0, b.1, b.2, r) <= 0.0 { 1.0 } else { 0.0 }
}

#[no_mangle]
pub extern "C" fn get_bounds_min_x() -> f32 {
    -(0.9 + 0.2)
}
#[no_mangle]
pub extern "C" fn get_bounds_min_y() -> f32 {
    -(0.6 + 0.2)
}
#[no_mangle]
pub extern "C" fn get_bounds_min_z() -> f32 {
    -(0.4 + 0.2)
}
#[no_mangle]
pub extern "C" fn get_bounds_max_x() -> f32 {
    0.9 + 0.2
}
#[no_mangle]
pub extern "C" fn get_bounds_max_y() -> f32 {
    0.6 + 0.2
}
#[no_mangle]
pub extern "C" fn get_bounds_max_z() -> f32 {
    0.4 + 0.2
}

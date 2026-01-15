//!
//! Uses a standard signed-distance formula and treats SDF <= 0 as inside.

#[inline]
fn abs3(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    (x.abs(), y.abs(), z.abs())
}

#[inline]
fn max_f32(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

#[inline]
fn min_f32(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

#[inline]
fn length3(x: f32, y: f32, z: f32) -> f32 {
    (x * x + y * y + z * z).sqrt()
}

#[inline]
fn sdf_rounded_box(x: f32, y: f32, z: f32, bx: f32, by: f32, bz: f32, r: f32) -> f32 {
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
pub extern "C" fn is_inside(x: f32, y: f32, z: f32) -> i32 {
    let b = (0.9, 0.6, 0.4);
    let r = 0.2;
    (sdf_rounded_box(x, y, z, b.0, b.1, b.2, r) <= 0.0) as i32
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

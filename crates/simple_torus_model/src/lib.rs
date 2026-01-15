//! Demo model: Torus (ring) centered at origin.
//!
//! Exports `is_inside(x,y,z) -> i32` and bounding-box getter functions.

#[inline]
fn inside_torus(x: f32, y: f32, z: f32) -> bool {
    // Implicit torus: (sqrt(x^2+z^2) - R)^2 + y^2 <= r^2
    // R: major radius, r: minor radius
    let r_major = 1.0;
    let r_minor = 0.35;
    let q = (x * x + z * z).sqrt() - r_major;
    q * q + y * y <= r_minor * r_minor
}

#[no_mangle]
pub extern "C" fn is_inside(x: f32, y: f32, z: f32) -> i32 {
    inside_torus(x, y, z) as i32
}

// Bounds: x/z in [-R-r, R+r], y in [-r, r]
#[no_mangle]
pub extern "C" fn get_bounds_min_x() -> f32 {
    -(1.0 + 0.35)
}
#[no_mangle]
pub extern "C" fn get_bounds_min_y() -> f32 {
    -0.35
}
#[no_mangle]
pub extern "C" fn get_bounds_min_z() -> f32 {
    -(1.0 + 0.35)
}
#[no_mangle]
pub extern "C" fn get_bounds_max_x() -> f32 {
    1.0 + 0.35
}
#[no_mangle]
pub extern "C" fn get_bounds_max_y() -> f32 {
    0.35
}
#[no_mangle]
pub extern "C" fn get_bounds_max_z() -> f32 {
    1.0 + 0.35
}

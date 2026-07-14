#![cfg_attr(target_arch = "wasm32", no_std)]

//! Demo model: Mandelbulb fractal (3D Mandelbrot-like set).
//!
//! This WASM module exports the N-dimensional model ABI:
//! - `get_dimensions() -> u32`: Returns 3
//! - `get_io_ptr() -> i32`: Returns the model-owned IO buffer (2n f64s)
//! - `get_bounds(out_ptr: i32)`: Writes interleaved min/max bounds
//! - `sample(pos_ptr: i32) -> f32`: Reads position, returns density
//! - `memory`: Linear memory export
//! - `get_metadata() -> i64`: Optional catalog metadata (CBOR, see ABI.md)
//!
//! We treat a point `c` as inside if the iterative sequence
//!   z_{n+1} = z_n^p + c,  z_0 = 0
//! does not escape past a bailout radius within a fixed iteration count.

#[cfg(target_arch = "wasm32")]
use core::panic::PanicInfo;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

const BOUND: f64 = 1.35;
const MIN_X: f64 = -BOUND;
const MAX_X: f64 = BOUND;
const MIN_Y: f64 = -BOUND;
const MAX_Y: f64 = BOUND;
const MIN_Z: f64 = -BOUND;
const MAX_Z: f64 = BOUND;

#[inline]
fn length3(x: f64, y: f64, z: f64) -> f64 {
    libm::sqrt(x * x + y * y + z * z)
}

#[inline]
fn mandelbulb_inside(cx: f64, cy: f64, cz: f64) -> bool {
    let power: f64 = 8.0;
    let max_iter: u32 = 18;
    let bailout: f64 = 2.0;

    let mut x = 0.0f64;
    let mut y = 0.0f64;
    let mut z = 0.0f64;

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

        let theta = libm::acos((z / r).clamp(-1.0, 1.0));
        let phi = libm::atan2(y, x);

        let rp = libm::pow(r, power);
        let thetap = theta * power;
        let phip = phi * power;

        let sin_t = libm::sin(thetap);
        let nx = rp * sin_t * libm::cos(phip);
        let ny = rp * sin_t * libm::sin(phip);
        let nz = rp * libm::cos(thetap);

        x = nx + cx;
        y = ny + cy;
        z = nz + cz;
    }

    true
}

/// Returns the number of dimensions (3 for this model).
#[unsafe(no_mangle)]
pub extern "C" fn get_dimensions() -> u32 {
    3
}

/// Model-owned IO scratch buffer: the host writes sample positions here and
/// reads bounds back from here. Must hold at least 2 * dims f64s.
static mut IO_BUFFER: [f64; 6] = [0.0; 6];

/// Returns a pointer to the model's IO buffer.
#[unsafe(no_mangle)]
pub extern "C" fn get_io_ptr() -> i32 {
    (&raw mut IO_BUFFER) as i32
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
#[unsafe(no_mangle)]
pub extern "C" fn sample(pos_ptr: i32) -> f32 {
    let ptr = pos_ptr as *const f64;
    let (x, y, z) = unsafe { (*ptr, *ptr.add(1), *ptr.add(2)) };

    // Slight scale so the interesting structure fills the bounds better.
    let s = 0.9f64;
    if mandelbulb_inside(x * s, y * s, z * s) {
        1.0
    } else {
        0.0
    }
}

/// CBOR catalog metadata, precomputed by build.rs so the module carries no
/// runtime CBOR encoder (most models are `no_std` on wasm32).
static METADATA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/metadata.cbor"));

/// Catalog metadata export (see ABI.md): `(ptr, len)` of CBOR-encoded
/// `OperatorMetadata` with empty inputs/outputs, packed as `ptr | (len << 32)`.
/// Read by host catalogs; never called during execution.
#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    let ptr = METADATA.as_ptr() as u32 as u64;
    let len = METADATA.len() as u32 as u64;
    (ptr | (len << 32)) as i64
}

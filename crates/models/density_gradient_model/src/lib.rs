//! Test model: a solid cube carrying a typed density channel.
//!
//! Exercises the optional typed-channel model ABI (see `volumetric_abi`):
//! occupancy is 1.0 inside the cube [-1, 1]^3, and a second `Density` channel
//! carries an x-gradient `clamp(0.5 + 0.5 * x, 0, 1)` — position-dependent,
//! so transform wrappers that fail to rewrite `sample_channels` positions
//! show up as wrong densities in tests.
//!
//! Exports the N-dimensional model ABI plus the typed-channel extension:
//! - `get_sample_format() -> i64`: [Occupancy, Density]
//! - `sample_channels(pos_ptr: i32, out_ptr: i32)`: writes both channels

use std::sync::OnceLock;
use volumetric_abi::{ChannelKind, SampleChannel, SampleFormat};

const MIN: f64 = -1.0;
const MAX: f64 = 1.0;

/// Returns the number of dimensions (3 for this model).
#[unsafe(no_mangle)]
pub extern "C" fn get_dimensions() -> u32 {
    3
}

/// Model-owned IO scratch buffer (2 * dims f64s). The host writes sample
/// positions into the first half and reads channel output from the second.
static mut IO_BUFFER: [f64; 6] = [0.0; 6];

/// Returns a pointer to the model's IO buffer.
#[unsafe(no_mangle)]
pub extern "C" fn get_io_ptr() -> i32 {
    (&raw mut IO_BUFFER) as i32
}

/// Writes the bounding box to memory at out_ptr, interleaved min/max.
#[unsafe(no_mangle)]
pub extern "C" fn get_bounds(out_ptr: i32) {
    let ptr = out_ptr as *mut f64;
    for axis in 0..3 {
        unsafe {
            *ptr.add(axis * 2) = MIN;
            *ptr.add(axis * 2 + 1) = MAX;
        }
    }
}

fn read_position(pos_ptr: i32) -> (f64, f64, f64) {
    let ptr = pos_ptr as *const f64;
    unsafe { (*ptr, *ptr.add(1), *ptr.add(2)) }
}

fn occupancy(x: f64, y: f64, z: f64) -> f32 {
    let inside = (MIN..=MAX).contains(&x) && (MIN..=MAX).contains(&y) && (MIN..=MAX).contains(&z);
    if inside { 1.0 } else { 0.0 }
}

fn density(x: f64) -> f32 {
    (0.5 + 0.5 * x).clamp(0.0, 1.0) as f32
}

/// Sample the occupancy at the position read from pos_ptr (canonical 1.0/0.0).
#[unsafe(no_mangle)]
pub extern "C" fn sample(pos_ptr: i32) -> f32 {
    let (x, y, z) = read_position(pos_ptr);
    occupancy(x, y, z)
}

/// Sample every declared channel: [occupancy, density] as f32s at out_ptr.
#[unsafe(no_mangle)]
pub extern "C" fn sample_channels(pos_ptr: i32, out_ptr: i32) {
    let (x, y, z) = read_position(pos_ptr);
    let out = out_ptr as *mut f32;
    unsafe {
        *out.add(0) = occupancy(x, y, z);
        *out.add(1) = density(x);
    }
}

/// Declare the per-sample format: occupancy plus a density channel.
#[unsafe(no_mangle)]
pub extern "C" fn get_sample_format() -> i64 {
    static FORMAT: OnceLock<Vec<u8>> = OnceLock::new();
    volumetric_abi::sample_format_reply(&FORMAT, || SampleFormat {
        channels: vec![
            SampleChannel {
                name: "occupancy".to_string(),
                kind: ChannelKind::Occupancy,
            },
            SampleChannel {
                name: "infill".to_string(),
                kind: ChannelKind::Density,
            },
        ],
    })
}

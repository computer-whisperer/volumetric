//! The brim model template: a complete, *stateless* 3D Model ABI
//! implementation over a baked 2D brim field extruded along a bed-plane
//! chart. `brim_operator` patches a copy of this module with the field it
//! bakes from scanning its input model's print-bed footprint.
//!
//! The bed plane is an arbitrary orthonormal chart: an origin, two
//! in-plane basis vectors b1/b2, and the unit `up` direction the brim
//! grows along (for axis-aligned beds these are unit axis vectors). The
//! field is a `gridfield_model_core` payload over the chart's (u, v)
//! coordinates holding `brim_width - distance_to_footprint` (world
//! units — the chart is an isometry), so its bilinear zero crossing is
//! the brim contour: a point is occupied when its height off the bed
//! `w = dot(p - origin, up)` lies in `0 <= w <= height` and the field at
//! `(dot(p - origin, b1), dot(p - origin, b2))` is >= 0.
//!
//! # Patch contract
//!
//! - `brim_payload_slot() -> i32` returns the address of a 4-byte slot.
//!   The operator overwrites the slot (via an active data segment) with
//!   the little-endian base address of the grid-field payload it appends
//!   in freshly reserved memory pages. The slot must be read volatilely —
//!   its compiled initializer is zero, and a folded read would make the
//!   template permanently empty.
//! - `brim_config_slot() -> i32` returns the address of a 104-byte slot
//!   holding 13 little-endian f64s: `origin` (3), `b1` (3), `b2` (3),
//!   `up` (3), then the brim `height`.
//! - An unpatched template (slots still zero) or a corrupt payload behaves
//!   as an empty model: bounds all zero, every sample 0.0 — the ABI's
//!   errors-read-as-outside convention.
//!
//! Beyond the Model ABI, `brim_sample(x, y, z) -> f32` is exported so the
//! operator's combined-output merge glue can evaluate the brim per
//! position without touching the IO buffer.
//!
//! # Regenerating the checked-in binary
//!
//! The operator embeds a prebuilt copy of this module (see
//! `crates/operators/brim_operator/template/`). After changing this crate
//! or `gridfield_model_core`'s read side:
//!
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p brim_model_template
//! cp target/wasm32-unknown-unknown/release/brim_model_template.wasm \
//!    crates/operators/brim_operator/template/
//! ```

use gridfield_model_core::PayloadView;

/// Model-owned IO scratch region (2 * 3 f64s), per the Model ABI.
static mut IO_BUFFER: [f64; 6] = [0.0; 6];

/// The 4-byte payload-base slot the operator patches. Aligned so the u32
/// read is well-formed.
#[repr(align(4))]
struct PayloadSlot([u8; 4]);
static PAYLOAD_SLOT: PayloadSlot = PayloadSlot([0; 4]);

/// The 104-byte chart-config slot the operator patches: origin, b1, b2,
/// up (3 f64s each), then the brim height — 13 little-endian f64s.
/// Aligned so the u64 reads are well-formed.
#[repr(align(8))]
struct ConfigSlot([u8; 104]);
static CONFIG_SLOT: ConfigSlot = ConfigSlot([0; 104]);

/// The patched field payload, if the slot has been filled and the header
/// validates.
fn payload() -> Option<PayloadView<'static>> {
    // Volatile: the slot's compile-time value is zero, and the whole point
    // is that the bytes change after compilation.
    let base = unsafe { core::ptr::read_volatile(PAYLOAD_SLOT.0.as_ptr() as *const u32) } as usize;
    if base == 0 {
        return None;
    }
    // payload_len lives at header offset 12; trust it only as far as
    // PayloadView's structural validation.
    let len = unsafe { core::ptr::read_volatile((base + 12) as *const u32) } as usize;
    let bytes = unsafe { core::slice::from_raw_parts(base as *const u8, len) };
    PayloadView::new(bytes).ok()
}

/// The patched bed chart (origin, b1, b2, up, height); all zeros while
/// unpatched, which behaves as an empty model.
fn chart() -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3], f64) {
    let base = CONFIG_SLOT.0.as_ptr() as *const u64;
    let read = |i: usize| f64::from_bits(unsafe { core::ptr::read_volatile(base.add(i)) });
    let vec3 = |o: usize| [read(o), read(o + 1), read(o + 2)];
    (vec3(0), vec3(3), vec3(6), vec3(9), read(12))
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[unsafe(no_mangle)]
pub extern "C" fn brim_payload_slot() -> i32 {
    PAYLOAD_SLOT.0.as_ptr() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn brim_config_slot() -> i32 {
    CONFIG_SLOT.0.as_ptr() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn get_dimensions() -> u32 {
    3
}

#[unsafe(no_mangle)]
pub extern "C" fn get_io_ptr() -> i32 {
    (&raw mut IO_BUFFER) as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn get_bounds(out_ptr: i32) {
    // The world AABB of the oriented brim box: the 8 corners of the
    // chart-space field bounds crossed with w in {0, height}.
    let field = payload().map(|p| p.bounds()).unwrap_or([0.0; 4]);
    let (origin, b1, b2, up, height) = chart();
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for &u in &field[..2] {
        for &v in &field[2..] {
            for w in [0.0, height] {
                for axis in 0..3 {
                    let c = origin[axis] + u * b1[axis] + v * b2[axis] + w * up[axis];
                    min[axis] = min[axis].min(c);
                    max[axis] = max[axis].max(c);
                }
            }
        }
    }
    for axis in 0..3 {
        let (lo, hi) = (min[axis], max[axis]);
        unsafe {
            core::ptr::write_unaligned((out_ptr as usize + axis * 16) as *mut f64, lo);
            core::ptr::write_unaligned((out_ptr as usize + axis * 16 + 8) as *mut f64, hi);
        }
    }
}

/// Occupancy at (x, y, z): within the brim height off the bed plane and
/// on the non-negative side of the baked field in chart coordinates.
/// Also called directly by merge glue.
#[unsafe(no_mangle)]
pub extern "C" fn brim_sample(x: f64, y: f64, z: f64) -> f32 {
    let (origin, b1, b2, up, height) = chart();
    let rel = [x - origin[0], y - origin[1], z - origin[2]];
    let w = dot(rel, up);
    if !(w >= 0.0 && w <= height) {
        return 0.0;
    }
    match payload().and_then(|view| view.sample(dot(rel, b1), dot(rel, b2))) {
        Some(value) if value >= 0.0 => 1.0,
        _ => 0.0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sample(pos_ptr: i32) -> f32 {
    let p: [f64; 3] = core::array::from_fn(|i| unsafe {
        core::ptr::read_unaligned((pos_ptr as usize + i * 8) as *const f64)
    });
    brim_sample(p[0], p[1], p[2])
}

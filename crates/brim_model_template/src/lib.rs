//! The brim model template: a complete, *stateless* 3D Model ABI
//! implementation over a baked 2D brim field extruded across a z slab.
//! `brim_operator` patches a copy of this module with the field it bakes
//! from scanning its input model's print-bed footprint.
//!
//! The field is a `gridfield_model_core` payload holding
//! `brim_width - distance_to_footprint` (world units), so its bilinear
//! zero crossing is the brim contour: a point is occupied when it lies in
//! the slab `bed_z <= z <= z_top` and the field at (x, y) is >= 0.
//!
//! # Patch contract
//!
//! - `brim_payload_slot() -> i32` returns the address of a 4-byte slot.
//!   The operator overwrites the slot (via an active data segment) with
//!   the little-endian base address of the grid-field payload it appends
//!   in freshly reserved memory pages. The slot must be read volatilely —
//!   its compiled initializer is zero, and a folded read would make the
//!   template permanently empty.
//! - `brim_config_slot() -> i32` returns the address of a 16-byte slot
//!   holding `bed_z` then `z_top` as little-endian f64s.
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

/// The 16-byte slab-config slot the operator patches: bed_z, z_top as
/// little-endian f64s. Aligned so the u64 reads are well-formed.
#[repr(align(8))]
struct ConfigSlot([u8; 16]);
static CONFIG_SLOT: ConfigSlot = ConfigSlot([0; 16]);

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

/// The patched (bed_z, z_top) slab; (0.0, 0.0) while unpatched.
fn slab() -> (f64, f64) {
    let base = CONFIG_SLOT.0.as_ptr() as *const u64;
    let bed_z = f64::from_bits(unsafe { core::ptr::read_volatile(base) });
    let z_top = f64::from_bits(unsafe { core::ptr::read_volatile(base.add(1)) });
    (bed_z, z_top)
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
    let xy = payload().map(|p| p.bounds()).unwrap_or([0.0; 4]);
    let (bed_z, z_top) = slab();
    let bounds = [xy[0], xy[1], xy[2], xy[3], bed_z, z_top];
    for (i, v) in bounds.iter().enumerate() {
        unsafe { core::ptr::write_unaligned((out_ptr as usize + i * 8) as *mut f64, *v) };
    }
}

/// Occupancy at (x, y, z): inside the slab and on the non-negative side
/// of the baked field. Also called directly by merge glue.
#[unsafe(no_mangle)]
pub extern "C" fn brim_sample(x: f64, y: f64, z: f64) -> f32 {
    let (bed_z, z_top) = slab();
    if !(z >= bed_z && z <= z_top) {
        return 0.0;
    }
    match payload().and_then(|view| view.sample(x, y)) {
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

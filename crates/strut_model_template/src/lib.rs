//! The strut-lattice model template: a complete, *stateless* Model ABI
//! implementation over a capsule-BVH payload that `strut_model_operator`
//! patches into a copy of this module as data segments.
//!
//! All geometry work (BVH construction, serialization) happens in the
//! operator via `strut_model_core`; this module only reads. Nothing here
//! memoizes or mutates between `sample` calls, so instances have no
//! warm-up and no state to invalidate.
//!
//! # Patch contract
//!
//! - `strut_payload_slot() -> i32` returns the address of a 4-byte slot.
//!   The operator overwrites the slot (via an active data segment) with the
//!   little-endian base address of the payload it appends in freshly
//!   reserved memory pages. The slot must be read volatilely — its compiled
//!   initializer is zero, and a folded read would make the template
//!   permanently empty.
//! - An unpatched template (slot still zero) or a corrupt payload behaves
//!   as an empty model: bounds all zero, every sample outside — the ABI's
//!   errors-read-as-outside convention.
//!
//! # Regenerating the checked-in binary
//!
//! The operator embeds a prebuilt copy of this module (see
//! `crates/operators/strut_model_operator/template/`). After changing this
//! crate or `strut_model_core`'s read side:
//!
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p strut_model_template
//! cp target/wasm32-unknown-unknown/release/strut_model_template.wasm \
//!    crates/operators/strut_model_operator/template/
//! ```

use strut_model_core::PayloadView;

/// Model-owned IO scratch region (2 * 3 f64s), per the Model ABI.
static mut IO_BUFFER: [f64; 6] = [0.0; 6];

/// The 4-byte payload-base slot the operator patches. Aligned so the u32
/// read is well-formed.
#[repr(align(4))]
struct Slot([u8; 4]);
static PAYLOAD_SLOT: Slot = Slot([0; 4]);

/// The patched payload, if the slot has been filled and the header
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

#[unsafe(no_mangle)]
pub extern "C" fn strut_payload_slot() -> i32 {
    PAYLOAD_SLOT.0.as_ptr() as i32
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
    let bounds = payload().map(|p| p.bounds()).unwrap_or([0.0; 6]);
    for (i, v) in bounds.iter().enumerate() {
        unsafe { core::ptr::write_unaligned((out_ptr as usize + i * 8) as *mut f64, *v) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sample(pos_ptr: i32) -> f32 {
    let p: [f64; 3] = core::array::from_fn(|i| unsafe {
        core::ptr::read_unaligned((pos_ptr as usize + i * 8) as *const f64)
    });
    match payload() {
        Some(view) if view.is_inside(p) => 1.0,
        _ => 0.0,
    }
}

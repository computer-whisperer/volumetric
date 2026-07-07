//! The FEA-deformation evaluator template: a wasm32 module
//! `fea_deform_operator` patches with a `deform_model_core` payload and
//! merges into an input model (via `model_merge_core`). Not a standalone
//! Model — it exports the pullback evaluator the merged module's `sample`
//! glue calls per position.
//!
//! # Patch contract
//!
//! - `deform_payload_slot() -> i32` returns the address of a 4-byte slot.
//!   The operator overwrites the slot (via an active data segment) with
//!   the little-endian base address of the payload it appends in freshly
//!   reserved memory pages, then drops this helper export. The slot must
//!   be read volatilely — its compiled initializer is zero.
//! - `deform_pull_back(x, y, z) -> i32` maps a deformed-space point to
//!   its material point: returns 1 and stores 3 f64s at the address
//!   `deform_result_ptr()` returns, or 0 when the point is outside the
//!   deformed mesh and skin (or the payload is unpatched/corrupt).
//! - `deform_result_ptr() -> i32` is a constant function the operator
//!   reads at merge time to bake the result address into its glue.
//! - `deform_bounds(i) -> f64` returns the payload's sample-domain bound
//!   `i` in the model ABI's interleaved min/max order (0 when unpatched).
//!
//! # Regenerating the checked-in binary
//!
//! The operator embeds a prebuilt copy of this module (see
//! `crates/operators/fea_deform_operator/template/`). After changing this
//! crate or `deform_model_core`'s read side:
//!
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p deform_model_template
//! cp target/wasm32-unknown-unknown/release/deform_model_template.wasm \
//!    crates/operators/fea_deform_operator/template/
//! ```

use deform_model_core::PayloadView;

/// The 4-byte payload-base slot the operator patches. Aligned so the u32
/// read is well-formed.
#[repr(align(4))]
struct Slot([u8; 4]);
static PAYLOAD_SLOT: Slot = Slot([0; 4]);

/// Where `deform_pull_back` leaves the material point for the glue.
static mut RESULT: [f64; 3] = [0.0; 3];

/// The patched payload, if the slot has been filled and the header
/// validates.
fn payload() -> Option<PayloadView<'static>> {
    // Volatile: the slot's compile-time value is zero; the operator
    // patches the bytes after compilation.
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
pub extern "C" fn deform_payload_slot() -> i32 {
    PAYLOAD_SLOT.0.as_ptr() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn deform_result_ptr() -> i32 {
    (&raw const RESULT) as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn deform_pull_back(x: f64, y: f64, z: f64) -> i32 {
    match payload().and_then(|view| view.pull_back([x, y, z])) {
        Some(material) => {
            unsafe { (&raw mut RESULT).write(material) };
            1
        }
        None => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn deform_bounds(i: i32) -> f64 {
    payload()
        .map(|view| view.bounds()[i as usize % 6])
        .unwrap_or(0.0)
}

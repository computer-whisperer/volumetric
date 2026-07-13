//! The island-mask evaluator template: a *stateless* multilinear reader
//! over a baked `voxelmask_model_core` payload in model coordinates.
//! `island_removal_operator` patches a copy of this module with the mask
//! of removed lattice points it computes by scanning its input model,
//! then merges it with the input — the merge glue evaluates the mask per
//! position to cut (or isolate) the removed geometry.
//!
//! Unlike `brim_model_template` this is NOT a standalone Model ABI
//! implementation: both operator outputs are merges with the input
//! model, so the template only provides the evaluator. Keeping it
//! ABI-free is also what lets it stay dimension-generic — the payload
//! header carries the lattice's dimensionality, while the Model ABI
//! surface (`get_dimensions`, bounds, IO buffer) comes from the input
//! model's side of the merge.
//!
//! # Patch contract
//!
//! - `island_payload_slot() -> i32` returns the address of a 4-byte
//!   slot. The operator overwrites the slot (via an active data segment)
//!   with the little-endian base address of the voxel-mask payload it
//!   appends in freshly reserved memory pages. The slot must be read
//!   volatilely — its compiled initializer is zero, and a folded read
//!   would make the mask permanently empty.
//! - `island_pos_slot() -> i32` returns the address of an 8 x f64
//!   position scratch. The merge glue stores the sample position's first
//!   `d` coordinates here (a cross-memory store — the merged module is
//!   multi-memory) before calling the evaluator.
//! - `island_sample() -> f32` evaluates the mask multilinearly at the
//!   scratch position: 1.0 deep inside removed regions, 0.0 outside,
//!   with the 0.5 level set halfway between a removed lattice point and
//!   a kept neighbor. An unpatched template (slot still zero), a corrupt
//!   payload, or a position outside the mask's bounds box reads as 0.0 —
//!   nothing removed there.
//!
//! # Regenerating the checked-in binary
//!
//! The operator embeds a prebuilt copy of this module (see
//! `crates/operators/island_removal_operator/template/`). After changing
//! this crate or `voxelmask_model_core`'s read side:
//!
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p island_model_template
//! cp target/wasm32-unknown-unknown/release/island_model_template.wasm \
//!    crates/operators/island_removal_operator/template/
//! ```

use voxelmask_model_core::{MAX_DIMS, PayloadView};

/// The 4-byte payload-base slot the operator patches. Aligned so the u32
/// read is well-formed.
#[repr(align(4))]
struct PayloadSlot([u8; 4]);
static PAYLOAD_SLOT: PayloadSlot = PayloadSlot([0; 4]);

/// The position scratch the merge glue writes into before each
/// `island_sample` call.
#[repr(align(8))]
struct PosSlot([u8; MAX_DIMS * 8]);
static POS_SLOT: PosSlot = PosSlot([0; MAX_DIMS * 8]);

/// The patched mask payload, if the slot has been filled and the header
/// validates.
fn payload() -> Option<PayloadView<'static>> {
    // Volatile: the slot's compile-time value is zero, and the whole point
    // is that the bytes change after compilation.
    let base = unsafe { core::ptr::read_volatile(PAYLOAD_SLOT.0.as_ptr() as *const u32) } as usize;
    if base == 0 {
        return None;
    }
    // payload_len lives at header offset 8; trust it only as far as
    // PayloadView's structural validation.
    let len = unsafe { core::ptr::read_volatile((base + 8) as *const u32) } as usize;
    let bytes = unsafe { core::slice::from_raw_parts(base as *const u8, len) };
    PayloadView::new(bytes).ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn island_payload_slot() -> i32 {
    PAYLOAD_SLOT.0.as_ptr() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn island_pos_slot() -> i32 {
    POS_SLOT.0.as_ptr() as i32
}

/// The mask value at the scratch position; 0.0 unpatched or outside the
/// mask's bounds box.
#[unsafe(no_mangle)]
pub extern "C" fn island_sample() -> f32 {
    let Some(view) = payload() else {
        return 0.0;
    };
    let base = POS_SLOT.0.as_ptr() as *const f64;
    // Volatile: the scratch is written externally (by the merge glue),
    // never from within this module.
    let mut pos = [0.0f64; MAX_DIMS];
    for (i, p) in pos[..view.dims()].iter_mut().enumerate() {
        *p = unsafe { core::ptr::read_volatile(base.add(i)) };
    }
    view.sample(&pos[..view.dims()]).unwrap_or(0.0)
}

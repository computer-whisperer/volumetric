//! The triangle-mesh model template: a complete, *stateless* Model ABI
//! implementation over a BVH payload that `mesh_to_model_operator` patches
//! into a copy of this module as data segments.
//!
//! All geometry work (BVH construction, texture compression,
//! serialization) happens in the operator via `trimesh_model_core`; this
//! module only reads. Nothing here memoizes or mutates between `sample`
//! calls, so instances have no warm-up and no state to invalidate.
//!
//! Besides the plain occupancy `sample`, the module exports the typed
//! channel pair `get_sample_format` / `sample_channels` (occupancy plus
//! the sRGB surface-colour trio, nearest-surface, defined everywhere).
//! The operator strips both exports when the payload has no colour
//! section, so only coloured models advertise the channels.
//!
//! # Patch contract
//!
//! - `mesh_payload_slot() -> i32` returns the address of a 4-byte slot.
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
//! `crates/operators/mesh_to_model_operator/template/`). After changing
//! this crate or `trimesh_model_core`'s read side:
//!
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p trimesh_model_template
//! cp target/wasm32-unknown-unknown/release/trimesh_model_template.wasm \
//!    crates/operators/mesh_to_model_operator/template/
//! ```

use trimesh_model_core::PayloadView;

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
pub extern "C" fn mesh_payload_slot() -> i32 {
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

/// Typed-channel format: occupancy plus the sRGB surface-color triple.
/// The operator strips this export (and `sample_channels`) from models
/// whose payload has no colour section, so only actually-colored models
/// advertise the channels.
#[unsafe(no_mangle)]
pub extern "C" fn get_sample_format() -> i64 {
    static FORMAT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::sample_format_reply(&FORMAT, || volumetric_abi::SampleFormat {
        channels: core::iter::once(volumetric_abi::SampleChannel {
            name: "occupancy".to_string(),
            kind: volumetric_abi::ChannelKind::Occupancy,
        })
        .chain(volumetric_abi::COLOR_CHANNEL_NAMES.iter().map(|name| {
            volumetric_abi::SampleChannel {
                name: name.to_string(),
                kind: volumetric_abi::ChannelKind::Custom(
                    volumetric_abi::COLOR_SRGB_CHANNEL_KIND.to_string(),
                ),
            }
        }))
        .collect(),
    })
}

/// Occupancy plus nearest-surface sRGB color, one f32 per channel. The
/// color triple is NaN when the template is unpatched or the payload has
/// no colour section (errors read as "no value").
#[unsafe(no_mangle)]
pub extern "C" fn sample_channels(pos_ptr: i32, out_ptr: i32) {
    let p: [f64; 3] = core::array::from_fn(|i| unsafe {
        core::ptr::read_unaligned((pos_ptr as usize + i * 8) as *const f64)
    });
    let (occupancy, color) = match payload() {
        Some(view) => (
            if view.is_inside(p) { 1.0f32 } else { 0.0 },
            view.nearest_color(p).unwrap_or([f32::NAN; 3]),
        ),
        None => (0.0, [f32::NAN; 3]),
    };
    let row = [occupancy, color[0], color[1], color[2]];
    for (i, v) in row.iter().enumerate() {
        unsafe { core::ptr::write_unaligned((out_ptr as usize + i * 4) as *mut f32, *v) };
    }
}

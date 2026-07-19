//! Stateless evaluator shell for a baked truncated signed distance field.
//!
//! `sdf_operator` patches an `ndfield_model_core` payload into this module,
//! merges it with the source model, and exports this module's memory as the
//! resulting model's public memory. The merge glue keeps the source model's
//! exact occupancy while this evaluator supplies the `signed_distance`
//! custom channel.
//!
//! # Patch contract
//!
//! - `sdf_payload_slot() -> i32` returns the address of the four-byte payload
//!   base slot. The operator removes this helper export and patches the slot.
//! - `sdf_sample(pos_ptr) -> f32` samples the patched field at an N-dimensional
//!   f64 position in this module's memory.
//! - `get_io_ptr()` returns a buffer large enough for the maximum supported
//!   position/bounds row and channel output.
//!
//! Regenerate the operator's checked-in template after changing this crate or
//! the read side of `ndfield_model_core`:
//!
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p sdf_model_template
//! cp target/wasm32-unknown-unknown/release/sdf_model_template.wasm \
//!    crates/operators/sdf_operator/template/
//! ```

use std::sync::OnceLock;

use ndfield_model_core::{MAX_DIMS, PayloadView};
use volumetric_abi::{
    ChannelKind, SIGNED_DISTANCE_CHANNEL_NAME, SampleChannel, SampleFormat, TSDF_CHANNEL_KIND,
};

/// Enough for 2*d f64 values at the format's maximum dimensionality.
static mut IO_BUFFER: [f64; MAX_DIMS * 2] = [0.0; MAX_DIMS * 2];

#[repr(align(4))]
struct PayloadSlot([u8; 4]);
static PAYLOAD_SLOT: PayloadSlot = PayloadSlot([0; 4]);

fn payload() -> Option<PayloadView<'static>> {
    let base = unsafe { core::ptr::read_volatile(PAYLOAD_SLOT.0.as_ptr() as *const u32) } as usize;
    if base == 0 {
        return None;
    }
    let len = unsafe { core::ptr::read_volatile((base + 8) as *const u32) } as usize;
    let bytes = unsafe { core::slice::from_raw_parts(base as *const u8, len) };
    PayloadView::new(bytes).ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn sdf_payload_slot() -> i32 {
    PAYLOAD_SLOT.0.as_ptr() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn get_io_ptr() -> i32 {
    (&raw mut IO_BUFFER) as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn sdf_sample(pos_ptr: i32) -> f32 {
    let Some(field) = payload() else {
        return 0.0;
    };
    let mut position = [0.0; MAX_DIMS];
    for (axis, coordinate) in position[..field.dimensions()].iter_mut().enumerate() {
        *coordinate =
            unsafe { core::ptr::read_unaligned((pos_ptr as usize + axis * 8) as *const f64) };
    }
    field.sample(&position[..field.dimensions()])
}

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
                name: SIGNED_DISTANCE_CHANNEL_NAME.to_string(),
                kind: ChannelKind::Custom(TSDF_CHANNEL_KIND.to_string()),
            },
        ],
    })
}

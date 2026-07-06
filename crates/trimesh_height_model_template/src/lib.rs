//! The mesh height-query model template: a *stateless* 2D Model ABI
//! implementation that answers `sample(u, v)` with the height of a
//! triangle mesh's surface over that lateral point, via the same BVH
//! payload `mesh_to_model_operator` uses (`trimesh_model_core`).
//! `mesh_height_operator` patches the payload and a small query config
//! into a copy of this module as data segments.
//!
//! # Patch contract
//!
//! - `mesh_payload_slot() -> i32` returns the address of a 4-byte slot the
//!   operator overwrites (via an active data segment) with the
//!   little-endian base address of the payload it appends in freshly
//!   reserved memory pages. Read volatilely — the compiled initializer is
//!   zero.
//! - `height_config_slot() -> i32` returns the address of a 12-byte slot:
//!   `axis: u32` (0/1/2 = x/y/z), `surface: u32` (0 = top, 1 = bottom),
//!   `miss: f32` (the sample value where the query line misses the mesh).
//!   The operator always patches all 12 bytes.
//! - An unpatched template or a corrupt payload behaves as an empty model:
//!   bounds all zero, every sample 0.0.
//!
//! The model's two coordinates are the mesh's non-height axes in ascending
//! order (axis z → (x, y), axis y → (x, z), axis x → (y, z)), matching the
//! FEA target-map convention. Its bounds are the mesh bounds projected onto
//! those axes.
//!
//! # Regenerating the checked-in binary
//!
//! The operator embeds a prebuilt copy of this module (see
//! `crates/operators/mesh_height_operator/template/`). After changing this
//! crate or `trimesh_model_core`'s read side:
//!
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p trimesh_height_model_template
//! cp target/wasm32-unknown-unknown/release/trimesh_height_model_template.wasm \
//!    crates/operators/mesh_height_operator/template/
//! ```

use trimesh_model_core::PayloadView;

/// Model-owned IO scratch region (2 * 2 f64s), per the Model ABI.
static mut IO_BUFFER: [f64; 4] = [0.0; 4];

/// The 4-byte payload-base slot the operator patches.
#[repr(align(4))]
struct Slot([u8; 4]);
static PAYLOAD_SLOT: Slot = Slot([0; 4]);

/// The 12-byte query-config slot the operator patches: axis u32,
/// surface u32, miss f32.
#[repr(align(4))]
struct ConfigSlot([u8; 12]);
static CONFIG_SLOT: ConfigSlot = ConfigSlot([0; 12]);

fn payload() -> Option<PayloadView<'static>> {
    // Volatile: the slots' compile-time values are zero, and the whole
    // point is that the bytes change after compilation.
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

/// `(axis, top, miss)` from the patched config slot.
fn config() -> (usize, bool, f32) {
    let ptr = CONFIG_SLOT.0.as_ptr();
    let axis = unsafe { core::ptr::read_volatile(ptr as *const u32) };
    let surface = unsafe { core::ptr::read_volatile(ptr.add(4) as *const u32) };
    let miss = unsafe { core::ptr::read_volatile(ptr.add(8) as *const f32) };
    ((axis as usize).min(2), surface == 0, miss)
}

#[unsafe(no_mangle)]
pub extern "C" fn mesh_payload_slot() -> i32 {
    PAYLOAD_SLOT.0.as_ptr() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn height_config_slot() -> i32 {
    CONFIG_SLOT.0.as_ptr() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn get_dimensions() -> u32 {
    2
}

#[unsafe(no_mangle)]
pub extern "C" fn get_io_ptr() -> i32 {
    (&raw mut IO_BUFFER) as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn get_bounds(out_ptr: i32) {
    let (axis, _, _) = config();
    let (u, v) = lateral(axis);
    let b = payload().map(|p| p.bounds()).unwrap_or([0.0; 6]);
    let out = [b[u * 2], b[u * 2 + 1], b[v * 2], b[v * 2 + 1]];
    for (i, val) in out.iter().enumerate() {
        unsafe { core::ptr::write_unaligned((out_ptr as usize + i * 8) as *mut f64, *val) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sample(pos_ptr: i32) -> f32 {
    let lat: [f64; 2] = core::array::from_fn(|i| unsafe {
        core::ptr::read_unaligned((pos_ptr as usize + i * 8) as *const f64)
    });
    let (axis, top, miss) = config();
    match payload().and_then(|view| view.height_at(lat, axis, top)) {
        Some(h) => h as f32,
        None => miss,
    }
}

/// The two non-height axes in ascending order.
fn lateral(axis: usize) -> (usize, usize) {
    match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

//! The lattice evaluator template: a wasm32 module `lattice_operator`
//! patches with its config and merges into a density model (via
//! `model_merge_core`). Not a standalone Model — it exports a single
//! evaluator the merged module's `sample` glue calls per position.
//!
//! # Patch contract
//!
//! - `lattice_config_slot() -> i32` returns the address of an 8-byte slot:
//!   `kind: u32` (the `lattice_model_core::LatticeKind` discriminant),
//!   `cell_size: f32` (pattern period in model units). The operator
//!   overwrites it via an active data segment and drops this helper export.
//! - `lattice_sample(x, y, z, density) -> f32` evaluates the configured
//!   lattice: 1.0 inside, 0.0 outside. Unpatched (zeroed) config reads as
//!   cell size 0 and answers 0.0 everywhere.
//!
//! # Regenerating the checked-in binary
//!
//! The operator embeds a prebuilt copy of this module (see
//! `crates/operators/lattice_operator/template/`). After changing this
//! crate or `lattice_model_core`:
//!
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p lattice_model_template
//! cp target/wasm32-unknown-unknown/release/lattice_model_template.wasm \
//!    crates/operators/lattice_operator/template/
//! ```

use lattice_model_core::{LatticeKind, lattice_occupied};

/// The 8-byte config slot the operator patches: kind u32, cell_size f32.
#[repr(align(4))]
struct ConfigSlot([u8; 8]);
static CONFIG_SLOT: ConfigSlot = ConfigSlot([0; 8]);

/// The patch address of the config slot (dropped from the merged module).
#[unsafe(no_mangle)]
pub extern "C" fn lattice_config_slot() -> i32 {
    CONFIG_SLOT.0.as_ptr() as i32
}

/// Evaluate the configured lattice at a position with the local density.
#[unsafe(no_mangle)]
pub extern "C" fn lattice_sample(x: f64, y: f64, z: f64, density: f32) -> f32 {
    // Volatile: the slot's compile-time value is zero; the operator patches
    // the bytes after compilation.
    let base = CONFIG_SLOT.0.as_ptr();
    let kind_raw = unsafe { core::ptr::read_volatile(base as *const u32) };
    let cell_size = unsafe { core::ptr::read_volatile(base.add(4) as *const f32) };
    let Some(kind) = LatticeKind::from_u32(kind_raw) else {
        return 0.0;
    };
    if lattice_occupied(kind, [x, y, z], f64::from(cell_size), density) {
        1.0
    } else {
        0.0
    }
}

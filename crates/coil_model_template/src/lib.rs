//! The coil pull-back evaluator template: a wasm32 module `coil_operator`
//! patches with its config and merges into an input model (via
//! `model_merge_core`). Not a standalone Model — it exports the
//! spiral-unwind evaluator the merged module's `sample` glue calls per
//! position.
//!
//! # Coordinate convention
//!
//! The input model is a flat sheet rolled up like a carpet. Its x axis
//! becomes arc length along an Archimedean spiral around the world y axis
//! (x min at the spiral's inner end, wrapping counterclockwise from +x
//! toward +z); its z axis becomes radial depth within each wrap (the z min
//! face on the inside); y passes through as the coil axis. The radial
//! advance per turn is `sheet thickness + gap`, so consecutive wraps never
//! intersect and `gap` survives as clearance between them.
//!
//! The spiral's mid-surface radius at unwound angle `phi` is
//! `r(phi) = inner_radius + b*phi` with `b = pitch / 2*pi`; arc length is
//! approximated by `s(phi) = integral of r = inner_radius*phi + b*phi^2/2`
//! (the `sqrt(1 + (b/r)^2)` stretch factor is dropped — well under 1% for
//! any pitch much smaller than the bore diameter).
//!
//! # Patch contract
//!
//! - `coil_config_slot() -> i32` returns the address of a 16-byte slot:
//!   `inner_radius: f64` (spiral start radius, metres) then `gap: f64`
//!   (radial clearance between wraps). The operator overwrites the slot
//!   via an active data segment and drops this helper export. Unpatched
//!   (zeroed) config pulls every point back to the sentinel and reports
//!   zero bounds.
//! - `coil_pull_back(x, y, z, x0, x1, z0, z1)` maps a world point to its
//!   flat sheet point, given the sheet's x and z bounds (read from the
//!   input model at sample time). Points with no material point — the
//!   bore, the gap between wraps, past the sheet's end — map to a
//!   sentinel far outside the sheet bounds, so the glue can sample the
//!   input unconditionally and let it answer "outside".
//! - `coil_result_ptr() -> i32` is a constant function the operator reads
//!   at merge time to bake the 3-f64 result address into its glue.
//! - `coil_bound(i, x0, x1, z0, z1, y_min, y_max) -> f64` returns world
//!   bound `i` in the model ABI's interleaved min/max order.
//!
//! # Regenerating the checked-in binary
//!
//! The operator embeds a prebuilt copy of this module (see
//! `crates/operators/coil_operator/template/`). After changing this crate:
//!
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p coil_model_template
//! cp target/wasm32-unknown-unknown/release/coil_model_template.wasm \
//!    crates/operators/coil_operator/template/
//! ```

const TAU: f64 = core::f64::consts::TAU;

/// The 16-byte config slot the operator patches: `[inner_radius, gap]`,
/// little-endian f64s. Aligned so the reads are well-formed.
#[repr(align(8))]
struct Slot([u8; 16]);
static CONFIG_SLOT: Slot = Slot([0; 16]);

/// Where `coil_pull_back` leaves the flat sheet point for the glue.
static mut RESULT: [f64; 3] = [0.0; 3];

/// The patched `(inner_radius, gap)`. Volatile: the slot's compile-time
/// value is zero; the operator patches the bytes after compilation.
fn config() -> (f64, f64) {
    let base = CONFIG_SLOT.0.as_ptr() as *const f64;
    unsafe {
        (
            core::ptr::read_volatile(base),
            core::ptr::read_volatile(base.add(1)),
        )
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn coil_config_slot() -> i32 {
    CONFIG_SLOT.0.as_ptr() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn coil_result_ptr() -> i32 {
    (&raw const RESULT) as i32
}

/// Unwind a world point to its flat sheet point, or `None` for points the
/// spiral never covers (the bore, the inter-wrap gap, or an invalid
/// config).
fn unwind(x: f64, y: f64, z: f64, x0: f64, z0: f64, z1: f64) -> Option<[f64; 3]> {
    let (inner_radius, gap) = config();
    let thickness = z1 - z0;
    let pitch = thickness + gap;
    // Strictly-positive check that also rejects NaN (which fails every
    // comparison): only `Some(Greater)` passes.
    let positive = |v: f64| v.partial_cmp(&0.0) == Some(std::cmp::Ordering::Greater);
    if !positive(inner_radius) || !positive(pitch) {
        return None;
    }
    let b = pitch / TAU;

    let radius = (x * x + z * z).sqrt();
    let mut angle = z.atan2(x);
    if angle < 0.0 {
        angle += TAU;
    }

    // The wrap this radius falls in: r(angle + TAU*wraps) <= radius.
    let wraps = ((radius - inner_radius - b * angle) / pitch).floor();
    let total = angle + TAU * wraps;
    if total < 0.0 {
        // Inside the bore, before the spiral's inner end.
        return None;
    }

    // Radial depth into the wrap, in [0, pitch); past the sheet's
    // thickness is the clearance gap between wraps.
    let depth = radius - (inner_radius + b * total);
    if depth > thickness {
        return None;
    }

    let arc = inner_radius * total + 0.5 * b * total * total;
    Some([x0 + arc, y, z0 + depth])
}

#[unsafe(no_mangle)]
pub extern "C" fn coil_pull_back(x: f64, y: f64, z: f64, x0: f64, x1: f64, z0: f64, z1: f64) {
    let flat = unwind(x, y, z, x0, z0, z1).unwrap_or_else(|| {
        // A point guaranteed outside the sheet's x and z ranges, whatever
        // their absolute scale.
        [x0 - (x1 - x0).abs() - 1.0, y, z0 - 1.0]
    });
    unsafe { (&raw mut RESULT).write(flat) };
}

#[unsafe(no_mangle)]
pub extern "C" fn coil_bound(
    i: i32,
    x0: f64,
    x1: f64,
    z0: f64,
    z1: f64,
    y_min: f64,
    y_max: f64,
) -> f64 {
    let (inner_radius, gap) = config();
    let thickness = (z1 - z0).max(0.0);
    let pitch = thickness + gap;
    let length = (x1 - x0).max(0.0);
    let outer = if inner_radius > 0.0 && pitch > 0.0 {
        let b = pitch / TAU;
        // The unwound angle where the sheet ends: s(total) = length.
        let total = ((inner_radius * inner_radius + 2.0 * b * length).sqrt() - inner_radius) / b;
        inner_radius + b * total + thickness
    } else {
        0.0
    };
    match i {
        0 => -outer,
        1 => outer,
        2 => y_min,
        3 => y_max,
        4 => -outer,
        5 => outer,
        _ => 0.0,
    }
}

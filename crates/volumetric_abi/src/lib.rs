//! The shared contract between the volumetric host and operator WASM modules.
//!
//! Both sides depend on this crate, so the operator metadata types are a
//! single definition instead of a hand-copied convention (CBOR enum encoding
//! is name-based; a rename that only lands on one side decodes as garbage).
//!
//! # Operator ABI
//!
//! Exports an operator must provide:
//! - `run()` — execute; read inputs and post outputs via the host imports
//! - `get_metadata() -> i64` — `(ptr, len)` of CBOR-encoded [`OperatorMetadata`],
//!   packed as `ptr | (len << 32)` (see [`pack_ptr_len`])
//!
//! Imports the host provides (module `"host"`), wrapped safely in [`host`]:
//! - `get_input_len(idx: i32) -> u32`
//! - `get_input_data(idx: i32, ptr: i32, len: i32)`
//! - `get_input_count() -> i32` — optional; how many input slots the step
//!   carries (see [`host::input_count`] and [`OperatorMetadata::variadic_input`])
//! - `post_output(idx: i32, ptr: i32, len: i32)`
//! - `post_error(ptr: i32, len: i32)` — optional; a run that posts an error
//!   fails with the message instead of returning outputs
//! - `post_warning(ptr: i32, len: i32)` — optional; non-fatal advisory the
//!   host attaches to the step's outputs (see [`host::post_warning`])
//! - `cancelled() -> i32` — optional; cooperative-cancellation poll for
//!   long-running operators (see [`host::cancelled`])
//!
//! ## Model-input sampling imports
//!
//! Operators that need to *evaluate* a `ModelWASM` input (rather than
//! rewrite it) use these imports; the host instantiates the input natively
//! and services the calls. All three fail soft (return 0) when the slot
//! doesn't hold a usable model. Only meaningful during `run()` — during
//! `get_metadata()` there are no inputs and they always fail.
//! - `input_model_dimensions(idx: i32) -> i32` — the model's dimension
//!   count `n`, or 0 on failure
//! - `input_model_bounds(idx: i32, out_ptr: i32) -> i32` — writes `2 * n`
//!   interleaved f64s `[min_0, max_0, ...]`; returns 1 on success
//! - `input_model_sample(idx: i32, pos_ptr: i32, count: i32, out_ptr: i32)
//!   -> i32` — reads `count * n` f64s at `pos_ptr`, writes `count` f32
//!   occupancies at `out_ptr` (classify with [`is_occupied`]; individual
//!   failed samples read 0.0 per the ABI error convention); returns 1 on
//!   success, 0 when the slot is not a model or a range is out of bounds
//!
//! # Model ABI (N-dimensional)
//!
//! Model WASM blobs (operator inputs/outputs of type `ModelWASM`) export:
//! - `get_dimensions() -> u32` — number of dimensions `n`
//! - `get_io_ptr() -> i32` — pointer to a model-owned IO scratch buffer of at
//!   least `2 * n` f64s. Callers use it as the position buffer for `sample`
//!   and the output buffer for `get_bounds`; the model's own layout decides
//!   where it lives, so callers never write to assumed offsets.
//! - `get_bounds(out_ptr: i32)` — writes `2 * n` interleaved f64s
//!   `[min_0, max_0, min_1, max_1, ...]` at `out_ptr`
//! - `sample(pos_ptr: i32) -> f32` — reads `n` f64s at `pos_ptr`, returns
//!   the occupancy value for that position (see below)
//! - `memory` — the linear memory the pointers above refer to
//!
//! `get_bounds` is the finite enclosure consumers use to traverse occupancy
//! geometry; it is not a clipping rule or a declaration that samples outside
//! the box are invalid. `sample` and `sample_channels` accept any finite
//! position. Whether an extra channel remains meaningful outside occupancy or
//! outside the geometry bounds is part of that channel kind's semantics.
//!
//! `sample` and `get_bounds` accept any pointer to a large-enough region of
//! the model's memory, and the model may clobber that region during the call
//! (transform wrappers rewrite the position in place). A caller that needs
//! the position after a call must keep its own copy.
//!
//! ## Sample semantics: occupancy, not a distance field
//!
//! `sample` returns an *occupancy* value. Only the classification against
//! [`OCCUPANCY_THRESHOLD`] is meaningful — never the magnitude:
//! - A point is inside iff `value > 0.5`; consumers classify with
//!   [`is_occupied`] and must not invent other thresholds.
//! - Models return the canonical values `1.0` (inside) and `0.0` (outside).
//! - A failed sample is reported as `0.0`: errors read as "outside".
//!
//! Models are deliberately *not* required to return signed distance. An
//! open composition chain can't preserve distance-ness (booleans, sweeps,
//! and non-uniform transforms all break it), and a magnitude consumers can't
//! trust is worse than none. Richer per-sample data goes through declared
//! channels instead.
//!
//! ## Optional: typed sample channels
//!
//! A model whose samples carry more than inside/outside (e.g. a material
//! density for variable-density printing) declares a per-sample format:
//! - `get_sample_format() -> i64` — `(ptr, len)` of CBOR-encoded
//!   [`SampleFormat`], packed as `ptr | (len << 32)` (see [`pack_ptr_len`]).
//!   A model without this export has the default format,
//!   [`SampleFormat::default`] (a single [`ChannelKind::Occupancy`] channel).
//! - `sample_channels(pos_ptr: i32, out_ptr: i32)` — reads `n` f64s at
//!   `pos_ptr`, writes one f32 per declared channel at `out_ptr` (any
//!   large-enough region of model memory; the clobber rule above applies).
//!   Required iff the format declares more than one channel.
//!
//! Channel 0 is always [`ChannelKind::Occupancy`] and must agree with what
//! `sample` returns at the same position — every consumer can classify
//! inside/outside through plain `sample` without ever reading the format.
//! Extra channels are strictly additive: each [`ChannelKind`] documents its
//! own value semantics, and a consumer ignores channels it doesn't
//! recognize. Operators that don't understand channels emit occupancy-only
//! models (channels are dropped, never silently mangled); position-only
//! wrappers like transforms forward the format and wrap `sample_channels`
//! exactly like `sample`.
//!
//! ## Optional: catalog metadata
//!
//! A model authored as a catalog entry (rather than produced mid-pipeline)
//! may export `get_metadata() -> i64` exactly like an operator — `(ptr,
//! len)` of CBOR-encoded [`OperatorMetadata`], packed with
//! [`pack_ptr_len`] — with empty `inputs`/`outputs` and the display fields
//! filled in. Hosts read it to list the model in Add catalogs; execution
//! never calls it. Derived blobs may carry a stale copy inherited from the
//! module they were built from (transform wrappers pass unknown exports
//! through), so hosts must only treat it as authoritative for modules
//! loaded from source files, never for pipeline outputs.

use std::sync::OnceLock;

pub mod annotations;
pub mod f64_map;
pub mod fea;
pub mod lua_parameters;
pub mod subspace;
pub mod threading;
pub mod trimesh;
pub mod wgsl_parameters;

/// The single inside/outside threshold for occupancy samples.
///
/// A sample is "inside" iff it is strictly greater than this. Every host,
/// operator, and generated model classifies with this one rule (via
/// [`is_occupied`]); models emit the canonical values `1.0`/`0.0`.
pub const OCCUPANCY_THRESHOLD: f32 = 0.5;

/// Classify an occupancy sample: `true` iff the point is inside.
///
/// This is the only correct way to interpret a `sample` return value.
/// `NaN` classifies as outside, matching the error convention (failed
/// samples report `0.0`).
#[inline]
pub fn is_occupied(sample: f32) -> bool {
    sample > OCCUPANCY_THRESHOLD
}

/// What one per-sample channel means. Each kind defines its own value
/// semantics; consumers ignore kinds they don't recognize.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum ChannelKind {
    /// Inside/outside classification: canonical `1.0`/`0.0`, classified with
    /// [`is_occupied`]. Channel 0 of every format is this kind, and must
    /// agree with the model's plain `sample` export.
    Occupancy,
    /// Fraction of solid material in `[0.0, 1.0]` (e.g. infill fraction for
    /// variable-density printing). Only meaningful where occupancy says
    /// inside.
    Density,
    /// An application-defined kind. Namespace the string (e.g.
    /// `"myapp.temperature"`) to avoid collisions. The kind defines its own
    /// domain; custom channels may remain meaningful where occupancy is zero
    /// and outside the model's advertised geometry bounds.
    Custom(String),
}

/// Conventional channel name emitted by the SDF-generation operator.
pub const SIGNED_DISTANCE_CHANNEL_NAME: &str = "signed_distance";

/// Namespaced [`ChannelKind::Custom`] identifier for one component of a
/// surface-color triple. Values are sRGB in `[0.0, 1.0]`, and the color at
/// a position is the color of the *nearest surface* of the model — defined
/// everywhere (inside, outside, and beyond the geometry bounds), so
/// consumers may sample it at meshed vertices that sit on or fractionally
/// off the boundary. A model declares all three components, named by
/// [`COLOR_CHANNEL_NAMES`] in r, g, b order; consumers locate the triple
/// with [`SampleFormat::color_trio`].
pub const COLOR_SRGB_CHANNEL_KIND: &str = "volumetric.color_srgb.v1";

/// Conventional channel names for the sRGB surface-color triple, in
/// declaration order (r, g, b).
pub const COLOR_CHANNEL_NAMES: [&str; 3] = ["color_r", "color_g", "color_b"];

/// Namespaced [`ChannelKind::Custom`] identifier for a truncated signed
/// distance field. Values are world-space distance, negative inside and
/// positive outside, clamped to a generator-declared symmetric band.
pub const TSDF_CHANNEL_KIND: &str = "volumetric.tsdf.v1";

/// One declared per-sample channel.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct SampleChannel {
    pub name: String,
    pub kind: ChannelKind,
}

/// A model's declared per-sample format: what `sample_channels` writes, one
/// f32 per channel, in order. Returned (CBOR-encoded) by the optional
/// `get_sample_format()` model export; models without the export have the
/// [`Default`] format.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct SampleFormat {
    pub channels: Vec<SampleChannel>,
}

impl Default for SampleFormat {
    /// The implicit format of a model with no `get_sample_format` export:
    /// a single occupancy channel.
    fn default() -> Self {
        Self {
            channels: vec![SampleChannel {
                name: "occupancy".to_string(),
                kind: ChannelKind::Occupancy,
            }],
        }
    }
}

impl SampleFormat {
    /// Check the structural rules: at least one channel, channel 0 is
    /// [`ChannelKind::Occupancy`], and channel names are non-empty and
    /// unique.
    pub fn validate(&self) -> Result<(), String> {
        let Some(first) = self.channels.first() else {
            return Err("sample format declares no channels".to_string());
        };
        if first.kind != ChannelKind::Occupancy {
            return Err(format!(
                "sample format channel 0 must be Occupancy, got {:?}",
                first.kind
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for channel in &self.channels {
            if channel.name.is_empty() {
                return Err("sample format has a channel with an empty name".to_string());
            }
            if !seen.insert(channel.name.as_str()) {
                return Err(format!("duplicate sample channel name {:?}", channel.name));
            }
        }
        Ok(())
    }

    /// Locate the sRGB surface-color triple: the channel indices of
    /// [`COLOR_CHANNEL_NAMES`] in r, g, b order, when all three are
    /// declared with kind [`COLOR_SRGB_CHANNEL_KIND`].
    pub fn color_trio(&self) -> Option<[usize; 3]> {
        let index_of = |name: &str| {
            self.channels.iter().position(|c| {
                c.name == name
                    && matches!(&c.kind, ChannelKind::Custom(kind) if kind == COLOR_SRGB_CHANNEL_KIND)
            })
        };
        Some([
            index_of(COLOR_CHANNEL_NAMES[0])?,
            index_of(COLOR_CHANNEL_NAMES[1])?,
            index_of(COLOR_CHANNEL_NAMES[2])?,
        ])
    }
}

/// CBOR-encode a sample format (the payload `get_sample_format()` points at).
pub fn encode_sample_format(format: &SampleFormat) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(format, &mut out)
        .expect("sample format CBOR serialization should not fail");
    out
}

/// Decode and structurally validate a `get_sample_format()` payload.
pub fn decode_sample_format(bytes: &[u8]) -> Result<SampleFormat, String> {
    let format: SampleFormat = ciborium::de::from_reader(std::io::Cursor::new(bytes))
        .map_err(|e| format!("failed to decode sample format CBOR: {e}"))?;
    format.validate()?;
    Ok(format)
}

/// Input slot declaration in an operator's metadata.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum OperatorMetadataInput {
    /// A model WASM blob (N-dimensional model ABI).
    ModelWASM,
    /// A CBOR-encoded configuration blob.
    ///
    /// The `String` is a CDDL snippet describing the expected CBOR structure.
    ///
    /// v0 convention (current host support): a single record/map like:
    /// `{ dx: float, dy: float, dz: float }`.
    ///
    /// The host UI uses this to generate widgets and encodes a CBOR map from
    /// field names to primitive values.
    CBORConfiguration(String),
    /// A Lua script source input.
    ///
    /// The `String` is a template/stub script showing the required function
    /// signatures. The host UI displays a multiline text editor pre-populated
    /// with this template. The script is passed as UTF-8 bytes to the operator.
    LuaSource(String),
    /// A WGSL script source input.
    ///
    /// Same host contract as [`OperatorMetadataInput::LuaSource`] — the
    /// `String` is a template/stub module, the host shows a text editor, and
    /// the script travels as UTF-8 bytes — but the source language is the
    /// WGSL model dialect (see `wgsl_script_operator`).
    WgslSource(String),
    /// A CBOR-encoded flat map from non-empty strings to finite f64 values.
    ///
    /// This is generic project data: hosts may offer an inline editor or
    /// route an `F64Map` asset produced elsewhere in the DAG. Any names,
    /// defaults, and ranges shown by a consumer are schema hints rather than
    /// part of the map's wire representation.
    F64Map,
    /// Raw binary data input (e.g., STL file data).
    ///
    /// The host UI should display a file picker allowing the user to select a
    /// file. The file contents are passed as raw bytes to the operator.
    Blob,
    /// A vector of f64 values with specified dimension.
    ///
    /// The `usize` specifies the expected dimension (e.g., 3 for vec3).
    /// The host UI allows either literal input (drag values) or asset
    /// reference. Data is encoded as raw bytes (8 bytes per f64,
    /// little-endian).
    VecF64(usize),
    /// A CBOR-encoded FEA mesh (explicit node positions, element
    /// connectivity, and named attribute arrays — not a sampleable field).
    ///
    /// The concrete schema ships with the first mesh-producing operator;
    /// the host UI offers a picker over FEA-mesh-typed assets.
    FeaMesh,
    /// A CBOR-encoded general-purpose triangle mesh (see [`crate::trimesh`]);
    /// explicit data with no manifold requirement, not a sampleable field.
    TriMesh,
    /// A CBOR-encoded affine subspace with an orthonormal chart (see
    /// [`crate::subspace`]); explicit data, not a sampleable field.
    Subspace,
}

/// Output slot declaration in an operator's metadata.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum OperatorMetadataOutput {
    /// A model WASM blob (N-dimensional model ABI).
    ModelWASM,
    /// A CBOR-encoded FEA mesh (see [`OperatorMetadataInput::FeaMesh`]).
    ///
    /// Unlike `ModelWASM`, this is explicit data: hosts must not feed it to
    /// the model executor (there is nothing to sample).
    FeaMesh,
    /// A CBOR-encoded triangle mesh (see [`OperatorMetadataInput::TriMesh`]);
    /// explicit data, never fed to the model executor.
    TriMesh,
    /// A CBOR-encoded affine subspace (see [`OperatorMetadataInput::Subspace`]);
    /// explicit data, never fed to the model executor.
    Subspace,
    /// A CBOR-encoded flat string-to-f64 map (see [`f64_map`]).
    F64Map,
}

/// Metadata a module returns from `get_metadata()`, CBOR-encoded.
///
/// Operators must export this; models may (with empty `inputs`/`outputs`)
/// so catalogs can display them without side tables. The display fields
/// are what host catalogs render; all of them default to empty so metadata
/// from modules built before they existed still decodes, and hosts treat
/// empty as "not declared".
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct OperatorMetadata {
    pub name: String,
    pub version: String,
    /// Human-readable catalog name (e.g. "Translate"); hosts fall back to
    /// `name` when empty.
    #[serde(default)]
    pub display_name: String,
    /// One-line summary for catalog rows/cards.
    #[serde(default)]
    pub description: String,
    /// Free-form catalog grouping (e.g. "Transforms"); hosts group entries
    /// by the verbatim string.
    #[serde(default)]
    pub category: String,
    /// Monochrome SVG icon source (24×24 viewBox, `currentColor` paint,
    /// lucide-style strokes); hosts tint it like a built-in icon and fall
    /// back to a stock glyph when empty. The [`icon_svg!`] macro wraps
    /// bare shape elements in the canonical document header.
    #[serde(default)]
    pub icon_svg: String,
    /// Full self-description as markdown (README style): what the operator
    /// models, each configuration parameter's physical meaning, units, and
    /// typical values, output fields, and failure modes. Rendered by host
    /// UIs and printed by the CLI; empty means undocumented. Convention:
    /// the crate keeps one `README.md` serving as both the rustdoc header
    /// (`#![doc = include_str!("../README.md")]`) and this field, so the
    /// docs cannot drift from the module.
    #[serde(default)]
    pub docs: String,
    pub inputs: Vec<OperatorMetadataInput>,
    /// Index into `inputs` of the one slot that accepts one or more values
    /// (`None`: every slot takes exactly one). A step wires as many inputs
    /// as it likes at that position and the slots after it shift along, so
    /// a step's input count is `inputs.len() - 1 + k` for `k >= 1` entries
    /// in the block; the helpers below own that arithmetic. The operator
    /// learns `k` from [`host::input_count`] and skips empty (unwired)
    /// entries. Defaulted so metadata built before the field existed still
    /// decodes as fixed-arity.
    #[serde(default)]
    pub variadic_input: Option<usize>,
    /// Human-readable labels for `inputs`, parallel by index; hosts show
    /// `input_names[i]` next to input slot `i`. Defaulted so metadata from
    /// operators built before this field existed still decodes (they get an
    /// empty list — hosts fall back to positional labels).
    #[serde(default)]
    pub input_names: Vec<String>,
    pub outputs: Vec<OperatorMetadataOutput>,
    /// Human-readable labels for `outputs`, parallel by index (e.g. "Ink",
    /// "Plate"). Hosts label output slots with them and derive default
    /// asset ids for multi-output operators. Defaulted like `input_names`,
    /// and single-output operators normally leave it empty.
    #[serde(default)]
    pub output_names: Vec<String>,
}

impl OperatorMetadata {
    /// The name catalogs display: `display_name`, falling back to `name`
    /// for modules that don't declare one.
    pub fn catalog_name(&self) -> &str {
        if self.display_name.is_empty() {
            &self.name
        } else {
            &self.display_name
        }
    }

    /// The declared label of input slot `idx`, if the operator provided a
    /// non-empty one.
    pub fn input_name(&self, idx: usize) -> Option<&str> {
        self.input_names
            .get(idx)
            .map(String::as_str)
            .filter(|name| !name.is_empty())
    }

    /// The declared label of output slot `idx`, if the operator provided a
    /// non-empty one.
    pub fn output_name(&self, idx: usize) -> Option<&str> {
        self.output_names
            .get(idx)
            .map(String::as_str)
            .filter(|name| !name.is_empty())
    }

    /// The variadic slot index, when it names a declared slot.
    pub fn variadic_slot(&self) -> Option<usize> {
        self.variadic_input.filter(|&slot| slot < self.inputs.len())
    }

    /// Whether a step carrying `count` inputs fits this declaration: exactly
    /// one per slot, or at least one per slot when a slot is variadic.
    pub fn accepts_input_count(&self, count: usize) -> bool {
        match self.variadic_slot() {
            Some(_) => count >= self.inputs.len(),
            None => count == self.inputs.len(),
        }
    }

    /// How many entries beyond one-per-slot a `count`-input step carries in
    /// the variadic block (0 for fixed-arity operators or a count that
    /// doesn't fit).
    fn extra_inputs(&self, count: usize) -> usize {
        match self.variadic_slot() {
            Some(_) if count >= self.inputs.len() => count - self.inputs.len(),
            _ => 0,
        }
    }

    /// The declared slot that input `idx` of a `count`-input step maps to,
    /// or `None` past the declaration. A count that doesn't fit maps
    /// positionally, so hosts can still describe a malformed step.
    pub fn slot_of_input(&self, idx: usize, count: usize) -> Option<usize> {
        let extra = self.extra_inputs(count);
        let slot = match self.variadic_slot() {
            Some(variadic) if idx > variadic => idx.saturating_sub(extra).max(variadic),
            _ => idx,
        };
        (slot < self.inputs.len()).then_some(slot)
    }

    /// The step input index where declared slot `slot` starts in a
    /// `count`-input step (its first entry, for the variadic slot).
    pub fn input_of_slot(&self, slot: usize, count: usize) -> usize {
        match self.variadic_slot() {
            Some(variadic) if slot > variadic => slot + self.extra_inputs(count),
            _ => slot,
        }
    }

    /// The step input indices occupying the variadic block of a
    /// `count`-input step, or `None` for fixed-arity operators.
    pub fn variadic_range(&self, count: usize) -> Option<std::ops::Range<usize>> {
        let variadic = self.variadic_slot()?;
        Some(variadic..variadic + 1 + self.extra_inputs(count))
    }

    /// The declared input type of input `idx` in a `count`-input step.
    pub fn input_type(&self, idx: usize, count: usize) -> Option<&OperatorMetadataInput> {
        self.slot_of_input(idx, count)
            .map(|slot| &self.inputs[slot])
    }

    /// Label for input `idx` of a `count`-input step: the slot's declared
    /// name, numbered within the variadic block ("Model 2").
    pub fn input_label(&self, idx: usize, count: usize) -> Option<String> {
        let slot = self.slot_of_input(idx, count)?;
        let name = self.input_name(slot)?;
        match self.variadic_range(count) {
            Some(range) if range.contains(&idx) => {
                Some(format!("{name} {}", idx - range.start + 1))
            }
            _ => Some(name.to_string()),
        }
    }
}

/// CBOR-encode operator metadata (the payload `get_metadata()` points at).
pub fn encode_metadata(metadata: &OperatorMetadata) -> Vec<u8> {
    let mut out = Vec::new();
    ciborium::ser::into_writer(metadata, &mut out)
        .expect("operator metadata CBOR serialization should not fail");
    out
}

/// Decode the CBOR metadata payload read back by the host.
pub fn decode_metadata(bytes: &[u8]) -> Result<OperatorMetadata, String> {
    ciborium::de::from_reader(std::io::Cursor::new(bytes))
        .map_err(|e| format!("failed to decode operator metadata CBOR: {e}"))
}

/// Wrap lucide-style shape elements in the canonical
/// [`OperatorMetadata::icon_svg`] document header: 24×24 viewBox, 2px
/// round strokes, no fill. Expands to a `&'static str`, so it works in
/// `const` position and in the build scripts no_std models precompute
/// their metadata CBOR from.
///
/// ```
/// let icon = volumetric_abi::icon_svg!(r##"<circle cx="12" cy="12" r="9"/>"##);
/// assert!(icon.starts_with("<svg ") && icon.ends_with("</svg>"));
/// ```
#[macro_export]
macro_rules! icon_svg {
    ($($body:literal),+ $(,)?) => {
        concat!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">"##,
            $($body,)+
            "</svg>"
        )
    };
}

/// Pack a metadata buffer's address and length into the `i64` that
/// `get_metadata()` returns: `ptr | (len << 32)`.
pub fn pack_ptr_len(bytes: &[u8]) -> i64 {
    let ptr = bytes.as_ptr() as u32 as u64;
    let len = bytes.len() as u32 as u64;
    (ptr | (len << 32)) as i64
}

/// Unpack the `(ptr, len)` a `get_metadata()` return value refers to.
pub fn unpack_ptr_len(packed: i64) -> (usize, usize) {
    let packed = packed as u64;
    ((packed & 0xFFFF_FFFF) as usize, (packed >> 32) as usize)
}

/// The complete `get_metadata()` body: encode once into `cell`, return the
/// packed pointer.
///
/// ```ignore
/// #[unsafe(no_mangle)]
/// pub extern "C" fn get_metadata() -> i64 {
///     static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
///     volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata { ... })
/// }
/// ```
pub fn metadata_reply(
    cell: &'static OnceLock<Vec<u8>>,
    build: impl FnOnce() -> OperatorMetadata,
) -> i64 {
    pack_ptr_len(cell.get_or_init(|| encode_metadata(&build())))
}

/// The complete `get_sample_format()` body: encode once into `cell`, return
/// the packed pointer.
///
/// ```ignore
/// #[unsafe(no_mangle)]
/// pub extern "C" fn get_sample_format() -> i64 {
///     static FORMAT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
///     volumetric_abi::sample_format_reply(&FORMAT, || SampleFormat { ... })
/// }
/// ```
pub fn sample_format_reply(
    cell: &'static OnceLock<Vec<u8>>,
    build: impl FnOnce() -> SampleFormat,
) -> i64 {
    pack_ptr_len(cell.get_or_init(|| encode_sample_format(&build())))
}

/// Safe wrappers over the host imports available to operator WASM modules.
///
/// Only meaningful when compiled to wasm32 and run under a volumetric host;
/// the raw imports resolve against import module `"host"`.
pub mod host {
    mod raw {
        #[link(wasm_import_module = "host")]
        unsafe extern "C" {
            pub fn get_input_len(idx: i32) -> u32;
            pub fn get_input_data(idx: i32, ptr: i32, len: i32);
            pub fn get_input_count() -> i32;
            pub fn post_output(output_idx: i32, ptr: i32, len: i32);
            pub fn post_error(ptr: i32, len: i32);
            pub fn post_warning(ptr: i32, len: i32);
            pub fn cancelled() -> i32;
            pub fn input_model_dimensions(idx: i32) -> i32;
            pub fn input_model_bounds(idx: i32, out_ptr: i32) -> i32;
            pub fn input_model_sample(idx: i32, pos_ptr: i32, count: i32, out_ptr: i32) -> i32;
        }
    }

    /// Read the full contents of input slot `idx` (empty if absent).
    pub fn read_input(idx: i32) -> Vec<u8> {
        let len = unsafe { raw::get_input_len(idx) } as usize;
        let mut buf = vec![0u8; len];
        if len > 0 {
            unsafe { raw::get_input_data(idx, buf.as_mut_ptr() as i32, len as i32) };
        }
        buf
    }

    /// How many input slots the running step carries — the upper bound for
    /// [`read_input`] indices, and for an operator with a variadic slot
    /// ([`crate::OperatorMetadata::variadic_input`]) the only way to learn
    /// how many entries it received. NOTE: calling this makes the module
    /// import `host.get_input_count`, which hosts older than the import
    /// reject at instantiation; only call it from operators built alongside
    /// their host.
    pub fn input_count() -> usize {
        unsafe { raw::get_input_count() }.max(0) as usize
    }

    /// Post `data` as the contents of output slot `idx`.
    pub fn post_output(idx: i32, data: &[u8]) {
        unsafe { raw::post_output(idx, data.as_ptr() as i32, data.len() as i32) }
    }

    /// Report a failure to the host; the run fails with this message instead
    /// of producing outputs. Only the first reported error is kept.
    pub fn report_error(msg: &str) {
        unsafe { raw::post_error(msg.as_ptr() as i32, msg.len() as i32) }
    }

    /// Report a non-fatal advisory to the host: the run still succeeds and
    /// its outputs stand, but the author should know something (a design
    /// loop that stopped short, a quality guarantee met only best-effort).
    /// Hosts collect warnings in call order and attach them to the step's
    /// outputs — through result caches and bakes — so they resurface
    /// wherever the result is reused. NOTE: calling this makes the module
    /// import `host.post_warning`, which hosts older than the import reject
    /// at instantiation; only call it from operators built alongside their
    /// host.
    pub fn post_warning(msg: &str) {
        unsafe { raw::post_warning(msg.as_ptr() as i32, msg.len() as i32) }
    }

    /// Whether the host wants this run to stop. Long-running operators
    /// should poll this between iterations and return early (with or
    /// without an error) — for threaded operators it is the only reliable
    /// cancellation path, since the host cannot safely interrupt a thread
    /// pool from outside. Hosts without mid-run cancellation return false.
    pub fn cancelled() -> bool {
        unsafe { raw::cancelled() != 0 }
    }

    /// The dimension count of the model in input slot `idx`, or `None` if
    /// the slot doesn't hold a usable model.
    pub fn input_model_dimensions(idx: i32) -> Option<u32> {
        let n = unsafe { raw::input_model_dimensions(idx) };
        (n > 0).then_some(n as u32)
    }

    /// The bounds of the model in input slot `idx`: `dimensions`
    /// interleaved `[min, max]` pairs.
    pub fn input_model_bounds(idx: i32, dimensions: usize) -> Option<Vec<f64>> {
        let mut bounds = vec![0.0f64; 2 * dimensions];
        let ok = unsafe { raw::input_model_bounds(idx, bounds.as_mut_ptr() as i32) };
        (ok == 1).then_some(bounds)
    }

    /// Sample the model in input slot `idx` at `positions` (`dimensions`
    /// f64s per sample, concatenated). Returns one occupancy value per
    /// sample — classify with [`crate::is_occupied`].
    pub fn input_model_sample(idx: i32, positions: &[f64], dimensions: usize) -> Option<Vec<f32>> {
        if dimensions == 0 || !positions.len().is_multiple_of(dimensions) {
            return None;
        }
        let count = positions.len() / dimensions;
        let mut out = vec![0.0f32; count];
        let ok = unsafe {
            raw::input_model_sample(
                idx,
                positions.as_ptr() as i32,
                count as i32,
                out.as_mut_ptr() as i32,
            )
        };
        (ok == 1).then_some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_round_trips() {
        let metadata = OperatorMetadata {
            name: "test_operator".to_string(),
            version: "1.2.3".to_string(),
            display_name: "Test Operator".to_string(),
            description: "Round-trips every input kind.".to_string(),
            category: "Testing".to_string(),
            icon_svg: "<svg viewBox=\"0 0 24 24\"><circle cx=\"12\" cy=\"12\" r=\"9\"/></svg>"
                .to_string(),
            docs: "# Test Operator\n\nRound-trips every input kind.".to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration("{ dx: float }".to_string()),
                OperatorMetadataInput::LuaSource("-- stub".to_string()),
                OperatorMetadataInput::F64Map,
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::VecF64(3),
                OperatorMetadataInput::FeaMesh,
                OperatorMetadataInput::TriMesh,
            ],
            variadic_input: None,
            input_names: vec![
                "Model".to_string(),
                "Config".to_string(),
                "Script".to_string(),
                "Values".to_string(),
                "File".to_string(),
                "Corner".to_string(),
                "Mesh".to_string(),
                "Surface".to_string(),
            ],
            outputs: vec![
                OperatorMetadataOutput::ModelWASM,
                OperatorMetadataOutput::FeaMesh,
                OperatorMetadataOutput::TriMesh,
                OperatorMetadataOutput::F64Map,
            ],
            output_names: vec![],
        };

        let decoded = decode_metadata(&encode_metadata(&metadata)).unwrap();
        assert_eq!(decoded, metadata);
        assert_eq!(metadata.input_name(0), Some("Model"));
        assert_eq!(metadata.input_name(8), None);
        assert_eq!(metadata.catalog_name(), "Test Operator");
    }

    /// Metadata CBOR from operators built before `input_names` (and later
    /// the display fields) existed must still decode, with the missing
    /// fields defaulted to empty.
    #[test]
    fn metadata_without_input_names_decodes() {
        #[derive(serde::Serialize)]
        struct OldMetadata {
            name: String,
            version: String,
            inputs: Vec<OperatorMetadataInput>,
            outputs: Vec<OperatorMetadataOutput>,
        }
        let mut old = Vec::new();
        ciborium::ser::into_writer(
            &OldMetadata {
                name: "legacy".to_string(),
                version: "0.1.0".to_string(),
                inputs: vec![OperatorMetadataInput::ModelWASM],
                outputs: vec![OperatorMetadataOutput::ModelWASM],
            },
            &mut old,
        )
        .unwrap();

        let decoded = decode_metadata(&old).unwrap();
        assert_eq!(decoded.name, "legacy");
        assert!(decoded.input_names.is_empty());
        assert!(decoded.output_names.is_empty());
        assert_eq!(decoded.input_name(0), None);
        assert!(decoded.display_name.is_empty());
        assert!(decoded.description.is_empty());
        assert!(decoded.category.is_empty());
        assert!(decoded.icon_svg.is_empty());
        assert_eq!(decoded.catalog_name(), "legacy");
    }

    fn variadic(
        inputs: Vec<OperatorMetadataInput>,
        variadic_input: Option<usize>,
    ) -> OperatorMetadata {
        OperatorMetadata {
            name: "v".to_string(),
            version: "0.0.0".to_string(),
            display_name: String::new(),
            description: String::new(),
            category: String::new(),
            icon_svg: String::new(),
            docs: String::new(),
            input_names: inputs
                .iter()
                .map(|input| match input {
                    OperatorMetadataInput::ModelWASM => "Model".to_string(),
                    OperatorMetadataInput::CBORConfiguration(_) => "Config".to_string(),
                    _ => "Other".to_string(),
                })
                .collect(),
            inputs,
            variadic_input,
            outputs: vec![OperatorMetadataOutput::ModelWASM],
            output_names: vec![],
        }
    }

    /// Fixed-arity metadata: exact count, positional mapping, plain labels.
    #[test]
    fn fixed_arity_maps_positionally() {
        let m = variadic(
            vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(String::new()),
            ],
            None,
        );
        assert!(m.accepts_input_count(2));
        assert!(!m.accepts_input_count(1));
        assert!(!m.accepts_input_count(3));
        assert_eq!(m.slot_of_input(1, 2), Some(1));
        assert_eq!(m.slot_of_input(2, 2), None);
        assert_eq!(m.input_of_slot(1, 2), 1);
        assert_eq!(m.variadic_range(2), None);
        assert_eq!(m.input_label(0, 2).as_deref(), Some("Model"));
    }

    /// A variadic slot in the middle: the slots after it shift by the
    /// block's extra entries, labels number the block, and a fresh
    /// one-per-slot step is the minimum.
    #[test]
    fn variadic_block_shifts_later_slots() {
        let m = variadic(
            vec![
                OperatorMetadataInput::CBORConfiguration(String::new()),
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(String::new()),
            ],
            Some(1),
        );
        assert!(!m.accepts_input_count(2));
        assert!(m.accepts_input_count(3));
        assert!(m.accepts_input_count(6));
        // Six inputs: config, four models, config.
        assert_eq!(m.variadic_range(6), Some(1..5));
        assert_eq!(m.slot_of_input(0, 6), Some(0));
        assert_eq!(m.slot_of_input(1, 6), Some(1));
        assert_eq!(m.slot_of_input(4, 6), Some(1));
        assert_eq!(m.slot_of_input(5, 6), Some(2));
        assert_eq!(m.slot_of_input(6, 6), None);
        assert_eq!(m.input_of_slot(2, 6), 5);
        assert_eq!(m.input_of_slot(1, 6), 1);
        assert_eq!(m.input_label(3, 6).as_deref(), Some("Model 3"));
        assert_eq!(m.input_label(5, 6).as_deref(), Some("Config"));
        // The minimum step has one entry in the block.
        assert_eq!(m.variadic_range(3), Some(1..2));
        assert_eq!(m.slot_of_input(2, 3), Some(2));
        // A count that doesn't fit describes positionally.
        assert_eq!(m.slot_of_input(1, 2), Some(1));
        assert_eq!(m.variadic_range(2), Some(1..2));
    }

    /// A variadic index past the declaration is ignored rather than trusted.
    #[test]
    fn out_of_range_variadic_index_is_fixed_arity() {
        let m = variadic(vec![OperatorMetadataInput::ModelWASM], Some(3));
        assert!(m.accepts_input_count(1));
        assert!(!m.accepts_input_count(2));
        assert_eq!(m.variadic_range(1), None);
    }

    #[test]
    fn sample_format_round_trips_and_validates() {
        let format = SampleFormat {
            channels: vec![
                SampleChannel {
                    name: "occupancy".to_string(),
                    kind: ChannelKind::Occupancy,
                },
                SampleChannel {
                    name: "infill".to_string(),
                    kind: ChannelKind::Density,
                },
                SampleChannel {
                    name: "temp".to_string(),
                    kind: ChannelKind::Custom("test.temperature".to_string()),
                },
            ],
        };
        let decoded = decode_sample_format(&encode_sample_format(&format)).unwrap();
        assert_eq!(decoded, format);

        assert!(SampleFormat::default().validate().is_ok());
        assert_eq!(SampleFormat::default().channels.len(), 1);

        // Channel 0 must be occupancy
        let bad = SampleFormat {
            channels: vec![SampleChannel {
                name: "d".to_string(),
                kind: ChannelKind::Density,
            }],
        };
        assert!(decode_sample_format(&encode_sample_format(&bad)).is_err());

        // Empty and duplicate-name formats rejected
        assert!(SampleFormat { channels: vec![] }.validate().is_err());
        let dup = SampleFormat {
            channels: vec![
                SampleChannel {
                    name: "x".to_string(),
                    kind: ChannelKind::Occupancy,
                },
                SampleChannel {
                    name: "x".to_string(),
                    kind: ChannelKind::Density,
                },
            ],
        };
        assert!(dup.validate().is_err());
    }

    #[test]
    fn color_trio_located_by_name_and_kind() {
        let color = |name: &str| SampleChannel {
            name: name.to_string(),
            kind: ChannelKind::Custom(COLOR_SRGB_CHANNEL_KIND.to_string()),
        };
        let format = SampleFormat {
            channels: vec![
                SampleChannel {
                    name: "occupancy".to_string(),
                    kind: ChannelKind::Occupancy,
                },
                color("color_r"),
                color("color_g"),
                color("color_b"),
            ],
        };
        assert_eq!(format.color_trio(), Some([1, 2, 3]));
        assert_eq!(SampleFormat::default().color_trio(), None);

        // Right names with the wrong kind don't count.
        let impostor = SampleFormat {
            channels: vec![
                SampleChannel {
                    name: "occupancy".to_string(),
                    kind: ChannelKind::Occupancy,
                },
                SampleChannel {
                    name: "color_r".to_string(),
                    kind: ChannelKind::Density,
                },
                color("color_g"),
                color("color_b"),
            ],
        };
        assert_eq!(impostor.color_trio(), None);
    }

    #[test]
    fn occupancy_classification() {
        assert!(is_occupied(1.0));
        assert!(!is_occupied(0.0));
        assert!(!is_occupied(0.5)); // strictly greater
        assert!(!is_occupied(0.3)); // the #3 disagreement case: outside, everywhere
        assert!(!is_occupied(f32::NAN));
    }

    #[test]
    fn icon_svg_macro_wraps_bodies_in_the_canonical_header() {
        const ICON: &str = icon_svg!(
            r##"<circle cx="12" cy="12" r="9"/>"##,
            r##"<path d="M3 12h18"/>"##,
        );
        assert!(ICON.starts_with(r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24""#));
        assert!(ICON.contains(r##"<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/>"##));
        assert!(ICON.ends_with("</svg>"));
    }

    #[test]
    fn ptr_len_round_trips() {
        let (ptr, len) = unpack_ptr_len(0x0000_0042_0000_1000);
        assert_eq!(ptr, 0x1000);
        assert_eq!(len, 0x42);

        // Pointers are 32-bit on the wasm32 target this ABI runs on; on
        // 64-bit test hosts only the low 32 bits round-trip.
        let bytes = vec![0u8; 1234];
        let (ptr, len) = unpack_ptr_len(pack_ptr_len(&bytes));
        assert_eq!(ptr, (bytes.as_ptr() as usize) & 0xFFFF_FFFF);
        assert_eq!(len, 1234);
    }
}

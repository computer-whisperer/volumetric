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
//! - `post_output(idx: i32, ptr: i32, len: i32)`
//! - `post_error(ptr: i32, len: i32)` — optional; a run that posts an error
//!   fails with the message instead of returning outputs
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

use std::sync::OnceLock;

pub mod fea;
pub mod trimesh;

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
    /// `"myapp.temperature"`) to avoid collisions.
    Custom(String),
}

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
}

/// Metadata an operator returns from `get_metadata()`, CBOR-encoded.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct OperatorMetadata {
    pub name: String,
    pub version: String,
    pub inputs: Vec<OperatorMetadataInput>,
    pub outputs: Vec<OperatorMetadataOutput>,
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
            pub fn post_output(output_idx: i32, ptr: i32, len: i32);
            pub fn post_error(ptr: i32, len: i32);
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

    /// Post `data` as the contents of output slot `idx`.
    pub fn post_output(idx: i32, data: &[u8]) {
        unsafe { raw::post_output(idx, data.as_ptr() as i32, data.len() as i32) }
    }

    /// Report a failure to the host; the run fails with this message instead
    /// of producing outputs. Only the first reported error is kept.
    pub fn report_error(msg: &str) {
        unsafe { raw::post_error(msg.as_ptr() as i32, msg.len() as i32) }
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
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration("{ dx: float }".to_string()),
                OperatorMetadataInput::LuaSource("-- stub".to_string()),
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::VecF64(3),
                OperatorMetadataInput::FeaMesh,
                OperatorMetadataInput::TriMesh,
            ],
            outputs: vec![
                OperatorMetadataOutput::ModelWASM,
                OperatorMetadataOutput::FeaMesh,
                OperatorMetadataOutput::TriMesh,
            ],
        };

        let decoded = decode_metadata(&encode_metadata(&metadata)).unwrap();
        assert_eq!(decoded, metadata);
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
    fn occupancy_classification() {
        assert!(is_occupied(1.0));
        assert!(!is_occupied(0.0));
        assert!(!is_occupied(0.5)); // strictly greater
        assert!(!is_occupied(0.3)); // the #3 disagreement case: outside, everywhere
        assert!(!is_occupied(f32::NAN));
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

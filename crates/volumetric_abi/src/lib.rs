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

use std::sync::OnceLock;

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
}

/// Output slot declaration in an operator's metadata.
#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum OperatorMetadataOutput {
    /// A model WASM blob (N-dimensional model ABI).
    ModelWASM,
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
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        };

        let decoded = decode_metadata(&encode_metadata(&metadata)).unwrap();
        assert_eq!(decoded, metadata);
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

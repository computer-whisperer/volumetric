//! Identity operator.
//!
//! Host ABI:
//! - `host.get_input_len(i32) -> u32`
//! - `host.get_input_data(i32, ptr, len)`
//! - `host.post_output(i32, ptr, len)`
//!
//! Operator ABI:
//! - `get_metadata() -> i64` returning `(ptr: u32, len: u32)` packed as `ptr | (len << 32)`
//!
//! Behavior:
//! - Reads all bytes from input 0
//! - Emits them unchanged to output 0

#[derive(Clone, Debug, serde::Serialize)]
enum OperatorMetadataInput {
    ModelWASM,
}

#[derive(Clone, Debug, serde::Serialize)]
enum OperatorMetadataOutput {
    ModelWASM,
}

#[derive(Clone, Debug, serde::Serialize)]
struct OperatorMetadata {
    name: String,
    version: String,
    inputs: Vec<OperatorMetadataInput>,
    outputs: Vec<OperatorMetadataOutput>,
}

#[link(wasm_import_module = "host")]
extern "C" {
    fn get_input_len(arg: i32) -> u32;
    fn get_input_data(arg: i32, ptr: i32, len: i32);
    fn post_output(output_idx: i32, ptr: i32, len: i32);
}

#[no_mangle]
pub extern "C" fn run() {
    let len = unsafe { get_input_len(0) } as usize;
    let mut buf = vec![0u8; len];

    if len > 0 {
        unsafe {
            get_input_data(0, buf.as_mut_ptr() as i32, len as i32);
        }
    }

    unsafe {
        post_output(0, buf.as_ptr() as i32, len as i32);
    }
}

#[no_mangle]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let bytes = METADATA.get_or_init(|| {
        let metadata = OperatorMetadata {
            name: "identity_operator".to_string(),
            version: "0.1.0".to_string(),
            inputs: vec![OperatorMetadataInput::ModelWASM],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        };

        let mut out = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut out)
            .expect("identity_operator metadata CBOR serialization should not fail");
        out
    });

    let ptr = bytes.as_ptr() as u32;
    let len = bytes.len() as u32;
    (ptr as u64 | ((len as u64) << 32)) as i64
}

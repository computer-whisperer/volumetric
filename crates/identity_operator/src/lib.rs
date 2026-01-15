//! Identity operator.
//!
//! Host ABI:
//! - `host.get_input_len(i32) -> u32`
//! - `host.get_input_data(i32, ptr, len)`
//! - `host.post_output(i32, ptr, len)`
//!
//! Behavior:
//! - Reads all bytes from input 0
//! - Emits them unchanged to output 0

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

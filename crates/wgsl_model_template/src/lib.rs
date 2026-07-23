//! Runtime base module for `wgsl_script_operator`-generated models.
//!
//! The operator embeds this module's prebuilt WASM, loads it with walrus,
//! compiles the WGSL script's functions into it (calling the `wgsl_*`
//! kernels below for the transcendentals WASM lacks), wires up the model
//! ABI exports, and then un-exports the kernels.
//!
//! Everything here must stay free of imports: generated models are
//! instantiated with no host environment.
//!
//! The embedded binary is regenerated with:
//!
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p wgsl_model_template
//! cp target/wasm32-unknown-unknown/release/wgsl_model_template.wasm \
//!    crates/operators/wgsl_script_operator/template/
//! ```

/// Model-ABI IO buffer: hosts write up to 3 f64 position components and
/// read up to 6 f64 bounds values through the address `get_io_ptr` returns.
static mut IO_BUFFER: [f64; 8] = [0.0; 8];

#[unsafe(no_mangle)]
pub extern "C" fn get_io_ptr() -> i32 {
    core::ptr::addr_of!(IO_BUFFER) as i32
}

macro_rules! unary_kernels {
    ($($name:ident => $func:path),* $(,)?) => {
        $(
            #[unsafe(no_mangle)]
            pub extern "C" fn $name(x: f64) -> f64 {
                $func(x)
            }
        )*
    };
}

unary_kernels! {
    wgsl_sin => libm::sin,
    wgsl_cos => libm::cos,
    wgsl_tan => libm::tan,
    wgsl_asin => libm::asin,
    wgsl_acos => libm::acos,
    wgsl_atan => libm::atan,
    wgsl_sinh => libm::sinh,
    wgsl_cosh => libm::cosh,
    wgsl_tanh => libm::tanh,
    wgsl_asinh => libm::asinh,
    wgsl_acosh => libm::acosh,
    wgsl_atanh => libm::atanh,
    wgsl_exp => libm::exp,
    wgsl_exp2 => libm::exp2,
    wgsl_log => libm::log,
    wgsl_log2 => libm::log2,
}

#[unsafe(no_mangle)]
pub extern "C" fn wgsl_atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

#[unsafe(no_mangle)]
pub extern "C" fn wgsl_pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

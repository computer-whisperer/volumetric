//! WGSL Script Operator: compiles a WGSL model-dialect module into a WASM
//! Model module.
//!
//! Host/operator ABI: see the `volumetric_abi` crate. Design:
//! `WGSL_SCRIPT_OPERATOR_PLAN.md` at the repo root.
//!
//! Input/Output:
//! - Input 0: UTF-8 WGSL source containing the required functions
//! - Input 1: optional CBOR [`volumetric_abi::f64_map::F64Map`] overrides
//! - Output 0: WASM bytes of a model module exporting the model ABI
//!
//! # The model dialect
//!
//! A script is a library-style WGSL module — no entry points, no bindings —
//! defining:
//!
//! ```wgsl
//! fn scene(p: vec3<f64>) -> bool   // presence predicate: true = inside
//! fn bounds_min() -> vec3<f64>
//! fn bounds_max() -> vec3<f64>
//! ```
//!
//! `scene(p: vec2<f64>) -> bool` (with `vec2` bounds) compiles a 2D sketch
//! instead. Occupancy is the primary model form: distance-style code ports
//! by ending with `<= 0.0`. Helper functions, `const` data (including
//! fixed-size arrays), and scalar `override` parameters are all fair game;
//! f64 is the working precision and every WGSL float builtin (including
//! trig) lowers to real libm kernels.
//!
//! Routed parameters annotate overrides, mirroring the Lua operator:
//!
//! ```wgsl
//! override radius: f64 = 1.0; // @param key="sphere.radius" min=0.01 max=4.0
//! ```
//!
//! Only annotated overrides are routed from the F64Map input; values are
//! range-checked against the annotation.

mod lower;
mod restrict;

use std::collections::HashMap;

use naga::valid::{Capabilities, ValidationFlags, Validator};

use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::wgsl_parameters;
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(thiserror::Error, Debug)]
pub enum CompileError {
    #[error("WGSL parse error:\n{0}")]
    Parse(String),
    #[error("WGSL validation error:\n{0}")]
    Validation(String),
    #[error("Unsupported: {0}")]
    Unsupported(String),
    #[error("Type error: {0}")]
    Type(String),
    #[error("Missing required function: {0}")]
    MissingFunction(&'static str),
    #[error("internal error: {0}")]
    Internal(String),
}

pub fn compile_wgsl_to_wasm(
    source: &str,
    routed_values: &volumetric_abi::f64_map::F64Map,
) -> Result<Vec<u8>, CompileError> {
    let parameters = wgsl_parameters::parse(source).map_err(CompileError::Type)?;

    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| CompileError::Parse(error.emit_to_string(source)))?;

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::FLOAT64);
    let info = validator
        .validate(&module)
        .map_err(|error| CompileError::Validation(error.emit_to_string(source)))?;

    restrict::check_module(&module)?;
    let model = restrict::find_model_functions(&module)?;
    let overrides = resolve_overrides(&module, &parameters, routed_values)?;

    lower::build(&module, &info, &model, &overrides)
}

/// Resolve every override to a concrete value: routed (and range-checked)
/// for annotated overrides whose key is present, otherwise the WGSL default.
fn resolve_overrides(
    module: &naga::Module,
    parameters: &[wgsl_parameters::WgslParameter],
    routed_values: &volumetric_abi::f64_map::F64Map,
) -> Result<HashMap<naga::Handle<naga::Override>, lower::ConstScalar>, CompileError> {
    let mut by_name = HashMap::new();
    for (handle, o) in module.overrides.iter() {
        if let Some(name) = &o.name {
            by_name.insert(name.clone(), handle);
        }
    }

    let mut annotated: HashMap<naga::Handle<naga::Override>, &wgsl_parameters::WgslParameter> =
        HashMap::new();
    for parameter in parameters {
        let Some(&handle) = by_name.get(&parameter.override_name) else {
            return Err(CompileError::Type(format!(
                "parameter `{}` does not annotate an `override` declaration",
                parameter.override_name
            )));
        };
        let is_f64 = matches!(
            module.types[module.overrides[handle].ty].inner,
            naga::TypeInner::Scalar(naga::Scalar {
                kind: naga::ScalarKind::Float,
                width: 8,
            })
        );
        if !is_f64 {
            return Err(CompileError::Type(format!(
                "annotated override `{}` must be f64",
                parameter.override_name
            )));
        }
        annotated.insert(handle, parameter);
    }

    let mut resolved = HashMap::new();
    for (handle, o) in module.overrides.iter() {
        let name = o.name.as_deref().unwrap_or("<unnamed>");
        if let Some(parameter) = annotated.get(&handle)
            && let Some(&value) = routed_values.get(&parameter.key)
        {
            parameter
                .validate_value(value)
                .map_err(CompileError::Type)?;
            resolved.insert(handle, lower::ConstScalar::F64(value));
            continue;
        }
        let Some(init) = o.init else {
            return Err(CompileError::Type(format!(
                "override `{name}` needs a default value (or a routed parameter)"
            )));
        };
        let cx = lower::ConstCx {
            module,
            overrides: &resolved,
        };
        let values = cx.eval(&module.global_expressions, init)?;
        let [value] = values.as_slice() else {
            return Err(CompileError::Type(format!(
                "override `{name}` must be scalar"
            )));
        };
        resolved.insert(handle, *value);
    }
    Ok(resolved)
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let buf = read_input(0);
    let src = match std::str::from_utf8(&buf) {
        Ok(s) => s,
        Err(_) => {
            report_error("WGSL source is not valid UTF-8");
            return;
        }
    };
    let routed_values = match volumetric_abi::f64_map::decode(&read_input(1)) {
        Ok(values) => values,
        Err(error) => {
            report_error(&format!("invalid F64Map parameters: {error}"));
            return;
        }
    };
    match compile_wgsl_to_wasm(src, &routed_values) {
        Ok(wasm) => post_output(0, &wasm),
        Err(error) => report_error(&format!("WGSL compile error: {error}")),
    }
}

/// Starter script for the UI editor: teaches the entry-point, bounds, and
/// parameter conventions, plus the alias vocabulary.
const WGSL_TEMPLATE: &str = r#"// Model dialect: `scene` is a presence predicate (true = inside).
// For a 2D sketch (extrude input) use vec2d and 2-component bounds.
alias float = f64;
alias vec2d = vec2<f64>;
alias vec3d = vec3<f64>;

override radius: float = 1.0; // @param key="sphere.radius" min=0.000001
const bounds_margin: float = 0.5;

fn scene(p: vec3d) -> bool {
    return length(p) <= radius;
}

// The sampled region: declared bounds must contain the model.
fn bounds_min() -> vec3d {
    return vec3d(-(radius + bounds_margin));
}

fn bounds_max() -> vec3d {
    return vec3d(radius + bounds_margin);
}
"#;

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata {
        name: "wgsl_script_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "WGSL Script".to_string(),
        description: "Compile a WGSL model-dialect script into a model module.".to_string(),
        category: "Scripting".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<path d="M4 5h16v12H4z"/>"##,
            r##"<path d="M8 9l-2 2 2 2"/>"##,
            r##"<path d="M16 9l2 2-2 2"/>"##,
            r##"<path d="M13 8l-2 6"/>"##,
        )
        .to_string(),
        inputs: vec![
            OperatorMetadataInput::WgslSource(WGSL_TEMPLATE.to_string()),
            OperatorMetadataInput::F64Map,
        ],
        input_names: vec!["Script".to_string(), "Parameters".to_string()],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::f64_map::F64Map;

    struct Model {
        store: wasmtime::Store<()>,
        instance: wasmtime::Instance,
        io_ptr: usize,
    }

    /// Instantiate a compiled model and drive it exactly like a host:
    /// positions written at `get_io_ptr`, `sample(io_ptr)` read back.
    fn model(bytes: &[u8]) -> Model {
        let engine = wasmtime::Engine::default();
        let module = wasmtime::Module::new(&engine, bytes).expect("compiled model must load");
        let mut store = wasmtime::Store::new(&engine, ());
        let instance =
            wasmtime::Instance::new(&mut store, &module, &[]).expect("instantiate model");
        let io_ptr = instance
            .get_typed_func::<(), i32>(&mut store, "get_io_ptr")
            .expect("get_io_ptr")
            .call(&mut store, ())
            .expect("io ptr") as usize;
        Model {
            store,
            instance,
            io_ptr,
        }
    }

    impl Model {
        fn sample(&mut self, pos: &[f64]) -> f32 {
            let memory = self
                .instance
                .get_memory(&mut self.store, "memory")
                .expect("memory");
            let mut buf = Vec::new();
            for value in pos {
                buf.extend_from_slice(&value.to_le_bytes());
            }
            memory
                .write(&mut self.store, self.io_ptr, &buf)
                .expect("write position");
            self.instance
                .get_typed_func::<i32, f32>(&mut self.store, "sample")
                .expect("sample")
                .call(&mut self.store, self.io_ptr as i32)
                .expect("sample call")
        }

        fn dimensions(&mut self) -> i32 {
            self.instance
                .get_typed_func::<(), i32>(&mut self.store, "get_dimensions")
                .expect("get_dimensions")
                .call(&mut self.store, ())
                .expect("dims")
        }

        fn bounds(&mut self, dims: usize) -> Vec<f64> {
            let memory = self
                .instance
                .get_memory(&mut self.store, "memory")
                .expect("memory");
            self.instance
                .get_typed_func::<i32, ()>(&mut self.store, "get_bounds")
                .expect("get_bounds")
                .call(&mut self.store, self.io_ptr as i32)
                .expect("bounds call");
            let mut buf = vec![0u8; dims * 16];
            memory
                .read(&mut self.store, self.io_ptr, &mut buf)
                .expect("read bounds");
            buf.chunks(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
    }

    fn compile(source: &str) -> Vec<u8> {
        let bytes = compile_wgsl_to_wasm(source, &F64Map::new())
            .unwrap_or_else(|e| panic!("compile failed: {e}"));
        // Every compiled module must reparse as valid wasm.
        walrus::Module::from_buffer(&bytes).expect("compiled model must reparse");
        bytes
    }

    const BOUNDS_3D: &str = "
fn bounds_min() -> vec3<f64> { return vec3<f64>(-10.0, -10.0, -10.0); }
fn bounds_max() -> vec3<f64> { return vec3<f64>(10.0, 10.0, 10.0); }
";

    /// Probe harness: `expr` computes an f64 from p.x/p.y; the scene is true
    /// iff it matches p.z to within `tol`.
    fn probe_scene(expr: &str, tol: f64) -> String {
        format!(
            "fn scene(p: vec3<f64>) -> bool {{
    let x = p.x;
    let y = p.y;
    let got = {expr};
    return abs(got - p.z) <= {tol:e};
}}
{BOUNDS_3D}"
        )
    }

    #[track_caller]
    fn assert_probe(model_: &mut Model, x: f64, y: f64, expected: f64) {
        assert_eq!(
            model_.sample(&[x, y, expected]),
            1.0,
            "probe at ({x}, {y}) should equal {expected}"
        );
        let off = expected.abs().max(1.0) * 1e-3;
        assert_eq!(
            model_.sample(&[x, y, expected + off]),
            0.0,
            "probe at ({x}, {y}) must reject {expected} + {off}"
        );
    }

    #[test]
    fn starter_template_is_a_unit_sphere() {
        let mut m = model(&compile(WGSL_TEMPLATE));
        assert_eq!(m.dimensions(), 3);
        assert_eq!(m.sample(&[0.0, 0.0, 0.0]), 1.0);
        assert_eq!(m.sample(&[0.99, 0.0, 0.0]), 1.0);
        assert_eq!(m.sample(&[1.01, 0.0, 0.0]), 0.0);
        assert_eq!(m.sample(&[0.6, 0.6, 0.6]), 0.0);
        let bounds = m.bounds(3);
        assert_eq!(bounds, vec![-1.5, 1.5, -1.5, 1.5, -1.5, 1.5]);
    }

    #[test]
    fn routed_overrides_rescale_the_model_and_its_bounds() {
        let routed = F64Map::from([("sphere.radius".to_string(), 2.0)]);
        let bytes = compile_wgsl_to_wasm(WGSL_TEMPLATE, &routed).expect("compile routed");
        let mut m = model(&bytes);
        assert_eq!(m.sample(&[1.9, 0.0, 0.0]), 1.0);
        assert_eq!(m.sample(&[2.1, 0.0, 0.0]), 0.0);
        assert_eq!(m.bounds(3)[..2], [-2.5, 2.5]);
    }

    #[test]
    fn out_of_range_routed_values_are_rejected() {
        let src = r#"
override radius: f64 = 1.0; // @param key="r" min=0.5 max=2.0
fn scene(p: vec3<f64>) -> bool { return length(p) <= radius; }
fn bounds_min() -> vec3<f64> { return vec3<f64>(-3.0); }
fn bounds_max() -> vec3<f64> { return vec3<f64>(3.0); }
"#;
        let routed = F64Map::from([("r".to_string(), 5.0)]);
        let error = compile_wgsl_to_wasm(src, &routed).expect_err("out of range must fail");
        assert!(
            matches!(error, CompileError::Type(ref m) if m.contains("above maximum")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn annotation_must_bind_an_override() {
        let src = format!(
            "const radius: f64 = 1.0; // @param key=\"r\"\n{}",
            "fn scene(p: vec3<f64>) -> bool { return length(p) <= 1.0; }
fn bounds_min() -> vec3<f64> { return vec3<f64>(-2.0); }
fn bounds_max() -> vec3<f64> { return vec3<f64>(2.0); }"
        );
        let error = compile_wgsl_to_wasm(&src, &F64Map::new()).expect_err("const @param must fail");
        assert!(
            matches!(error, CompileError::Type(ref m) if m.contains("must annotate an `override")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn sketches_compile_with_two_dimensions() {
        let src = "
fn scene(p: vec2<f64>) -> bool { return abs(p.x) <= 1.0 && abs(p.y) <= 0.5; }
fn bounds_min() -> vec2<f64> { return vec2<f64>(-1.5, -1.0); }
fn bounds_max() -> vec2<f64> { return vec2<f64>(1.5, 1.0); }
";
        let mut m = model(&compile(src));
        assert_eq!(m.dimensions(), 2);
        assert_eq!(m.sample(&[0.9, 0.4]), 1.0);
        assert_eq!(m.sample(&[0.9, 0.6]), 0.0);
        assert_eq!(m.bounds(2), vec![-1.5, 1.5, -1.0, 1.0]);
    }

    #[test]
    fn const_array_hole_table_with_runtime_indexing() {
        let src = "
const holes = array<vec2<f64>, 4>(
    vec2<f64>(1.0, 1.0),
    vec2<f64>(1.0, -1.0),
    vec2<f64>(-1.0, 1.0),
    vec2<f64>(-1.0, -1.0),
);

fn scene(p: vec3<f64>) -> bool {
    var hole = false;
    for (var i: i32 = 0; i < 4; i++) {
        hole = hole || (length(p.xy - holes[i]) <= 0.25);
    }
    let body = abs(p.x) <= 2.0 && abs(p.y) <= 2.0 && abs(p.z) <= 0.5;
    return body && !hole;
}
"
        .to_string()
            + BOUNDS_3D;
        let mut m = model(&compile(&src));
        assert_eq!(m.sample(&[0.0, 0.0, 0.0]), 1.0);
        assert_eq!(m.sample(&[1.0, 1.0, 0.0]), 0.0); // inside a hole
        assert_eq!(m.sample(&[-1.0, -1.1, 0.0]), 0.0); // hole radius reaches
        assert_eq!(m.sample(&[-1.0, -1.3, 0.0]), 1.0); // just past the hole
        assert_eq!(m.sample(&[2.1, 0.0, 0.0]), 0.0); // outside the body
    }

    #[test]
    fn mutable_local_arrays_reset_between_samples() {
        // The accumulator array must be zero at every sample call; if memory
        // persisted, the second sample of the same point would differ.
        let src = "
fn scene(p: vec3<f64>) -> bool {
    var acc = array<f64, 2>(0.0, 0.0);
    for (var i: i32 = 0; i < 3; i++) {
        acc[0] = acc[0] + p.x;
        acc[1] = acc[1] + 1.0;
    }
    return acc[0] == 3.0 * p.x && acc[1] == 3.0;
}
"
        .to_string()
            + BOUNDS_3D;
        let mut m = model(&compile(&src));
        for _ in 0..3 {
            assert_eq!(m.sample(&[2.5, 0.0, 0.0]), 1.0);
        }
    }

    #[test]
    fn loops_break_continue_and_switch_execute() {
        let src = "
fn classify(n: i32) -> f64 {
    switch n {
        case 0, 1: { return 10.0; }
        case 2: { return 20.0; }
        default: { return 30.0; }
    }
}

fn scene(p: vec3<f64>) -> bool {
    var total: f64 = 0.0;
    var i: i32 = 0;
    loop {
        if i >= 10 { break; }
        i++;
        if i % 2 == 0 { continue; }
        total = total + 1.0; // counts odd i: 1,3,5,7,9
    }
    let sum_ok = total == 5.0;
    let switch_ok = classify(0) == 10.0 && classify(1) == 10.0
        && classify(2) == 20.0 && classify(7) == 30.0;
    return sum_ok && switch_ok && length(p) <= 1.0;
}
"
        .to_string()
            + BOUNDS_3D;
        let mut m = model(&compile(&src));
        assert_eq!(m.sample(&[0.5, 0.0, 0.0]), 1.0);
        assert_eq!(m.sample(&[1.5, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn guarded_integer_division_follows_wgsl_semantics() {
        // x/0 == x, MIN/-1 == MIN, x%0 == 0 — and nothing traps.
        let src = "
fn scene(p: vec3<f64>) -> bool {
    let a = i32(p.x);
    let b = i32(p.y);
    let q = f64(a / b);
    let r = f64(a % b);
    return abs(q - p.z) <= 0.5 && abs(r) <= 1000000.0;
}
"
        .to_string()
            + BOUNDS_3D;
        let mut m = model(&compile(&src));
        // 7 / 0 == 7 per WGSL.
        assert_eq!(m.sample(&[7.0, 0.0, 7.0]), 1.0);
        // 7 / 2 == 3.
        assert_eq!(m.sample(&[7.0, 2.0, 3.0]), 1.0);
    }

    #[test]
    fn transcendentals_match_std_via_probe() {
        for (expr, f) in [
            ("sin(x)", f64::sin as fn(f64) -> f64),
            ("cos(x)", f64::cos),
            ("tan(x)", f64::tan),
            ("exp(x)", f64::exp),
            ("sqrt(abs(x))", |x: f64| x.abs().sqrt()),
            ("log(abs(x) + 1.0)", |x: f64| (x.abs() + 1.0).ln()),
            ("atan(x)", f64::atan),
        ] {
            let mut m = model(&compile(&probe_scene(expr, 1e-12)));
            for x in [-2.5, -1.0, -0.3, 0.0, 0.7, 1.9, 3.1] {
                assert_probe(&mut m, x, 0.0, f(x));
            }
        }
    }

    #[test]
    fn two_argument_math_matches_std_via_probe() {
        let mut m = model(&compile(&probe_scene("pow(abs(x), y)", 1e-9)));
        for (x, y) in [(2.0, 2.0), (2.0, 0.5), (3.0, -1.5), (0.5, 3.0)] {
            assert_probe(&mut m, x, y, x.abs().powf(y));
        }
        let mut m = model(&compile(&probe_scene("atan2(y, x)", 1e-12)));
        for (x, y) in [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (0.0, 2.0)] {
            assert_probe(&mut m, x, y, y.atan2(x));
        }
    }

    #[test]
    fn vector_builtins_and_swizzles_via_probe() {
        for (expr, f) in [
            (
                "length(vec3<f64>(x, y, 2.0))",
                (|x, y| (x * x + y * y + 4.0f64).sqrt()) as fn(f64, f64) -> f64,
            ),
            ("dot(vec2<f64>(x, y), vec2<f64>(y, x))", |x, y| 2.0 * x * y),
            ("distance(vec2<f64>(x, 0.0), vec2<f64>(0.0, y))", |x, y| {
                (x * x + y * y).sqrt()
            }),
            (
                "cross(vec3<f64>(x, y, 0.0), vec3<f64>(0.0, x, y)).x",
                |x, y| y * y,
            ),
            ("normalize(vec2<f64>(x, y)).x", |x, y: f64| {
                x / (x * x + y * y).sqrt()
            }),
            ("vec3<f64>(x, y, 0.0).yx.x", |_x, y| y),
            ("clamp(x, -1.0, 1.0) + mix(0.0, y, 0.25)", |x: f64, y| {
                x.clamp(-1.0, 1.0) + 0.25 * y
            }),
            ("select(x, y, x < y)", f64::max),
            ("min(x, y) + max(x, y)", |x, y| x + y),
        ] {
            let mut m = model(&compile(&probe_scene(expr, 1e-12)));
            for (x, y) in [(1.5, 2.5), (-0.5, 3.0), (2.0, -1.0)] {
                assert_probe(&mut m, x, y, f(x, y));
            }
        }
    }

    #[test]
    fn float_modulo_truncates_like_wgsl() {
        let mut m = model(&compile(&probe_scene("x % y", 1e-12)));
        for (x, y) in [(7.5, 2.0), (-7.5, 2.0), (7.5, -2.0)] {
            assert_probe(&mut m, x, y, x - y * (x / y).trunc());
        }
    }

    #[test]
    fn f32_arithmetic_stays_f32() {
        let src = "
fn scene(p: vec3<f64>) -> bool {
    let x = f32(p.x);
    let third = x / 3.0f;
    return abs(f64(third) - p.z) <= 1e-12;
}
"
        .to_string()
            + BOUNDS_3D;
        let mut m = model(&compile(&src));
        let expected = (1.0f32 / 3.0f32) as f64; // f32-rounded, not f64
        assert_eq!(m.sample(&[1.0, 0.0, expected]), 1.0);
        assert_eq!(m.sample(&[1.0, 0.0, 1.0 / 3.0]), 0.0);
    }

    #[test]
    fn exports_are_exactly_the_model_abi() {
        let bytes = compile(WGSL_TEMPLATE);
        let module = walrus::Module::from_buffer(&bytes).unwrap();
        let mut names: Vec<_> = module
            .exports
            .iter()
            .map(|export| export.name.clone())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "get_bounds",
                "get_dimensions",
                "get_io_ptr",
                "memory",
                "sample"
            ]
        );
    }

    #[test]
    fn scene_must_return_bool() {
        let src = "
fn scene(p: vec3<f64>) -> f64 { return length(p) - 1.0; }
fn bounds_min() -> vec3<f64> { return vec3<f64>(-2.0); }
fn bounds_max() -> vec3<f64> { return vec3<f64>(2.0); }
";
        let error = compile_wgsl_to_wasm(src, &F64Map::new()).expect_err("f64 scene must fail");
        assert!(
            matches!(error, CompileError::Type(ref m) if m.contains("occupancy is the primary")),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn dialect_restrictions_reject_shader_constructs() {
        let scene_and_bounds = "
fn scene(p: vec3<f64>) -> bool { return length(p) <= 1.0; }
fn bounds_min() -> vec3<f64> { return vec3<f64>(-2.0); }
fn bounds_max() -> vec3<f64> { return vec3<f64>(2.0); }
";
        // var<private> breaks the stateless-model contract.
        let src = format!("var<private> acc: f64 = 0.0;\n{scene_and_bounds}");
        let error = compile_wgsl_to_wasm(&src, &F64Map::new()).expect_err("private var");
        assert!(
            matches!(error, CompileError::Unsupported(ref m) if m.contains("stateless")),
            "unexpected error: {error}"
        );

        // Entry points are not part of the dialect.
        let src = format!("@compute @workgroup_size(1) fn main() {{}}\n{scene_and_bounds}");
        let error = compile_wgsl_to_wasm(&src, &F64Map::new()).expect_err("entry point");
        assert!(
            matches!(error, CompileError::Unsupported(ref m) if m.contains("entry point")),
            "unexpected error: {error}"
        );

        // Matrices are not lowered yet.
        let src =
            format!("fn twist(m: mat2x2<f32>) -> f32 {{ return m[0][0]; }}\n{scene_and_bounds}");
        let error = compile_wgsl_to_wasm(&src, &F64Map::new()).expect_err("matrix");
        assert!(
            matches!(error, CompileError::Unsupported(ref m) if m.contains("matrices")),
            "unexpected error: {error}"
        );

        // Missing bounds.
        let src = "fn scene(p: vec3<f64>) -> bool { return true; }";
        let error = compile_wgsl_to_wasm(src, &F64Map::new()).expect_err("missing bounds");
        assert!(matches!(error, CompileError::MissingFunction(_)));
    }

    #[test]
    fn fidget_spinner_port_matches_reference_geometry() {
        let src = include_str!("../../../../examples/fidget_spinner.wgsl");
        let mut m = model(&compile(src));
        assert_eq!(m.dimensions(), 3);
        // Center bearing hole.
        assert_eq!(m.sample(&[0.0, 0.0, 0.0]), 0.0);
        // Web between center and right lobe.
        assert_eq!(m.sample(&[0.0175, 0.0, 0.0]), 1.0);
        // Inside the right lobe ring: outside bearing, inside lobe.
        assert_eq!(m.sample(&[0.035 + 0.0125, 0.0, 0.0]), 1.0);
        // Inside the right bearing hole.
        assert_eq!(m.sample(&[0.035, 0.0, 0.0]), 0.0);
        // Above the body: outside by thickness.
        assert_eq!(m.sample(&[0.0175, 0.0, 0.004]), 0.0);
        // Far outside.
        assert_eq!(m.sample(&[0.08, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn raspberry_pi_tray_port_matches_reference_geometry() {
        let src = include_str!("../../../../examples/raspberry_pi_4_tray.wgsl");
        let mut m = model(&compile(src));
        // Floor of the tray (inside the wall footprint, below floor top).
        assert_eq!(m.sample(&[0.040, 0.028, 0.001]), 0.0); // floor vent region is open
        assert_eq!(m.sample(&[0.040, 0.005, 0.001]), 1.0); // solid floor
        // Cavity above the floor is open.
        assert_eq!(m.sample(&[0.040, 0.028, 0.010]), 0.0);
        // Wall on the long edge (y just outside the board envelope).
        assert_eq!(m.sample(&[0.020, -0.0015, 0.010]), 1.0);
        // Mounting post at the official hole center, at standoff height.
        assert_eq!(m.sample(&[0.0035 + 0.002, 0.0035, 0.004]), 1.0);
        // Screw hole pierces the post center.
        assert_eq!(m.sample(&[0.0035, 0.0035, 0.004]), 0.0);
        // USB-C aperture in the lower wall: open where the port sits.
        assert_eq!(m.sample(&[0.010, -0.0015, 0.007]), 0.0);
        // Outside the case entirely.
        assert_eq!(m.sample(&[0.100, 0.028, 0.010]), 0.0);
    }
}

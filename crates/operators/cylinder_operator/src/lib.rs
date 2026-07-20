//! Cylinder Operator.
//!
//! Emitter operator that generates an analytic cylinder (or capsule)
//! between two axis endpoints — the round primitive the housing/
//! fastening workflows need (screw holes, bosses, pins, snap nubs).
//! Mirrors `rectangular_prism_operator`: the model wasm is generated
//! directly with walrus, evaluating the exact quadric at sample time.
//!
//! Inputs:
//! - Input 0: CBOR configuration:
//!   - `radius` (metres, default 5e-3)
//!   - `cap`: `"flat"` (true cylinder, default) or `"round"` (capsule —
//!     hemispherical ends past the endpoints)
//! - Input 1: VecF64(3) — axis endpoint A
//! - Input 2: VecF64(3) — axis endpoint B
//!
//! Output 0: ModelWASM (3D). Classification: with `d = p - A`,
//! `t = d·ab / |ab|²` the axial parameter — flat caps require
//! `0 <= t <= 1` and squared perpendicular distance `|d|² - t²|ab|²`
//! within `r²`; round caps clamp `t` and test the squared distance to
//! the segment. Bounds are the endpoint box padded by `r` (exact for
//! capsules, conservative for flat caps).

use walrus::{FunctionBuilder, Module, ValType};

use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct CylinderConfig {
    radius: f64,
    cap: String,
}

impl Default for CylinderConfig {
    fn default() -> Self {
        Self {
            radius: 5e-3,
            cap: "flat".to_string(),
        }
    }
}

/// Decode a VecF64(3) (8 bytes per f64, little-endian).
fn decode_vec3(data: &[u8]) -> Option<[f64; 3]> {
    if data.len() < 24 {
        return None;
    }
    Some([
        f64::from_le_bytes(data[0..8].try_into().unwrap()),
        f64::from_le_bytes(data[8..16].try_into().unwrap()),
        f64::from_le_bytes(data[16..24].try_into().unwrap()),
    ])
}

pub fn generate_wasm(
    a: [f64; 3],
    b: [f64; 3],
    radius: f64,
    round_cap: bool,
) -> Result<Vec<u8>, String> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(format!("radius must be finite and positive, got {radius}"));
    }
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    if !(ab2.is_finite() && ab2 > 0.0) {
        return Err("axis endpoints must be distinct and finite".to_string());
    }
    let inv_ab2 = 1.0 / ab2;
    let r2 = radius * radius;

    let mut module = Module::default();
    let memory_id = module.memories.add_local(false, false, 1, None, None);
    module.exports.add("memory", memory_id);

    {
        let mut builder = FunctionBuilder::new(&mut module.types, &[], &[ValType::I32]);
        builder.func_body().i32_const(3);
        let func_id = builder.finish(vec![], &mut module.funcs);
        module.exports.add("get_dimensions", func_id);
    }
    {
        // Stateless module: the IO buffer is the start of the page
        // (offset 8, nonzero so it never aliases a null pointer).
        let mut builder = FunctionBuilder::new(&mut module.types, &[], &[ValType::I32]);
        builder.func_body().i32_const(8);
        let func_id = builder.finish(vec![], &mut module.funcs);
        module.exports.add("get_io_ptr", func_id);
    }
    {
        // Endpoint box padded by the radius; exact for capsules,
        // conservative for flat caps.
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let out_ptr = module.locals.add(ValType::I32);
        for axis in 0..3 {
            for (slot, value) in [
                (2 * axis, a[axis].min(b[axis]) - radius),
                (2 * axis + 1, a[axis].max(b[axis]) + radius),
            ] {
                builder
                    .func_body()
                    .local_get(out_ptr)
                    .f64_const(value)
                    .store(
                        memory_id,
                        walrus::ir::StoreKind::F64,
                        walrus::ir::MemArg {
                            align: 3,
                            offset: (slot * 8) as u64,
                        },
                    );
            }
        }
        let func_id = builder.finish(vec![out_ptr], &mut module.funcs);
        module.exports.add("get_bounds", func_id);
    }
    {
        let mut builder = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);
        let d = [
            module.locals.add(ValType::F64),
            module.locals.add(ValType::F64),
            module.locals.add(ValType::F64),
        ];
        let t = module.locals.add(ValType::F64);
        let acc = module.locals.add(ValType::F64);

        let mut body = builder.func_body();
        // d = p - A
        for axis in 0..3 {
            body.local_get(pos_ptr)
                .load(
                    memory_id,
                    walrus::ir::LoadKind::F64,
                    walrus::ir::MemArg {
                        align: 3,
                        offset: (axis * 8) as u64,
                    },
                )
                .f64_const(a[axis])
                .binop(walrus::ir::BinaryOp::F64Sub)
                .local_set(d[axis]);
        }
        // t = (d . ab) * inv_ab2
        body.local_get(d[0])
            .f64_const(ab[0])
            .binop(walrus::ir::BinaryOp::F64Mul);
        for axis in 1..3 {
            body.local_get(d[axis])
                .f64_const(ab[axis])
                .binop(walrus::ir::BinaryOp::F64Mul)
                .binop(walrus::ir::BinaryOp::F64Add);
        }
        body.f64_const(inv_ab2)
            .binop(walrus::ir::BinaryOp::F64Mul)
            .local_set(t);

        if round_cap {
            // t = clamp(t, 0, 1); acc = |d - t*ab|^2; inside = acc <= r2
            body.local_get(t)
                .f64_const(0.0)
                .binop(walrus::ir::BinaryOp::F64Max)
                .f64_const(1.0)
                .binop(walrus::ir::BinaryOp::F64Min)
                .local_set(t);
            body.f64_const(0.0).local_set(acc);
            for axis in 0..3 {
                body.local_get(acc)
                    .local_get(d[axis])
                    .local_get(t)
                    .f64_const(ab[axis])
                    .binop(walrus::ir::BinaryOp::F64Mul)
                    .binop(walrus::ir::BinaryOp::F64Sub)
                    .local_tee(d[axis])
                    .local_get(d[axis])
                    .binop(walrus::ir::BinaryOp::F64Mul)
                    .binop(walrus::ir::BinaryOp::F64Add)
                    .local_set(acc);
            }
            body.local_get(acc)
                .f64_const(r2)
                .binop(walrus::ir::BinaryOp::F64Le);
        } else {
            // acc = |d|^2 - t^2 * ab2 (squared perpendicular distance);
            // inside = 0 <= t <= 1 && acc <= r2
            body.f64_const(0.0).local_set(acc);
            for &component in &d {
                body.local_get(acc)
                    .local_get(component)
                    .local_get(component)
                    .binop(walrus::ir::BinaryOp::F64Mul)
                    .binop(walrus::ir::BinaryOp::F64Add)
                    .local_set(acc);
            }
            body.local_get(acc)
                .local_get(t)
                .local_get(t)
                .binop(walrus::ir::BinaryOp::F64Mul)
                .f64_const(ab2)
                .binop(walrus::ir::BinaryOp::F64Mul)
                .binop(walrus::ir::BinaryOp::F64Sub)
                .f64_const(r2)
                .binop(walrus::ir::BinaryOp::F64Le);
            body.local_get(t)
                .f64_const(0.0)
                .binop(walrus::ir::BinaryOp::F64Ge)
                .binop(walrus::ir::BinaryOp::I32And);
            body.local_get(t)
                .f64_const(1.0)
                .binop(walrus::ir::BinaryOp::F64Le)
                .binop(walrus::ir::BinaryOp::I32And);
        }
        body.if_else(
            ValType::F32,
            |then| {
                then.f32_const(1.0);
            },
            |otherwise| {
                otherwise.f32_const(0.0);
            },
        );
        let func_id = builder.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", func_id);
    }

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config: CylinderConfig = {
        let bytes = read_input(0);
        if bytes.is_empty() {
            CylinderConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(bytes)) {
                Ok(config) => config,
                Err(error) => {
                    report_error(&format!("invalid cylinder configuration: {error}"));
                    return;
                }
            }
        }
    };
    let round_cap = match config.cap.as_str() {
        "flat" => false,
        "round" => true,
        other => {
            report_error(&format!("cap must be \"flat\" or \"round\", got {other:?}"));
            return;
        }
    };
    let Some(a) = decode_vec3(&read_input(1)) else {
        report_error("endpoint A: expected 3 f64 values");
        return;
    };
    let Some(b) = decode_vec3(&read_input(2)) else {
        report_error("endpoint B: expected 3 f64 values");
        return;
    };
    match generate_wasm(a, b, config.radius, round_cap) {
        Ok(wasm) => post_output(0, &wasm),
        Err(error) => report_error(&format!("cylinder generation failed: {error}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata {
        name: "cylinder_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "Cylinder".to_string(),
        description: "Analytic cylinder or capsule between two axis endpoints.".to_string(),
        category: "Primitives".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<ellipse cx="12" cy="6" rx="7" ry="3"/>"##,
            r##"<path d="M5 6v12"/>"##,
            r##"<path d="M19 6v12"/>"##,
            r##"<path d="M5 18c0 1.7 3.1 3 7 3s7-1.3 7-3"/>"##,
        )
        .to_string(),
        inputs: vec![
            OperatorMetadataInput::CBORConfiguration(
                r#"{ radius: float .default 5e-3, cap: "flat" / "round" .default "flat" }"#
                    .to_string(),
            ),
            OperatorMetadataInput::VecF64(3),
            OperatorMetadataInput::VecF64(3),
        ],
        input_names: vec![
            "Config".to_string(),
            "Endpoint A".to_string(),
            "Endpoint B".to_string(),
        ],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_module_has_the_model_abi() {
        for round_cap in [false, true] {
            let wasm = generate_wasm([0.0, 0.0, 0.0], [0.0, 0.0, 0.01], 0.002, round_cap).unwrap();
            let module = walrus::Module::from_buffer(&wasm).expect("emitted wasm parses");
            let names: Vec<&str> = module.exports.iter().map(|e| e.name.as_str()).collect();
            for required in [
                "sample",
                "get_bounds",
                "get_dimensions",
                "get_io_ptr",
                "memory",
            ] {
                assert!(names.contains(&required), "missing export {required}");
            }
        }
    }

    #[test]
    fn rejects_degenerate_inputs() {
        assert!(generate_wasm([0.0; 3], [0.0; 3], 0.01, false).is_err());
        assert!(generate_wasm([0.0; 3], [1.0, 0.0, 0.0], 0.0, false).is_err());
        assert!(generate_wasm([0.0; 3], [1.0, 0.0, 0.0], f64::NAN, true).is_err());
    }
}

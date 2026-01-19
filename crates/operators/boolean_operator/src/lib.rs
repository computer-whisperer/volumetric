//! Boolean operator.
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
//! - Reads WASM model A bytes from input 0
//! - Reads WASM model B bytes from input 1
//! - Reads CBOR configuration from input 2 (schema declared in metadata)
//! - Produces a merged WASM model with `is_inside`/bounds implementing union/subtract/intersect

use wasm_encoder::{CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module, TypeSection, ValType};

#[derive(Clone, Debug, serde::Serialize)]
enum OperatorMetadataInput {
    ModelWASM,
    CBORConfiguration(String),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BooleanOp {
    Union,
    Subtract,
    Intersect,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum BooleanOpConfig {
    Union,
    Subtract,
    Intersect,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct BooleanConfig {
    op: Option<BooleanOpConfig>,
}

impl Default for BooleanConfig {
    fn default() -> Self {
        Self { op: Some(BooleanOpConfig::Union) }
    }
}

impl From<BooleanOpConfig> for BooleanOp {
    fn from(value: BooleanOpConfig) -> Self {
        match value {
            BooleanOpConfig::Union => BooleanOp::Union,
            BooleanOpConfig::Subtract => BooleanOp::Subtract,
            BooleanOpConfig::Intersect => BooleanOp::Intersect,
        }
    }
}

#[link(wasm_import_module = "host")]
unsafe extern "C" {
    fn get_input_len(arg: i32) -> u32;
    fn get_input_data(arg: i32, ptr: i32, len: i32);
    fn post_output(output_idx: i32, ptr: i32, len: i32);
}

const ABI_FUNCTIONS: &[&str] = &[
    "is_inside",
    "get_bounds_min_x",
    "get_bounds_min_y",
    "get_bounds_min_z",
    "get_bounds_max_x",
    "get_bounds_max_y",
    "get_bounds_max_z",
];

#[derive(Debug)]
struct AbiExports {
    is_inside: u32,
    bmin_x: u32,
    bmin_y: u32,
    bmin_z: u32,
    bmax_x: u32,
    bmax_y: u32,
    bmax_z: u32,
}

fn parse_abi_exports(wasm: &[u8]) -> Result<AbiExports, String> {
    let mut map: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload.map_err(|e| e.to_string())? {
            wasmparser::Payload::ImportSection(s) => {
                if s.count() > 0 {
                    return Err("Model must not import anything".to_string());
                }
            }
            wasmparser::Payload::ExportSection(s) => {
                for e in s {
                    let e = e.map_err(|e| e.to_string())?;
                    if e.kind == wasmparser::ExternalKind::Func && ABI_FUNCTIONS.contains(&e.name) {
                        map.insert(e.name.to_string(), e.index);
                    }
                }
            }
            _ => {}
        }
    }

    let get = |name: &str| map.get(name).copied().ok_or_else(|| format!("Model missing export `{name}`"));
    Ok(AbiExports {
        is_inside: get("is_inside")?,
        bmin_x: get("get_bounds_min_x")?,
        bmin_y: get("get_bounds_min_y")?,
        bmin_z: get("get_bounds_min_z")?,
        bmax_x: get("get_bounds_max_x")?,
        bmax_y: get("get_bounds_max_y")?,
        bmax_z: get("get_bounds_max_z")?,
    })
}

#[derive(Default)]
struct Counts {
    types: u32,
    funcs: u32,
    globals: u32,
    tables: u32,
    memories: u32,
    elements: u32,
    data: u32,
    has_data_count: bool,
}

fn count_sections(wasm: &[u8]) -> Result<Counts, String> {
    let mut counts = Counts::default();
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload.map_err(|e| e.to_string())? {
            wasmparser::Payload::TypeSection(s) => counts.types = counts.types.saturating_add(s.count()),
            wasmparser::Payload::FunctionSection(s) => counts.funcs = counts.funcs.saturating_add(s.count()),
            wasmparser::Payload::GlobalSection(s) => counts.globals = counts.globals.saturating_add(s.count()),
            wasmparser::Payload::TableSection(s) => counts.tables = counts.tables.saturating_add(s.count()),
            wasmparser::Payload::MemorySection(s) => counts.memories = counts.memories.saturating_add(s.count()),
            wasmparser::Payload::ElementSection(s) => counts.elements = counts.elements.saturating_add(s.count()),
            wasmparser::Payload::DataSection(s) => counts.data = counts.data.saturating_add(s.count()),
            wasmparser::Payload::DataCountSection { .. } => counts.has_data_count = true,
            wasmparser::Payload::ImportSection(s) => {
                if s.count() > 0 {
                    return Err("Model must not import anything".to_string());
                }
            }
            wasmparser::Payload::StartSection { .. } => {
                return Err("Model must not define a start function".to_string());
            }
            _ => {}
        }
    }
    Ok(counts)
}

struct OffsetReencoder {
    type_offset: u32,
    func_offset: u32,
    global_offset: u32,
    table_offset: u32,
    memory_offset: u32,
}

impl wasm_encoder::reencode::Reencode for OffsetReencoder {
    type Error = std::convert::Infallible;

    fn type_index(&mut self, ty: u32) -> u32 {
        self.type_offset + ty
    }

    fn function_index(&mut self, func: u32) -> u32 {
        self.func_offset + func
    }

    fn global_index(&mut self, global: u32) -> u32 {
        self.global_offset + global
    }

    fn table_index(&mut self, table: u32) -> u32 {
        self.table_offset + table
    }

    fn memory_index(&mut self, memory: u32) -> u32 {
        self.memory_offset + memory
    }

    fn data_index(&mut self, data: u32) -> u32 {
        data
    }

    fn element_index(&mut self, element: u32) -> u32 {
        element
    }
}

fn append_module_sections(
    wasm: &[u8],
    re: &mut dyn wasm_encoder::reencode::Reencode<Error = std::convert::Infallible>,
    types: &mut TypeSection,
    funcs: &mut FunctionSection,
    code: &mut CodeSection,
    globals: &mut wasm_encoder::GlobalSection,
    tables: &mut wasm_encoder::TableSection,
    memories: &mut wasm_encoder::MemorySection,
    elements: &mut wasm_encoder::ElementSection,
    data: &mut wasm_encoder::DataSection,
) -> Result<(), String> {
    for payload in wasmparser::Parser::new(0).parse_all(wasm) {
        match payload.map_err(|e| e.to_string())? {
            wasmparser::Payload::TypeSection(s) => {
                wasm_encoder::reencode::utils::parse_type_section(re, types, s)
                    .map_err(|e| format!("type section reencode failed: {e}"))?;
            }
            wasmparser::Payload::FunctionSection(s) => {
                wasm_encoder::reencode::utils::parse_function_section(re, funcs, s)
                    .map_err(|e| format!("function section reencode failed: {e}"))?;
            }
            wasmparser::Payload::CodeSectionEntry(body) => {
                wasm_encoder::reencode::utils::parse_function_body(re, code, body)
                    .map_err(|e| format!("code reencode failed: {e}"))?;
            }
            wasmparser::Payload::GlobalSection(s) => {
                wasm_encoder::reencode::utils::parse_global_section(re, globals, s)
                    .map_err(|e| format!("global section reencode failed: {e}"))?;
            }
            wasmparser::Payload::TableSection(s) => {
                wasm_encoder::reencode::utils::parse_table_section(re, tables, s)
                    .map_err(|e| format!("table section reencode failed: {e}"))?;
            }
            wasmparser::Payload::MemorySection(s) => {
                wasm_encoder::reencode::utils::parse_memory_section(re, memories, s)
                    .map_err(|e| format!("memory section reencode failed: {e}"))?;
            }
            wasmparser::Payload::ElementSection(s) => {
                wasm_encoder::reencode::utils::parse_element_section(re, elements, s)
                    .map_err(|e| format!("element section reencode failed: {e}"))?;
            }
            wasmparser::Payload::DataSection(s) => {
                wasm_encoder::reencode::utils::parse_data_section(re, data, s)
                    .map_err(|e| format!("data section reencode failed: {e}"))?;
            }
            wasmparser::Payload::ImportSection(s) => {
                if s.count() > 0 {
                    return Err("Model must not import anything".to_string());
                }
            }
            wasmparser::Payload::ExportSection(_)
            | wasmparser::Payload::DataCountSection { .. }
            | wasmparser::Payload::StartSection { .. }
            | wasmparser::Payload::CustomSection(_) => {
                // Strictly ignore: exports are replaced; data_count is recreated; start/custom are not supported.
            }
            wasmparser::Payload::End(_) => break,
            _ => {}
        }
    }

    Ok(())
}

fn add_bounds_wrapper(
    types: &mut TypeSection,
    funcs: &mut FunctionSection,
    code: &mut CodeSection,
    exports: &mut ExportSection,
    export_name: &str,
    a_idx: u32,
    b_idx: u32,
    op: BooleanOp,
    is_min: bool,
) -> u32 {
    let ty = types.len();
    types.ty().function([], [ValType::F64]);
    funcs.function(ty);

    let mut f = Function::new([]);
    f.instruction(&Instruction::Call(a_idx));
    if op == BooleanOp::Subtract {
        // keep A only
    } else {
        f.instruction(&Instruction::Call(b_idx));
        match (op, is_min) {
            (BooleanOp::Union, true) => {
                // min(a, b)
                f.instruction(&Instruction::F64Min);
            }
            (BooleanOp::Union, false) => {
                // max(a, b)
                f.instruction(&Instruction::F64Max);
            }
            (BooleanOp::Intersect, true) => {
                // max(min_a, min_b)
                f.instruction(&Instruction::F64Max);
            }
            (BooleanOp::Intersect, false) => {
                // min(max_a, max_b)
                f.instruction(&Instruction::F64Min);
            }
            (BooleanOp::Subtract, _) => unreachable!(),
        }
    }
    f.instruction(&Instruction::End);
    code.function(&f);

    let func_index = (funcs.len() - 1) as u32;
    exports.export(export_name, ExportKind::Func, func_index);
    func_index
}

fn add_is_inside_wrapper(
    types: &mut TypeSection,
    funcs: &mut FunctionSection,
    code: &mut CodeSection,
    exports: &mut ExportSection,
    a_idx: u32,
    b_idx: u32,
    op: BooleanOp,
) {
    let ty = types.len();
    // New ABI: (f64,f64,f64) -> f32 (density). We keep boolean semantics by thresholding at 0.5
    types.ty().function([ValType::F64, ValType::F64, ValType::F64], [ValType::F32]);
    funcs.function(ty);

    let mut f = Function::new([]);
    // a_bool = (a_density(x,y,z) > 0.5)
    f.instruction(&Instruction::LocalGet(0));
    f.instruction(&Instruction::LocalGet(1));
    f.instruction(&Instruction::LocalGet(2));
    f.instruction(&Instruction::Call(a_idx));
    f.instruction(&Instruction::F32Const(0.5));
    f.instruction(&Instruction::F32Gt);

    if op == BooleanOp::Subtract {
        // b_bool = (b_density > 0.5)
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::LocalGet(1));
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::Call(b_idx));
        f.instruction(&Instruction::F32Const(0.5));
        f.instruction(&Instruction::F32Gt);
        // a && !b
        f.instruction(&Instruction::I32Eqz);
        f.instruction(&Instruction::I32And);
    } else if op == BooleanOp::Union {
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::LocalGet(1));
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::Call(b_idx));
        f.instruction(&Instruction::F32Const(0.5));
        f.instruction(&Instruction::F32Gt);
        f.instruction(&Instruction::I32Or);
    } else {
        // intersect
        f.instruction(&Instruction::LocalGet(0));
        f.instruction(&Instruction::LocalGet(1));
        f.instruction(&Instruction::LocalGet(2));
        f.instruction(&Instruction::Call(b_idx));
        f.instruction(&Instruction::F32Const(0.5));
        f.instruction(&Instruction::F32Gt);
        f.instruction(&Instruction::I32And);
    }

    // convert boolean i32 to f32 0.0/1.0
    f.instruction(&Instruction::F32ConvertI32S);
    f.instruction(&Instruction::End);
    code.function(&f);

    let func_index = (funcs.len() - 1) as u32;
    exports.export("is_inside", ExportKind::Func, func_index);
}

fn merge_models(a_wasm: &[u8], b_wasm: &[u8], op: BooleanOp) -> Result<Vec<u8>, String> {
    let a_counts = count_sections(a_wasm)?;
    let b_counts = count_sections(b_wasm)?;

    let a_exports = parse_abi_exports(a_wasm)?;
    let b_exports = parse_abi_exports(b_wasm)?;

    let mut types = TypeSection::new();
    let mut funcs = FunctionSection::new();
    let mut code = CodeSection::new();
    let mut globals = wasm_encoder::GlobalSection::new();
    let mut tables = wasm_encoder::TableSection::new();
    let mut memories = wasm_encoder::MemorySection::new();
    let mut elements = wasm_encoder::ElementSection::new();
    let mut data = wasm_encoder::DataSection::new();

    // Append A with identity mapping.
    let mut re_a = OffsetReencoder {
        type_offset: 0,
        func_offset: 0,
        global_offset: 0,
        table_offset: 0,
        memory_offset: 0,
    };
    append_module_sections(
        a_wasm,
        &mut re_a,
        &mut types,
        &mut funcs,
        &mut code,
        &mut globals,
        &mut tables,
        &mut memories,
        &mut elements,
        &mut data,
    )?;

    // Append B with index offsets.
    let mut re_b = OffsetReencoder {
        type_offset: a_counts.types,
        func_offset: a_counts.funcs,
        global_offset: a_counts.globals,
        table_offset: a_counts.tables,
        memory_offset: a_counts.memories,
    };
    append_module_sections(
        b_wasm,
        &mut re_b,
        &mut types,
        &mut funcs,
        &mut code,
        &mut globals,
        &mut tables,
        &mut memories,
        &mut elements,
        &mut data,
    )?;

    // Build exports with wrappers.
    let mut exports = ExportSection::new();

    add_is_inside_wrapper(
        &mut types,
        &mut funcs,
        &mut code,
        &mut exports,
        a_exports.is_inside,
        b_exports.is_inside + a_counts.funcs,
        op,
    );

    add_bounds_wrapper(&mut types, &mut funcs, &mut code, &mut exports, "get_bounds_min_x", a_exports.bmin_x, b_exports.bmin_x + a_counts.funcs, op, true);
    add_bounds_wrapper(&mut types, &mut funcs, &mut code, &mut exports, "get_bounds_min_y", a_exports.bmin_y, b_exports.bmin_y + a_counts.funcs, op, true);
    add_bounds_wrapper(&mut types, &mut funcs, &mut code, &mut exports, "get_bounds_min_z", a_exports.bmin_z, b_exports.bmin_z + a_counts.funcs, op, true);
    add_bounds_wrapper(&mut types, &mut funcs, &mut code, &mut exports, "get_bounds_max_x", a_exports.bmax_x, b_exports.bmax_x + a_counts.funcs, op, false);
    add_bounds_wrapper(&mut types, &mut funcs, &mut code, &mut exports, "get_bounds_max_y", a_exports.bmax_y, b_exports.bmax_y + a_counts.funcs, op, false);
    add_bounds_wrapper(&mut types, &mut funcs, &mut code, &mut exports, "get_bounds_max_z", a_exports.bmax_z, b_exports.bmax_z + a_counts.funcs, op, false);

    let mut out = Module::new();
    if types.len() > 0 {
        out.section(&types);
    }
    if funcs.len() > 0 {
        out.section(&funcs);
    }
    if tables.len() > 0 {
        out.section(&tables);
    }
    if memories.len() > 0 {
        out.section(&memories);
    }
    if globals.len() > 0 {
        out.section(&globals);
    }
    out.section(&exports);
    if elements.len() > 0 {
        out.section(&elements);
    }

    if a_counts.has_data_count || b_counts.has_data_count {
        out.section(&wasm_encoder::DataCountSection {
            count: a_counts.data + b_counts.data,
        });
    }
    out.section(&code);
    if data.len() > 0 {
        out.section(&data);
    }

    Ok(out.finish())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let len_a = unsafe { get_input_len(0) } as usize;
    let mut a_buf = vec![0u8; len_a];
    if len_a > 0 {
        unsafe { get_input_data(0, a_buf.as_mut_ptr() as i32, len_a as i32) };
    }

    let len_b = unsafe { get_input_len(1) } as usize;
    let mut b_buf = vec![0u8; len_b];
    if len_b > 0 {
        unsafe { get_input_data(1, b_buf.as_mut_ptr() as i32, len_b as i32) };
    }

    let cfg = {
        let cfg_len = unsafe { get_input_len(2) } as usize;
        if cfg_len == 0 {
            BooleanConfig::default()
        } else {
            let mut cfg_buf = vec![0u8; cfg_len];
            unsafe { get_input_data(2, cfg_buf.as_mut_ptr() as i32, cfg_len as i32) };
            let mut cursor = std::io::Cursor::new(&cfg_buf);
            ciborium::de::from_reader::<BooleanConfig, _>(&mut cursor).unwrap_or_default()
        }
    };
    let op = cfg.op.unwrap_or(BooleanOpConfig::Union).into();

    let output = merge_models(&a_buf, &b_buf, op).unwrap_or_else(|_| a_buf);
    unsafe {
        post_output(0, output.as_ptr() as i32, output.len() as i32);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let bytes = METADATA.get_or_init(|| {
        let schema = "{ op: \"union\" / \"subtract\" / \"intersect\" .default \"union\" }".to_string();
        let metadata = OperatorMetadata {
            name: "boolean_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        };

        let mut out = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut out)
            .expect("boolean_operator metadata CBOR serialization should not fail");
        out
    });

    let ptr = bytes.as_ptr() as u32;
    let len = bytes.len() as u32;
    (ptr as u64 | ((len as u64) << 32)) as i64
}

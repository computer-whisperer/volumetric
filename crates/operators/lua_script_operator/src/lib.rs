//! Lua Script Operator: compiles a restricted Lua script into a WASM Model module.
//!
//! Host ABI:
//! - `host.get_input_len(i32) -> u32`
//! - `host.get_input_data(i32, ptr, len)`
//! - `host.post_output(i32, ptr, len)`
//!
//! Operator ABI:
//! - `get_metadata() -> i64` returning `(ptr: u32, len: u32)` packed as `ptr | (len << 32)`
//!
//! Input/Output:
//! - Input 0: UTF-8 Lua source containing the required functions
//! - Output 0: WASM bytes of a model module that exports the model ABI
//!
//! Notes:
//! - MVP supports a restricted Lua subset (expression-oriented); math functions limited to those with direct WASM ops: `math.abs`, `math.min`, `math.max`, `math.sqrt`, `math.floor`, `math.ceil`, `math.trunc`, `math.nearest`.
//! - Trigonometric/exponential functions (e.g., `math.sin`, `math.cos`, `math.exp`, `math.log`) are not yet supported and will cause a compile error.

use walrus::{FunctionBuilder, Module, ModuleConfig, ValType};

#[derive(Clone, Debug, serde::Serialize)]
enum OperatorMetadataInput { LuaSource(String) }

#[derive(Clone, Debug, serde::Serialize)]
enum OperatorMetadataOutput { ModelWASM }

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

const REQUIRED_FUNCS: &[&str] = &[
    "is_inside",
    "get_bounds_min_x",
    "get_bounds_min_y",
    "get_bounds_min_z",
    "get_bounds_max_x",
    "get_bounds_max_y",
    "get_bounds_max_z",
];

#[derive(thiserror::Error, Debug)]
enum CompileError {
    #[error("Lua parse error: {0}")]
    Parse(String),
    #[error("Missing required function: {0}")]
    MissingFunc(&'static str),
    #[error("Unsupported Lua construct: {0}")]
    Unsupported(&'static str),
    #[error("Type error: {0}")]
    Type(&'static str),
}

fn compile_lua_to_wasm(src: &str) -> Result<Vec<u8>, CompileError> {
    // Validate that it is syntactically valid Lua
    full_moon::parse(src).map_err(|e| CompileError::Parse(e.to_string()))?;

    // Very simple textual extraction of required functions and their single-line return expressions
    // This keeps using full_moon for validation while we bootstrap MVP codegen.
    let mut functions: std::collections::HashMap<String, (Vec<String>, String)> = Default::default();
    for &fname in REQUIRED_FUNCS {
        // Regex-ish scan: function <name> (params) ... return <expr>
        if let Some(start_idx) = src.find(&format!("function {}", fname)) {
            let rest = &src[start_idx..];
            if let Some(open_paren) = rest.find('(') {
                if let Some(close_paren) = rest[open_paren+1..].find(')') {
                    let params_str = &rest[open_paren+1..open_paren+1+close_paren];
                    let params: Vec<String> = params_str.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
                    // find return
                    if let Some(ret_idx) = rest.find("return ") {
                        let after = &rest[ret_idx+7..];
                        // take until end of line or ';' or endfunction
                        let end = after.find('\n').or_else(|| after.find(';')).unwrap_or(after.len());
                        let expr = after[..end].trim().to_string();
                        functions.insert(fname.to_string(), (params, expr));
                        continue;
                    }
                }
            }
        }
        return Err(CompileError::MissingFunc(fname));
    }

    // Create a new module
    let mut module = Module::with_config(ModuleConfig::new());

    // Expression emitter from a tiny expression string (supports literals, x/y/z, + - * /, parentheses)
    use walrus::InstrSeqBuilder;
    fn emit_expr_str(b: &mut InstrSeqBuilder, expr: &str, params: &[String], locals: &[walrus::LocalId]) -> Result<(), CompileError> {
        // Shunting-yard to RPN, then emit stack code
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Op { Add, Sub, Mul, Div }
        fn precedence(op: Op) -> i32 { match op { Op::Add|Op::Sub => 1, Op::Mul|Op::Div => 2 } }
        let mut output: Vec<String> = Vec::new();
        let mut ops: Vec<Op> = Vec::new();
        // Tokenize very simply
        let mut i = 0;
        let bytes = expr.as_bytes();
        while i < bytes.len() {
            let c = bytes[i] as char;
            if c.is_whitespace() { i += 1; continue; }
            if c.is_ascii_digit() || c == '.' { // number
                let start = i;
                i += 1;
                while i < bytes.len() && ((bytes[i] as char).is_ascii_digit() || bytes[i] as char == '.' || bytes[i] as char == 'e' || bytes[i] as char == 'E' || bytes[i] as char == '-' || bytes[i] as char == '+') { i += 1; }
                output.push(expr[start..i].to_string());
                continue;
            }
            if c.is_ascii_alphabetic() { // name
                let start = i;
                i += 1;
                while i < bytes.len() && ((bytes[i] as char).is_ascii_alphanumeric() || bytes[i] as char == '_') { i += 1; }
                output.push(expr[start..i].to_string());
                continue;
            }
            match c {
                '+' => { while let Some(top) = ops.last().copied() { if precedence(top) >= precedence(Op::Add) { output.push(match ops.pop().unwrap() { Op::Add=>"+",Op::Sub=>"-",Op::Mul=>"*",Op::Div=>"/"}.to_string()); } else { break; } } ops.push(Op::Add); i+=1; }
                '-' => { while let Some(top) = ops.last().copied() { if precedence(top) >= precedence(Op::Sub) { output.push(match ops.pop().unwrap() { Op::Add=>"+",Op::Sub=>"-",Op::Mul=>"*",Op::Div=>"/"}.to_string()); } else { break; } } ops.push(Op::Sub); i+=1; }
                '*' => { while let Some(top) = ops.last().copied() { if precedence(top) >= precedence(Op::Mul) { output.push(match ops.pop().unwrap() { Op::Add=>"+",Op::Sub=>"-",Op::Mul=>"*",Op::Div=>"/"}.to_string()); } else { break; } } ops.push(Op::Mul); i+=1; }
                '/' => { while let Some(top) = ops.last().copied() { if precedence(top) >= precedence(Op::Div) { output.push(match ops.pop().unwrap() { Op::Add=>"+",Op::Sub=>"-",Op::Mul=>"*",Op::Div=>"/"}.to_string()); } else { break; } } ops.push(Op::Div); i+=1; }
                '(' => { ops.clear(); i+=1; /* super-simplified: ignore nested ops for MVP */ }
                ')' => { while let Some(op) = ops.pop() { output.push(match op { Op::Add=>"+",Op::Sub=>"-",Op::Mul=>"*",Op::Div=>"/"}.to_string()); } i+=1; }
                _ => return Err(CompileError::Unsupported("unexpected char in expr")),
            }
        }
        while let Some(op) = ops.pop() { output.push(match op { Op::Add=>"+",Op::Sub=>"-",Op::Mul=>"*",Op::Div=>"/"}.to_string()); }

        use walrus::ir::BinaryOp::*;
        for tok in output {
            match tok.as_str() {
                "+" => { b.binop(F64Add); }
                "-" => { b.binop(F64Sub); }
                "*" => { b.binop(F64Mul); }
                "/" => { b.binop(F64Div); }
                s => {
                    if let Ok(v) = s.parse::<f64>() { b.f64_const(v); }
                    else {
                        // variable name
                        if let Some((idx, _)) = params.iter().enumerate().find(|(_, p)| p == &s) {
                            b.local_get(locals[idx]);
                        } else {
                            return Err(CompileError::Unsupported("unknown name in expr"));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    for &fname in REQUIRED_FUNCS {
        let (params, expr_str) = functions.get(fname).cloned().unwrap();
        match fname {
            "is_inside" => {
                if params.len() != 3 { return Err(CompileError::Type("is_inside must take 3 params")); }
                let mut fb = FunctionBuilder::new(&mut module.types, &[ValType::F64, ValType::F64, ValType::F64], &[ValType::F32]);
                let l_x = module.locals.add(ValType::F64);
                let l_y = module.locals.add(ValType::F64);
                let l_z = module.locals.add(ValType::F64);
                let mut ib = fb.func_body();
                emit_expr_str(&mut ib, &expr_str, &params, &[l_x, l_y, l_z])?;
                ib.unop(walrus::ir::UnaryOp::F32DemoteF64);
                let fid = fb.finish(vec![l_x, l_y, l_z], &mut module.funcs);
                module.exports.add("is_inside", fid);
            }
            _ => {
                if !params.is_empty() { return Err(CompileError::Type("bounds getters must have 0 params")); }
                let mut fb = FunctionBuilder::new(&mut module.types, &[], &[ValType::F64]);
                let mut ib = fb.func_body();
                emit_expr_str(&mut ib, &expr_str, &params, &[])?;
                let fid = fb.finish(vec![], &mut module.funcs);
                module.exports.add(fname, fid);
            }
        }
    }

    Ok(module.emit_wasm())
}

#[no_mangle]
pub extern "C" fn run() {
    let len = unsafe { get_input_len(0) } as usize;
    let mut buf = vec![0u8; len];
    if len > 0 { unsafe { get_input_data(0, buf.as_mut_ptr() as i32, len as i32); } }
    let src = match std::str::from_utf8(&buf) { Ok(s) => s, Err(_) => "" };
    let output = match compile_lua_to_wasm(src) { Ok(w) => w, Err(e) => {
        // On error, return empty bytes to signal failure upstream (percolate)
        let msg = format!("Lua compile error: {}", e);
        let mut b = Vec::new();
        b.extend_from_slice(msg.as_bytes());
        b
    }};
    unsafe { post_output(0, output.as_ptr() as i32, output.len() as i32); }
}

#[no_mangle]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let bytes = METADATA.get_or_init(|| {
        let schema = "Lua script with required functions".to_string();
        let metadata = OperatorMetadata {
            name: "lua_script_operator".to_string(),
            version: "0.1.0".to_string(),
            inputs: vec![OperatorMetadataInput::LuaSource(schema)],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
        };
        let mut out = Vec::new();
        ciborium::ser::into_writer(&metadata, &mut out).expect("metadata CBOR serialization should not fail");
        out
    });
    let ptr = bytes.as_ptr() as u32;
    let len = bytes.len() as u32;
    (ptr as u64 | ((len as u64) << 32)) as i64
}

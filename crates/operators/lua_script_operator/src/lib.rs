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
    use full_moon::ast::{self, Stmt, LastStmt, Expression, BinOp, UnOp, Var, Parameter, Prefix, Suffix, Call, FunctionArgs, Index};
    use std::collections::HashMap;
    use walrus::InstrSeqBuilder;
    use walrus::ir::{BinaryOp, UnaryOp};

    // Parse the Lua source into an AST
    let ast = full_moon::parse(src).map_err(|errors| {
        let msg = errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ");
        CompileError::Parse(msg)
    })?;

    // Extract function declarations from the AST
    // Maps function name -> (parameter names, return expression)
    let mut functions: HashMap<String, (Vec<String>, Expression)> = HashMap::new();

    for stmt in ast.nodes().stmts() {
        if let Stmt::FunctionDeclaration(func_decl) = stmt {
            // Get function name (only support simple names, not methods like a:b or a.b)
            let name = func_decl.name().names().iter()
                .map(|t| t.token().to_string())
                .collect::<Vec<_>>()
                .join(".");
            
            // Get parameters
            let params: Vec<String> = func_decl.body().parameters().iter()
                .filter_map(|p| match p {
                    Parameter::Name(token) => Some(token.token().to_string()),
                    Parameter::Ellipsis(_) => None, // varargs not supported
                    _ => None, // handle non-exhaustive enum
                })
                .collect();

            // Get return expression from the function body
            let block = func_decl.body().block();
            if let Some(LastStmt::Return(ret)) = block.last_stmt() {
                // We expect exactly one return expression for our simple functions
                let mut returns_iter = ret.returns().iter();
                if let Some(expr) = returns_iter.next() {
                    if returns_iter.next().is_none() {
                        functions.insert(name, (params, expr.clone()));
                    }
                }
            }
        }
    }

    // Verify all required functions are present
    for &fname in REQUIRED_FUNCS {
        if !functions.contains_key(fname) {
            return Err(CompileError::MissingFunc(fname));
        }
    }

    // Create a new WASM module
    let mut module = Module::with_config(ModuleConfig::new());

    // Emit WASM instructions for an expression from the AST
    fn emit_expr(
        b: &mut InstrSeqBuilder,
        expr: &Expression,
        params: &[String],
        locals: &[walrus::LocalId],
    ) -> Result<(), CompileError> {
        match expr {
            Expression::Number(token) => {
                let num_str = token.token().to_string();
                let v: f64 = num_str.parse()
                    .map_err(|_| CompileError::Parse(format!("invalid number: {}", num_str)))?;
                b.f64_const(v);
            }
            Expression::Var(var) => {
                match var {
                    Var::Name(token) => {
                        let name = token.token().to_string();
                        if let Some((idx, _)) = params.iter().enumerate().find(|(_, p)| **p == name) {
                            b.local_get(locals[idx]);
                        } else {
                            return Err(CompileError::Unsupported("unknown variable name"));
                        }
                    }
                    Var::Expression(_) => {
                        return Err(CompileError::Unsupported("complex variable expressions not supported"));
                    }
                    _ => return Err(CompileError::Unsupported("unsupported variable type")),
                }
            }
            Expression::Parentheses { expression, .. } => {
                emit_expr(b, expression, params, locals)?;
            }
            Expression::UnaryOperator { unop, expression } => {
                match unop {
                    UnOp::Minus(_) => {
                        emit_expr(b, expression, params, locals)?;
                        b.unop(UnaryOp::F64Neg);
                    }
                    UnOp::Not(_) => {
                        return Err(CompileError::Unsupported("logical not operator"));
                    }
                    UnOp::Hash(_) => {
                        return Err(CompileError::Unsupported("length operator"));
                    }
                    _ => return Err(CompileError::Unsupported("unsupported unary operator")),
                }
            }
            Expression::BinaryOperator { lhs, binop, rhs } => {
                emit_expr(b, lhs, params, locals)?;
                emit_expr(b, rhs, params, locals)?;
                match binop {
                    BinOp::Plus(_) => { b.binop(BinaryOp::F64Add); }
                    BinOp::Minus(_) => { b.binop(BinaryOp::F64Sub); }
                    BinOp::Star(_) => { b.binop(BinaryOp::F64Mul); }
                    BinOp::Slash(_) => { b.binop(BinaryOp::F64Div); }
                    BinOp::Caret(_) => {
                        return Err(CompileError::Unsupported("exponentiation operator (use math.pow if available)"));
                    }
                    BinOp::Percent(_) => {
                        return Err(CompileError::Unsupported("modulo operator"));
                    }
                    BinOp::TwoDots(_) => {
                        return Err(CompileError::Unsupported("string concatenation"));
                    }
                    BinOp::LessThan(_) | BinOp::LessThanEqual(_) |
                    BinOp::GreaterThan(_) | BinOp::GreaterThanEqual(_) |
                    BinOp::TwoEqual(_) | BinOp::TildeEqual(_) => {
                        return Err(CompileError::Unsupported("comparison operators in expressions"));
                    }
                    BinOp::And(_) | BinOp::Or(_) => {
                        return Err(CompileError::Unsupported("logical operators"));
                    }
                    _ => return Err(CompileError::Unsupported("unsupported binary operator")),
                }
            }
            Expression::FunctionCall(call) => {
                // Handle math.* function calls
                let (obj_name, method_name) = extract_method_call(call)?;
                if obj_name != "math" {
                    return Err(CompileError::Unsupported("only math.* function calls supported"));
                }
                
                // Get arguments
                let args = extract_call_args(call)?;
                
                match method_name.as_str() {
                    "abs" => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.abs requires 1 argument"));
                        }
                        emit_expr(b, &args[0], params, locals)?;
                        b.unop(UnaryOp::F64Abs);
                    }
                    "sqrt" => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.sqrt requires 1 argument"));
                        }
                        emit_expr(b, &args[0], params, locals)?;
                        b.unop(UnaryOp::F64Sqrt);
                    }
                    "floor" => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.floor requires 1 argument"));
                        }
                        emit_expr(b, &args[0], params, locals)?;
                        b.unop(UnaryOp::F64Floor);
                    }
                    "ceil" => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.ceil requires 1 argument"));
                        }
                        emit_expr(b, &args[0], params, locals)?;
                        b.unop(UnaryOp::F64Ceil);
                    }
                    "trunc" => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.trunc requires 1 argument"));
                        }
                        emit_expr(b, &args[0], params, locals)?;
                        b.unop(UnaryOp::F64Trunc);
                    }
                    "nearest" => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.nearest requires 1 argument"));
                        }
                        emit_expr(b, &args[0], params, locals)?;
                        b.unop(UnaryOp::F64Nearest);
                    }
                    "min" => {
                        if args.len() != 2 {
                            return Err(CompileError::Type("math.min requires 2 arguments"));
                        }
                        emit_expr(b, &args[0], params, locals)?;
                        emit_expr(b, &args[1], params, locals)?;
                        b.binop(BinaryOp::F64Min);
                    }
                    "max" => {
                        if args.len() != 2 {
                            return Err(CompileError::Type("math.max requires 2 arguments"));
                        }
                        emit_expr(b, &args[0], params, locals)?;
                        emit_expr(b, &args[1], params, locals)?;
                        b.binop(BinaryOp::F64Max);
                    }
                    _ => {
                        return Err(CompileError::Unsupported("unsupported math function"));
                    }
                }
            }
            Expression::Symbol(token) => {
                let sym = token.token().to_string();
                match sym.as_str() {
                    "true" => { b.f64_const(1.0); }
                    "false" => { b.f64_const(0.0); }
                    _ => return Err(CompileError::Unsupported("unsupported symbol")),
                }
            }
            _ => {
                return Err(CompileError::Unsupported("unsupported expression type"));
            }
        }
        Ok(())
    }

    // Helper to extract object.method pattern from a function call
    fn extract_method_call(call: &ast::FunctionCall) -> Result<(String, String), CompileError> {
        let prefix = call.prefix();
        let suffixes: Vec<_> = call.suffixes().collect();
        
        // Pattern: math.abs(x) -> prefix=Name("math"), suffixes=[Index::Dot("abs"), Call(...)]
        if let Prefix::Name(obj_token) = prefix {
            let obj_name = obj_token.token().to_string();
            if suffixes.len() >= 2 {
                if let Suffix::Index(Index::Dot { name, .. }) = &suffixes[0] {
                    let method_name = name.token().to_string();
                    return Ok((obj_name, method_name));
                }
            }
        }
        Err(CompileError::Unsupported("unsupported function call pattern"))
    }

    // Helper to extract arguments from a function call
    fn extract_call_args(call: &ast::FunctionCall) -> Result<Vec<Expression>, CompileError> {
        let suffixes: Vec<_> = call.suffixes().collect();
        
        // Find the Call suffix (should be the last one for math.func(args) pattern)
        for suffix in suffixes.iter().rev() {
            if let Suffix::Call(call_suffix) = suffix {
                match call_suffix {
                    Call::AnonymousCall(func_args) => {
                        match func_args {
                            FunctionArgs::Parentheses { arguments, .. } => {
                                return Ok(arguments.iter().cloned().collect());
                            }
                            FunctionArgs::String(s) => {
                                return Ok(vec![Expression::String(s.clone())]);
                            }
                            FunctionArgs::TableConstructor(t) => {
                                return Ok(vec![Expression::TableConstructor(t.clone())]);
                            }
                            _ => {}
                        }
                    }
                    Call::MethodCall(_) => {
                        return Err(CompileError::Unsupported("method call syntax not supported"));
                    }
                    _ => {}
                }
            }
        }
        Err(CompileError::Unsupported("could not extract function arguments"))
    }

    // Generate WASM functions for each required Lua function
    for &fname in REQUIRED_FUNCS {
        let (params, expr) = functions.get(fname).unwrap();
        match fname {
            "is_inside" => {
                if params.len() != 3 {
                    return Err(CompileError::Type("is_inside must take 3 params"));
                }
                let mut fb = FunctionBuilder::new(
                    &mut module.types,
                    &[ValType::F64, ValType::F64, ValType::F64],
                    &[ValType::F32],
                );
                let l_x = module.locals.add(ValType::F64);
                let l_y = module.locals.add(ValType::F64);
                let l_z = module.locals.add(ValType::F64);
                let mut ib = fb.func_body();
                emit_expr(&mut ib, expr, params, &[l_x, l_y, l_z])?;
                ib.unop(UnaryOp::F32DemoteF64);
                let fid = fb.finish(vec![l_x, l_y, l_z], &mut module.funcs);
                module.exports.add("is_inside", fid);
            }
            _ => {
                if !params.is_empty() {
                    return Err(CompileError::Type("bounds getters must have 0 params"));
                }
                let mut fb = FunctionBuilder::new(&mut module.types, &[], &[ValType::F64]);
                let mut ib = fb.func_body();
                emit_expr(&mut ib, expr, params, &[])?;
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

/// Minimal stub template for the Lua script input.
/// This template shows all required function signatures that the script must implement.
const LUA_TEMPLATE: &str = r#"-- Density field: 0.0 = empty, 1.0 = full
function is_inside(x, y, z)
    -- Example: unit sphere centered at origin
    -- Returns 1.0 if inside (r < 1), 0.0 if outside
    return math.max(0.0, 1.0 - math.sqrt(x*x + y*y + z*z))
end

-- Bounding box functions define the region to sample
function get_bounds_min_x()
    return -1.5
end

function get_bounds_min_y()
    return -1.5
end

function get_bounds_min_z()
    return -1.5
end

function get_bounds_max_x()
    return 1.5
end

function get_bounds_max_y()
    return 1.5
end

function get_bounds_max_z()
    return 1.5
end
"#;

#[no_mangle]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let bytes = METADATA.get_or_init(|| {
        let metadata = OperatorMetadata {
            name: "lua_script_operator".to_string(),
            version: "0.1.0".to_string(),
            inputs: vec![OperatorMetadataInput::LuaSource(LUA_TEMPLATE.to_string())],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_simple_sphere() {
        let lua_src = r#"
function is_inside(x, y, z)
    return x*x + y*y + z*z - 1.0
end

function get_bounds_min_x()
    return -1.5
end

function get_bounds_min_y()
    return -1.5
end

function get_bounds_min_z()
    return -1.5
end

function get_bounds_max_x()
    return 1.5
end

function get_bounds_max_y()
    return 1.5
end

function get_bounds_max_z()
    return 1.5
end
"#;
        let result = compile_lua_to_wasm(lua_src);
        assert!(result.is_ok(), "Failed to compile: {:?}", result.err());
        let wasm = result.unwrap();
        // Check that we got valid WASM (starts with magic number)
        assert!(wasm.len() > 8, "WASM output too short");
        assert_eq!(&wasm[0..4], b"\0asm", "Invalid WASM magic number");
    }

    #[test]
    fn test_compile_with_math_functions() {
        let lua_src = r#"
function is_inside(x, y, z)
    return math.sqrt(x*x + y*y + z*z) - 1.0
end

function get_bounds_min_x()
    return -2.0
end

function get_bounds_min_y()
    return -2.0
end

function get_bounds_min_z()
    return -2.0
end

function get_bounds_max_x()
    return 2.0
end

function get_bounds_max_y()
    return 2.0
end

function get_bounds_max_z()
    return 2.0
end
"#;
        let result = compile_lua_to_wasm(lua_src);
        assert!(result.is_ok(), "Failed to compile with math.sqrt: {:?}", result.err());
    }

    #[test]
    fn test_compile_with_unary_minus() {
        let lua_src = r#"
function is_inside(x, y, z)
    return -x + y - z
end

function get_bounds_min_x()
    return -1.0
end

function get_bounds_min_y()
    return -1.0
end

function get_bounds_min_z()
    return -1.0
end

function get_bounds_max_x()
    return 1.0
end

function get_bounds_max_y()
    return 1.0
end

function get_bounds_max_z()
    return 1.0
end
"#;
        let result = compile_lua_to_wasm(lua_src);
        assert!(result.is_ok(), "Failed to compile with unary minus: {:?}", result.err());
    }

    #[test]
    fn test_missing_function_error() {
        let lua_src = r#"
function is_inside(x, y, z)
    return x + y + z
end
"#;
        let result = compile_lua_to_wasm(lua_src);
        assert!(result.is_err(), "Should fail with missing functions");
        if let Err(CompileError::MissingFunc(_)) = result {
            // Expected
        } else {
            panic!("Expected MissingFunc error, got: {:?}", result);
        }
    }

    #[test]
    fn test_parse_error() {
        let lua_src = "function broken(";
        let result = compile_lua_to_wasm(lua_src);
        assert!(result.is_err(), "Should fail with parse error");
        if let Err(CompileError::Parse(_)) = result {
            // Expected
        } else {
            panic!("Expected Parse error, got: {:?}", result);
        }
    }
}

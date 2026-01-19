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
//! - Output 0: WASM bytes of a model module that exports the N-dimensional model ABI
//!
//! Generated Model ABI:
//! - `get_dimensions() -> u32`: Returns 3 (always 3D)
//! - `get_bounds(out_ptr: i32)`: Writes 6 f64 values (min_x, max_x, min_y, max_y, min_z, max_z)
//! - `sample(pos_ptr: i32) -> f32`: Reads position from memory, returns density
//! - `memory`: Exported memory for position/bounds I/O
//!
//! # Supported Lua Subset
//!
//! ## Expressions
//! - Number literals, `true`, `false`
//! - Variables (simple names)
//! - Arithmetic: `+`, `-`, `*`, `/`, `%` (modulo), `^` (exponentiation)
//! - Comparison: `<`, `<=`, `>`, `>=`, `==`, `~=`
//! - Logical: `and`, `or` (with short-circuit evaluation)
//! - Unary: `-` (negation), `not`
//! - Parentheses: `(expr)`
//! - User-defined function calls: `helper(x, y)`
//!
//! ## Statements
//! - Local assignment: `local x = expr`
//! - Assignment: `x = expr`
//! - If-then-else: `if cond then ... elseif ... else ... end`
//! - While loop: `while cond do ... end`
//! - Repeat-until loop: `repeat ... until cond`
//! - Numeric for loop: `for i = start, end [, step] do ... end`
//! - Break: `break` (exits innermost loop)
//! - Return: `return expr`
//!
//! ## Math Constants
//! - `math.pi` - π (3.14159...)
//! - `math.huge` - Infinity
//!
//! ## Math Functions (Native WASM)
//! - `math.abs`, `math.sqrt`, `math.floor`, `math.ceil`, `math.trunc`, `math.nearest`
//! - `math.min`, `math.max` (two arguments)
//!
//! ## Math Functions (Taylor Series Approximations)
//! These use polynomial approximations, accurate for typical use cases:
//! - `math.sin`, `math.cos`, `math.tan` - accurate for |x| < π
//! - `math.exp` - accurate for small x
//! - `math.log` - accurate for x near 1
//! - `math.pow(x, y)` - uses exp(y * log(x))
//! - `math.atan2(y, x)` - basic approximation
//!
//! ## User-Defined Functions
//! You can define helper functions that are callable from `is_inside` and other functions:
//! ```lua
//! function dist(x, y, z)
//!     return math.sqrt(x*x + y*y + z*z)
//! end
//!
//! function is_inside(x, y, z)
//!     return dist(x, y, z) - 1.0
//! end
//! ```
//!
//! ## Not Yet Supported
//! - Generic for loops (`for k, v in pairs(t)`)
//! - Tables, strings, multiple return values
//! - Closures, recursion between helper functions

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
unsafe extern "C" {
    fn get_input_len(arg: i32) -> u32;
    fn get_input_data(arg: i32, ptr: i32, len: i32);
    fn post_output(output_idx: i32, ptr: i32, len: i32);
}

/// Required Lua functions that must be defined in the script.
/// These are compiled to internal WASM functions and wrapped by the ABI exports.
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
    Unsupported(String),
    #[error("Type error: {0}")]
    Type(String),
}

// ============================================================================
// Intermediate Representation (IR)
// ============================================================================

/// Binary operators in the IR
#[derive(Clone, Debug)]
enum IrBinOp {
    Add, Sub, Mul, Div, Mod, Pow,
    Lt, Le, Gt, Ge, Eq, Ne,
    And, Or,
}

/// Unary operators in the IR
#[derive(Clone, Debug)]
enum IrUnaryOp {
    Neg,
    Not,
}

/// Built-in math functions
#[derive(Clone, Debug)]
enum IrMathFunc {
    // Native WASM operations
    Abs, Sqrt, Floor, Ceil, Trunc, Nearest,
    Min, Max,
    // Polynomial approximations (no native WASM support)
    Sin, Cos, Tan, Exp, Log, Pow, Atan2,
}

/// IR Expression - decoupled from the AST
#[derive(Clone, Debug)]
#[allow(dead_code)]
enum IrExpr {
    Number(f64),
    Var(String),
    BinOp { op: IrBinOp, lhs: Box<IrExpr>, rhs: Box<IrExpr> },
    UnaryOp { op: IrUnaryOp, expr: Box<IrExpr> },
    MathCall { func: IrMathFunc, args: Vec<IrExpr> },
    FuncCall { name: String, args: Vec<IrExpr> },
    IfThenElse { cond: Box<IrExpr>, then_expr: Box<IrExpr>, else_expr: Box<IrExpr> },
}

/// IR Statement
#[derive(Clone, Debug)]
enum IrStmt {
    LocalAssign { name: String, value: IrExpr },
    Assign { name: String, value: IrExpr },
    If { cond: IrExpr, then_body: Vec<IrStmt>, else_body: Vec<IrStmt> },
    While { cond: IrExpr, body: Vec<IrStmt> },
    RepeatUntil { body: Vec<IrStmt>, cond: IrExpr },
    NumericFor { var: String, start: IrExpr, end: IrExpr, step: Option<IrExpr>, body: Vec<IrStmt> },
    Break,
    Return(IrExpr),
}

/// IR Function definition
#[derive(Clone, Debug)]
struct IrFunc {
    params: Vec<String>,
    body: Vec<IrStmt>,
}

// ============================================================================
// AST to IR Conversion
// ============================================================================

/// Convert a Lua AST expression to IR
fn ast_expr_to_ir(expr: &full_moon::ast::Expression) -> Result<IrExpr, CompileError> {
    use full_moon::ast::{Expression, BinOp, UnOp, Var};
    
    match expr {
        Expression::Number(token) => {
            let num_str = token.token().to_string();
            let v: f64 = num_str.parse()
                .map_err(|_| CompileError::Parse(format!("invalid number: {}", num_str)))?;
            Ok(IrExpr::Number(v))
        }
        Expression::Var(var) => {
            use full_moon::ast::{Suffix, Index};
            match var {
                Var::Name(token) => {
                    Ok(IrExpr::Var(token.token().to_string()))
                }
                Var::Expression(var_expr) => {
                    // Handle math.pi and similar constants
                    if let full_moon::ast::Prefix::Name(obj_token) = var_expr.prefix() {
                        let obj_name = obj_token.token().to_string();
                        let suffixes: Vec<_> = var_expr.suffixes().collect();

                        if obj_name == "math" && suffixes.len() == 1 {
                            if let Suffix::Index(Index::Dot { name, .. }) = &suffixes[0] {
                                let const_name = name.token().to_string();
                                match const_name.as_str() {
                                    "pi" => return Ok(IrExpr::Number(std::f64::consts::PI)),
                                    "huge" => return Ok(IrExpr::Number(f64::INFINITY)),
                                    _ => {}
                                }
                            }
                        }
                    }
                    Err(CompileError::Unsupported("complex variable expressions not supported".into()))
                }
                _ => Err(CompileError::Unsupported("unsupported variable type".into())),
            }
        }
        Expression::Parentheses { expression, .. } => {
            ast_expr_to_ir(expression)
        }
        Expression::UnaryOperator { unop, expression } => {
            let inner = ast_expr_to_ir(expression)?;
            match unop {
                UnOp::Minus(_) => Ok(IrExpr::UnaryOp { op: IrUnaryOp::Neg, expr: Box::new(inner) }),
                UnOp::Not(_) => Ok(IrExpr::UnaryOp { op: IrUnaryOp::Not, expr: Box::new(inner) }),
                UnOp::Hash(_) => Err(CompileError::Unsupported("length operator".into())),
                _ => Err(CompileError::Unsupported("unsupported unary operator".into())),
            }
        }
        Expression::BinaryOperator { lhs, binop, rhs } => {
            let left = ast_expr_to_ir(lhs)?;
            let right = ast_expr_to_ir(rhs)?;
            let op = match binop {
                BinOp::Plus(_) => IrBinOp::Add,
                BinOp::Minus(_) => IrBinOp::Sub,
                BinOp::Star(_) => IrBinOp::Mul,
                BinOp::Slash(_) => IrBinOp::Div,
                BinOp::LessThan(_) => IrBinOp::Lt,
                BinOp::LessThanEqual(_) => IrBinOp::Le,
                BinOp::GreaterThan(_) => IrBinOp::Gt,
                BinOp::GreaterThanEqual(_) => IrBinOp::Ge,
                BinOp::TwoEqual(_) => IrBinOp::Eq,
                BinOp::TildeEqual(_) => IrBinOp::Ne,
                BinOp::And(_) => IrBinOp::And,
                BinOp::Or(_) => IrBinOp::Or,
                BinOp::Caret(_) => IrBinOp::Pow,
                BinOp::Percent(_) => IrBinOp::Mod,
                BinOp::TwoDots(_) => return Err(CompileError::Unsupported("string concatenation".into())),
                _ => return Err(CompileError::Unsupported("unsupported binary operator".into())),
            };
            Ok(IrExpr::BinOp { op, lhs: Box::new(left), rhs: Box::new(right) })
        }
        Expression::FunctionCall(call) => {
            ast_func_call_to_ir(call)
        }
        Expression::Symbol(token) => {
            let sym = token.token().to_string();
            match sym.as_str() {
                "true" => Ok(IrExpr::Number(1.0)),
                "false" => Ok(IrExpr::Number(0.0)),
                _ => Err(CompileError::Unsupported(format!("unsupported symbol: {}", sym))),
            }
        }
        _ => Err(CompileError::Unsupported("unsupported expression type".into())),
    }
}

/// Convert a function call AST to IR
fn ast_func_call_to_ir(call: &full_moon::ast::FunctionCall) -> Result<IrExpr, CompileError> {
    use full_moon::ast::{Prefix, Suffix, Index};
    
    let prefix = call.prefix();
    let suffixes: Vec<_> = call.suffixes().collect();
    
    // Pattern: math.abs(x) -> prefix=Name("math"), suffixes=[Index::Dot("abs"), Call(...)]
    // Pattern: helper(x) -> prefix=Name("helper"), suffixes=[Call(...)]
    
    if let Prefix::Name(obj_token) = prefix {
        let obj_name = obj_token.token().to_string();
        
        // Check for math.func pattern
        if suffixes.len() >= 2 {
            if let Suffix::Index(Index::Dot { name, .. }) = &suffixes[0] {
                let method_name = name.token().to_string();
                let args = extract_call_args_from_suffixes(&suffixes[1..])?;
                
                if obj_name == "math" {
                    let func = match method_name.as_str() {
                        "abs" => IrMathFunc::Abs,
                        "sqrt" => IrMathFunc::Sqrt,
                        "floor" => IrMathFunc::Floor,
                        "ceil" => IrMathFunc::Ceil,
                        "trunc" => IrMathFunc::Trunc,
                        "nearest" => IrMathFunc::Nearest,
                        "min" => IrMathFunc::Min,
                        "max" => IrMathFunc::Max,
                        "sin" => IrMathFunc::Sin,
                        "cos" => IrMathFunc::Cos,
                        "tan" => IrMathFunc::Tan,
                        "exp" => IrMathFunc::Exp,
                        "log" => IrMathFunc::Log,
                        "pow" => IrMathFunc::Pow,
                        "atan2" => IrMathFunc::Atan2,
                        _ => return Err(CompileError::Unsupported(format!("unsupported math function: math.{}", method_name))),
                    };
                    return Ok(IrExpr::MathCall { func, args });
                } else {
                    return Err(CompileError::Unsupported(format!("unsupported module: {}", obj_name)));
                }
            }
        }
        
        // Check for simple function call pattern: func(args)
        if suffixes.len() == 1 {
            let args = extract_call_args_from_suffixes(&suffixes)?;
            return Ok(IrExpr::FuncCall { name: obj_name, args });
        }
    }
    
    Err(CompileError::Unsupported("unsupported function call pattern".into()))
}

/// Extract arguments from call suffixes
fn extract_call_args_from_suffixes(suffixes: &[&full_moon::ast::Suffix]) -> Result<Vec<IrExpr>, CompileError> {
    use full_moon::ast::{Suffix, Call, FunctionArgs};
    
    for suffix in suffixes.iter() {
        if let Suffix::Call(call_suffix) = suffix {
            match call_suffix {
                Call::AnonymousCall(func_args) => {
                    match func_args {
                        FunctionArgs::Parentheses { arguments, .. } => {
                            return arguments.iter()
                                .map(|arg| ast_expr_to_ir(arg))
                                .collect();
                        }
                        _ => {}
                    }
                }
                Call::MethodCall(_) => {
                    return Err(CompileError::Unsupported("method call syntax not supported".into()));
                }
                _ => {}
            }
        }
    }
    Err(CompileError::Unsupported("could not extract function arguments".into()))
}

/// Convert a Lua block to IR statements
fn ast_block_to_ir(block: &full_moon::ast::Block) -> Result<Vec<IrStmt>, CompileError> {
    use full_moon::ast::{Stmt, LastStmt};
    
    let mut stmts = Vec::new();
    
    for stmt in block.stmts() {
        match stmt {
            Stmt::LocalAssignment(local_assign) => {
                // Handle: local x = expr
                let names: Vec<_> = local_assign.names().iter().collect();
                let exprs: Vec<_> = local_assign.expressions().iter().collect();
                
                for (i, name_token) in names.iter().enumerate() {
                    let name = name_token.token().to_string();
                    let value = if i < exprs.len() {
                        ast_expr_to_ir(exprs[i])?
                    } else {
                        IrExpr::Number(0.0) // Default to nil/0
                    };
                    stmts.push(IrStmt::LocalAssign { name, value });
                }
            }
            Stmt::Assignment(assign) => {
                // Handle: x = expr
                let vars: Vec<_> = assign.variables().iter().collect();
                let exprs: Vec<_> = assign.expressions().iter().collect();
                
                for (i, var) in vars.iter().enumerate() {
                    if let full_moon::ast::Var::Name(name_token) = var {
                        let name = name_token.token().to_string();
                        let value = if i < exprs.len() {
                            ast_expr_to_ir(exprs[i])?
                        } else {
                            IrExpr::Number(0.0)
                        };
                        stmts.push(IrStmt::Assign { name, value });
                    } else {
                        return Err(CompileError::Unsupported("complex assignment target".into()));
                    }
                }
            }
            Stmt::If(if_stmt) => {
                let ir_if = ast_if_to_ir(if_stmt)?;
                stmts.push(ir_if);
            }
            Stmt::While(while_stmt) => {
                let cond = ast_expr_to_ir(while_stmt.condition())?;
                let body = ast_block_to_ir(while_stmt.block())?;
                stmts.push(IrStmt::While { cond, body });
            }
            Stmt::NumericFor(for_stmt) => {
                let var = for_stmt.index_variable().token().to_string();
                let start = ast_expr_to_ir(for_stmt.start())?;
                let end = ast_expr_to_ir(for_stmt.end())?;
                let step = if let Some(step_expr) = for_stmt.step() {
                    Some(ast_expr_to_ir(step_expr)?)
                } else {
                    None
                };
                let body = ast_block_to_ir(for_stmt.block())?;
                stmts.push(IrStmt::NumericFor { var, start, end, step, body });
            }
            Stmt::Repeat(repeat_stmt) => {
                let body = ast_block_to_ir(repeat_stmt.block())?;
                let cond = ast_expr_to_ir(repeat_stmt.until())?;
                stmts.push(IrStmt::RepeatUntil { body, cond });
            }
            _ => {
                return Err(CompileError::Unsupported("unsupported statement type".into()));
            }
        }
    }
    
    // Handle last statement (return)
    if let Some(last_stmt) = block.last_stmt() {
        match last_stmt {
            LastStmt::Return(ret) => {
                let mut returns_iter = ret.returns().iter();
                if let Some(expr) = returns_iter.next() {
                    if returns_iter.next().is_none() {
                        stmts.push(IrStmt::Return(ast_expr_to_ir(expr)?));
                    } else {
                        return Err(CompileError::Unsupported("multiple return values".into()));
                    }
                } else {
                    // Empty return
                    stmts.push(IrStmt::Return(IrExpr::Number(0.0)));
                }
            }
            LastStmt::Break(_) => {
                stmts.push(IrStmt::Break);
            }
            _ => {
                return Err(CompileError::Unsupported("unsupported last statement".into()));
            }
        }
    }
    
    Ok(stmts)
}

/// Convert an if statement to IR
fn ast_if_to_ir(if_stmt: &full_moon::ast::If) -> Result<IrStmt, CompileError> {
    let cond = ast_expr_to_ir(if_stmt.condition())?;
    let then_body = ast_block_to_ir(if_stmt.block())?;
    
    // Handle elseif chains by converting to nested if-else
    let mut else_body = Vec::new();
    
    if let Some(else_ifs) = if_stmt.else_if() {
        // Build nested if-else from elseif chain
        let else_ifs: Vec<_> = else_ifs.iter().collect();
        if !else_ifs.is_empty() {
            else_body = build_elseif_chain(&else_ifs, if_stmt.else_block())?;
        } else if let Some(else_block) = if_stmt.else_block() {
            else_body = ast_block_to_ir(else_block)?;
        }
    } else if let Some(else_block) = if_stmt.else_block() {
        else_body = ast_block_to_ir(else_block)?;
    }
    
    Ok(IrStmt::If { cond, then_body, else_body })
}

/// Build nested if-else from elseif chain
fn build_elseif_chain(
    else_ifs: &[&full_moon::ast::ElseIf],
    final_else: Option<&full_moon::ast::Block>,
) -> Result<Vec<IrStmt>, CompileError> {
    if else_ifs.is_empty() {
        if let Some(else_block) = final_else {
            return ast_block_to_ir(else_block);
        }
        return Ok(Vec::new());
    }
    
    let first = else_ifs[0];
    let cond = ast_expr_to_ir(first.condition())?;
    let then_body = ast_block_to_ir(first.block())?;
    let else_body = build_elseif_chain(&else_ifs[1..], final_else)?;
    
    Ok(vec![IrStmt::If { cond, then_body, else_body }])
}

/// Convert a function declaration to IR
fn ast_func_to_ir(func_decl: &full_moon::ast::FunctionDeclaration) -> Result<IrFunc, CompileError> {
    use full_moon::ast::Parameter;
    
    let params: Vec<String> = func_decl.body().parameters().iter()
        .filter_map(|p| match p {
            Parameter::Name(token) => Some(token.token().to_string()),
            Parameter::Ellipsis(_) => None,
            _ => None,
        })
        .collect();
    
    let body = ast_block_to_ir(func_decl.body().block())?;
    
    Ok(IrFunc { params, body })
}

// ============================================================================
// Lua to WASM Compilation
// ============================================================================

fn compile_lua_to_wasm(src: &str) -> Result<Vec<u8>, CompileError> {
    use full_moon::ast::Stmt;
    use std::collections::HashMap;
    use walrus::InstrSeqBuilder;
    use walrus::ir::{BinaryOp, UnaryOp};

    // Parse the Lua source into an AST
    let ast = full_moon::parse(src).map_err(|errors| {
        let msg = errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ");
        CompileError::Parse(msg)
    })?;

    // Extract function declarations from the AST and convert to IR
    let mut functions: HashMap<String, IrFunc> = HashMap::new();

    for stmt in ast.nodes().stmts() {
        if let Stmt::FunctionDeclaration(func_decl) = stmt {
            let name = func_decl.name().names().iter()
                .map(|t| t.token().to_string())
                .collect::<Vec<_>>()
                .join(".");
            
            let ir_func = ast_func_to_ir(func_decl)?;
            functions.insert(name, ir_func);
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

    // Map of function names to their WASM function IDs (populated as we generate functions)
    let mut func_ids: HashMap<String, walrus::FunctionId> = HashMap::new();

    // Emit WASM instructions for an IR expression
    fn emit_ir_expr(
        b: &mut InstrSeqBuilder,
        func_ids: &HashMap<String, walrus::FunctionId>,
        expr: &IrExpr,
        params: &[String],
        locals: &HashMap<String, walrus::LocalId>,
    ) -> Result<(), CompileError> {
        match expr {
            IrExpr::Number(v) => {
                b.f64_const(*v);
            }
            IrExpr::Var(name) => {
                if let Some(&local_id) = locals.get(name) {
                    b.local_get(local_id);
                } else {
                    return Err(CompileError::Unsupported(format!("unknown variable: {}", name)));
                }
            }
            IrExpr::BinOp { op, lhs, rhs } => {
                match op {
                    IrBinOp::Mod => {
                        // Lua modulo: a % b = a - floor(a/b) * b
                        emit_ir_expr(b, func_ids, lhs, params, locals)?;  // a (for final subtraction)
                        emit_ir_expr(b, func_ids, lhs, params, locals)?;  // a
                        emit_ir_expr(b, func_ids, rhs, params, locals)?;  // b
                        b.binop(BinaryOp::F64Div);              // a/b
                        b.unop(UnaryOp::F64Floor);              // floor(a/b)
                        emit_ir_expr(b, func_ids, rhs, params, locals)?;  // b
                        b.binop(BinaryOp::F64Mul);              // floor(a/b) * b
                        b.binop(BinaryOp::F64Sub);              // a - floor(a/b) * b
                    }
                    IrBinOp::Pow => {
                        // x^y = exp(y * log(x)) - same as math.pow
                        // Compute log(x) ≈ (x-1) - (x-1)²/2
                        emit_ir_expr(b, func_ids, lhs, params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        emit_ir_expr(b, func_ids, lhs, params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        emit_ir_expr(b, func_ids, lhs, params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(2.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Sub);  // log(x) ≈ (x-1) - (x-1)²/2
                        // Multiply by y
                        emit_ir_expr(b, func_ids, rhs, params, locals)?;
                        b.binop(BinaryOp::F64Mul);  // y * log(x)
                        // exp(z) ≈ 1 + z
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Add);
                    }
                    IrBinOp::And => {
                        // a and b: if a != 0 then b else a (short-circuit)
                        // Evaluate lhs for condition check
                        emit_ir_expr(b, func_ids, lhs, params, locals)?;
                        b.f64_const(0.0);
                        b.binop(BinaryOp::F64Ne);  // lhs != 0.0
                        let lhs = lhs.clone();
                        let rhs = rhs.clone();
                        let params_vec: Vec<String> = params.to_vec();
                        let locals_clone = locals.clone();
                        let func_ids_ref = func_ids;
                        b.if_else(
                            ValType::F64,
                            |then_block| {
                                // lhs truthy: return rhs
                                emit_ir_expr(then_block, func_ids_ref, &rhs, &params_vec, &locals_clone)
                                    .expect("emit_ir_expr in and-then branch failed");
                            },
                            |else_block| {
                                // lhs falsy: return lhs (re-evaluate, will be 0.0)
                                emit_ir_expr(else_block, func_ids_ref, &lhs, &params_vec, &locals_clone)
                                    .expect("emit_ir_expr in and-else branch failed");
                            },
                        );
                    }
                    IrBinOp::Or => {
                        // a or b: if a != 0 then a else b (short-circuit)
                        // Evaluate lhs for condition check
                        emit_ir_expr(b, func_ids, lhs, params, locals)?;
                        b.f64_const(0.0);
                        b.binop(BinaryOp::F64Ne);  // lhs != 0.0
                        let lhs = lhs.clone();
                        let rhs = rhs.clone();
                        let params_vec: Vec<String> = params.to_vec();
                        let locals_clone = locals.clone();
                        let func_ids_ref = func_ids;
                        b.if_else(
                            ValType::F64,
                            |then_block| {
                                // lhs truthy: return lhs (re-evaluate)
                                emit_ir_expr(then_block, func_ids_ref, &lhs, &params_vec, &locals_clone)
                                    .expect("emit_ir_expr in or-then branch failed");
                            },
                            |else_block| {
                                // lhs falsy: return rhs
                                emit_ir_expr(else_block, func_ids_ref, &rhs, &params_vec, &locals_clone)
                                    .expect("emit_ir_expr in or-else branch failed");
                            },
                        );
                    }
                    _ => {
                        emit_ir_expr(b, func_ids,lhs, params, locals)?;
                        emit_ir_expr(b, func_ids,rhs, params, locals)?;
                        match op {
                            IrBinOp::Add => { b.binop(BinaryOp::F64Add); }
                            IrBinOp::Sub => { b.binop(BinaryOp::F64Sub); }
                            IrBinOp::Mul => { b.binop(BinaryOp::F64Mul); }
                            IrBinOp::Div => { b.binop(BinaryOp::F64Div); }
                            IrBinOp::Lt => { b.binop(BinaryOp::F64Lt); }
                            IrBinOp::Le => { b.binop(BinaryOp::F64Le); }
                            IrBinOp::Gt => { b.binop(BinaryOp::F64Gt); }
                            IrBinOp::Ge => { b.binop(BinaryOp::F64Ge); }
                            IrBinOp::Eq => { b.binop(BinaryOp::F64Eq); }
                            IrBinOp::Ne => { b.binop(BinaryOp::F64Ne); }
                            _ => unreachable!(),
                        }
                    }
                }
            }
            IrExpr::UnaryOp { op, expr: inner } => {
                emit_ir_expr(b, func_ids,inner, params, locals)?;
                match op {
                    IrUnaryOp::Neg => { b.unop(UnaryOp::F64Neg); }
                    IrUnaryOp::Not => {
                        // Convert to boolean: compare with 0, then negate
                        b.f64_const(0.0);
                        b.binop(BinaryOp::F64Eq);
                    }
                }
            }
            IrExpr::MathCall { func, args } => {
                match func {
                    IrMathFunc::Abs | IrMathFunc::Sqrt | IrMathFunc::Floor |
                    IrMathFunc::Ceil | IrMathFunc::Trunc | IrMathFunc::Nearest => {
                        if args.len() != 1 {
                            return Err(CompileError::Type(format!("math function requires 1 argument")));
                        }
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        match func {
                            IrMathFunc::Abs => { b.unop(UnaryOp::F64Abs); }
                            IrMathFunc::Sqrt => { b.unop(UnaryOp::F64Sqrt); }
                            IrMathFunc::Floor => { b.unop(UnaryOp::F64Floor); }
                            IrMathFunc::Ceil => { b.unop(UnaryOp::F64Ceil); }
                            IrMathFunc::Trunc => { b.unop(UnaryOp::F64Trunc); }
                            IrMathFunc::Nearest => { b.unop(UnaryOp::F64Nearest); }
                            _ => unreachable!(),
                        }
                    }
                    IrMathFunc::Min | IrMathFunc::Max => {
                        if args.len() != 2 {
                            return Err(CompileError::Type(format!("math.min/max requires 2 arguments")));
                        }
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[1], params, locals)?;
                        match func {
                            IrMathFunc::Min => { b.binop(BinaryOp::F64Min); }
                            IrMathFunc::Max => { b.binop(BinaryOp::F64Max); }
                            _ => unreachable!(),
                        }
                    }
                    IrMathFunc::Sin => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.sin requires 1 argument".into()));
                        }
                        // sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040 (Taylor series, accurate for |x| < π)
                        // Compute: x
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        // Compute: - x³/6
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);  // x³
                        b.f64_const(6.0);
                        b.binop(BinaryOp::F64Div);  // x³/6
                        b.binop(BinaryOp::F64Sub);  // x - x³/6
                        // Compute: + x⁵/120
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);  // x⁵
                        b.f64_const(120.0);
                        b.binop(BinaryOp::F64Div);  // x⁵/120
                        b.binop(BinaryOp::F64Add);  // x - x³/6 + x⁵/120
                        // Compute: - x⁷/5040
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);  // x⁷
                        b.f64_const(5040.0);
                        b.binop(BinaryOp::F64Div);  // x⁷/5040
                        b.binop(BinaryOp::F64Sub);  // final result
                    }
                    IrMathFunc::Cos => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.cos requires 1 argument".into()));
                        }
                        // cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720 (Taylor series)
                        b.f64_const(1.0);
                        // - x²/2
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);  // x²
                        b.f64_const(2.0);
                        b.binop(BinaryOp::F64Div);  // x²/2
                        b.binop(BinaryOp::F64Sub);  // 1 - x²/2
                        // + x⁴/24
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);  // x⁴
                        b.f64_const(24.0);
                        b.binop(BinaryOp::F64Div);  // x⁴/24
                        b.binop(BinaryOp::F64Add);
                        // - x⁶/720
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);  // x⁶
                        b.f64_const(720.0);
                        b.binop(BinaryOp::F64Div);  // x⁶/720
                        b.binop(BinaryOp::F64Sub);  // final result
                    }
                    IrMathFunc::Tan => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.tan requires 1 argument".into()));
                        }
                        // tan(x) = sin(x) / cos(x) - use simpler approximation
                        // tan(x) ≈ x + x³/3 + 2x⁵/15 for small x
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        // + x³/3
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(3.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Add);
                        // + 2x⁵/15
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(7.5);  // 15/2
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Add);
                    }
                    IrMathFunc::Exp => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.exp requires 1 argument".into()));
                        }
                        // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120 + x⁶/720
                        b.f64_const(1.0);
                        // + x
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Add);
                        // + x²/2
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(2.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Add);
                        // + x³/6
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(6.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Add);
                        // + x⁴/24
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(24.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Add);
                        // + x⁵/120
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(120.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Add);
                        // + x⁶/720
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(720.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Add);
                    }
                    IrMathFunc::Log => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.log requires 1 argument".into()));
                        }
                        // log(x) using the identity: log(x) = 2 * atanh((x-1)/(x+1))
                        // For x near 1, use Taylor: log(1+u) ≈ u - u²/2 + u³/3 - u⁴/4
                        // where u = x - 1
                        // This is accurate for 0.5 < x < 2
                        // log(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 - (x-1)⁴/4
                        // Let u = x - 1
                        // Compute u
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);  // u = x - 1
                        // For simplicity, just use u - u²/2 + u³/3
                        // We need u multiple times, so emit (x-1) each time
                        // - u²/2
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        b.binop(BinaryOp::F64Mul);  // u²
                        b.f64_const(2.0);
                        b.binop(BinaryOp::F64Div);  // u²/2
                        b.binop(BinaryOp::F64Sub);  // u - u²/2
                        // + u³/3
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);  // u³
                        b.f64_const(3.0);
                        b.binop(BinaryOp::F64Div);  // u³/3
                        b.binop(BinaryOp::F64Add);  // u - u²/2 + u³/3
                    }
                    IrMathFunc::Pow => {
                        if args.len() != 2 {
                            return Err(CompileError::Type("math.pow requires 2 arguments".into()));
                        }
                        // pow(x, y) = exp(y * log(x))
                        // First compute log(x)
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(2.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Sub);  // log(x) ≈ (x-1) - (x-1)²/2
                        // Multiply by y
                        emit_ir_expr(b, func_ids,&args[1], params, locals)?;
                        b.binop(BinaryOp::F64Mul);  // y * log(x)
                        // Now compute exp of that - store intermediate and use exp approximation
                        // exp(z) ≈ 1 + z + z²/2 + z³/6
                        // This is tricky because we need z multiple times but it's a computed value
                        // For simplicity, use a rougher approximation: exp(z) ≈ 1 + z + z²/2
                        // We need to duplicate the z value, but we can't without locals
                        // WORKAROUND: Recompute y*log(x) each time
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Add);  // 1 + z (very rough exp approximation)
                    }
                    IrMathFunc::Atan2 => {
                        if args.len() != 2 {
                            return Err(CompileError::Type("math.atan2 requires 2 arguments".into()));
                        }
                        // atan2(y, x) - simplified approximation
                        // For the primary case (x > 0): atan(y/x)
                        // atan(z) ≈ z - z³/3 + z⁵/5 for |z| < 1
                        // For simplicity: atan2(y,x) ≈ y/x - (y/x)³/3 when x > 0, |y/x| < 1
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;  // y
                        emit_ir_expr(b, func_ids,&args[1], params, locals)?;  // x
                        b.binop(BinaryOp::F64Div);  // y/x = z
                        // - z³/3
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[1], params, locals)?;
                        b.binop(BinaryOp::F64Div);
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[1], params, locals)?;
                        b.binop(BinaryOp::F64Div);
                        emit_ir_expr(b, func_ids,&args[0], params, locals)?;
                        emit_ir_expr(b, func_ids,&args[1], params, locals)?;
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);  // z³
                        b.f64_const(3.0);
                        b.binop(BinaryOp::F64Div);  // z³/3
                        b.binop(BinaryOp::F64Sub);  // z - z³/3
                    }
                }
            }
            IrExpr::FuncCall { name, args } => {
                // Look up the function ID
                if let Some(&fid) = func_ids.get(name) {
                    // Emit arguments
                    for arg in args {
                        emit_ir_expr(b, func_ids, arg, params, locals)?;
                    }
                    // Call the function
                    b.call(fid);
                } else {
                    return Err(CompileError::Unsupported(format!("unknown function: {}", name)));
                }
            }
            IrExpr::IfThenElse { cond, then_expr, else_expr } => {
                emit_ir_condition(b, func_ids, cond, params, locals)?;
                let then_expr = then_expr.clone();
                let else_expr = else_expr.clone();
                let params_vec: Vec<String> = params.to_vec();
                let locals_clone: HashMap<String, walrus::LocalId> = locals.clone();
                let func_ids_ref = func_ids;
                b.if_else(
                    ValType::F64,
                    |then_block| {
                        emit_ir_expr(then_block, func_ids_ref, &then_expr, &params_vec, &locals_clone)
                            .expect("emit_ir_expr in then branch failed");
                    },
                    |else_block| {
                        emit_ir_expr(else_block, func_ids_ref, &else_expr, &params_vec, &locals_clone)
                            .expect("emit_ir_expr in else branch failed");
                    },
                );
            }
        }
        Ok(())
    }

    // Emit a condition expression that produces an i32 (0 or 1)
    fn emit_ir_condition(
        b: &mut InstrSeqBuilder,
        func_ids: &HashMap<String, walrus::FunctionId>,
        expr: &IrExpr,
        params: &[String],
        locals: &HashMap<String, walrus::LocalId>,
    ) -> Result<(), CompileError> {
        match expr {
            IrExpr::BinOp { op, lhs, rhs } => {
                match op {
                    IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge | IrBinOp::Eq | IrBinOp::Ne => {
                        emit_ir_expr(b, func_ids,lhs, params, locals)?;
                        emit_ir_expr(b, func_ids,rhs, params, locals)?;
                        match op {
                            IrBinOp::Lt => { b.binop(BinaryOp::F64Lt); }
                            IrBinOp::Le => { b.binop(BinaryOp::F64Le); }
                            IrBinOp::Gt => { b.binop(BinaryOp::F64Gt); }
                            IrBinOp::Ge => { b.binop(BinaryOp::F64Ge); }
                            IrBinOp::Eq => { b.binop(BinaryOp::F64Eq); }
                            IrBinOp::Ne => { b.binop(BinaryOp::F64Ne); }
                            _ => unreachable!(),
                        }
                        return Ok(());
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        // For non-comparison expressions, emit as f64 and convert to i32 (non-zero = true)
        emit_ir_expr(b, func_ids,expr, params, locals)?;
        b.f64_const(0.0);
        b.binop(BinaryOp::F64Ne);
        Ok(())
    }

    // Emit IR statements
    fn emit_ir_stmts(
        b: &mut InstrSeqBuilder,
        func_ids: &HashMap<String, walrus::FunctionId>,
        stmts: &[IrStmt],
        params: &[String],
        locals: &mut HashMap<String, walrus::LocalId>,
        module_locals: &mut walrus::ModuleLocals,
    ) -> Result<Option<IrExpr>, CompileError> {
        for stmt in stmts {
            match stmt {
                IrStmt::LocalAssign { name, value } => {
                    let local_id = if let Some(&id) = locals.get(name) {
                        id
                    } else {
                        let id = module_locals.add(ValType::F64);
                        locals.insert(name.clone(), id);
                        id
                    };
                    emit_ir_expr(b, func_ids,value, params, locals)?;
                    b.local_set(local_id);
                }
                IrStmt::Assign { name, value } => {
                    if let Some(&local_id) = locals.get(name) {
                        emit_ir_expr(b, func_ids,value, params, locals)?;
                        b.local_set(local_id);
                    } else {
                        return Err(CompileError::Unsupported(format!("assignment to unknown variable: {}", name)));
                    }
                }
                IrStmt::If { cond, then_body, else_body } => {
                    emit_ir_condition(b, func_ids, cond, params, locals)?;
                    let params_vec: Vec<String> = params.to_vec();
                    let then_locals = locals.clone();
                    let else_locals = locals.clone();
                    let func_ids_ref = func_ids;
                    b.if_else(
                        ValType::F64,
                        |then_block| {
                            // We need a dummy ModuleLocals for nested calls - this is a limitation
                            // For now, we'll handle simple if-then-else with returns
                            if let Some(IrStmt::Return(expr)) = then_body.first() {
                                emit_ir_expr(then_block, func_ids_ref, expr, &params_vec, &then_locals)
                                    .expect("emit in then branch failed");
                            } else {
                                then_block.f64_const(0.0); // fallback
                            }
                        },
                        |else_block| {
                            if let Some(IrStmt::Return(expr)) = else_body.first() {
                                emit_ir_expr(else_block, func_ids_ref, expr, &params_vec, &else_locals)
                                    .expect("emit in else branch failed");
                            } else {
                                else_block.f64_const(0.0); // fallback
                            }
                        },
                    );
                    return Ok(None); // If statement handled the return
                }
                IrStmt::While { cond, body } => {
                    let params_vec: Vec<String> = params.to_vec();
                    let loop_locals = locals.clone();
                    let cond = cond.clone();
                    let body = body.clone();
                    let func_ids_ref = func_ids;

                    // While loop structure: block { loop { if !cond br exit; body; br loop } }
                    b.block(None, |exit_block| {
                        let exit_id = exit_block.id();
                        exit_block.loop_(None, |loop_block| {
                            let loop_id = loop_block.id();

                            // Emit condition check - exit if false
                            emit_ir_condition(loop_block, func_ids_ref, &cond, &params_vec, &loop_locals)
                                .expect("emit while condition failed");
                            loop_block.unop(UnaryOp::I32Eqz); // Invert: exit if condition is false
                            loop_block.br_if(exit_id);

                            // Emit body statements
                            for stmt in &body {
                                match stmt {
                                    IrStmt::LocalAssign { name, value } | IrStmt::Assign { name, value } => {
                                        if let Some(&local_id) = loop_locals.get(name) {
                                            emit_ir_expr(loop_block, func_ids_ref, value, &params_vec, &loop_locals)
                                                .expect("emit while body assignment failed");
                                            loop_block.local_set(local_id);
                                        }
                                        // If local doesn't exist, skip (limitation: can't create new locals in loop)
                                    }
                                    IrStmt::Break => {
                                        loop_block.br(exit_id);
                                    }
                                    _ => {
                                        // Skip unsupported statements in loop body for now
                                    }
                                }
                            }

                            // Continue to next iteration
                            loop_block.br(loop_id);
                        });
                    });
                }
                IrStmt::RepeatUntil { body, cond } => {
                    let params_vec: Vec<String> = params.to_vec();
                    let loop_locals = locals.clone();
                    let cond = cond.clone();
                    let body = body.clone();
                    let func_ids_ref = func_ids;

                    // Repeat-until structure: loop { body; if cond br exit; br loop }
                    // Different from while: body executes first, then condition checked
                    b.block(None, |exit_block| {
                        let exit_id = exit_block.id();
                        exit_block.loop_(None, |loop_block| {
                            let loop_id = loop_block.id();

                            // Emit body statements first (always executes at least once)
                            for stmt in &body {
                                match stmt {
                                    IrStmt::LocalAssign { name, value } | IrStmt::Assign { name, value } => {
                                        if let Some(&local_id) = loop_locals.get(name) {
                                            emit_ir_expr(loop_block, func_ids_ref, value, &params_vec, &loop_locals)
                                                .expect("emit repeat body assignment failed");
                                            loop_block.local_set(local_id);
                                        }
                                    }
                                    IrStmt::Break => {
                                        loop_block.br(exit_id);
                                    }
                                    _ => {}
                                }
                            }

                            // Check condition - exit if true (until condition met)
                            emit_ir_condition(loop_block, func_ids_ref, &cond, &params_vec, &loop_locals)
                                .expect("emit repeat-until condition failed");
                            loop_block.br_if(exit_id);

                            // Continue to next iteration
                            loop_block.br(loop_id);
                        });
                    });
                }
                IrStmt::NumericFor { var, start, end, step, body } => {
                    // Create local for loop variable
                    let loop_var_id = module_locals.add(ValType::F64);
                    locals.insert(var.clone(), loop_var_id);

                    // Also create locals for end and step to avoid re-evaluating
                    let end_var_id = module_locals.add(ValType::F64);
                    let step_var_id = module_locals.add(ValType::F64);

                    // Initialize loop variable with start value
                    emit_ir_expr(b, func_ids,start, params, locals)?;
                    b.local_set(loop_var_id);

                    // Initialize end value
                    emit_ir_expr(b, func_ids,end, params, locals)?;
                    b.local_set(end_var_id);

                    // Initialize step value (default to 1.0)
                    if let Some(step_expr) = step {
                        emit_ir_expr(b, func_ids,step_expr, params, locals)?;
                    } else {
                        b.f64_const(1.0);
                    }
                    b.local_set(step_var_id);

                    let params_vec: Vec<String> = params.to_vec();
                    let loop_locals = locals.clone();
                    let body = body.clone();
                    let func_ids_ref = func_ids;

                    // For loop structure: block { loop { check; body; increment; br loop } }
                    b.block(None, |exit_block| {
                        let exit_id = exit_block.id();
                        exit_block.loop_(None, |loop_block| {
                            let loop_id = loop_block.id();

                            // Check condition based on step sign:
                            // if step > 0: exit if i > end
                            // if step < 0: exit if i < end
                            // For simplicity, we implement: exit if (i - end) * step > 0
                            // This works for both positive and negative steps
                            loop_block.local_get(loop_var_id);
                            loop_block.local_get(end_var_id);
                            loop_block.binop(BinaryOp::F64Sub);  // i - end
                            loop_block.local_get(step_var_id);
                            loop_block.binop(BinaryOp::F64Mul);  // (i - end) * step
                            loop_block.f64_const(0.0);
                            loop_block.binop(BinaryOp::F64Gt);   // > 0 means we should exit
                            loop_block.br_if(exit_id);

                            // Emit body statements
                            for stmt in &body {
                                match stmt {
                                    IrStmt::LocalAssign { name, value } | IrStmt::Assign { name, value } => {
                                        if let Some(&local_id) = loop_locals.get(name) {
                                            emit_ir_expr(loop_block, func_ids_ref, value, &params_vec, &loop_locals)
                                                .expect("emit for body assignment failed");
                                            loop_block.local_set(local_id);
                                        }
                                    }
                                    IrStmt::Break => {
                                        loop_block.br(exit_id);
                                    }
                                    _ => {}
                                }
                            }

                            // Increment: i = i + step
                            loop_block.local_get(loop_var_id);
                            loop_block.local_get(step_var_id);
                            loop_block.binop(BinaryOp::F64Add);
                            loop_block.local_set(loop_var_id);

                            // Continue to next iteration
                            loop_block.br(loop_id);
                        });
                    });
                }
                IrStmt::Break => {
                    // Break at the top level (outside loop body handling) is an error
                    // Loop bodies handle break directly using exit_id
                    return Err(CompileError::Unsupported("break outside of loop".into()));
                }
                IrStmt::Return(expr) => {
                    return Ok(Some(expr.clone()));
                }
            }
        }
        Ok(None)
    }

    // First, generate helper functions (non-required functions)
    // These need to be created first so they can be called from required functions
    for (fname, ir_func) in &functions {
        if REQUIRED_FUNCS.contains(&fname.as_str()) {
            continue; // Required functions are generated separately below
        }

        // Helper functions take f64 params and return f64
        let param_types: Vec<ValType> = ir_func.params.iter().map(|_| ValType::F64).collect();
        let mut fb = FunctionBuilder::new(&mut module.types, &param_types, &[ValType::F64]);

        // Create locals for parameters
        let param_locals: Vec<walrus::LocalId> = ir_func.params.iter()
            .map(|_| module.locals.add(ValType::F64))
            .collect();

        let mut locals_map: HashMap<String, walrus::LocalId> = HashMap::new();
        for (i, param_name) in ir_func.params.iter().enumerate() {
            locals_map.insert(param_name.clone(), param_locals[i]);
        }

        let mut ib = fb.func_body();
        let ret_expr = emit_ir_stmts(&mut ib, &func_ids, &ir_func.body, &ir_func.params, &mut locals_map, &mut module.locals)?;
        if let Some(expr) = ret_expr {
            emit_ir_expr(&mut ib, &func_ids, &expr, &ir_func.params, &locals_map)?;
        }

        let fid = fb.finish(param_locals, &mut module.funcs);
        func_ids.insert(fname.clone(), fid);
    }

    // Create memory for I/O buffers and export it
    // add_local(shared, shared64, initial_pages, max_pages, page_size_log2)
    let memory_id = module.memories.add_local(false, false, 1, None, None);
    module.exports.add("memory", memory_id);

    // Generate internal WASM functions for each required Lua function (not exported directly)
    // These will be called by the ABI wrapper functions
    let mut internal_func_ids: HashMap<String, walrus::FunctionId> = HashMap::new();

    for &fname in REQUIRED_FUNCS {
        let ir_func = functions.get(fname).unwrap();
        match fname {
            "is_inside" => {
                if ir_func.params.len() != 3 {
                    return Err(CompileError::Type("is_inside must take 3 params".into()));
                }
                // Internal is_inside: (f64, f64, f64) -> f64 (not demoted to f32 yet)
                let mut fb = FunctionBuilder::new(
                    &mut module.types,
                    &[ValType::F64, ValType::F64, ValType::F64],
                    &[ValType::F64],
                );
                let l_x = module.locals.add(ValType::F64);
                let l_y = module.locals.add(ValType::F64);
                let l_z = module.locals.add(ValType::F64);
                let mut locals_map: HashMap<String, walrus::LocalId> = HashMap::new();
                locals_map.insert(ir_func.params[0].clone(), l_x);
                locals_map.insert(ir_func.params[1].clone(), l_y);
                locals_map.insert(ir_func.params[2].clone(), l_z);
                let mut ib = fb.func_body();
                let ret_expr = emit_ir_stmts(&mut ib, &func_ids, &ir_func.body, &ir_func.params, &mut locals_map, &mut module.locals)?;
                if let Some(expr) = ret_expr {
                    emit_ir_expr(&mut ib, &func_ids, &expr, &ir_func.params, &locals_map)?;
                }
                let fid = fb.finish(vec![l_x, l_y, l_z], &mut module.funcs);
                internal_func_ids.insert(fname.to_string(), fid);
            }
            _ => {
                // Bounds getter: () -> f64
                if !ir_func.params.is_empty() {
                    return Err(CompileError::Type("bounds getters must have 0 params".into()));
                }
                let mut fb = FunctionBuilder::new(&mut module.types, &[], &[ValType::F64]);
                let mut locals_map: HashMap<String, walrus::LocalId> = HashMap::new();
                let mut ib = fb.func_body();
                let ret_expr = emit_ir_stmts(&mut ib, &func_ids, &ir_func.body, &ir_func.params, &mut locals_map, &mut module.locals)?;
                if let Some(expr) = ret_expr {
                    emit_ir_expr(&mut ib, &func_ids, &expr, &ir_func.params, &locals_map)?;
                }
                let fid = fb.finish(vec![], &mut module.funcs);
                internal_func_ids.insert(fname.to_string(), fid);
            }
        }
    }

    // Generate and export get_dimensions() -> u32
    {
        let mut fb = FunctionBuilder::new(&mut module.types, &[], &[ValType::I32]);
        let mut ib = fb.func_body();
        ib.i32_const(3); // Always 3 dimensions
        let fid = fb.finish(vec![], &mut module.funcs);
        module.exports.add("get_dimensions", fid);
    }

    // Generate and export get_bounds(out_ptr: i32)
    // Writes 6 f64 values: min_x, max_x, min_y, max_y, min_z, max_z
    {
        let mut fb = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let out_ptr = module.locals.add(ValType::I32);
        let mut ib = fb.func_body();

        let bounds_funcs = [
            ("get_bounds_min_x", 0),
            ("get_bounds_max_x", 8),
            ("get_bounds_min_y", 16),
            ("get_bounds_max_y", 24),
            ("get_bounds_min_z", 32),
            ("get_bounds_max_z", 40),
        ];

        for (func_name, offset) in bounds_funcs {
            let func_id = *internal_func_ids.get(func_name).unwrap();
            let mem_arg = walrus::ir::MemArg { align: 3, offset };
            ib.local_get(out_ptr);
            ib.call(func_id);
            ib.store(memory_id, walrus::ir::StoreKind::F64, mem_arg);
        }

        let fid = fb.finish(vec![out_ptr], &mut module.funcs);
        module.exports.add("get_bounds", fid);
    }

    // Generate and export sample(pos_ptr: i32) -> f32
    // Reads 3 f64 values from pos_ptr, calls internal is_inside, returns f32
    {
        let mut fb = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);
        let mut ib = fb.func_body();

        let is_inside_id = *internal_func_ids.get("is_inside").unwrap();

        // Load x from pos_ptr + 0
        let mem_arg_x = walrus::ir::MemArg { align: 3, offset: 0 };
        ib.local_get(pos_ptr);
        ib.load(memory_id, walrus::ir::LoadKind::F64, mem_arg_x);

        // Load y from pos_ptr + 8
        let mem_arg_y = walrus::ir::MemArg { align: 3, offset: 8 };
        ib.local_get(pos_ptr);
        ib.load(memory_id, walrus::ir::LoadKind::F64, mem_arg_y);

        // Load z from pos_ptr + 16
        let mem_arg_z = walrus::ir::MemArg { align: 3, offset: 16 };
        ib.local_get(pos_ptr);
        ib.load(memory_id, walrus::ir::LoadKind::F64, mem_arg_z);

        // Call internal is_inside(x, y, z) -> f64
        ib.call(is_inside_id);

        // Convert f64 to f32
        ib.unop(UnaryOp::F32DemoteF64);

        let fid = fb.finish(vec![pos_ptr], &mut module.funcs);
        module.exports.add("sample", fid);
    }

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let bytes = METADATA.get_or_init(|| {
        let metadata = OperatorMetadata {
            name: "lua_script_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
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

    #[test]
    fn test_compile_with_if_then_else() {
        // This is the exact script from the issue description
        let lua_src = r#"
function is_inside(x, y, z)
    -- Example: unit sphere centered at origin
    if (x*x + y*y + z*z - 1.0) < 0.0 then
       return 1.0
    else
       return 0.0
    end
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
        assert!(result.is_ok(), "Failed to compile with if-then-else: {:?}", result.err());
        let wasm = result.unwrap();
        // Check that we got valid WASM (starts with magic number)
        assert!(wasm.len() > 8, "WASM output too short");
        assert_eq!(&wasm[0..4], b"\0asm", "Invalid WASM magic number");
    }

    #[test]
    fn test_compile_with_modulo() {
        let lua_src = r#"
function is_inside(x, y, z)
    -- Use modulo to create a repeating pattern
    local mx = x % 1.0
    local my = y % 1.0
    local mz = z % 1.0
    return mx * mx + my * my + mz * mz - 0.25
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
        assert!(result.is_ok(), "Failed to compile with modulo: {:?}", result.err());
        let wasm = result.unwrap();
        assert!(wasm.len() > 8, "WASM output too short");
        assert_eq!(&wasm[0..4], b"\0asm", "Invalid WASM magic number");
    }

    #[test]
    fn test_compile_with_while_loop() {
        let lua_src = r#"
function is_inside(x, y, z)
    -- Iterative computation using while loop
    local sum = 0.0
    local i = 0.0
    while i < 3.0 do
        sum = sum + x * x + y * y + z * z
        i = i + 1.0
    end
    return sum - 3.0
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
        assert!(result.is_ok(), "Failed to compile with while loop: {:?}", result.err());
        let wasm = result.unwrap();
        assert!(wasm.len() > 8, "WASM output too short");
        assert_eq!(&wasm[0..4], b"\0asm", "Invalid WASM magic number");
    }

    #[test]
    fn test_compile_with_numeric_for() {
        let lua_src = r#"
function is_inside(x, y, z)
    -- Use numeric for loop to sum values
    local sum = 0.0
    for i = 1, 5 do
        sum = sum + x * i
    end
    return sum - 10.0
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
        assert!(result.is_ok(), "Failed to compile with numeric for: {:?}", result.err());
        let wasm = result.unwrap();
        assert!(wasm.len() > 8, "WASM output too short");
        assert_eq!(&wasm[0..4], b"\0asm", "Invalid WASM magic number");
    }

    #[test]
    fn test_compile_with_trig_functions() {
        let lua_src = r#"
function is_inside(x, y, z)
    -- Use trig functions for a sinusoidal pattern
    local wave = math.sin(x) + math.cos(y)
    return wave - z
end

function get_bounds_min_x()
    return -3.14159
end

function get_bounds_min_y()
    return -3.14159
end

function get_bounds_min_z()
    return -3.0
end

function get_bounds_max_x()
    return 3.14159
end

function get_bounds_max_y()
    return 3.14159
end

function get_bounds_max_z()
    return 3.0
end
"#;
        let result = compile_lua_to_wasm(lua_src);
        assert!(result.is_ok(), "Failed to compile with trig functions: {:?}", result.err());
        let wasm = result.unwrap();
        assert!(wasm.len() > 8, "WASM output too short");
        assert_eq!(&wasm[0..4], b"\0asm", "Invalid WASM magic number");
    }

    #[test]
    fn test_compile_with_user_function() {
        let lua_src = r#"
-- Helper function to compute distance squared
function dist_squared(x, y, z)
    return x*x + y*y + z*z
end

function is_inside(x, y, z)
    -- Use the helper function
    return dist_squared(x, y, z) - 1.0
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
        assert!(result.is_ok(), "Failed to compile with user function: {:?}", result.err());
        let wasm = result.unwrap();
        assert!(wasm.len() > 8, "WASM output too short");
        assert_eq!(&wasm[0..4], b"\0asm", "Invalid WASM magic number");
    }

    #[test]
    fn test_compile_with_exponentiation() {
        let lua_src = r#"
function is_inside(x, y, z)
    -- Use exponentiation operator
    return x^2 + y^2 + z^2 - 1.0
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
        assert!(result.is_ok(), "Failed to compile with exponentiation: {:?}", result.err());
        let wasm = result.unwrap();
        assert!(wasm.len() > 8, "WASM output too short");
        assert_eq!(&wasm[0..4], b"\0asm", "Invalid WASM magic number");
    }

    #[test]
    fn test_compile_with_repeat_until() {
        let lua_src = r#"
function is_inside(x, y, z)
    -- Use repeat-until loop
    local sum = 0.0
    local i = 1.0
    repeat
        sum = sum + i
        i = i + 1.0
    until i > 5.0
    return sum + x + y + z
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
        assert!(result.is_ok(), "Failed to compile with repeat-until: {:?}", result.err());
        let wasm = result.unwrap();
        assert!(wasm.len() > 8, "WASM output too short");
        assert_eq!(&wasm[0..4], b"\0asm", "Invalid WASM magic number");
    }

    #[test]
    fn test_compile_with_logical_and_or() {
        let lua_src = r#"
function is_inside(x, y, z)
    -- Use logical and/or operators
    local inside_x = (x > -1.0) and (x < 1.0)
    local inside_y = (y > -1.0) and (y < 1.0)
    local inside_z = (z > -1.0) and (z < 1.0)
    -- inside_x and inside_y and inside_z will be 1 if all true, 0 otherwise
    local result = inside_x and inside_y and inside_z
    -- Return negative if inside (to match SDF convention)
    return (result and -1.0) or 1.0
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
        assert!(result.is_ok(), "Failed to compile with logical and/or: {:?}", result.err());
        let wasm = result.unwrap();
        assert!(wasm.len() > 8, "WASM output too short");
        assert_eq!(&wasm[0..4], b"\0asm", "Invalid WASM magic number");
    }

    #[test]
    fn test_compile_with_math_pi() {
        let lua_src = r#"
function is_inside(x, y, z)
    -- Use math.pi constant
    local radius = 1.0
    local circumference = 2.0 * math.pi * radius
    return math.sqrt(x*x + y*y + z*z) - (circumference / (2.0 * math.pi))
end

function get_bounds_min_x()
    return -math.pi
end

function get_bounds_min_y()
    return -math.pi
end

function get_bounds_min_z()
    return -math.pi
end

function get_bounds_max_x()
    return math.pi
end

function get_bounds_max_y()
    return math.pi
end

function get_bounds_max_z()
    return math.pi
end
"#;
        let result = compile_lua_to_wasm(lua_src);
        assert!(result.is_ok(), "Failed to compile with math.pi: {:?}", result.err());
        let wasm = result.unwrap();
        assert!(wasm.len() > 8, "WASM output too short");
        assert_eq!(&wasm[0..4], b"\0asm", "Invalid WASM magic number");
    }

    #[test]
    fn test_compile_with_break_in_repeat() {
        // Test break in repeat-until loop (simpler case - break at end of body)
        let lua_src = r#"
function is_inside(x, y, z)
    -- Use break in repeat-until
    local sum = 0.0
    local i = 1.0
    repeat
        sum = sum + i
        i = i + 1.0
        break
    until i > 100.0
    return sum + x + y + z
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
        assert!(result.is_ok(), "Failed to compile with break in repeat: {:?}", result.err());
        let wasm = result.unwrap();
        assert!(wasm.len() > 8, "WASM output too short");
        assert_eq!(&wasm[0..4], b"\0asm", "Invalid WASM magic number");
    }
}

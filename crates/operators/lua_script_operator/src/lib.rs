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
    Add, Sub, Mul, Div,
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
    Abs, Sqrt, Floor, Ceil, Trunc, Nearest,
    Min, Max,
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
            match var {
                Var::Name(token) => {
                    Ok(IrExpr::Var(token.token().to_string()))
                }
                Var::Expression(_) => {
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
                BinOp::Caret(_) => return Err(CompileError::Unsupported("exponentiation operator (use math.pow if available)".into())),
                BinOp::Percent(_) => return Err(CompileError::Unsupported("modulo operator".into())),
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
                return Err(CompileError::Unsupported("break statement".into()));
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

    // Emit WASM instructions for an IR expression
    fn emit_ir_expr(
        b: &mut InstrSeqBuilder,
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
                emit_ir_expr(b, lhs, params, locals)?;
                emit_ir_expr(b, rhs, params, locals)?;
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
                    IrBinOp::And | IrBinOp::Or => {
                        return Err(CompileError::Unsupported("logical operators in expressions".into()));
                    }
                }
            }
            IrExpr::UnaryOp { op, expr: inner } => {
                emit_ir_expr(b, inner, params, locals)?;
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
                        emit_ir_expr(b, &args[0], params, locals)?;
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
                        emit_ir_expr(b, &args[0], params, locals)?;
                        emit_ir_expr(b, &args[1], params, locals)?;
                        match func {
                            IrMathFunc::Min => { b.binop(BinaryOp::F64Min); }
                            IrMathFunc::Max => { b.binop(BinaryOp::F64Max); }
                            _ => unreachable!(),
                        }
                    }
                }
            }
            IrExpr::FuncCall { name, args: _ } => {
                return Err(CompileError::Unsupported(format!("user function calls not yet supported: {}", name)));
            }
            IrExpr::IfThenElse { cond, then_expr, else_expr } => {
                emit_ir_condition(b, cond, params, locals)?;
                let then_expr = then_expr.clone();
                let else_expr = else_expr.clone();
                let params_vec: Vec<String> = params.to_vec();
                let locals_clone: HashMap<String, walrus::LocalId> = locals.clone();
                b.if_else(
                    ValType::F64,
                    |then_block| {
                        emit_ir_expr(then_block, &then_expr, &params_vec, &locals_clone)
                            .expect("emit_ir_expr in then branch failed");
                    },
                    |else_block| {
                        emit_ir_expr(else_block, &else_expr, &params_vec, &locals_clone)
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
        expr: &IrExpr,
        params: &[String],
        locals: &HashMap<String, walrus::LocalId>,
    ) -> Result<(), CompileError> {
        match expr {
            IrExpr::BinOp { op, lhs, rhs } => {
                match op {
                    IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge | IrBinOp::Eq | IrBinOp::Ne => {
                        emit_ir_expr(b, lhs, params, locals)?;
                        emit_ir_expr(b, rhs, params, locals)?;
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
        emit_ir_expr(b, expr, params, locals)?;
        b.f64_const(0.0);
        b.binop(BinaryOp::F64Ne);
        Ok(())
    }

    // Emit IR statements
    fn emit_ir_stmts(
        b: &mut InstrSeqBuilder,
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
                    emit_ir_expr(b, value, params, locals)?;
                    b.local_set(local_id);
                }
                IrStmt::Assign { name, value } => {
                    if let Some(&local_id) = locals.get(name) {
                        emit_ir_expr(b, value, params, locals)?;
                        b.local_set(local_id);
                    } else {
                        return Err(CompileError::Unsupported(format!("assignment to unknown variable: {}", name)));
                    }
                }
                IrStmt::If { cond, then_body, else_body } => {
                    emit_ir_condition(b, cond, params, locals)?;
                    let params_vec: Vec<String> = params.to_vec();
                    let then_locals = locals.clone();
                    let else_locals = locals.clone();
                    b.if_else(
                        ValType::F64,
                        |then_block| {
                            // We need a dummy ModuleLocals for nested calls - this is a limitation
                            // For now, we'll handle simple if-then-else with returns
                            if let Some(IrStmt::Return(expr)) = then_body.first() {
                                emit_ir_expr(then_block, expr, &params_vec, &then_locals)
                                    .expect("emit in then branch failed");
                            } else {
                                then_block.f64_const(0.0); // fallback
                            }
                        },
                        |else_block| {
                            if let Some(IrStmt::Return(expr)) = else_body.first() {
                                emit_ir_expr(else_block, expr, &params_vec, &else_locals)
                                    .expect("emit in else branch failed");
                            } else {
                                else_block.f64_const(0.0); // fallback
                            }
                        },
                    );
                    return Ok(None); // If statement handled the return
                }
                IrStmt::Return(expr) => {
                    return Ok(Some(expr.clone()));
                }
            }
        }
        Ok(None)
    }

    // Generate WASM functions for each required Lua function
    for &fname in REQUIRED_FUNCS {
        let ir_func = functions.get(fname).unwrap();
        match fname {
            "is_inside" => {
                if ir_func.params.len() != 3 {
                    return Err(CompileError::Type("is_inside must take 3 params".into()));
                }
                let mut fb = FunctionBuilder::new(
                    &mut module.types,
                    &[ValType::F64, ValType::F64, ValType::F64],
                    &[ValType::F32],
                );
                let l_x = module.locals.add(ValType::F64);
                let l_y = module.locals.add(ValType::F64);
                let l_z = module.locals.add(ValType::F64);
                let mut locals_map: HashMap<String, walrus::LocalId> = HashMap::new();
                locals_map.insert(ir_func.params[0].clone(), l_x);
                locals_map.insert(ir_func.params[1].clone(), l_y);
                locals_map.insert(ir_func.params[2].clone(), l_z);
                let mut ib = fb.func_body();
                let ret_expr = emit_ir_stmts(&mut ib, &ir_func.body, &ir_func.params, &mut locals_map, &mut module.locals)?;
                if let Some(expr) = ret_expr {
                    emit_ir_expr(&mut ib, &expr, &ir_func.params, &locals_map)?;
                }
                ib.unop(UnaryOp::F32DemoteF64);
                let fid = fb.finish(vec![l_x, l_y, l_z], &mut module.funcs);
                module.exports.add("is_inside", fid);
            }
            _ => {
                if !ir_func.params.is_empty() {
                    return Err(CompileError::Type("bounds getters must have 0 params".into()));
                }
                let mut fb = FunctionBuilder::new(&mut module.types, &[], &[ValType::F64]);
                let mut locals_map: HashMap<String, walrus::LocalId> = HashMap::new();
                let mut ib = fb.func_body();
                let ret_expr = emit_ir_stmts(&mut ib, &ir_func.body, &ir_func.params, &mut locals_map, &mut module.locals)?;
                if let Some(expr) = ret_expr {
                    emit_ir_expr(&mut ib, &expr, &ir_func.params, &locals_map)?;
                }
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
}

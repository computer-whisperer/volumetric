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
    use full_moon::ast::{Expression, BinOp, UnOp, Var, Prefix, Suffix, Call, FunctionArgs, Index};
    
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
    use full_moon::ast::{Prefix, Suffix, Call, FunctionArgs, Index};
    
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

    // Helper function to extract a single return expression from a block
    fn extract_return_expr(block: &Block) -> Option<Expression> {
        // Block should have no statements (or only comments) and a return as last_stmt
        if block.stmts().next().is_some() {
            return None; // Has statements other than return
        }
        if let Some(LastStmt::Return(ret)) = block.last_stmt() {
            let mut returns_iter = ret.returns().iter();
            if let Some(expr) = returns_iter.next() {
                if returns_iter.next().is_none() {
                    return Some(expr.clone());
                }
            }
        }
        None
    }

    // Extract function declarations from the AST
    // Maps function name -> (parameter names, function body)
    let mut functions: HashMap<String, (Vec<String>, FunctionBody)> = HashMap::new();

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

            // Get function body - either a simple return or an if-then-else
            let block = func_decl.body().block();
            
            // First, try to extract a simple return expression
            if let Some(expr) = extract_return_expr(block) {
                functions.insert(name, (params, FunctionBody::SimpleReturn(expr)));
                continue;
            }
            
            // Next, try to handle if-then-else with returns in each branch
            let mut stmts_iter = block.stmts();
            if let Some(first_stmt) = stmts_iter.next() {
                if stmts_iter.next().is_none() && block.last_stmt().is_none() {
                    // Single statement, no last_stmt - check if it's an if statement
                    if let Stmt::If(if_stmt) = first_stmt {
                        // We support: if condition then return X else return Y end
                        // No elseif support for now
                        if if_stmt.else_if().is_none() || if_stmt.else_if().map(|v| v.is_empty()).unwrap_or(true) {
                            if let (Some(then_expr), Some(else_block)) = (
                                extract_return_expr(if_stmt.block()),
                                if_stmt.else_block()
                            ) {
                                if let Some(else_expr) = extract_return_expr(else_block) {
                                    let condition = if_stmt.condition().clone();
                                    functions.insert(name, (params, FunctionBody::IfThenElse {
                                        condition,
                                        then_expr,
                                        else_expr,
                                    }));
                                    continue;
                                }
                            }
                        }
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

    // Emit a condition expression that produces an i32 (0 or 1) for use in if/select
    fn emit_condition(
        b: &mut InstrSeqBuilder,
        expr: &Expression,
        params: &[String],
        locals: &[walrus::LocalId],
    ) -> Result<(), CompileError> {
        // Handle comparison expressions specially - they produce i32 results
        if let Expression::BinaryOperator { lhs, binop, rhs } = expr {
            match binop {
                BinOp::LessThan(_) => {
                    emit_expr(b, lhs, params, locals)?;
                    emit_expr(b, rhs, params, locals)?;
                    b.binop(BinaryOp::F64Lt);
                    return Ok(());
                }
                BinOp::LessThanEqual(_) => {
                    emit_expr(b, lhs, params, locals)?;
                    emit_expr(b, rhs, params, locals)?;
                    b.binop(BinaryOp::F64Le);
                    return Ok(());
                }
                BinOp::GreaterThan(_) => {
                    emit_expr(b, lhs, params, locals)?;
                    emit_expr(b, rhs, params, locals)?;
                    b.binop(BinaryOp::F64Gt);
                    return Ok(());
                }
                BinOp::GreaterThanEqual(_) => {
                    emit_expr(b, lhs, params, locals)?;
                    emit_expr(b, rhs, params, locals)?;
                    b.binop(BinaryOp::F64Ge);
                    return Ok(());
                }
                BinOp::TwoEqual(_) => {
                    emit_expr(b, lhs, params, locals)?;
                    emit_expr(b, rhs, params, locals)?;
                    b.binop(BinaryOp::F64Eq);
                    return Ok(());
                }
                BinOp::TildeEqual(_) => {
                    emit_expr(b, lhs, params, locals)?;
                    emit_expr(b, rhs, params, locals)?;
                    b.binop(BinaryOp::F64Ne);
                    return Ok(());
                }
                _ => {}
            }
        }
        // Handle parenthesized conditions
        if let Expression::Parentheses { expression, .. } = expr {
            return emit_condition(b, expression, params, locals);
        }
        // For non-comparison expressions, emit as f64 and convert to i32 (non-zero = true)
        emit_expr(b, expr, params, locals)?;
        b.f64_const(0.0);
        b.binop(BinaryOp::F64Ne);
        Ok(())
    }

    // Emit code for a function body (either simple return or if-then-else)
    fn emit_function_body(
        b: &mut InstrSeqBuilder,
        body: &FunctionBody,
        params: &[String],
        locals: &[walrus::LocalId],
    ) -> Result<(), CompileError> {
        match body {
            FunctionBody::SimpleReturn(expr) => {
                emit_expr(b, expr, params, locals)?;
            }
            FunctionBody::IfThenElse { condition, then_expr, else_expr } => {
                // Emit: if condition then return then_expr else return else_expr end
                // Using WASM if-else block structure
                emit_condition(b, condition, params, locals)?;
                // Clone expressions for use in closures
                let then_expr = then_expr.clone();
                let else_expr = else_expr.clone();
                let params_vec: Vec<String> = params.to_vec();
                let locals_vec: Vec<walrus::LocalId> = locals.to_vec();
                b.if_else(
                    ValType::F64,
                    |then_block| {
                        emit_expr(then_block, &then_expr, &params_vec, &locals_vec)
                            .expect("emit_expr in then branch failed");
                    },
                    |else_block| {
                        emit_expr(else_block, &else_expr, &params_vec, &locals_vec)
                            .expect("emit_expr in else branch failed");
                    },
                );
            }
        }
        Ok(())
    }

    // Generate WASM functions for each required Lua function
    for &fname in REQUIRED_FUNCS {
        let (params, body) = functions.get(fname).unwrap();
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
                emit_function_body(&mut ib, body, params, &[l_x, l_y, l_z])?;
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
                emit_function_body(&mut ib, body, params, &[])?;
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

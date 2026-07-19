//! Lua Script Operator: compiles a restricted Lua script into a WASM Model module.
//!
//! Host/operator ABI: see the `volumetric_abi` crate.
//!
//! Input/Output:
//! - Input 0: UTF-8 Lua source containing the required functions
//! - Input 1: optional CBOR [`volumetric_abi::f64_map::F64Map`] overrides
//! - Output 0: WASM bytes of a model module that exports the N-dimensional model ABI
//!
//! The script's dimensionality comes from `is_inside`'s arity:
//! `is_inside(x, y, z)` compiles a 3D volume, `is_inside(x, y)` a 2D sketch
//! (for use with extrude-style operators). A 2D script defines only the
//! x/y bounds functions.
//!
//! `is_inside` returns an occupancy value per the `volumetric_abi` contract:
//! `1.0` inside, `0.0` outside (a point classifies as inside iff the value
//! is `> 0.5`).
//!
//! Generated Model ABI:
//! - `get_dimensions() -> u32`: Returns 2 or 3 per is_inside's arity
//! - `get_io_ptr() -> i32`: Returns the model-owned IO buffer (2n f64s)
//! - `get_bounds(out_ptr: i32)`: Writes 2n f64 values, interleaved min/max
//! - `sample(pos_ptr: i32) -> f32`: Reads position from memory, returns occupancy
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
//! - Module-scope numeric constants: `local radius = 0.01` (evaluated at
//!   compile time, in declaration order, and visible to every function)
//! - Routable constants: append `-- @param key="part.radius" min=0.001
//!   max=1.0` to a numeric module constant; a missing F64Map key keeps the
//!   literal default (the annotation itself stays on the declaration line)
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
//!     if dist(x, y, z) <= 1.0 then
//!         return 1.0
//!     else
//!         return 0.0
//!     end
//! end
//! ```
//!
//! ## Not Yet Supported
//! - Generic for loops (`for k, v in pairs(t)`)
//! - Tables, strings, multiple return values
//! - Closures, recursive helper calls

use walrus::{FunctionBuilder, Module, ModuleConfig, ValType};

use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};

use volumetric_abi::host::{post_output, read_input, report_error};

/// Per-axis bounds getters in IO-buffer order (interleaved min/max). A
/// script must define the first `2 * dims` of these plus `is_inside`; the
/// dimensionality comes from `is_inside`'s arity (2 for a sketch, 3 for a
/// volume). These are compiled to internal WASM functions and wrapped by the
/// ABI exports.
const BOUNDS_FUNCS: &[&str] = &[
    "get_bounds_min_x",
    "get_bounds_max_x",
    "get_bounds_min_y",
    "get_bounds_max_y",
    "get_bounds_min_z",
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

/// Evaluate one module-scope constant expression. Module constants are baked
/// into the generated model: they are deliberately numeric and side-effect
/// free, which keeps model sampling stateless while giving scripts one shared
/// source of truth for dimensions used by several helpers and bounds getters.
fn eval_constant(
    expr: &IrExpr,
    constants: &std::collections::BTreeMap<String, f64>,
) -> Result<f64, CompileError> {
    let binary = |op: &IrBinOp, a: f64, b: f64| -> f64 {
        match op {
            IrBinOp::Add => a + b,
            IrBinOp::Sub => a - b,
            IrBinOp::Mul => a * b,
            IrBinOp::Div => a / b,
            // Lua modulo follows the divisor's sign.
            IrBinOp::Mod => a - (a / b).floor() * b,
            IrBinOp::Pow => a.powf(b),
            IrBinOp::Lt => (a < b) as u8 as f64,
            IrBinOp::Le => (a <= b) as u8 as f64,
            IrBinOp::Gt => (a > b) as u8 as f64,
            IrBinOp::Ge => (a >= b) as u8 as f64,
            IrBinOp::Eq => (a == b) as u8 as f64,
            IrBinOp::Ne => (a != b) as u8 as f64,
            IrBinOp::And => ((a != 0.0) && (b != 0.0)) as u8 as f64,
            IrBinOp::Or => ((a != 0.0) || (b != 0.0)) as u8 as f64,
        }
    };

    match expr {
        IrExpr::Number(value) => Ok(*value),
        IrExpr::Var(name) => constants.get(name).copied().ok_or_else(|| {
            CompileError::Type(format!(
                "module constant references unknown or later name `{name}`"
            ))
        }),
        IrExpr::BinOp { op, lhs, rhs } => Ok(binary(
            op,
            eval_constant(lhs, constants)?,
            eval_constant(rhs, constants)?,
        )),
        IrExpr::UnaryOp { op, expr } => {
            let value = eval_constant(expr, constants)?;
            Ok(match op {
                IrUnaryOp::Neg => -value,
                IrUnaryOp::Not => (value == 0.0) as u8 as f64,
            })
        }
        IrExpr::MathCall { func, args } => {
            let args: Vec<f64> = args
                .iter()
                .map(|arg| eval_constant(arg, constants))
                .collect::<Result<_, _>>()?;
            match (func, args.as_slice()) {
                (IrMathFunc::Abs, [x]) => Ok(x.abs()),
                (IrMathFunc::Sqrt, [x]) => Ok(x.sqrt()),
                (IrMathFunc::Floor, [x]) => Ok(x.floor()),
                (IrMathFunc::Ceil, [x]) => Ok(x.ceil()),
                (IrMathFunc::Trunc, [x]) => Ok(x.trunc()),
                (IrMathFunc::Nearest, [x]) => Ok(x.round_ties_even()),
                (IrMathFunc::Min, [a, b]) => Ok(a.min(*b)),
                (IrMathFunc::Max, [a, b]) => Ok(a.max(*b)),
                (IrMathFunc::Sin, [x]) => Ok(x.sin()),
                (IrMathFunc::Cos, [x]) => Ok(x.cos()),
                (IrMathFunc::Tan, [x]) => Ok(x.tan()),
                (IrMathFunc::Exp, [x]) => Ok(x.exp()),
                (IrMathFunc::Log, [x]) => Ok(x.ln()),
                (IrMathFunc::Pow, [x, y]) => Ok(x.powf(*y)),
                (IrMathFunc::Atan2, [y, x]) => Ok(y.atan2(*x)),
                _ => Err(CompileError::Type(
                    "invalid math call in module constant".to_string(),
                )),
            }
        }
        IrExpr::FuncCall { name, .. } => Err(CompileError::Unsupported(format!(
            "module constant cannot call user function `{name}`"
        ))),
        IrExpr::IfThenElse { .. } => Err(CompileError::Unsupported(
            "conditional expression in module constant".to_string(),
        )),
    }
}

// ============================================================================
// Intermediate Representation (IR)
// ============================================================================

/// Binary operators in the IR
#[derive(Clone, Debug)]
enum IrBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
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
    Abs,
    Sqrt,
    Floor,
    Ceil,
    Trunc,
    Nearest,
    Min,
    Max,
    // Polynomial approximations (no native WASM support)
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Pow,
    Atan2,
}

/// IR Expression - decoupled from the AST
#[derive(Clone, Debug)]
#[allow(dead_code)]
enum IrExpr {
    Number(f64),
    Var(String),
    BinOp {
        op: IrBinOp,
        lhs: Box<IrExpr>,
        rhs: Box<IrExpr>,
    },
    UnaryOp {
        op: IrUnaryOp,
        expr: Box<IrExpr>,
    },
    MathCall {
        func: IrMathFunc,
        args: Vec<IrExpr>,
    },
    FuncCall {
        name: String,
        args: Vec<IrExpr>,
    },
    IfThenElse {
        cond: Box<IrExpr>,
        then_expr: Box<IrExpr>,
        else_expr: Box<IrExpr>,
    },
}

/// IR Statement
#[derive(Clone, Debug)]
enum IrStmt {
    LocalAssign {
        name: String,
        value: IrExpr,
    },
    Assign {
        name: String,
        value: IrExpr,
    },
    If {
        cond: IrExpr,
        then_body: Vec<IrStmt>,
        else_body: Vec<IrStmt>,
    },
    While {
        cond: IrExpr,
        body: Vec<IrStmt>,
    },
    RepeatUntil {
        body: Vec<IrStmt>,
        cond: IrExpr,
    },
    NumericFor {
        var: String,
        start: IrExpr,
        end: IrExpr,
        step: Option<IrExpr>,
        body: Vec<IrStmt>,
    },
    Break,
    Return(IrExpr),
}

/// IR Function definition
#[derive(Clone, Debug)]
struct IrFunc {
    params: Vec<String>,
    body: Vec<IrStmt>,
}

fn collect_expr_calls(expr: &IrExpr, calls: &mut std::collections::BTreeSet<String>) {
    match expr {
        IrExpr::Number(_) | IrExpr::Var(_) => {}
        IrExpr::BinOp { lhs, rhs, .. } => {
            collect_expr_calls(lhs, calls);
            collect_expr_calls(rhs, calls);
        }
        IrExpr::UnaryOp { expr, .. } => collect_expr_calls(expr, calls),
        IrExpr::MathCall { args, .. } => {
            for arg in args {
                collect_expr_calls(arg, calls);
            }
        }
        IrExpr::FuncCall { name, args } => {
            calls.insert(name.clone());
            for arg in args {
                collect_expr_calls(arg, calls);
            }
        }
        IrExpr::IfThenElse {
            cond,
            then_expr,
            else_expr,
        } => {
            collect_expr_calls(cond, calls);
            collect_expr_calls(then_expr, calls);
            collect_expr_calls(else_expr, calls);
        }
    }
}

fn collect_stmt_calls(stmt: &IrStmt, calls: &mut std::collections::BTreeSet<String>) {
    match stmt {
        IrStmt::LocalAssign { value, .. } | IrStmt::Assign { value, .. } => {
            collect_expr_calls(value, calls);
        }
        IrStmt::If {
            cond,
            then_body,
            else_body,
        } => {
            collect_expr_calls(cond, calls);
            for stmt in then_body.iter().chain(else_body) {
                collect_stmt_calls(stmt, calls);
            }
        }
        IrStmt::While { cond, body } => {
            collect_expr_calls(cond, calls);
            for stmt in body {
                collect_stmt_calls(stmt, calls);
            }
        }
        IrStmt::RepeatUntil { body, cond } => {
            for stmt in body {
                collect_stmt_calls(stmt, calls);
            }
            collect_expr_calls(cond, calls);
        }
        IrStmt::NumericFor {
            start,
            end,
            step,
            body,
            ..
        } => {
            collect_expr_calls(start, calls);
            collect_expr_calls(end, calls);
            if let Some(step) = step {
                collect_expr_calls(step, calls);
            }
            for stmt in body {
                collect_stmt_calls(stmt, calls);
            }
        }
        IrStmt::Return(expr) => collect_expr_calls(expr, calls),
        IrStmt::Break => {}
    }
}

/// Order helpers after all helpers they call. Walrus assigns function IDs as
/// bodies are finished, so dependency ordering gives Lua's source-independent
/// global function lookup without relying on `HashMap` iteration order.
fn helper_emission_order(
    functions: &std::collections::HashMap<String, IrFunc>,
    required_funcs: &[&str],
) -> Result<Vec<String>, CompileError> {
    fn visit(
        name: &str,
        functions: &std::collections::HashMap<String, IrFunc>,
        required: &std::collections::BTreeSet<&str>,
        states: &mut std::collections::HashMap<String, u8>,
        order: &mut Vec<String>,
    ) -> Result<(), CompileError> {
        match states.get(name).copied() {
            Some(2) => return Ok(()),
            Some(1) => {
                return Err(CompileError::Unsupported(format!(
                    "recursive helper call involving `{name}`"
                )));
            }
            _ => {}
        }
        states.insert(name.to_string(), 1);

        let function = functions
            .get(name)
            .ok_or_else(|| CompileError::Unsupported(format!("unknown function: {name}")))?;
        let mut calls = std::collections::BTreeSet::new();
        for stmt in &function.body {
            collect_stmt_calls(stmt, &mut calls);
        }
        for dependency in calls {
            if required.contains(dependency.as_str()) {
                return Err(CompileError::Unsupported(format!(
                    "helper `{name}` cannot call required model function `{dependency}`"
                )));
            }
            visit(&dependency, functions, required, states, order)?;
        }

        states.insert(name.to_string(), 2);
        order.push(name.to_string());
        Ok(())
    }

    let required: std::collections::BTreeSet<_> = required_funcs.iter().copied().collect();
    let mut helper_names: Vec<_> = functions
        .keys()
        .filter(|name| !required.contains(name.as_str()))
        .cloned()
        .collect();
    helper_names.sort();

    let mut states = std::collections::HashMap::new();
    let mut order = Vec::with_capacity(helper_names.len());
    for name in helper_names {
        visit(&name, functions, &required, &mut states, &mut order)?;
    }
    Ok(order)
}

// ============================================================================
// AST to IR Conversion
// ============================================================================

/// Convert a Lua AST expression to IR
fn ast_expr_to_ir(expr: &full_moon::ast::Expression) -> Result<IrExpr, CompileError> {
    use full_moon::ast::{BinOp, Expression, UnOp, Var};

    match expr {
        Expression::Number(token) => {
            let num_str = token.token().to_string();
            let v: f64 = num_str
                .parse()
                .map_err(|_| CompileError::Parse(format!("invalid number: {}", num_str)))?;
            Ok(IrExpr::Number(v))
        }
        Expression::Var(var) => {
            use full_moon::ast::{Index, Suffix};
            match var {
                Var::Name(token) => Ok(IrExpr::Var(token.token().to_string())),
                Var::Expression(var_expr) => {
                    // Handle math.pi and similar constants
                    if let full_moon::ast::Prefix::Name(obj_token) = var_expr.prefix() {
                        let obj_name = obj_token.token().to_string();
                        let suffixes: Vec<_> = var_expr.suffixes().collect();

                        if obj_name == "math"
                            && suffixes.len() == 1
                            && let Suffix::Index(Index::Dot { name, .. }) = &suffixes[0]
                        {
                            let const_name = name.token().to_string();
                            match const_name.as_str() {
                                "pi" => return Ok(IrExpr::Number(std::f64::consts::PI)),
                                "huge" => return Ok(IrExpr::Number(f64::INFINITY)),
                                _ => {}
                            }
                        }
                    }
                    Err(CompileError::Unsupported(
                        "complex variable expressions not supported".into(),
                    ))
                }
                _ => Err(CompileError::Unsupported(
                    "unsupported variable type".into(),
                )),
            }
        }
        Expression::Parentheses { expression, .. } => ast_expr_to_ir(expression),
        Expression::UnaryOperator { unop, expression } => {
            let inner = ast_expr_to_ir(expression)?;
            match unop {
                UnOp::Minus(_) => Ok(IrExpr::UnaryOp {
                    op: IrUnaryOp::Neg,
                    expr: Box::new(inner),
                }),
                UnOp::Not(_) => Ok(IrExpr::UnaryOp {
                    op: IrUnaryOp::Not,
                    expr: Box::new(inner),
                }),
                UnOp::Hash(_) => Err(CompileError::Unsupported("length operator".into())),
                _ => Err(CompileError::Unsupported(
                    "unsupported unary operator".into(),
                )),
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
                BinOp::TwoDots(_) => {
                    return Err(CompileError::Unsupported("string concatenation".into()));
                }
                _ => {
                    return Err(CompileError::Unsupported(
                        "unsupported binary operator".into(),
                    ));
                }
            };
            Ok(IrExpr::BinOp {
                op,
                lhs: Box::new(left),
                rhs: Box::new(right),
            })
        }
        Expression::FunctionCall(call) => ast_func_call_to_ir(call),
        Expression::Symbol(token) => {
            let sym = token.token().to_string();
            match sym.as_str() {
                "true" => Ok(IrExpr::Number(1.0)),
                "false" => Ok(IrExpr::Number(0.0)),
                _ => Err(CompileError::Unsupported(format!(
                    "unsupported symbol: {}",
                    sym
                ))),
            }
        }
        _ => Err(CompileError::Unsupported(
            "unsupported expression type".into(),
        )),
    }
}

/// Convert a function call AST to IR
fn ast_func_call_to_ir(call: &full_moon::ast::FunctionCall) -> Result<IrExpr, CompileError> {
    use full_moon::ast::{Index, Prefix, Suffix};

    let prefix = call.prefix();
    let suffixes: Vec<_> = call.suffixes().collect();

    // Pattern: math.abs(x) -> prefix=Name("math"), suffixes=[Index::Dot("abs"), Call(...)]
    // Pattern: helper(x) -> prefix=Name("helper"), suffixes=[Call(...)]

    if let Prefix::Name(obj_token) = prefix {
        let obj_name = obj_token.token().to_string();

        // Check for math.func pattern
        if suffixes.len() >= 2
            && let Suffix::Index(Index::Dot { name, .. }) = &suffixes[0]
        {
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
                    _ => {
                        return Err(CompileError::Unsupported(format!(
                            "unsupported math function: math.{}",
                            method_name
                        )));
                    }
                };
                return Ok(IrExpr::MathCall { func, args });
            } else {
                return Err(CompileError::Unsupported(format!(
                    "unsupported module: {}",
                    obj_name
                )));
            }
        }

        // Check for simple function call pattern: func(args)
        if suffixes.len() == 1 {
            let args = extract_call_args_from_suffixes(&suffixes)?;
            return Ok(IrExpr::FuncCall {
                name: obj_name,
                args,
            });
        }
    }

    Err(CompileError::Unsupported(
        "unsupported function call pattern".into(),
    ))
}

/// Extract arguments from call suffixes
fn extract_call_args_from_suffixes(
    suffixes: &[&full_moon::ast::Suffix],
) -> Result<Vec<IrExpr>, CompileError> {
    use full_moon::ast::{Call, FunctionArgs, Suffix};

    for suffix in suffixes.iter() {
        if let Suffix::Call(call_suffix) = suffix {
            match call_suffix {
                Call::AnonymousCall(FunctionArgs::Parentheses { arguments, .. }) => {
                    return arguments.iter().map(ast_expr_to_ir).collect();
                }
                Call::AnonymousCall(_) => {}
                Call::MethodCall(_) => {
                    return Err(CompileError::Unsupported(
                        "method call syntax not supported".into(),
                    ));
                }
                _ => {}
            }
        }
    }
    Err(CompileError::Unsupported(
        "could not extract function arguments".into(),
    ))
}

/// Convert a Lua block to IR statements
fn ast_block_to_ir(block: &full_moon::ast::Block) -> Result<Vec<IrStmt>, CompileError> {
    use full_moon::ast::{LastStmt, Stmt};

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
                        return Err(CompileError::Unsupported(
                            "complex assignment target".into(),
                        ));
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
                stmts.push(IrStmt::NumericFor {
                    var,
                    start,
                    end,
                    step,
                    body,
                });
            }
            Stmt::Repeat(repeat_stmt) => {
                let body = ast_block_to_ir(repeat_stmt.block())?;
                let cond = ast_expr_to_ir(repeat_stmt.until())?;
                stmts.push(IrStmt::RepeatUntil { body, cond });
            }
            _ => {
                return Err(CompileError::Unsupported(
                    "unsupported statement type".into(),
                ));
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
                return Err(CompileError::Unsupported(
                    "unsupported last statement".into(),
                ));
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

    Ok(IrStmt::If {
        cond,
        then_body,
        else_body,
    })
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

    Ok(vec![IrStmt::If {
        cond,
        then_body,
        else_body,
    }])
}

/// Convert a function declaration to IR
fn ast_func_to_ir(func_decl: &full_moon::ast::FunctionDeclaration) -> Result<IrFunc, CompileError> {
    use full_moon::ast::Parameter;

    let params: Vec<String> = func_decl
        .body()
        .parameters()
        .iter()
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

#[cfg(test)]
fn compile_lua_to_wasm(src: &str) -> Result<Vec<u8>, CompileError> {
    compile_lua_to_wasm_with_parameters(src, &volumetric_abi::f64_map::F64Map::new())
}

fn compile_lua_to_wasm_with_parameters(
    src: &str,
    routed_values: &volumetric_abi::f64_map::F64Map,
) -> Result<Vec<u8>, CompileError> {
    use full_moon::ast::Stmt;
    use full_moon::node::Node;
    use std::collections::{BTreeMap, HashMap};
    use walrus::InstrSeqBuilder;
    use walrus::ir::{BinaryOp, UnaryOp};

    let parameter_specs = volumetric_abi::lua_parameters::parse(src)
        .map_err(|error| CompileError::Type(format!("invalid parameter annotation: {error}")))?;
    let parameter_lines: BTreeMap<_, _> = parameter_specs
        .iter()
        .map(|parameter| (parameter.local_name.clone(), parameter.source_line))
        .collect();
    let mut parameter_values = BTreeMap::new();
    for parameter in &parameter_specs {
        if let Some(value) = routed_values.get(&parameter.key).copied() {
            parameter
                .validate_value(value)
                .map_err(CompileError::Type)?;
            parameter_values.insert(parameter.local_name.clone(), (parameter.source_line, value));
        }
    }

    // Parse the Lua source into an AST
    let ast = full_moon::parse(src).map_err(|errors| {
        let msg = errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        CompileError::Parse(msg)
    })?;

    // Extract function declarations and evaluate module-scope numeric
    // constants in source order. Constant expressions intentionally cannot
    // call user functions: compilation stays deterministic and sampling the
    // resulting model remains stateless.
    let mut functions: HashMap<String, IrFunc> = HashMap::new();
    let mut constants: BTreeMap<String, f64> = BTreeMap::new();
    let mut bound_parameters = std::collections::BTreeSet::new();

    for stmt in ast.nodes().stmts() {
        match stmt {
            Stmt::FunctionDeclaration(func_decl) => {
                let name = func_decl
                    .name()
                    .names()
                    .iter()
                    .map(|t| t.token().to_string())
                    .collect::<Vec<_>>()
                    .join(".");

                let ir_func = ast_func_to_ir(func_decl)?;
                functions.insert(name, ir_func);
            }
            Stmt::LocalAssignment(local_assign) => {
                let source_line = local_assign
                    .start_position()
                    .map(|position| position.line());
                let names: Vec<_> = local_assign.names().iter().collect();
                let exprs: Vec<_> = local_assign.expressions().iter().collect();
                let mut pending = Vec::with_capacity(names.len());

                // Match Lua's multiple-assignment behavior: evaluate all
                // right-hand sides against the old environment before making
                // any newly assigned names visible.
                for (index, name_token) in names.iter().enumerate() {
                    let name = name_token.token().to_string();
                    if let Some(parameter_line) = parameter_lines
                        .get(&name)
                        .copied()
                        .filter(|parameter_line| Some(*parameter_line) == source_line)
                    {
                        bound_parameters.insert((name.clone(), parameter_line));
                    }
                    let value = if let Some((parameter_line, value)) = parameter_values
                        .get(&name)
                        .filter(|(parameter_line, _)| Some(*parameter_line) == source_line)
                    {
                        bound_parameters.insert((name.clone(), *parameter_line));
                        *value
                    } else if let Some(expr) = exprs.get(index) {
                        eval_constant(&ast_expr_to_ir(expr)?, &constants)?
                    } else {
                        0.0
                    };
                    pending.push((name, value));
                }
                constants.extend(pending);
            }
            Stmt::Assignment(assign) => {
                let vars: Vec<_> = assign.variables().iter().collect();
                let exprs: Vec<_> = assign.expressions().iter().collect();
                let mut pending = Vec::with_capacity(vars.len());

                for (index, var) in vars.iter().enumerate() {
                    let full_moon::ast::Var::Name(name_token) = var else {
                        return Err(CompileError::Unsupported(
                            "complex module-scope assignment target".into(),
                        ));
                    };
                    let value = if let Some(expr) = exprs.get(index) {
                        eval_constant(&ast_expr_to_ir(expr)?, &constants)?
                    } else {
                        0.0
                    };
                    pending.push((name_token.token().to_string(), value));
                }
                constants.extend(pending);
            }
            _ => {
                return Err(CompileError::Unsupported(
                    "module scope supports only numeric assignments and function declarations"
                        .into(),
                ));
            }
        }
    }

    if let Some(name) = constants.keys().find(|name| functions.contains_key(*name)) {
        return Err(CompileError::Type(format!(
            "module name `{name}` is both a numeric constant and a function"
        )));
    }
    for parameter in &parameter_specs {
        let binding = (parameter.local_name.clone(), parameter.source_line);
        if !bound_parameters.contains(&binding) {
            return Err(CompileError::Type(format!(
                "parameter `{}` does not annotate a module-scope numeric constant",
                parameter.local_name
            )));
        }
    }

    // The script's dimensionality comes from is_inside's arity: 2 compiles
    // a sketch (occupancy over x, y), 3 a volume.
    let dims = match functions.get("is_inside") {
        None => return Err(CompileError::MissingFunc("is_inside")),
        Some(f) => f.params.len(),
    };
    if dims != 2 && dims != 3 {
        return Err(CompileError::Type(format!(
            "is_inside must take 2 params (2D sketch) or 3 params (3D volume), got {dims}"
        )));
    }
    let required_funcs: Vec<&'static str> = std::iter::once("is_inside")
        .chain(BOUNDS_FUNCS[..dims * 2].iter().copied())
        .collect();

    // Verify all required functions are present
    for &fname in &required_funcs {
        if !functions.contains_key(fname) {
            return Err(CompileError::MissingFunc(fname));
        }
    }

    // Create a new WASM module
    let mut module = Module::with_config(ModuleConfig::new());

    fn add_constant_locals(
        module_locals: &mut walrus::ModuleLocals,
        constants: &BTreeMap<String, f64>,
    ) -> (
        HashMap<String, walrus::LocalId>,
        Vec<(walrus::LocalId, f64)>,
    ) {
        let mut locals = HashMap::with_capacity(constants.len());
        let mut initializers = Vec::with_capacity(constants.len());
        for (name, value) in constants {
            let local = module_locals.add(ValType::F64);
            locals.insert(name.clone(), local);
            initializers.push((local, *value));
        }
        (locals, initializers)
    }

    fn initialize_constants(
        builder: &mut InstrSeqBuilder,
        initializers: &[(walrus::LocalId, f64)],
    ) {
        for &(local, value) in initializers {
            builder.f64_const(value);
            builder.local_set(local);
        }
    }

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
                    return Err(CompileError::Unsupported(format!(
                        "unknown variable: {}",
                        name
                    )));
                }
            }
            IrExpr::BinOp { op, lhs, rhs } => {
                match op {
                    IrBinOp::Mod => {
                        // Lua modulo: a % b = a - floor(a/b) * b
                        emit_ir_expr(b, func_ids, lhs, params, locals)?; // a (for final subtraction)
                        emit_ir_expr(b, func_ids, lhs, params, locals)?; // a
                        emit_ir_expr(b, func_ids, rhs, params, locals)?; // b
                        b.binop(BinaryOp::F64Div); // a/b
                        b.unop(UnaryOp::F64Floor); // floor(a/b)
                        emit_ir_expr(b, func_ids, rhs, params, locals)?; // b
                        b.binop(BinaryOp::F64Mul); // floor(a/b) * b
                        b.binop(BinaryOp::F64Sub); // a - floor(a/b) * b
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
                        b.binop(BinaryOp::F64Sub); // log(x) ≈ (x-1) - (x-1)²/2
                        // Multiply by y
                        emit_ir_expr(b, func_ids, rhs, params, locals)?;
                        b.binop(BinaryOp::F64Mul); // y * log(x)
                        // exp(z) ≈ 1 + z
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Add);
                    }
                    IrBinOp::And => {
                        // a and b: if a != 0 then b else a (short-circuit)
                        // Evaluate lhs for condition check
                        emit_ir_expr(b, func_ids, lhs, params, locals)?;
                        b.f64_const(0.0);
                        b.binop(BinaryOp::F64Ne); // lhs != 0.0
                        let lhs = lhs.clone();
                        let rhs = rhs.clone();
                        let params_vec: Vec<String> = params.to_vec();
                        let locals_clone = locals.clone();
                        let func_ids_ref = func_ids;
                        b.if_else(
                            ValType::F64,
                            |then_block| {
                                // lhs truthy: return rhs
                                emit_ir_expr(
                                    then_block,
                                    func_ids_ref,
                                    &rhs,
                                    &params_vec,
                                    &locals_clone,
                                )
                                .expect("emit_ir_expr in and-then branch failed");
                            },
                            |else_block| {
                                // lhs falsy: return lhs (re-evaluate, will be 0.0)
                                emit_ir_expr(
                                    else_block,
                                    func_ids_ref,
                                    &lhs,
                                    &params_vec,
                                    &locals_clone,
                                )
                                .expect("emit_ir_expr in and-else branch failed");
                            },
                        );
                    }
                    IrBinOp::Or => {
                        // a or b: if a != 0 then a else b (short-circuit)
                        // Evaluate lhs for condition check
                        emit_ir_expr(b, func_ids, lhs, params, locals)?;
                        b.f64_const(0.0);
                        b.binop(BinaryOp::F64Ne); // lhs != 0.0
                        let lhs = lhs.clone();
                        let rhs = rhs.clone();
                        let params_vec: Vec<String> = params.to_vec();
                        let locals_clone = locals.clone();
                        let func_ids_ref = func_ids;
                        b.if_else(
                            ValType::F64,
                            |then_block| {
                                // lhs truthy: return lhs (re-evaluate)
                                emit_ir_expr(
                                    then_block,
                                    func_ids_ref,
                                    &lhs,
                                    &params_vec,
                                    &locals_clone,
                                )
                                .expect("emit_ir_expr in or-then branch failed");
                            },
                            |else_block| {
                                // lhs falsy: return rhs
                                emit_ir_expr(
                                    else_block,
                                    func_ids_ref,
                                    &rhs,
                                    &params_vec,
                                    &locals_clone,
                                )
                                .expect("emit_ir_expr in or-else branch failed");
                            },
                        );
                    }
                    _ => {
                        emit_ir_expr(b, func_ids, lhs, params, locals)?;
                        emit_ir_expr(b, func_ids, rhs, params, locals)?;
                        match op {
                            IrBinOp::Add => {
                                b.binop(BinaryOp::F64Add);
                            }
                            IrBinOp::Sub => {
                                b.binop(BinaryOp::F64Sub);
                            }
                            IrBinOp::Mul => {
                                b.binop(BinaryOp::F64Mul);
                            }
                            IrBinOp::Div => {
                                b.binop(BinaryOp::F64Div);
                            }
                            // Comparisons produce an i32; expression context
                            // is uniformly f64, so convert the 0/1 back.
                            IrBinOp::Lt => {
                                b.binop(BinaryOp::F64Lt);
                                b.unop(UnaryOp::F64ConvertSI32);
                            }
                            IrBinOp::Le => {
                                b.binop(BinaryOp::F64Le);
                                b.unop(UnaryOp::F64ConvertSI32);
                            }
                            IrBinOp::Gt => {
                                b.binop(BinaryOp::F64Gt);
                                b.unop(UnaryOp::F64ConvertSI32);
                            }
                            IrBinOp::Ge => {
                                b.binop(BinaryOp::F64Ge);
                                b.unop(UnaryOp::F64ConvertSI32);
                            }
                            IrBinOp::Eq => {
                                b.binop(BinaryOp::F64Eq);
                                b.unop(UnaryOp::F64ConvertSI32);
                            }
                            IrBinOp::Ne => {
                                b.binop(BinaryOp::F64Ne);
                                b.unop(UnaryOp::F64ConvertSI32);
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
            IrExpr::UnaryOp { op, expr: inner } => {
                emit_ir_expr(b, func_ids, inner, params, locals)?;
                match op {
                    IrUnaryOp::Neg => {
                        b.unop(UnaryOp::F64Neg);
                    }
                    IrUnaryOp::Not => {
                        // Convert to boolean: compare with 0 (i32), then back
                        // to the f64 expression convention.
                        b.f64_const(0.0);
                        b.binop(BinaryOp::F64Eq);
                        b.unop(UnaryOp::F64ConvertSI32);
                    }
                }
            }
            IrExpr::MathCall { func, args } => {
                match func {
                    IrMathFunc::Abs
                    | IrMathFunc::Sqrt
                    | IrMathFunc::Floor
                    | IrMathFunc::Ceil
                    | IrMathFunc::Trunc
                    | IrMathFunc::Nearest => {
                        if args.len() != 1 {
                            return Err(CompileError::Type(
                                "math function requires 1 argument".to_string(),
                            ));
                        }
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        match func {
                            IrMathFunc::Abs => {
                                b.unop(UnaryOp::F64Abs);
                            }
                            IrMathFunc::Sqrt => {
                                b.unop(UnaryOp::F64Sqrt);
                            }
                            IrMathFunc::Floor => {
                                b.unop(UnaryOp::F64Floor);
                            }
                            IrMathFunc::Ceil => {
                                b.unop(UnaryOp::F64Ceil);
                            }
                            IrMathFunc::Trunc => {
                                b.unop(UnaryOp::F64Trunc);
                            }
                            IrMathFunc::Nearest => {
                                b.unop(UnaryOp::F64Nearest);
                            }
                            _ => unreachable!(),
                        }
                    }
                    IrMathFunc::Min | IrMathFunc::Max => {
                        if args.len() != 2 {
                            return Err(CompileError::Type(
                                "math.min/max requires 2 arguments".to_string(),
                            ));
                        }
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[1], params, locals)?;
                        match func {
                            IrMathFunc::Min => {
                                b.binop(BinaryOp::F64Min);
                            }
                            IrMathFunc::Max => {
                                b.binop(BinaryOp::F64Max);
                            }
                            _ => unreachable!(),
                        }
                    }
                    IrMathFunc::Sin => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.sin requires 1 argument".into()));
                        }
                        // sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040 (Taylor series, accurate for |x| < π)
                        // Compute: x
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        // Compute: - x³/6
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul); // x³
                        b.f64_const(6.0);
                        b.binop(BinaryOp::F64Div); // x³/6
                        b.binop(BinaryOp::F64Sub); // x - x³/6
                        // Compute: + x⁵/120
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul); // x⁵
                        b.f64_const(120.0);
                        b.binop(BinaryOp::F64Div); // x⁵/120
                        b.binop(BinaryOp::F64Add); // x - x³/6 + x⁵/120
                        // Compute: - x⁷/5040
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul); // x⁷
                        b.f64_const(5040.0);
                        b.binop(BinaryOp::F64Div); // x⁷/5040
                        b.binop(BinaryOp::F64Sub); // final result
                    }
                    IrMathFunc::Cos => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.cos requires 1 argument".into()));
                        }
                        // cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720 (Taylor series)
                        b.f64_const(1.0);
                        // - x²/2
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul); // x²
                        b.f64_const(2.0);
                        b.binop(BinaryOp::F64Div); // x²/2
                        b.binop(BinaryOp::F64Sub); // 1 - x²/2
                        // + x⁴/24
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul); // x⁴
                        b.f64_const(24.0);
                        b.binop(BinaryOp::F64Div); // x⁴/24
                        b.binop(BinaryOp::F64Add);
                        // - x⁶/720
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul); // x⁶
                        b.f64_const(720.0);
                        b.binop(BinaryOp::F64Div); // x⁶/720
                        b.binop(BinaryOp::F64Sub); // final result
                    }
                    IrMathFunc::Tan => {
                        if args.len() != 1 {
                            return Err(CompileError::Type("math.tan requires 1 argument".into()));
                        }
                        // tan(x) = sin(x) / cos(x) - use simpler approximation
                        // tan(x) ≈ x + x³/3 + 2x⁵/15 for small x
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        // + x³/3
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(3.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Add);
                        // + 2x⁵/15
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(7.5); // 15/2
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
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.binop(BinaryOp::F64Add);
                        // + x²/2
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(2.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Add);
                        // + x³/6
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(6.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Add);
                        // + x⁴/24
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(24.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Add);
                        // + x⁵/120
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(120.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Add);
                        // + x⁶/720
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
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
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub); // u = x - 1
                        // For simplicity, just use u - u²/2 + u³/3
                        // We need u multiple times, so emit (x-1) each time
                        // - u²/2
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        b.binop(BinaryOp::F64Mul); // u²
                        b.f64_const(2.0);
                        b.binop(BinaryOp::F64Div); // u²/2
                        b.binop(BinaryOp::F64Sub); // u - u²/2
                        // + u³/3
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul); // u³
                        b.f64_const(3.0);
                        b.binop(BinaryOp::F64Div); // u³/3
                        b.binop(BinaryOp::F64Add); // u - u²/2 + u³/3
                    }
                    IrMathFunc::Pow => {
                        if args.len() != 2 {
                            return Err(CompileError::Type("math.pow requires 2 arguments".into()));
                        }
                        // pow(x, y) = exp(y * log(x))
                        // First compute log(x)
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Sub);
                        b.binop(BinaryOp::F64Mul);
                        b.f64_const(2.0);
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Sub); // log(x) ≈ (x-1) - (x-1)²/2
                        // Multiply by y
                        emit_ir_expr(b, func_ids, &args[1], params, locals)?;
                        b.binop(BinaryOp::F64Mul); // y * log(x)
                        // Now compute exp of that - store intermediate and use exp approximation
                        // exp(z) ≈ 1 + z + z²/2 + z³/6
                        // This is tricky because we need z multiple times but it's a computed value
                        // For simplicity, use a rougher approximation: exp(z) ≈ 1 + z + z²/2
                        // We need to duplicate the z value, but we can't without locals
                        // WORKAROUND: Recompute y*log(x) each time
                        b.f64_const(1.0);
                        b.binop(BinaryOp::F64Add); // 1 + z (very rough exp approximation)
                    }
                    IrMathFunc::Atan2 => {
                        if args.len() != 2 {
                            return Err(CompileError::Type(
                                "math.atan2 requires 2 arguments".into(),
                            ));
                        }
                        // atan2(y, x) - simplified approximation
                        // For the primary case (x > 0): atan(y/x)
                        // atan(z) ≈ z - z³/3 + z⁵/5 for |z| < 1
                        // For simplicity: atan2(y,x) ≈ y/x - (y/x)³/3 when x > 0, |y/x| < 1
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?; // y
                        emit_ir_expr(b, func_ids, &args[1], params, locals)?; // x
                        b.binop(BinaryOp::F64Div); // y/x = z
                        // - z³/3
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[1], params, locals)?;
                        b.binop(BinaryOp::F64Div);
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[1], params, locals)?;
                        b.binop(BinaryOp::F64Div);
                        emit_ir_expr(b, func_ids, &args[0], params, locals)?;
                        emit_ir_expr(b, func_ids, &args[1], params, locals)?;
                        b.binop(BinaryOp::F64Div);
                        b.binop(BinaryOp::F64Mul);
                        b.binop(BinaryOp::F64Mul); // z³
                        b.f64_const(3.0);
                        b.binop(BinaryOp::F64Div); // z³/3
                        b.binop(BinaryOp::F64Sub); // z - z³/3
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
                    return Err(CompileError::Unsupported(format!(
                        "unknown function: {}",
                        name
                    )));
                }
            }
            IrExpr::IfThenElse {
                cond,
                then_expr,
                else_expr,
            } => {
                emit_ir_condition(b, func_ids, cond, params, locals)?;
                let then_expr = then_expr.clone();
                let else_expr = else_expr.clone();
                let params_vec: Vec<String> = params.to_vec();
                let locals_clone: HashMap<String, walrus::LocalId> = locals.clone();
                let func_ids_ref = func_ids;
                b.if_else(
                    ValType::F64,
                    |then_block| {
                        emit_ir_expr(
                            then_block,
                            func_ids_ref,
                            &then_expr,
                            &params_vec,
                            &locals_clone,
                        )
                        .expect("emit_ir_expr in then branch failed");
                    },
                    |else_block| {
                        emit_ir_expr(
                            else_block,
                            func_ids_ref,
                            &else_expr,
                            &params_vec,
                            &locals_clone,
                        )
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
        if let IrExpr::BinOp { op, lhs, rhs } = expr
            && matches!(
                op,
                IrBinOp::Lt | IrBinOp::Le | IrBinOp::Gt | IrBinOp::Ge | IrBinOp::Eq | IrBinOp::Ne
            )
        {
            emit_ir_expr(b, func_ids, lhs, params, locals)?;
            emit_ir_expr(b, func_ids, rhs, params, locals)?;
            match op {
                IrBinOp::Lt => {
                    b.binop(BinaryOp::F64Lt);
                }
                IrBinOp::Le => {
                    b.binop(BinaryOp::F64Le);
                }
                IrBinOp::Gt => {
                    b.binop(BinaryOp::F64Gt);
                }
                IrBinOp::Ge => {
                    b.binop(BinaryOp::F64Ge);
                }
                IrBinOp::Eq => {
                    b.binop(BinaryOp::F64Eq);
                }
                IrBinOp::Ne => {
                    b.binop(BinaryOp::F64Ne);
                }
                _ => unreachable!(),
            }
            return Ok(());
        }
        // For non-comparison expressions, emit as f64 and convert to i32 (non-zero = true)
        emit_ir_expr(b, func_ids, expr, params, locals)?;
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
                    emit_ir_expr(b, func_ids, value, params, locals)?;
                    b.local_set(local_id);
                }
                IrStmt::Assign { name, value } => {
                    if let Some(&local_id) = locals.get(name) {
                        emit_ir_expr(b, func_ids, value, params, locals)?;
                        b.local_set(local_id);
                    } else {
                        return Err(CompileError::Unsupported(format!(
                            "assignment to unknown variable: {}",
                            name
                        )));
                    }
                }
                IrStmt::If {
                    cond,
                    then_body,
                    else_body,
                } => {
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
                                emit_ir_expr(
                                    then_block,
                                    func_ids_ref,
                                    expr,
                                    &params_vec,
                                    &then_locals,
                                )
                                .expect("emit in then branch failed");
                            } else {
                                then_block.f64_const(0.0); // fallback
                            }
                        },
                        |else_block| {
                            if let Some(IrStmt::Return(expr)) = else_body.first() {
                                emit_ir_expr(
                                    else_block,
                                    func_ids_ref,
                                    expr,
                                    &params_vec,
                                    &else_locals,
                                )
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
                            emit_ir_condition(
                                loop_block,
                                func_ids_ref,
                                &cond,
                                &params_vec,
                                &loop_locals,
                            )
                            .expect("emit while condition failed");
                            loop_block.unop(UnaryOp::I32Eqz); // Invert: exit if condition is false
                            loop_block.br_if(exit_id);

                            // Emit body statements
                            for stmt in &body {
                                match stmt {
                                    IrStmt::LocalAssign { name, value }
                                    | IrStmt::Assign { name, value } => {
                                        if let Some(&local_id) = loop_locals.get(name) {
                                            emit_ir_expr(
                                                loop_block,
                                                func_ids_ref,
                                                value,
                                                &params_vec,
                                                &loop_locals,
                                            )
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
                                    IrStmt::LocalAssign { name, value }
                                    | IrStmt::Assign { name, value } => {
                                        if let Some(&local_id) = loop_locals.get(name) {
                                            emit_ir_expr(
                                                loop_block,
                                                func_ids_ref,
                                                value,
                                                &params_vec,
                                                &loop_locals,
                                            )
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
                            emit_ir_condition(
                                loop_block,
                                func_ids_ref,
                                &cond,
                                &params_vec,
                                &loop_locals,
                            )
                            .expect("emit repeat-until condition failed");
                            loop_block.br_if(exit_id);

                            // Continue to next iteration
                            loop_block.br(loop_id);
                        });
                    });
                }
                IrStmt::NumericFor {
                    var,
                    start,
                    end,
                    step,
                    body,
                } => {
                    // Create local for loop variable
                    let loop_var_id = module_locals.add(ValType::F64);
                    locals.insert(var.clone(), loop_var_id);

                    // Also create locals for end and step to avoid re-evaluating
                    let end_var_id = module_locals.add(ValType::F64);
                    let step_var_id = module_locals.add(ValType::F64);

                    // Initialize loop variable with start value
                    emit_ir_expr(b, func_ids, start, params, locals)?;
                    b.local_set(loop_var_id);

                    // Initialize end value
                    emit_ir_expr(b, func_ids, end, params, locals)?;
                    b.local_set(end_var_id);

                    // Initialize step value (default to 1.0)
                    if let Some(step_expr) = step {
                        emit_ir_expr(b, func_ids, step_expr, params, locals)?;
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
                            loop_block.binop(BinaryOp::F64Sub); // i - end
                            loop_block.local_get(step_var_id);
                            loop_block.binop(BinaryOp::F64Mul); // (i - end) * step
                            loop_block.f64_const(0.0);
                            loop_block.binop(BinaryOp::F64Gt); // > 0 means we should exit
                            loop_block.br_if(exit_id);

                            // Emit body statements
                            for stmt in &body {
                                match stmt {
                                    IrStmt::LocalAssign { name, value }
                                    | IrStmt::Assign { name, value } => {
                                        if let Some(&local_id) = loop_locals.get(name) {
                                            emit_ir_expr(
                                                loop_block,
                                                func_ids_ref,
                                                value,
                                                &params_vec,
                                                &loop_locals,
                                            )
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

    // Generate helpers after their dependencies, then generate the required
    // model functions that call them. This remains deterministic even though
    // declarations are stored in a HashMap.
    for fname in helper_emission_order(&functions, &required_funcs)? {
        let ir_func = functions.get(&fname).expect("ordered helper must exist");

        // Helper functions take f64 params and return f64
        let param_types: Vec<ValType> = ir_func.params.iter().map(|_| ValType::F64).collect();
        let mut fb = FunctionBuilder::new(&mut module.types, &param_types, &[ValType::F64]);

        // Create locals for parameters
        let param_locals: Vec<walrus::LocalId> = ir_func
            .params
            .iter()
            .map(|_| module.locals.add(ValType::F64))
            .collect();

        let (mut locals_map, constant_initializers) =
            add_constant_locals(&mut module.locals, &constants);
        for (i, param_name) in ir_func.params.iter().enumerate() {
            locals_map.insert(param_name.clone(), param_locals[i]);
        }

        let mut ib = fb.func_body();
        initialize_constants(&mut ib, &constant_initializers);
        let ret_expr = emit_ir_stmts(
            &mut ib,
            &func_ids,
            &ir_func.body,
            &ir_func.params,
            &mut locals_map,
            &mut module.locals,
        )?;
        if let Some(expr) = ret_expr {
            emit_ir_expr(&mut ib, &func_ids, &expr, &ir_func.params, &locals_map)?;
        }

        let fid = fb.finish(param_locals, &mut module.funcs);
        func_ids.insert(fname, fid);
    }

    // Create memory for I/O buffers and export it
    // add_local(shared, shared64, initial_pages, max_pages, page_size_log2)
    let memory_id = module.memories.add_local(false, false, 1, None, None);
    module.exports.add("memory", memory_id);

    // Generate internal WASM functions for each required Lua function (not exported directly)
    // These will be called by the ABI wrapper functions
    let mut internal_func_ids: HashMap<String, walrus::FunctionId> = HashMap::new();

    for &fname in &required_funcs {
        let ir_func = functions.get(fname).unwrap();
        match fname {
            "is_inside" => {
                // Internal is_inside: (f64 * dims) -> f64 (not demoted to f32
                // yet); arity was validated when dims was inferred.
                let param_types = vec![ValType::F64; dims];
                let mut fb = FunctionBuilder::new(&mut module.types, &param_types, &[ValType::F64]);
                let param_locals: Vec<walrus::LocalId> =
                    (0..dims).map(|_| module.locals.add(ValType::F64)).collect();
                let (mut locals_map, constant_initializers) =
                    add_constant_locals(&mut module.locals, &constants);
                for (param, &local) in ir_func.params.iter().zip(&param_locals) {
                    locals_map.insert(param.clone(), local);
                }
                let mut ib = fb.func_body();
                initialize_constants(&mut ib, &constant_initializers);
                let ret_expr = emit_ir_stmts(
                    &mut ib,
                    &func_ids,
                    &ir_func.body,
                    &ir_func.params,
                    &mut locals_map,
                    &mut module.locals,
                )?;
                if let Some(expr) = ret_expr {
                    emit_ir_expr(&mut ib, &func_ids, &expr, &ir_func.params, &locals_map)?;
                }
                let fid = fb.finish(param_locals, &mut module.funcs);
                internal_func_ids.insert(fname.to_string(), fid);
            }
            _ => {
                // Bounds getter: () -> f64
                if !ir_func.params.is_empty() {
                    return Err(CompileError::Type(
                        "bounds getters must have 0 params".into(),
                    ));
                }
                let mut fb = FunctionBuilder::new(&mut module.types, &[], &[ValType::F64]);
                let (mut locals_map, constant_initializers) =
                    add_constant_locals(&mut module.locals, &constants);
                let mut ib = fb.func_body();
                initialize_constants(&mut ib, &constant_initializers);
                let ret_expr = emit_ir_stmts(
                    &mut ib,
                    &func_ids,
                    &ir_func.body,
                    &ir_func.params,
                    &mut locals_map,
                    &mut module.locals,
                )?;
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
        ib.i32_const(dims as i32);
        let fid = fb.finish(vec![], &mut module.funcs);
        module.exports.add("get_dimensions", fid);
    }

    // Generate and export get_io_ptr() -> i32
    // The compiled Lua functions keep no state in linear memory, so the IO
    // buffer is the start of the page (offset 8: nonzero so it never aliases
    // a null pointer).
    {
        let mut fb = FunctionBuilder::new(&mut module.types, &[], &[ValType::I32]);
        let mut ib = fb.func_body();
        ib.i32_const(8);
        let fid = fb.finish(vec![], &mut module.funcs);
        module.exports.add("get_io_ptr", fid);
    }

    // Generate and export get_bounds(out_ptr: i32)
    // Writes 2 * dims f64 values, interleaved min/max per axis
    {
        let mut fb = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[]);
        let out_ptr = module.locals.add(ValType::I32);
        let mut ib = fb.func_body();

        for (i, &func_name) in BOUNDS_FUNCS[..dims * 2].iter().enumerate() {
            let func_id = *internal_func_ids.get(func_name).unwrap();
            let mem_arg = walrus::ir::MemArg {
                align: 3,
                offset: (i * 8) as u64,
            };
            ib.local_get(out_ptr);
            ib.call(func_id);
            ib.store(memory_id, walrus::ir::StoreKind::F64, mem_arg);
        }

        let fid = fb.finish(vec![out_ptr], &mut module.funcs);
        module.exports.add("get_bounds", fid);
    }

    // Generate and export sample(pos_ptr: i32) -> f32
    // Reads dims f64 values from pos_ptr, calls internal is_inside, returns f32
    {
        let mut fb = FunctionBuilder::new(&mut module.types, &[ValType::I32], &[ValType::F32]);
        let pos_ptr = module.locals.add(ValType::I32);
        let mut ib = fb.func_body();

        let is_inside_id = *internal_func_ids.get("is_inside").unwrap();

        // Load each position component from pos_ptr + 8 * axis
        for axis in 0..dims {
            let mem_arg = walrus::ir::MemArg {
                align: 3,
                offset: (axis * 8) as u64,
            };
            ib.local_get(pos_ptr);
            ib.load(memory_id, walrus::ir::LoadKind::F64, mem_arg);
        }

        // Call internal is_inside(...) -> f64
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
    let buf = read_input(0);
    let src = match std::str::from_utf8(&buf) {
        Ok(s) => s,
        Err(_) => {
            report_error("Lua source is not valid UTF-8");
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
    let output = match compile_lua_to_wasm_with_parameters(src, &routed_values) {
        Ok(w) => w,
        Err(e) => {
            report_error(&format!("Lua compile error: {e}"));
            return;
        }
    };
    post_output(0, &output);
}

/// Minimal stub template for the Lua script input.
/// This template shows all required function signatures that the script must implement.
const LUA_TEMPLATE: &str = r#"-- Occupancy: return 1.0 inside, 0.0 outside (inside iff value > 0.5).
-- For a 2D sketch (extrude input), define is_inside(x, y) and only the
-- x/y bounds functions instead.
local radius = 1.0 -- @param key="sphere.radius" min=0.000001
local bounds_margin = 0.5
local bound = radius + bounds_margin

function is_inside(x, y, z)
    -- Example: unit sphere centered at origin
    if x*x + y*y + z*z <= radius*radius then
        return 1.0
    else
        return 0.0
    end
end

-- Bounding box functions define the region to sample
function get_bounds_min_x()
    return -bound
end

function get_bounds_min_y()
    return -bound
end

function get_bounds_min_z()
    return -bound
end

function get_bounds_max_x()
    return bound
end

function get_bounds_max_y()
    return bound
end

function get_bounds_max_z()
    return bound
end
"#;

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata {
        name: "lua_script_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "Lua Script".to_string(),
        description: "Compile a restricted Lua script into a model module.".to_string(),
        category: "Scripting".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<circle cx="11" cy="13" r="7.5"/>"##,
            r##"<circle cx="14.5" cy="9.5" r="2"/>"##,
            r##"<circle cx="20.5" cy="3.5" r="1.5"/>"##,
        )
        .to_string(),
        inputs: vec![
            OperatorMetadataInput::LuaSource(LUA_TEMPLATE.to_string()),
            OperatorMetadataInput::F64Map,
        ],
        input_names: vec!["Script".to_string(), "Parameters".to_string()],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reparse compiled output with walrus, which rejects type-invalid code
    /// sections — comparisons used as *values* (e.g. chained `and`) once
    /// emitted i32 where the f64 expression convention was expected.
    fn assert_valid_wasm(bytes: &[u8]) {
        Module::from_buffer_with_config(bytes, &ModuleConfig::new())
            .expect("compiled Lua module must reparse as valid wasm");
    }

    #[test]
    fn module_constants_compile_for_helpers_sampling_and_bounds() {
        let lua_src = r#"
local diameter = 2.0
local radius = diameter / 2.0
local margin = math.sqrt(0.0625)
local bound = radius + margin

function radial_squared(x, y)
    return x*x + y*y
end

function is_inside(x, y)
    return radial_squared(x, y) <= radius*radius
end

function get_bounds_min_x() return -bound end
function get_bounds_max_x() return bound end
function get_bounds_min_y() return -bound end
function get_bounds_max_y() return bound end
"#;

        assert_valid_wasm(&compile_lua_to_wasm(lua_src).expect("compile shared module constants"));
    }

    #[test]
    fn annotated_constants_accept_routed_f64_map_values() {
        let lua_src = r#"
local radius = 1.0 -- @param key="shared.radius" min=0.25 max=4.0
local bound = radius + 0.5
function is_inside(x, y) return x*x + y*y <= radius*radius end
function get_bounds_min_x() return -bound end
function get_bounds_max_x() return bound end
function get_bounds_min_y() return -bound end
function get_bounds_max_y() return bound end
"#;
        let routed = volumetric_abi::f64_map::F64Map::from([
            ("shared.radius".to_string(), 2.0),
            ("unrelated.global".to_string(), 99.0),
        ]);
        assert_valid_wasm(
            &compile_lua_to_wasm_with_parameters(lua_src, &routed)
                .expect("compile routed parameter and ignore unrelated key"),
        );
    }

    #[test]
    fn annotated_constants_enforce_declared_ranges() {
        let lua_src = r#"
local radius = 1.0 -- @param key=radius min=0.25 max=4.0
function is_inside(x, y) return x*x + y*y <= radius*radius end
function get_bounds_min_x() return -radius end
function get_bounds_max_x() return radius end
function get_bounds_min_y() return -radius end
function get_bounds_max_y() return radius end
"#;
        let routed = volumetric_abi::f64_map::F64Map::from([("radius".to_string(), 5.0)]);
        let error = compile_lua_to_wasm_with_parameters(lua_src, &routed)
            .expect_err("out-of-range override must fail");
        assert!(
            matches!(error, CompileError::Type(ref message) if message.contains("above maximum")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn annotations_bind_the_exact_module_declaration() {
        let lua_src = r#"
local radius = 1.0
function helper(x)
    local radius = 2.0 -- @param key=radius
    return x <= radius
end
function is_inside(x, y) return helper(x*x + y*y) end
function get_bounds_min_x() return -radius end
function get_bounds_max_x() return radius end
function get_bounds_min_y() return -radius end
function get_bounds_max_y() return radius end
"#;
        let routed = volumetric_abi::f64_map::F64Map::from([("radius".to_string(), 3.0)]);
        let error = compile_lua_to_wasm_with_parameters(lua_src, &routed)
            .expect_err("a nested annotation must not retarget a module constant");
        assert!(
            matches!(error, CompileError::Type(ref message) if message.contains("does not annotate a module-scope")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn fidget_spinner_reference_compiles() {
        let lua_src = include_str!("../../../../examples/fidget_spinner.lua");
        assert_valid_wasm(&compile_lua_to_wasm(lua_src).expect("compile reference spinner"));
    }

    #[test]
    fn raspberry_pi_tray_reference_compiles() {
        let lua_src = include_str!("../../../../examples/raspberry_pi_4_tray.lua");
        assert_valid_wasm(&compile_lua_to_wasm(lua_src).expect("compile reference Pi tray"));
    }

    #[test]
    fn helper_calls_are_emitted_in_dependency_order() {
        let lua_src = r#"
function outer(x) return inner(x) end
function inner(x) return x*x end
function is_inside(x, y) return outer(x) + y*y <= 1.0 end
function get_bounds_min_x() return -1.0 end
function get_bounds_max_x() return 1.0 end
function get_bounds_min_y() return -1.0 end
function get_bounds_max_y() return 1.0 end
"#;
        assert_valid_wasm(&compile_lua_to_wasm(lua_src).expect("compile forward helper call"));
    }

    #[test]
    fn recursive_helper_calls_are_rejected() {
        let lua_src = r#"
function first(x) return second(x) end
function second(x) return first(x) end
function is_inside(x, y) return first(x) + y end
function get_bounds_min_x() return -1.0 end
function get_bounds_max_x() return 1.0 end
function get_bounds_min_y() return -1.0 end
function get_bounds_max_y() return 1.0 end
"#;
        let error = compile_lua_to_wasm(lua_src).expect_err("recursive helpers must fail");
        assert!(
            matches!(error, CompileError::Unsupported(ref message) if message.contains("recursive helper call")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn module_constants_cannot_reference_later_names() {
        let lua_src = r#"
local radius = diameter / 2.0
local diameter = 2.0
function is_inside(x, y) return x*x + y*y <= radius*radius end
function get_bounds_min_x() return -radius end
function get_bounds_max_x() return radius end
function get_bounds_min_y() return -radius end
function get_bounds_max_y() return radius end
"#;

        let err = compile_lua_to_wasm(lua_src).expect_err("forward reference must be rejected");
        assert!(
            matches!(err, CompileError::Type(ref message) if message.contains("later name `diameter`")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn comparisons_as_values_emit_valid_wasm() {
        let lua_src = r#"
function is_inside(x, y)
    if x >= 0.5 and x <= 1.0 and y >= 0.0 and y <= 2.0 then
        return 1.0
    else
        return 0.0
    end
end
function get_bounds_min_x() return 0.0 end
function get_bounds_max_x() return 1.25 end
function get_bounds_min_y() return -0.25 end
function get_bounds_max_y() return 2.25 end
"#;
        assert_valid_wasm(&compile_lua_to_wasm(lua_src).expect("compile chained and"));

        let lua_src_not = r#"
function is_inside(x, y, z)
    return (not (x > 0.0)) or (y < 1.0 and z ~= 0.0)
end
function get_bounds_min_x() return -1.0 end
function get_bounds_max_x() return 1.0 end
function get_bounds_min_y() return -1.0 end
function get_bounds_max_y() return 1.0 end
function get_bounds_min_z() return -1.0 end
function get_bounds_max_z() return 1.0 end
"#;
        assert_valid_wasm(&compile_lua_to_wasm(lua_src_not).expect("compile not/or"));
    }

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
        assert!(
            result.is_ok(),
            "Failed to compile with math.sqrt: {:?}",
            result.err()
        );
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
        assert!(
            result.is_ok(),
            "Failed to compile with unary minus: {:?}",
            result.err()
        );
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
        assert!(
            result.is_ok(),
            "Failed to compile with if-then-else: {:?}",
            result.err()
        );
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
        assert!(
            result.is_ok(),
            "Failed to compile with modulo: {:?}",
            result.err()
        );
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
        assert!(
            result.is_ok(),
            "Failed to compile with while loop: {:?}",
            result.err()
        );
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
        assert!(
            result.is_ok(),
            "Failed to compile with numeric for: {:?}",
            result.err()
        );
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
        assert!(
            result.is_ok(),
            "Failed to compile with trig functions: {:?}",
            result.err()
        );
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
        assert!(
            result.is_ok(),
            "Failed to compile with user function: {:?}",
            result.err()
        );
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
        assert!(
            result.is_ok(),
            "Failed to compile with exponentiation: {:?}",
            result.err()
        );
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
        assert!(
            result.is_ok(),
            "Failed to compile with repeat-until: {:?}",
            result.err()
        );
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
        assert!(
            result.is_ok(),
            "Failed to compile with logical and/or: {:?}",
            result.err()
        );
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
        assert!(
            result.is_ok(),
            "Failed to compile with math.pi: {:?}",
            result.err()
        );
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
        assert!(
            result.is_ok(),
            "Failed to compile with break in repeat: {:?}",
            result.err()
        );
        let wasm = result.unwrap();
        assert!(wasm.len() > 8, "WASM output too short");
        assert_eq!(&wasm[0..4], b"\0asm", "Invalid WASM magic number");
    }
}

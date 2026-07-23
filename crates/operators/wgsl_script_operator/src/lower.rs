//! naga IR → walrus lowering.
//!
//! The compiled module starts life as the prebuilt `wgsl_model_template`
//! (libm kernels, IO buffer, memory); script functions are built directly
//! into it and the model-ABI exports wired on top.
//!
//! Value representation: every scalar is one wasm value (f64, f32, or i32 —
//! bools and both integer signednesses share i32); vectors are flattened
//! into per-component values. Function signatures flatten the same way and
//! use multi-value returns. Fixed-size arrays never live in wasm locals:
//! module-`const`-initialized read-only arrays become data segments, and
//! mutable function-local arrays get static frames in linear memory above
//! the template's `__heap_base` (sound because WGSL has no recursion, and
//! each thread instantiates its own model).

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};

use walrus::ir::{BinaryOp, MemArg, UnaryOp, Value};
use walrus::{ConstExpr, FunctionBuilder, FunctionId, InstrSeqBuilder, LocalId, ValType};

use naga::Handle;

use crate::CompileError;
use crate::restrict::ModelFunctions;

/// The prebuilt template module (see crate docs for regeneration).
static TEMPLATE: &[u8] = include_bytes!("../template/wgsl_model_template.wasm");

const PAGE: u64 = 65536;

// ---------------------------------------------------------------------------
// Scalar classification and constant values
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScalarTy {
    F64,
    F32,
    I32,
    U32,
    Bool,
}

impl ScalarTy {
    fn val_type(self) -> ValType {
        match self {
            ScalarTy::F64 => ValType::F64,
            ScalarTy::F32 => ValType::F32,
            ScalarTy::I32 | ScalarTy::U32 | ScalarTy::Bool => ValType::I32,
        }
    }

    fn byte_width(self) -> u32 {
        match self {
            ScalarTy::F64 => 8,
            _ => 4,
        }
    }

    fn classify(scalar: naga::Scalar) -> Result<ScalarTy, CompileError> {
        use naga::ScalarKind as Sk;
        match (scalar.kind, scalar.width) {
            (Sk::Float, 8) => Ok(ScalarTy::F64),
            (Sk::Float, 4) => Ok(ScalarTy::F32),
            (Sk::Sint, 4) => Ok(ScalarTy::I32),
            (Sk::Uint, 4) => Ok(ScalarTy::U32),
            (Sk::Bool, _) => Ok(ScalarTy::Bool),
            _ => Err(CompileError::Unsupported(format!(
                "scalar type {:?} width {} is not part of the model dialect",
                scalar.kind, scalar.width
            ))),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ConstScalar {
    F64(f64),
    F32(f32),
    I32(i32),
    U32(u32),
    Bool(bool),
}

impl ConstScalar {
    fn ty(self) -> ScalarTy {
        match self {
            ConstScalar::F64(_) => ScalarTy::F64,
            ConstScalar::F32(_) => ScalarTy::F32,
            ConstScalar::I32(_) => ScalarTy::I32,
            ConstScalar::U32(_) => ScalarTy::U32,
            ConstScalar::Bool(_) => ScalarTy::Bool,
        }
    }

    fn as_f64(self) -> Result<f64, CompileError> {
        match self {
            ConstScalar::F64(v) => Ok(v),
            ConstScalar::F32(v) => Ok(v as f64),
            _ => Err(CompileError::Type(
                "expected a floating-point constant".to_string(),
            )),
        }
    }

    fn write_le(self, out: &mut Vec<u8>) {
        match self {
            ConstScalar::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
            ConstScalar::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
            ConstScalar::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
            ConstScalar::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
            ConstScalar::Bool(v) => out.extend_from_slice(&(v as u32).to_le_bytes()),
        }
    }

    fn emit(self, b: &mut InstrSeqBuilder) {
        match self {
            ConstScalar::F64(v) => {
                b.f64_const(v);
            }
            ConstScalar::F32(v) => {
                b.f32_const(v);
            }
            ConstScalar::I32(v) => {
                b.i32_const(v);
            }
            ConstScalar::U32(v) => {
                b.i32_const(v as i32);
            }
            ConstScalar::Bool(v) => {
                b.i32_const(v as i32);
            }
        }
    }
}

fn literal_const(lit: naga::Literal) -> Result<ConstScalar, CompileError> {
    match lit {
        naga::Literal::F64(v) => Ok(ConstScalar::F64(v)),
        naga::Literal::F32(v) => Ok(ConstScalar::F32(v)),
        naga::Literal::I32(v) => Ok(ConstScalar::I32(v)),
        naga::Literal::U32(v) => Ok(ConstScalar::U32(v)),
        naga::Literal::Bool(v) => Ok(ConstScalar::Bool(v)),
        other => Err(CompileError::Unsupported(format!(
            "literal {other:?} is not part of the model dialect"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Type flattening
// ---------------------------------------------------------------------------

/// What a (non-pointer) expression flattens to.
fn value_components(
    module: &naga::Module,
    inner: &naga::TypeInner,
) -> Result<Vec<ScalarTy>, CompileError> {
    use naga::TypeInner as Ti;
    match inner {
        Ti::Scalar(scalar) => Ok(vec![ScalarTy::classify(*scalar)?]),
        Ti::Vector { size, scalar } => Ok(vec![ScalarTy::classify(*scalar)?; *size as usize]),
        Ti::Array { .. } => Err(CompileError::Unsupported(
            "array values must live in `var`s (pass elements, not whole arrays)".to_string(),
        )),
        other => Err(CompileError::Unsupported(format!(
            "values of type {} are not supported",
            type_label(module, other)
        ))),
    }
}

fn type_label(_module: &naga::Module, inner: &naga::TypeInner) -> String {
    format!("{inner:?}")
}

/// What a pointer expression points at.
#[derive(Clone, Copy, Debug)]
enum Pointee {
    Scalar(ScalarTy),
    Vector(ScalarTy, usize),
    Array {
        elem: ScalarTy,
        elem_len: usize,
        stride: u32,
        count: u32,
    },
}

impl Pointee {
    fn of_inner(module: &naga::Module, inner: &naga::TypeInner) -> Result<Pointee, CompileError> {
        use naga::TypeInner as Ti;
        match inner {
            Ti::Scalar(scalar) => Ok(Pointee::Scalar(ScalarTy::classify(*scalar)?)),
            Ti::Vector { size, scalar } => Ok(Pointee::Vector(
                ScalarTy::classify(*scalar)?,
                *size as usize,
            )),
            Ti::Array { base, size, stride } => {
                let count = match size {
                    naga::ArraySize::Constant(n) => n.get(),
                    _ => {
                        return Err(CompileError::Unsupported(
                            "runtime-sized arrays are not supported".to_string(),
                        ));
                    }
                };
                let (elem, elem_len) = match &module.types[*base].inner {
                    Ti::Scalar(scalar) => (ScalarTy::classify(*scalar)?, 1),
                    Ti::Vector { size, scalar } => (ScalarTy::classify(*scalar)?, *size as usize),
                    other => {
                        return Err(CompileError::Unsupported(format!(
                            "arrays of {} are not supported",
                            type_label(module, other)
                        )));
                    }
                };
                Ok(Pointee::Array {
                    elem,
                    elem_len,
                    stride: *stride,
                    count,
                })
            }
            other => Err(CompileError::Unsupported(format!(
                "pointers to {} are not supported",
                type_label(module, other)
            ))),
        }
    }

    fn components(self) -> Result<Vec<ScalarTy>, CompileError> {
        match self {
            Pointee::Scalar(ty) => Ok(vec![ty]),
            Pointee::Vector(ty, n) => Ok(vec![ty; n]),
            Pointee::Array { .. } => Err(CompileError::Unsupported(
                "whole-array loads/stores are not supported; index the array instead".to_string(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Constant evaluation (module consts, override defaults, local var inits)
// ---------------------------------------------------------------------------

pub struct ConstCx<'a> {
    pub module: &'a naga::Module,
    pub overrides: &'a HashMap<Handle<naga::Override>, ConstScalar>,
}

impl ConstCx<'_> {
    /// Evaluate an expression from `arena` to flattened scalar components.
    pub fn eval(
        &self,
        arena: &naga::Arena<naga::Expression>,
        handle: Handle<naga::Expression>,
    ) -> Result<Vec<ConstScalar>, CompileError> {
        use naga::Expression as Ex;
        match &arena[handle] {
            Ex::Literal(lit) => Ok(vec![literal_const(*lit)?]),
            Ex::Constant(constant) => {
                let init = self.module.constants[*constant].init;
                self.eval(&self.module.global_expressions, init)
            }
            Ex::Override(handle) => {
                let value = self.overrides.get(handle).ok_or_else(|| {
                    CompileError::Internal("override evaluated before resolution".to_string())
                })?;
                Ok(vec![*value])
            }
            Ex::ZeroValue(ty) => {
                let comps =
                    value_components(self.module, &self.module.types[*ty].inner).or_else(|_| {
                        Pointee::of_inner(self.module, &self.module.types[*ty].inner).and_then(
                            |pointee| match pointee {
                                Pointee::Array {
                                    elem,
                                    elem_len,
                                    count,
                                    ..
                                } => Ok(vec![elem; elem_len * count as usize]),
                                _ => Err(CompileError::Unsupported(
                                    "unsupported zero-value type".to_string(),
                                )),
                            },
                        )
                    })?;
                Ok(comps.iter().map(|ty| zero_of(*ty)).collect())
            }
            Ex::Compose { components, .. } => {
                let mut out = Vec::new();
                for component in components {
                    out.extend(self.eval(arena, *component)?);
                }
                Ok(out)
            }
            Ex::Splat { size, value } => {
                let value = self.eval_scalar(arena, *value)?;
                Ok(vec![value; *size as usize])
            }
            Ex::Unary { op, expr } => {
                let value = self.eval_scalar(arena, *expr)?;
                const_unary(*op, value)
            }
            Ex::Binary { op, left, right } => {
                let left = self.eval_scalar(arena, *left)?;
                let right = self.eval_scalar(arena, *right)?;
                const_binary(*op, left, right)
            }
            Ex::Math { fun, arg, arg1, .. } => {
                let x = self.eval_scalar(arena, *arg)?.as_f64()?;
                let y = arg1
                    .map(|a| self.eval_scalar(arena, a).and_then(ConstScalar::as_f64))
                    .transpose()?;
                const_math(*fun, x, y).map(|v| vec![ConstScalar::F64(v)])
            }
            Ex::As {
                expr,
                kind,
                convert,
            } => {
                let value = self.eval_scalar(arena, *expr)?;
                const_convert(value, *kind, *convert).map(|v| vec![v])
            }
            other => Err(CompileError::Unsupported(format!(
                "unsupported constant expression: {other:?}"
            ))),
        }
    }

    fn eval_scalar(
        &self,
        arena: &naga::Arena<naga::Expression>,
        handle: Handle<naga::Expression>,
    ) -> Result<ConstScalar, CompileError> {
        let values = self.eval(arena, handle)?;
        match values.as_slice() {
            [value] => Ok(*value),
            _ => Err(CompileError::Unsupported(
                "expected a scalar constant".to_string(),
            )),
        }
    }
}

fn zero_of(ty: ScalarTy) -> ConstScalar {
    match ty {
        ScalarTy::F64 => ConstScalar::F64(0.0),
        ScalarTy::F32 => ConstScalar::F32(0.0),
        ScalarTy::I32 => ConstScalar::I32(0),
        ScalarTy::U32 => ConstScalar::U32(0),
        ScalarTy::Bool => ConstScalar::Bool(false),
    }
}

fn const_unary(op: naga::UnaryOperator, v: ConstScalar) -> Result<Vec<ConstScalar>, CompileError> {
    use naga::UnaryOperator as Uo;
    let out = match (op, v) {
        (Uo::Negate, ConstScalar::F64(x)) => ConstScalar::F64(-x),
        (Uo::Negate, ConstScalar::F32(x)) => ConstScalar::F32(-x),
        (Uo::Negate, ConstScalar::I32(x)) => ConstScalar::I32(x.wrapping_neg()),
        (Uo::LogicalNot, ConstScalar::Bool(x)) => ConstScalar::Bool(!x),
        (Uo::BitwiseNot, ConstScalar::I32(x)) => ConstScalar::I32(!x),
        (Uo::BitwiseNot, ConstScalar::U32(x)) => ConstScalar::U32(!x),
        _ => {
            return Err(CompileError::Unsupported(format!(
                "unsupported constant unary {op:?} on {v:?}"
            )));
        }
    };
    Ok(vec![out])
}

fn const_binary(
    op: naga::BinaryOperator,
    l: ConstScalar,
    r: ConstScalar,
) -> Result<Vec<ConstScalar>, CompileError> {
    use naga::BinaryOperator as Bo;
    let out = match (l, r) {
        (ConstScalar::F64(a), ConstScalar::F64(b)) => match op {
            Bo::Add => ConstScalar::F64(a + b),
            Bo::Subtract => ConstScalar::F64(a - b),
            Bo::Multiply => ConstScalar::F64(a * b),
            Bo::Divide => ConstScalar::F64(a / b),
            Bo::Modulo => ConstScalar::F64(a - b * (a / b).trunc()),
            Bo::Less => ConstScalar::Bool(a < b),
            Bo::LessEqual => ConstScalar::Bool(a <= b),
            Bo::Greater => ConstScalar::Bool(a > b),
            Bo::GreaterEqual => ConstScalar::Bool(a >= b),
            Bo::Equal => ConstScalar::Bool(a == b),
            Bo::NotEqual => ConstScalar::Bool(a != b),
            _ => {
                return Err(CompileError::Unsupported(format!(
                    "unsupported constant f64 binary {op:?}"
                )));
            }
        },
        (ConstScalar::F32(a), ConstScalar::F32(b)) => {
            return const_binary(op, ConstScalar::F64(a as f64), ConstScalar::F64(b as f64)).map(
                |v| {
                    v.into_iter()
                        .map(|c| match c {
                            ConstScalar::F64(x) => ConstScalar::F32(x as f32),
                            other => other,
                        })
                        .collect()
                },
            );
        }
        (ConstScalar::I32(a), ConstScalar::I32(b)) => match op {
            Bo::Add => ConstScalar::I32(a.wrapping_add(b)),
            Bo::Subtract => ConstScalar::I32(a.wrapping_sub(b)),
            Bo::Multiply => ConstScalar::I32(a.wrapping_mul(b)),
            Bo::Divide => ConstScalar::I32(if b == 0 || (a == i32::MIN && b == -1) {
                a
            } else {
                a / b
            }),
            Bo::Modulo => ConstScalar::I32(if b == 0 || (a == i32::MIN && b == -1) {
                0
            } else {
                a % b
            }),
            Bo::Less => ConstScalar::Bool(a < b),
            Bo::LessEqual => ConstScalar::Bool(a <= b),
            Bo::Greater => ConstScalar::Bool(a > b),
            Bo::GreaterEqual => ConstScalar::Bool(a >= b),
            Bo::Equal => ConstScalar::Bool(a == b),
            Bo::NotEqual => ConstScalar::Bool(a != b),
            _ => {
                return Err(CompileError::Unsupported(format!(
                    "unsupported constant i32 binary {op:?}"
                )));
            }
        },
        (ConstScalar::U32(a), ConstScalar::U32(b)) => match op {
            Bo::Add => ConstScalar::U32(a.wrapping_add(b)),
            Bo::Subtract => ConstScalar::U32(a.wrapping_sub(b)),
            Bo::Multiply => ConstScalar::U32(a.wrapping_mul(b)),
            // WGSL semantics: x/0 == x, x%0 == 0.
            Bo::Divide => ConstScalar::U32(a.checked_div(b).unwrap_or(a)),
            Bo::Modulo => ConstScalar::U32(a.checked_rem(b).unwrap_or(0)),
            Bo::Less => ConstScalar::Bool(a < b),
            Bo::LessEqual => ConstScalar::Bool(a <= b),
            Bo::Greater => ConstScalar::Bool(a > b),
            Bo::GreaterEqual => ConstScalar::Bool(a >= b),
            Bo::Equal => ConstScalar::Bool(a == b),
            Bo::NotEqual => ConstScalar::Bool(a != b),
            _ => {
                return Err(CompileError::Unsupported(format!(
                    "unsupported constant u32 binary {op:?}"
                )));
            }
        },
        _ => {
            return Err(CompileError::Unsupported(format!(
                "unsupported constant binary {op:?} on {l:?} and {r:?}"
            )));
        }
    };
    Ok(vec![out])
}

fn const_math(fun: naga::MathFunction, x: f64, y: Option<f64>) -> Result<f64, CompileError> {
    use naga::MathFunction as Mf;
    let y = || y.unwrap_or(0.0);
    Ok(match fun {
        Mf::Abs => x.abs(),
        Mf::Min => x.min(y()),
        Mf::Max => x.max(y()),
        Mf::Sqrt => x.sqrt(),
        Mf::InverseSqrt => 1.0 / x.sqrt(),
        Mf::Floor => x.floor(),
        Mf::Ceil => x.ceil(),
        Mf::Trunc => x.trunc(),
        Mf::Round => x.round_ties_even(),
        Mf::Fract => x - x.floor(),
        Mf::Sin => x.sin(),
        Mf::Cos => x.cos(),
        Mf::Tan => x.tan(),
        Mf::Asin => x.asin(),
        Mf::Acos => x.acos(),
        Mf::Atan => x.atan(),
        Mf::Atan2 => x.atan2(y()),
        Mf::Exp => x.exp(),
        Mf::Exp2 => x.exp2(),
        Mf::Log => x.ln(),
        Mf::Log2 => x.log2(),
        Mf::Pow => x.powf(y()),
        Mf::Radians => x.to_radians(),
        Mf::Degrees => x.to_degrees(),
        Mf::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(CompileError::Unsupported(format!(
                "unsupported constant math function {other:?}"
            )));
        }
    })
}

fn const_convert(
    v: ConstScalar,
    kind: naga::ScalarKind,
    convert: Option<u8>,
) -> Result<ConstScalar, CompileError> {
    use naga::ScalarKind as Sk;
    if convert.is_none() {
        return Err(CompileError::Unsupported(
            "bitcast in constant expressions is not supported".to_string(),
        ));
    }
    let width = convert.unwrap_or(4);
    let as_f64 = match v {
        ConstScalar::F64(x) => x,
        ConstScalar::F32(x) => x as f64,
        ConstScalar::I32(x) => x as f64,
        ConstScalar::U32(x) => x as f64,
        ConstScalar::Bool(x) => x as u8 as f64,
    };
    Ok(match (kind, width) {
        (Sk::Float, 8) => ConstScalar::F64(as_f64),
        (Sk::Float, 4) => ConstScalar::F32(as_f64 as f32),
        (Sk::Sint, 4) => ConstScalar::I32(as_f64 as i32),
        (Sk::Uint, 4) => ConstScalar::U32(as_f64 as u32),
        (Sk::Bool, _) => ConstScalar::Bool(as_f64 != 0.0),
        _ => {
            return Err(CompileError::Unsupported(format!(
                "unsupported constant conversion to {kind:?} width {width}"
            )));
        }
    })
}

// ---------------------------------------------------------------------------
// Module-level lowering
// ---------------------------------------------------------------------------

/// Storage assigned to one naga local variable.
#[derive(Clone)]
enum VarStorage {
    /// Scalar/vector variable: one wasm local per component.
    Locals(Vec<LocalId>),
    /// Array variable at a static linear-memory address.
    Mem { addr: u32 },
}

struct ArrayPlan {
    addr: u32,
    baked: bool,
    /// Entry-time contents when not baked: Some(consts) for a const init,
    /// None for WGSL zero-initialization.
    init: Option<Option<Vec<ConstScalar>>>,
}

pub struct Lowerer<'a> {
    module: &'a naga::Module,
    info: &'a naga::valid::ModuleInfo,
    overrides: &'a HashMap<Handle<naga::Override>, ConstScalar>,
    wasm: walrus::Module,
    memory: walrus::MemoryId,
    kernels: HashMap<String, FunctionId>,
    func_ids: HashMap<Handle<naga::Function>, FunctionId>,
    /// Per function, per local-variable: the static array plan.
    array_plans: HashMap<(Handle<naga::Function>, Handle<naga::LocalVariable>), ArrayPlan>,
    /// Module-scope array constants baked as data segments, by handle.
    const_segments: HashMap<Handle<naga::Constant>, u32>,
    scratch_end: u32,
}

pub fn build(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    model: &ModelFunctions,
    overrides: &HashMap<Handle<naga::Override>, ConstScalar>,
) -> Result<Vec<u8>, CompileError> {
    let wasm = walrus::Module::from_buffer(TEMPLATE)
        .map_err(|e| CompileError::Internal(format!("template does not parse: {e}")))?;

    let mut lowerer = Lowerer::new(module, info, overrides, wasm)?;
    lowerer.plan_arrays()?;
    lowerer.lower_all_functions()?;
    lowerer.add_abi_exports(model)?;
    lowerer.finish()
}

impl<'a> Lowerer<'a> {
    fn new(
        module: &'a naga::Module,
        info: &'a naga::valid::ModuleInfo,
        overrides: &'a HashMap<Handle<naga::Override>, ConstScalar>,
        wasm: walrus::Module,
    ) -> Result<Self, CompileError> {
        let mut kernels = HashMap::new();
        let mut memory = None;
        let mut heap_base = None;
        for export in wasm.exports.iter() {
            match export.item {
                walrus::ExportItem::Function(id) if export.name.starts_with("wgsl_") => {
                    kernels.insert(export.name.clone(), id);
                }
                walrus::ExportItem::Memory(id) if export.name == "memory" => {
                    memory = Some(id);
                }
                walrus::ExportItem::Global(id) if export.name == "__heap_base" => {
                    if let walrus::GlobalKind::Local(ConstExpr::Value(Value::I32(v))) =
                        wasm.globals.get(id).kind
                    {
                        heap_base = Some(v as u32);
                    }
                }
                _ => {}
            }
        }
        let memory = memory
            .ok_or_else(|| CompileError::Internal("template exports no memory".to_string()))?;
        let heap_base = heap_base
            .ok_or_else(|| CompileError::Internal("template exports no __heap_base".to_string()))?;

        Ok(Lowerer {
            module,
            info,
            overrides,
            wasm,
            memory,
            kernels,
            func_ids: HashMap::new(),
            array_plans: HashMap::new(),
            const_segments: HashMap::new(),
            scratch_end: heap_base.next_multiple_of(16),
        })
    }

    fn const_cx(&self) -> ConstCx<'a> {
        ConstCx {
            module: self.module,
            overrides: self.overrides,
        }
    }

    /// Assign static addresses to every array-typed local variable, bake
    /// read-only const arrays as data segments, and record entry-time init
    /// plans for the rest. Module-scope array constants get segments too so
    /// value-position indexing (`holes[i]`) can load from them.
    fn plan_arrays(&mut self) -> Result<(), CompileError> {
        for (const_handle, constant) in self.module.constants.iter() {
            let inner = &self.module.types[constant.ty].inner;
            if !matches!(inner, naga::TypeInner::Array { .. }) {
                continue;
            }
            let pointee = Pointee::of_inner(self.module, inner)?;
            let Pointee::Array { stride, count, .. } = pointee else {
                unreachable!()
            };
            let consts = self
                .const_cx()
                .eval(&self.module.global_expressions, constant.init)?;
            let mut bytes = Vec::with_capacity((stride * count) as usize);
            write_array_bytes(&mut bytes, &consts, pointee)?;
            let addr = self.scratch_end;
            self.scratch_end = (self.scratch_end + stride * count).next_multiple_of(16);
            self.wasm.data.add(
                walrus::DataKind::Active {
                    memory: self.memory,
                    offset: ConstExpr::Value(Value::I32(addr as i32)),
                },
                bytes,
            );
            self.const_segments.insert(const_handle, addr);
        }

        for (fn_handle, function) in self.module.functions.iter() {
            let mutated = collect_mutated_vars(function);
            for (var_handle, var) in function.local_variables.iter() {
                let inner = &self.module.types[var.ty].inner;
                if !matches!(inner, naga::TypeInner::Array { .. }) {
                    continue;
                }
                let pointee = Pointee::of_inner(self.module, inner)?;
                let Pointee::Array { stride, count, .. } = pointee else {
                    unreachable!()
                };
                let size = stride * count;
                let addr = self.scratch_end;
                self.scratch_end = (self.scratch_end + size).next_multiple_of(16);

                let init_consts = var
                    .init
                    .map(|init| self.const_cx().eval(&function.expressions, init))
                    .transpose()?;
                let read_only = !mutated.contains(&var_handle);

                let plan = match init_consts {
                    Some(consts) if read_only => {
                        let mut bytes = Vec::with_capacity(size as usize);
                        write_array_bytes(&mut bytes, &consts, pointee)?;
                        self.wasm.data.add(
                            walrus::DataKind::Active {
                                memory: self.memory,
                                offset: ConstExpr::Value(Value::I32(addr as i32)),
                            },
                            bytes,
                        );
                        ArrayPlan {
                            addr,
                            baked: true,
                            init: None,
                        }
                    }
                    init_consts => ArrayPlan {
                        addr,
                        baked: false,
                        init: Some(init_consts),
                    },
                };
                self.array_plans.insert((fn_handle, var_handle), plan);
            }
        }
        Ok(())
    }

    /// Lower every function, callees before callers.
    fn lower_all_functions(&mut self) -> Result<(), CompileError> {
        for handle in call_order(self.module)? {
            self.lower_function(handle)?;
        }
        Ok(())
    }

    fn flat_signature(
        &self,
        function: &naga::Function,
    ) -> Result<(Vec<Vec<ScalarTy>>, Vec<ScalarTy>), CompileError> {
        let mut params = Vec::new();
        for argument in &function.arguments {
            params.push(value_components(
                self.module,
                &self.module.types[argument.ty].inner,
            )?);
        }
        let results = match &function.result {
            Some(result) => value_components(self.module, &self.module.types[result.ty].inner)?,
            None => Vec::new(),
        };
        Ok((params, results))
    }

    fn lower_function(&mut self, fn_handle: Handle<naga::Function>) -> Result<(), CompileError> {
        let function = &self.module.functions[fn_handle];
        let fn_info = &self.info[fn_handle];
        let (param_tys, result_tys) = self.flat_signature(function)?;

        let wasm_params: Vec<ValType> =
            param_tys.iter().flatten().map(|ty| ty.val_type()).collect();
        let wasm_results: Vec<ValType> = result_tys.iter().map(|ty| ty.val_type()).collect();

        let mut builder = FunctionBuilder::new(&mut self.wasm.types, &wasm_params, &wasm_results);
        if let Some(name) = &function.name {
            builder.name(format!("wgsl_script::{name}"));
        }

        // Parameter locals, grouped per naga argument.
        let mut param_locals: Vec<Vec<LocalId>> = Vec::new();
        let mut flat_param_locals = Vec::new();
        for tys in &param_tys {
            let group: Vec<LocalId> = tys
                .iter()
                .map(|ty| self.wasm.locals.add(ty.val_type()))
                .collect();
            flat_param_locals.extend(group.iter().copied());
            param_locals.push(group);
        }

        // Storage for local variables.
        let mut var_storage: HashMap<Handle<naga::LocalVariable>, VarStorage> = HashMap::new();
        let mut entry_inits: Vec<(Vec<LocalId>, Vec<ConstScalar>)> = Vec::new();
        let mut entry_mem_inits: Vec<(u32, Option<Vec<ConstScalar>>, u32)> = Vec::new();
        for (var_handle, var) in function.local_variables.iter() {
            let inner = &self.module.types[var.ty].inner;
            if let Some(plan) = self.array_plans.get(&(fn_handle, var_handle)) {
                var_storage.insert(var_handle, VarStorage::Mem { addr: plan.addr });
                if !plan.baked {
                    let pointee = Pointee::of_inner(self.module, inner)?;
                    let Pointee::Array { stride, count, .. } = pointee else {
                        unreachable!()
                    };
                    entry_mem_inits.push((plan.addr, plan.init.clone().flatten(), stride * count));
                }
                continue;
            }
            let comps = value_components(self.module, inner)?;
            let locals: Vec<LocalId> = comps
                .iter()
                .map(|ty| self.wasm.locals.add(ty.val_type()))
                .collect();
            if let Some(init) = var.init {
                let consts = self.const_cx().eval(&function.expressions, init)?;
                entry_inits.push((locals.clone(), consts));
            }
            // No-init locals rely on wasm zero-initialization per call.
            var_storage.insert(var_handle, VarStorage::Locals(locals));
        }

        let cx = FnCx {
            module: self.module,
            fn_info,
            function,
            overrides: self.overrides,
            kernels: &self.kernels,
            func_ids: &self.func_ids,
            const_segments: &self.const_segments,
            memory: self.memory,
            param_locals,
            var_storage,
            locals: RefCell::new(&mut self.wasm.locals),
            cache: RefCell::new(HashMap::new()),
            frames: RefCell::new(Vec::new()),
            error: RefCell::new(None),
        };

        let mut body = builder.func_body();
        // Entry-time initialization of `var`s with constant initializers,
        // and per-call reset of memory-backed arrays (memory persists
        // across sample calls; WGSL vars are fresh per invocation).
        for (addr, init, size) in &entry_mem_inits {
            match init {
                Some(consts) => {
                    let mut offset = 0u32;
                    for value in consts {
                        body.i32_const(*addr as i32);
                        value.emit(&mut body);
                        body.store(
                            self.memory,
                            store_kind(value.ty()),
                            MemArg {
                                align: value.ty().byte_width().trailing_zeros(),
                                offset: u64::from(offset),
                            },
                        );
                        offset += value.ty().byte_width();
                    }
                }
                None => {
                    body.i32_const(*addr as i32);
                    body.i32_const(0);
                    body.i32_const(*size as i32);
                    body.memory_fill(self.memory);
                }
            }
        }
        for (locals, consts) in &entry_inits {
            for (local, value) in locals.iter().zip(consts) {
                value.emit(&mut body);
                body.local_set(*local);
            }
        }

        cx.emit_block(&mut body, &function.body);
        if let Some(error) = cx.error.into_inner() {
            return Err(error);
        }
        if !wasm_results.is_empty() {
            // naga guarantees value-returning bodies return on all paths;
            // the trailing end is unreachable and must type-check as such.
            body.unreachable();
        }

        let id = builder.finish(flat_param_locals, &mut self.wasm.funcs);
        self.func_ids.insert(fn_handle, id);
        Ok(())
    }

    fn add_abi_exports(&mut self, model: &ModelFunctions) -> Result<(), CompileError> {
        let dims = model.dims;
        let scene = *self
            .func_ids
            .get(&model.scene)
            .ok_or_else(|| CompileError::Internal("scene was not lowered".to_string()))?;
        let bounds_min = self.func_ids[&model.bounds_min];
        let bounds_max = self.func_ids[&model.bounds_max];

        // get_dimensions() -> i32
        {
            let mut fb = FunctionBuilder::new(&mut self.wasm.types, &[], &[ValType::I32]);
            fb.func_body().i32_const(dims as i32);
            let id = fb.finish(vec![], &mut self.wasm.funcs);
            self.wasm.exports.add("get_dimensions", id);
        }

        // get_bounds(out_ptr: i32) — interleaved min/max per axis.
        {
            let mut fb = FunctionBuilder::new(&mut self.wasm.types, &[ValType::I32], &[]);
            let out_ptr = self.wasm.locals.add(ValType::I32);
            let min_locals: Vec<LocalId> = (0..dims)
                .map(|_| self.wasm.locals.add(ValType::F64))
                .collect();
            let max_locals: Vec<LocalId> = (0..dims)
                .map(|_| self.wasm.locals.add(ValType::F64))
                .collect();
            let mut b = fb.func_body();
            b.call(bounds_min);
            for local in min_locals.iter().rev() {
                b.local_set(*local);
            }
            b.call(bounds_max);
            for local in max_locals.iter().rev() {
                b.local_set(*local);
            }
            for axis in 0..dims {
                for (slot, local) in [(0, min_locals[axis]), (1, max_locals[axis])] {
                    b.local_get(out_ptr);
                    b.local_get(local);
                    b.store(
                        self.memory,
                        walrus::ir::StoreKind::F64,
                        MemArg {
                            align: 3,
                            offset: (axis * 16 + slot * 8) as u64,
                        },
                    );
                }
            }
            let id = fb.finish(vec![out_ptr], &mut self.wasm.funcs);
            self.wasm.exports.add("get_bounds", id);
        }

        // sample(pos_ptr: i32) -> f32 — canonical 1.0/0.0 occupancy.
        {
            let mut fb =
                FunctionBuilder::new(&mut self.wasm.types, &[ValType::I32], &[ValType::F32]);
            let pos_ptr = self.wasm.locals.add(ValType::I32);
            let mut b = fb.func_body();
            for axis in 0..dims {
                b.local_get(pos_ptr);
                b.load(
                    self.memory,
                    walrus::ir::LoadKind::F64,
                    MemArg {
                        align: 3,
                        offset: (axis * 8) as u64,
                    },
                );
            }
            b.call(scene);
            let cond = self.wasm.locals.add(ValType::I32);
            b.local_set(cond);
            b.f32_const(1.0);
            b.f32_const(0.0);
            b.local_get(cond);
            b.select(Some(ValType::F32));
            let id = fb.finish(vec![pos_ptr], &mut self.wasm.funcs);
            self.wasm.exports.add("sample", id);
        }

        Ok(())
    }

    fn finish(mut self) -> Result<Vec<u8>, CompileError> {
        // Grow memory to cover the static array frames.
        let needed_pages = (u64::from(self.scratch_end)).div_ceil(PAGE);
        let memory = self.wasm.memories.get_mut(self.memory);
        if memory.initial < needed_pages {
            memory.initial = needed_pages;
        }

        // Drop the template's internal exports; models expose only the ABI.
        let to_delete: Vec<_> = self
            .wasm
            .exports
            .iter()
            .filter(|export| {
                export.name.starts_with("wgsl_")
                    || export.name == "__heap_base"
                    || export.name == "__data_end"
            })
            .map(|export| export.id())
            .collect();
        for id in to_delete {
            self.wasm.exports.delete(id);
        }

        walrus::passes::gc::run(&mut self.wasm);
        Ok(self.wasm.emit_wasm())
    }
}

fn store_kind(ty: ScalarTy) -> walrus::ir::StoreKind {
    match ty {
        ScalarTy::F64 => walrus::ir::StoreKind::F64,
        ScalarTy::F32 => walrus::ir::StoreKind::F32,
        _ => walrus::ir::StoreKind::I32 { atomic: false },
    }
}

fn load_kind(ty: ScalarTy) -> walrus::ir::LoadKind {
    match ty {
        ScalarTy::F64 => walrus::ir::LoadKind::F64,
        ScalarTy::F32 => walrus::ir::LoadKind::F32,
        _ => walrus::ir::LoadKind::I32 { atomic: false },
    }
}

fn write_array_bytes(
    out: &mut Vec<u8>,
    consts: &[ConstScalar],
    pointee: Pointee,
) -> Result<(), CompileError> {
    let Pointee::Array {
        elem,
        elem_len,
        stride,
        count,
    } = pointee
    else {
        return Err(CompileError::Internal("expected array pointee".to_string()));
    };
    if consts.len() != elem_len * count as usize {
        return Err(CompileError::Internal(format!(
            "array init has {} components, expected {}",
            consts.len(),
            elem_len * count as usize
        )));
    }
    let elem_bytes = elem.byte_width() as usize * elem_len;
    let stride = stride as usize;
    for (index, chunk) in consts.chunks(elem_len).enumerate() {
        let start = index * stride;
        out.resize(start, 0);
        for value in chunk {
            value.write_le(out);
        }
        debug_assert!(out.len() - start <= elem_bytes.max(stride));
    }
    out.resize(stride * count as usize, 0);
    Ok(())
}

/// Which local variables does the function ever store through?
fn collect_mutated_vars(function: &naga::Function) -> BTreeSet<Handle<naga::LocalVariable>> {
    fn pointer_root(
        function: &naga::Function,
        mut handle: Handle<naga::Expression>,
    ) -> Option<Handle<naga::LocalVariable>> {
        loop {
            match &function.expressions[handle] {
                naga::Expression::LocalVariable(var) => return Some(*var),
                naga::Expression::Access { base, .. }
                | naga::Expression::AccessIndex { base, .. } => handle = *base,
                _ => return None,
            }
        }
    }
    fn walk(
        function: &naga::Function,
        block: &naga::Block,
        out: &mut BTreeSet<Handle<naga::LocalVariable>>,
    ) {
        use naga::Statement as St;
        for stmt in block.iter() {
            match stmt {
                St::Store { pointer, .. } => {
                    if let Some(var) = pointer_root(function, *pointer) {
                        out.insert(var);
                    }
                }
                St::Block(inner) => walk(function, inner, out),
                St::If { accept, reject, .. } => {
                    walk(function, accept, out);
                    walk(function, reject, out);
                }
                St::Switch { cases, .. } => {
                    for case in cases {
                        walk(function, &case.body, out);
                    }
                }
                St::Loop {
                    body, continuing, ..
                } => {
                    walk(function, body, out);
                    walk(function, continuing, out);
                }
                _ => {}
            }
        }
    }
    let mut out = BTreeSet::new();
    walk(function, &function.body, &mut out);
    out
}

/// Topological order over the call graph (callees first). WGSL rejects
/// recursion at parse, so a cycle here is an internal error.
fn call_order(module: &naga::Module) -> Result<Vec<Handle<naga::Function>>, CompileError> {
    fn callees(function: &naga::Function, out: &mut BTreeSet<Handle<naga::Function>>) {
        fn walk(block: &naga::Block, out: &mut BTreeSet<Handle<naga::Function>>) {
            use naga::Statement as St;
            for stmt in block.iter() {
                match stmt {
                    St::Call { function, .. } => {
                        out.insert(*function);
                    }
                    St::Block(inner) => walk(inner, out),
                    St::If { accept, reject, .. } => {
                        walk(accept, out);
                        walk(reject, out);
                    }
                    St::Switch { cases, .. } => {
                        for case in cases {
                            walk(&case.body, out);
                        }
                    }
                    St::Loop {
                        body, continuing, ..
                    } => {
                        walk(body, out);
                        walk(continuing, out);
                    }
                    _ => {}
                }
            }
        }
        walk(&function.body, out);
    }

    let mut order = Vec::new();
    let mut state: HashMap<Handle<naga::Function>, u8> = HashMap::new();
    fn visit(
        module: &naga::Module,
        handle: Handle<naga::Function>,
        state: &mut HashMap<Handle<naga::Function>, u8>,
        order: &mut Vec<Handle<naga::Function>>,
    ) -> Result<(), CompileError> {
        match state.get(&handle) {
            Some(2) => return Ok(()),
            Some(1) => {
                return Err(CompileError::Internal(
                    "recursive call graph survived validation".to_string(),
                ));
            }
            _ => {}
        }
        state.insert(handle, 1);
        let mut called = BTreeSet::new();
        callees(&module.functions[handle], &mut called);
        for callee in called {
            visit(module, callee, state, order)?;
        }
        state.insert(handle, 2);
        order.push(handle);
        Ok(())
    }
    for (handle, _) in module.functions.iter() {
        visit(module, handle, &mut state, &mut order)?;
    }
    Ok(order)
}

// ---------------------------------------------------------------------------
// Per-function lowering
// ---------------------------------------------------------------------------

/// Break/continue targets. `Break` branches to the innermost frame's target;
/// `Continue` to the innermost `Loop` frame's continue target.
enum Frame {
    Loop {
        break_to: walrus::ir::InstrSeqId,
        continue_to: walrus::ir::InstrSeqId,
    },
    Switch {
        break_to: walrus::ir::InstrSeqId,
    },
}

/// A resolved pointer destination.
enum Place {
    Locals(Vec<LocalId>),
    Mem {
        base: u32,
        dynamic: Option<LocalId>,
        pointee: Pointee,
    },
    /// Dynamically-indexed vector held in locals.
    VecDyn {
        locals: Vec<LocalId>,
        index: LocalId,
        comp: ScalarTy,
    },
}

struct FnCx<'a, 'm> {
    module: &'a naga::Module,
    fn_info: &'a naga::valid::FunctionInfo,
    function: &'a naga::Function,
    overrides: &'a HashMap<Handle<naga::Override>, ConstScalar>,
    kernels: &'a HashMap<String, FunctionId>,
    func_ids: &'a HashMap<Handle<naga::Function>, FunctionId>,
    const_segments: &'a HashMap<Handle<naga::Constant>, u32>,
    memory: walrus::MemoryId,
    param_locals: Vec<Vec<LocalId>>,
    var_storage: HashMap<Handle<naga::LocalVariable>, VarStorage>,
    locals: RefCell<&'m mut walrus::ModuleLocals>,
    cache: RefCell<HashMap<Handle<naga::Expression>, Vec<LocalId>>>,
    frames: RefCell<Vec<Frame>>,
    error: RefCell<Option<CompileError>>,
}

type EResult<T> = Result<T, CompileError>;

impl FnCx<'_, '_> {
    fn fail(&self, error: CompileError) {
        let mut slot = self.error.borrow_mut();
        if slot.is_none() {
            *slot = Some(error);
        }
    }

    fn failed(&self) -> bool {
        self.error.borrow().is_some()
    }

    fn add_local(&self, ty: ValType) -> LocalId {
        self.locals.borrow_mut().add(ty)
    }

    fn const_cx(&self) -> ConstCx<'_> {
        ConstCx {
            module: self.module,
            overrides: self.overrides,
        }
    }

    /// The (value) components of an expression's type.
    fn expr_components(&self, handle: Handle<naga::Expression>) -> EResult<Vec<ScalarTy>> {
        let inner = self.fn_info[handle].ty.inner_with(&self.module.types);
        value_components(self.module, inner)
    }

    fn pointee_of(&self, handle: Handle<naga::Expression>) -> EResult<Pointee> {
        use naga::TypeInner as Ti;
        let inner = self.fn_info[handle].ty.inner_with(&self.module.types);
        match inner {
            Ti::Pointer { base, .. } => {
                Pointee::of_inner(self.module, &self.module.types[*base].inner)
            }
            Ti::ValuePointer { size, scalar, .. } => match size {
                Some(size) => Ok(Pointee::Vector(
                    ScalarTy::classify(*scalar)?,
                    *size as usize,
                )),
                None => Ok(Pointee::Scalar(ScalarTy::classify(*scalar)?)),
            },
            other => Err(CompileError::Internal(format!(
                "expected pointer type, found {other:?}"
            ))),
        }
    }

    // -- expression emission ------------------------------------------------

    /// Push the expression's flattened components onto the wasm stack.
    fn emit_expr(&self, b: &mut InstrSeqBuilder, handle: Handle<naga::Expression>) -> EResult<()> {
        use naga::Expression as Ex;
        if let Some(locals) = self.cache.borrow().get(&handle) {
            for local in locals {
                b.local_get(*local);
            }
            return Ok(());
        }
        match &self.function.expressions[handle] {
            Ex::Literal(lit) => {
                literal_const(*lit)?.emit(b);
                Ok(())
            }
            Ex::Constant(_) | Ex::ZeroValue(_) | Ex::Override(_) => {
                for value in self.const_cx().eval(&self.function.expressions, handle)? {
                    value.emit(b);
                }
                Ok(())
            }
            Ex::FunctionArgument(index) => {
                for local in &self.param_locals[*index as usize] {
                    b.local_get(*local);
                }
                Ok(())
            }
            Ex::Load { pointer } => {
                let place = self.resolve_place(b, *pointer)?;
                self.emit_load(b, place)
            }
            Ex::Compose { components, .. } => {
                for component in components {
                    self.emit_expr(b, *component)?;
                }
                Ok(())
            }
            Ex::Splat { size, value } => {
                self.emit_expr(b, *value)?;
                let ty = self.expr_components(*value)?[0];
                let tmp = self.add_local(ty.val_type());
                b.local_tee(tmp);
                for _ in 1..*size as usize {
                    b.local_get(tmp);
                }
                Ok(())
            }
            Ex::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let locals = self.operand_locals(b, *vector)?;
                for component in &pattern[..*size as usize] {
                    b.local_get(locals[*component as usize]);
                }
                Ok(())
            }
            Ex::AccessIndex { base, index } => {
                if let Some(pointee) = self.array_value_pointee(*base)? {
                    // Constant array indexed with a constant: fold directly.
                    let Pointee::Array {
                        elem_len, count, ..
                    } = pointee
                    else {
                        unreachable!()
                    };
                    if *index >= count {
                        return Err(CompileError::Type(format!(
                            "constant array index {index} out of bounds (length {count})"
                        )));
                    }
                    let consts = self.const_cx().eval(&self.function.expressions, *base)?;
                    let start = *index as usize * elem_len;
                    for value in &consts[start..start + elem_len] {
                        value.emit(b);
                    }
                    return Ok(());
                }
                // Value-position indexing of a vector.
                let locals = self.operand_locals(b, *base)?;
                let index = *index as usize;
                if index >= locals.len() {
                    return Err(CompileError::Internal(
                        "component index out of bounds".to_string(),
                    ));
                }
                b.local_get(locals[index]);
                Ok(())
            }
            Ex::Access { base, index } => {
                if let Some(pointee) = self.array_value_pointee(*base)? {
                    return self.emit_const_array_index(b, *base, *index, pointee);
                }
                // Value-position dynamic indexing of a vector.
                let comps = self.expr_components(*base)?;
                let locals = self.operand_locals(b, *base)?;
                let index_local = self.emit_clamped_index(b, *index, comps.len() as u32)?;
                self.emit_vec_dyn_load(b, &locals, index_local, comps[0]);
                Ok(())
            }
            Ex::Unary { op, expr } => self.emit_unary(b, *op, *expr),
            Ex::Binary { op, left, right } => self.emit_binary(b, *op, *left, *right),
            Ex::Select {
                condition,
                accept,
                reject,
            } => self.emit_select(b, *condition, *accept, *reject),
            Ex::Relational { fun, argument } => self.emit_relational(b, *fun, *argument),
            Ex::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => self.emit_math(b, *fun, *arg, *arg1, *arg2, *arg3),
            Ex::As {
                expr,
                kind,
                convert,
            } => self.emit_as(b, *expr, *kind, *convert),
            Ex::CallResult(_) => Err(CompileError::Internal(
                "call result read before the call executed".to_string(),
            )),
            Ex::LocalVariable(_) | Ex::GlobalVariable(_) => Err(CompileError::Internal(
                "pointer expression in value position".to_string(),
            )),
            other => Err(CompileError::Unsupported(format!(
                "expression {other:?} is not part of the model dialect"
            ))),
        }
    }

    /// If the expression is an array-typed *value*, classify it. Only
    /// const-evaluable arrays (module constants, zero values) are legal in
    /// value position; anything else was spilled to a `var` by the frontend.
    fn array_value_pointee(&self, handle: Handle<naga::Expression>) -> EResult<Option<Pointee>> {
        let inner = self.fn_info[handle].ty.inner_with(&self.module.types);
        if !matches!(inner, naga::TypeInner::Array { .. }) {
            return Ok(None);
        }
        use naga::Expression as Ex;
        match &self.function.expressions[handle] {
            Ex::Constant(_) | Ex::ZeroValue(_) => Ok(Some(Pointee::of_inner(self.module, inner)?)),
            other => Err(CompileError::Unsupported(format!(
                "array values must live in `var`s or `const`s (found {other:?})"
            ))),
        }
    }

    /// Dynamic indexing into a const-evaluable array value: zero arrays fold
    /// to zeros; module constants load from their baked data segment.
    fn emit_const_array_index(
        &self,
        b: &mut InstrSeqBuilder,
        base: Handle<naga::Expression>,
        index: Handle<naga::Expression>,
        pointee: Pointee,
    ) -> EResult<()> {
        use naga::Expression as Ex;
        let Pointee::Array {
            elem,
            elem_len,
            stride,
            count,
        } = pointee
        else {
            unreachable!()
        };
        match &self.function.expressions[base] {
            Ex::ZeroValue(_) => {
                for _ in 0..elem_len {
                    zero_of(elem).emit(b);
                }
                Ok(())
            }
            Ex::Constant(constant) => {
                let addr = self.const_segments.get(constant).copied().ok_or_else(|| {
                    CompileError::Internal("array constant has no data segment".to_string())
                })?;
                let index_local = self.emit_clamped_index(b, index, count)?;
                let offset_local = self.add_local(ValType::I32);
                b.local_get(index_local);
                b.i32_const(stride as i32);
                b.binop(BinaryOp::I32Mul);
                b.local_set(offset_local);
                for i in 0..elem_len {
                    b.local_get(offset_local);
                    b.load(
                        self.memory,
                        load_kind(elem),
                        MemArg {
                            align: elem.byte_width().trailing_zeros(),
                            offset: u64::from(addr + i as u32 * elem.byte_width()),
                        },
                    );
                }
                Ok(())
            }
            other => Err(CompileError::Unsupported(format!(
                "array values must live in `var`s or `const`s (found {other:?})"
            ))),
        }
    }

    /// Emit the expression's components into (possibly cached) locals.
    fn operand_locals(
        &self,
        b: &mut InstrSeqBuilder,
        handle: Handle<naga::Expression>,
    ) -> EResult<Vec<LocalId>> {
        if let Some(locals) = self.cache.borrow().get(&handle) {
            return Ok(locals.clone());
        }
        let comps = self.expr_components(handle)?;
        self.emit_expr(b, handle)?;
        let locals: Vec<LocalId> = comps
            .iter()
            .map(|ty| self.add_local(ty.val_type()))
            .collect();
        for local in locals.iter().rev() {
            b.local_set(*local);
        }
        Ok(locals)
    }

    /// Evaluate the expression and cache its components (Emit statements).
    fn cache_expr(&self, b: &mut InstrSeqBuilder, handle: Handle<naga::Expression>) -> EResult<()> {
        use naga::Expression as Ex;
        // Pointer-typed and leaf expressions don't need caching; loads and
        // computations do. Resolve places lazily at their use sites.
        match &self.function.expressions[handle] {
            Ex::LocalVariable(_)
            | Ex::GlobalVariable(_)
            | Ex::FunctionArgument(_)
            | Ex::Literal(_)
            | Ex::Constant(_)
            | Ex::Override(_)
            | Ex::ZeroValue(_)
            | Ex::CallResult(_) => return Ok(()),
            Ex::Access { base, .. } | Ex::AccessIndex { base, .. } => {
                // Pointer-position access chains are resolved at Load/Store;
                // only value-position accesses are worth caching.
                let base_inner = self.fn_info[*base].ty.inner_with(&self.module.types);
                if matches!(
                    base_inner,
                    naga::TypeInner::Pointer { .. } | naga::TypeInner::ValuePointer { .. }
                ) {
                    return Ok(());
                }
            }
            _ => {}
        }
        // Array-typed values (const array literals and the like) are only
        // meaningful at their use sites (indexing, var init, store).
        let inner = self.fn_info[handle].ty.inner_with(&self.module.types);
        if matches!(
            inner,
            naga::TypeInner::Array { .. }
                | naga::TypeInner::Pointer { .. }
                | naga::TypeInner::ValuePointer { .. }
        ) {
            return Ok(());
        }
        let comps = self.expr_components(handle)?;
        self.emit_expr(b, handle)?;
        let locals: Vec<LocalId> = comps
            .iter()
            .map(|ty| self.add_local(ty.val_type()))
            .collect();
        for local in locals.iter().rev() {
            b.local_set(*local);
        }
        self.cache.borrow_mut().insert(handle, locals);
        Ok(())
    }

    // -- places -------------------------------------------------------------

    fn resolve_place(
        &self,
        b: &mut InstrSeqBuilder,
        handle: Handle<naga::Expression>,
    ) -> EResult<Place> {
        use naga::Expression as Ex;
        match &self.function.expressions[handle] {
            Ex::LocalVariable(var) => match self.var_storage.get(var) {
                Some(VarStorage::Locals(locals)) => Ok(Place::Locals(locals.clone())),
                Some(VarStorage::Mem { addr }) => {
                    let pointee = self.pointee_of(handle)?;
                    Ok(Place::Mem {
                        base: *addr,
                        dynamic: None,
                        pointee,
                    })
                }
                None => Err(CompileError::Internal(
                    "local variable without storage".to_string(),
                )),
            },
            Ex::AccessIndex { base, index } => {
                let place = self.resolve_place(b, *base)?;
                self.narrow_place_const(place, *index)
            }
            Ex::Access { base, index } => {
                let place = self.resolve_place(b, *base)?;
                self.narrow_place_dyn(b, place, *index)
            }
            other => Err(CompileError::Unsupported(format!(
                "unsupported pointer expression: {other:?}"
            ))),
        }
    }

    fn narrow_place_const(&self, place: Place, index: u32) -> EResult<Place> {
        match place {
            Place::Locals(locals) => {
                let index = index as usize;
                if index >= locals.len() {
                    return Err(CompileError::Internal(
                        "constant index out of bounds".to_string(),
                    ));
                }
                Ok(Place::Locals(vec![locals[index]]))
            }
            Place::Mem {
                base,
                dynamic,
                pointee,
            } => match pointee {
                Pointee::Array {
                    elem,
                    elem_len,
                    stride,
                    count,
                } => {
                    if index >= count {
                        return Err(CompileError::Type(format!(
                            "constant array index {index} out of bounds (length {count})"
                        )));
                    }
                    let pointee = if elem_len == 1 {
                        Pointee::Scalar(elem)
                    } else {
                        Pointee::Vector(elem, elem_len)
                    };
                    Ok(Place::Mem {
                        base: base + index * stride,
                        dynamic,
                        pointee,
                    })
                }
                Pointee::Vector(comp, len) => {
                    if index as usize >= len {
                        return Err(CompileError::Internal(
                            "vector index out of bounds".to_string(),
                        ));
                    }
                    Ok(Place::Mem {
                        base: base + index * comp.byte_width(),
                        dynamic,
                        pointee: Pointee::Scalar(comp),
                    })
                }
                Pointee::Scalar(_) => {
                    Err(CompileError::Internal("indexing into a scalar".to_string()))
                }
            },
            Place::VecDyn { .. } => Err(CompileError::Unsupported(
                "indexing a dynamically-indexed component is not supported".to_string(),
            )),
        }
    }

    fn narrow_place_dyn(
        &self,
        b: &mut InstrSeqBuilder,
        place: Place,
        index: Handle<naga::Expression>,
    ) -> EResult<Place> {
        match place {
            Place::Locals(locals) => {
                // Dynamic index into a vector held in locals.
                let comp = {
                    let tys: Vec<ScalarTy> = locals
                        .iter()
                        .map(|_| ScalarTy::F64) // placeholder, refined below
                        .collect();
                    let _ = tys;
                    // The component type comes from the pointee of the base;
                    // locals all share it, so read from storage via index 0.
                    // (Locals places always come from scalar/vector vars.)
                    self.local_val_ty(locals[0])
                };
                let index_local = self.emit_clamped_index(b, index, locals.len() as u32)?;
                Ok(Place::VecDyn {
                    locals,
                    index: index_local,
                    comp,
                })
            }
            Place::Mem {
                base,
                dynamic,
                pointee,
            } => {
                let (stride, next) = match pointee {
                    Pointee::Array {
                        elem,
                        elem_len,
                        stride,
                        count,
                    } => {
                        let index_local = self.emit_clamped_index(b, index, count)?;
                        b.local_get(index_local);
                        b.i32_const(stride as i32);
                        b.binop(BinaryOp::I32Mul);
                        let next = if elem_len == 1 {
                            Pointee::Scalar(elem)
                        } else {
                            Pointee::Vector(elem, elem_len)
                        };
                        (index_local, next)
                    }
                    Pointee::Vector(comp, len) => {
                        let index_local = self.emit_clamped_index(b, index, len as u32)?;
                        b.local_get(index_local);
                        b.i32_const(comp.byte_width() as i32);
                        b.binop(BinaryOp::I32Mul);
                        (index_local, Pointee::Scalar(comp))
                    }
                    Pointee::Scalar(_) => {
                        return Err(CompileError::Internal("indexing into a scalar".to_string()));
                    }
                };
                let _ = stride;
                // Accumulate into a fresh dynamic-offset local.
                let offset_local = self.add_local(ValType::I32);
                if let Some(previous) = dynamic {
                    b.local_get(previous);
                    b.binop(BinaryOp::I32Add);
                }
                b.local_set(offset_local);
                Ok(Place::Mem {
                    base,
                    dynamic: Some(offset_local),
                    pointee: next,
                })
            }
            Place::VecDyn { .. } => Err(CompileError::Unsupported(
                "indexing a dynamically-indexed component is not supported".to_string(),
            )),
        }
    }

    fn local_val_ty(&self, local: LocalId) -> ScalarTy {
        match self.locals.borrow().get(local).ty() {
            ValType::F64 => ScalarTy::F64,
            ValType::F32 => ScalarTy::F32,
            _ => ScalarTy::I32,
        }
    }

    /// Emit `index` clamped to `[0, count)` into an i32 local (WGSL sanctions
    /// clamping for out-of-bounds accesses).
    fn emit_clamped_index(
        &self,
        b: &mut InstrSeqBuilder,
        index: Handle<naga::Expression>,
        count: u32,
    ) -> EResult<LocalId> {
        let signed = matches!(self.expr_components(index)?[0], ScalarTy::I32);
        self.emit_expr(b, index)?;
        let raw = self.add_local(ValType::I32);
        b.local_set(raw);
        let max = (count - 1) as i32;
        if signed {
            // max(i, 0)
            b.local_get(raw);
            b.i32_const(0);
            b.local_get(raw);
            b.i32_const(0);
            b.binop(BinaryOp::I32GtS);
            b.select(Some(ValType::I32));
            b.local_set(raw);
        }
        // min(i, max) — unsigned compare also handles huge u32 indices.
        b.local_get(raw);
        b.i32_const(max);
        b.local_get(raw);
        b.i32_const(max);
        b.binop(BinaryOp::I32LtU);
        b.select(Some(ValType::I32));
        let out = self.add_local(ValType::I32);
        b.local_set(out);
        Ok(out)
    }

    fn emit_load(&self, b: &mut InstrSeqBuilder, place: Place) -> EResult<()> {
        match place {
            Place::Locals(locals) => {
                for local in locals {
                    b.local_get(local);
                }
                Ok(())
            }
            Place::Mem {
                base,
                dynamic,
                pointee,
            } => {
                let comps = pointee.components()?;
                for (i, comp) in comps.iter().enumerate() {
                    match dynamic {
                        Some(local) => b.local_get(local),
                        None => b.i32_const(0),
                    };
                    b.load(
                        self.memory,
                        load_kind(*comp),
                        MemArg {
                            align: comp.byte_width().trailing_zeros(),
                            offset: u64::from(base + i as u32 * comp.byte_width()),
                        },
                    );
                }
                Ok(())
            }
            Place::VecDyn {
                locals,
                index,
                comp,
            } => {
                self.emit_vec_dyn_load_typed(b, &locals, index, comp);
                Ok(())
            }
        }
    }

    fn emit_vec_dyn_load(
        &self,
        b: &mut InstrSeqBuilder,
        locals: &[LocalId],
        index: LocalId,
        comp: ScalarTy,
    ) {
        self.emit_vec_dyn_load_typed(b, locals, index, comp)
    }

    fn emit_vec_dyn_load_typed(
        &self,
        b: &mut InstrSeqBuilder,
        locals: &[LocalId],
        index: LocalId,
        comp: ScalarTy,
    ) {
        let acc = self.add_local(comp.val_type());
        b.local_get(locals[locals.len() - 1]);
        b.local_set(acc);
        for i in (0..locals.len() - 1).rev() {
            b.local_get(locals[i]);
            b.local_get(acc);
            b.local_get(index);
            b.i32_const(i as i32);
            b.binop(BinaryOp::I32Eq);
            b.select(Some(comp.val_type()));
            b.local_set(acc);
        }
        b.local_get(acc);
    }

    fn emit_store(
        &self,
        b: &mut InstrSeqBuilder,
        place: Place,
        value: Handle<naga::Expression>,
    ) -> EResult<()> {
        // Whole-array assignment (`var a = array(...)`, `a = b`): elementwise.
        if let Place::Mem {
            base,
            dynamic,
            pointee: pointee @ Pointee::Array { .. },
        } = &place
        {
            if dynamic.is_some() {
                return Err(CompileError::Internal(
                    "dynamic whole-array store".to_string(),
                ));
            }
            return self.emit_whole_array_store(b, *base, *pointee, value);
        }
        let value_locals = self.operand_locals(b, value)?;
        match place {
            Place::Locals(locals) => {
                if locals.len() != value_locals.len() {
                    return Err(CompileError::Internal(
                        "store component mismatch".to_string(),
                    ));
                }
                for (dst, src) in locals.iter().zip(&value_locals) {
                    b.local_get(*src);
                    b.local_set(*dst);
                }
                Ok(())
            }
            Place::Mem {
                base,
                dynamic,
                pointee,
            } => {
                let comps = pointee.components()?;
                if comps.len() != value_locals.len() {
                    return Err(CompileError::Internal(
                        "store component mismatch".to_string(),
                    ));
                }
                for (i, (comp, src)) in comps.iter().zip(&value_locals).enumerate() {
                    match dynamic {
                        Some(local) => b.local_get(local),
                        None => b.i32_const(0),
                    };
                    b.local_get(*src);
                    b.store(
                        self.memory,
                        store_kind(*comp),
                        MemArg {
                            align: comp.byte_width().trailing_zeros(),
                            offset: u64::from(base + i as u32 * comp.byte_width()),
                        },
                    );
                }
                Ok(())
            }
            Place::VecDyn {
                locals,
                index,
                comp,
            } => {
                let src = value_locals[0];
                for (i, dst) in locals.iter().enumerate() {
                    b.local_get(src);
                    b.local_get(*dst);
                    b.local_get(index);
                    b.i32_const(i as i32);
                    b.binop(BinaryOp::I32Eq);
                    b.select(Some(comp.val_type()));
                    b.local_set(*dst);
                }
                Ok(())
            }
        }
    }

    fn emit_whole_array_store(
        &self,
        b: &mut InstrSeqBuilder,
        base: u32,
        pointee: Pointee,
        value: Handle<naga::Expression>,
    ) -> EResult<()> {
        use naga::Expression as Ex;
        let Pointee::Array {
            elem,
            elem_len,
            stride,
            count,
        } = pointee
        else {
            unreachable!()
        };
        let store_elem_consts = |b: &mut InstrSeqBuilder, consts: &[ConstScalar]| -> EResult<()> {
            if consts.len() != elem_len * count as usize {
                return Err(CompileError::Internal(
                    "array store component mismatch".to_string(),
                ));
            }
            for (i, chunk) in consts.chunks(elem_len).enumerate() {
                for (j, value) in chunk.iter().enumerate() {
                    b.i32_const(0);
                    value.emit(b);
                    b.store(
                        self.memory,
                        store_kind(elem),
                        MemArg {
                            align: elem.byte_width().trailing_zeros(),
                            offset: u64::from(
                                base + i as u32 * stride + j as u32 * elem.byte_width(),
                            ),
                        },
                    );
                }
            }
            Ok(())
        };
        match &self.function.expressions[value] {
            Ex::Constant(_) | Ex::ZeroValue(_) => {
                let consts = self.const_cx().eval(&self.function.expressions, value)?;
                store_elem_consts(b, &consts)
            }
            Ex::Compose { components, .. } => {
                if components.len() != count as usize {
                    return Err(CompileError::Internal(
                        "array compose arity mismatch".to_string(),
                    ));
                }
                for (i, component) in components.iter().enumerate() {
                    let locals = self.operand_locals(b, *component)?;
                    for (j, local) in locals.iter().enumerate() {
                        b.i32_const(0);
                        b.local_get(*local);
                        b.store(
                            self.memory,
                            store_kind(elem),
                            MemArg {
                                align: elem.byte_width().trailing_zeros(),
                                offset: u64::from(
                                    base + i as u32 * stride + j as u32 * elem.byte_width(),
                                ),
                            },
                        );
                    }
                }
                Ok(())
            }
            other => Err(CompileError::Unsupported(format!(
                "whole-array assignment from {other:?} is not supported; \
                 assign elements individually"
            ))),
        }
    }

    // -- operators ----------------------------------------------------------

    fn emit_unary(
        &self,
        b: &mut InstrSeqBuilder,
        op: naga::UnaryOperator,
        expr: Handle<naga::Expression>,
    ) -> EResult<()> {
        use naga::UnaryOperator as Uo;
        let comps = self.expr_components(expr)?;
        let locals = self.operand_locals(b, expr)?;
        for (comp, local) in comps.iter().zip(&locals) {
            b.local_get(*local);
            match (op, comp) {
                (Uo::Negate, ScalarTy::F64) => {
                    b.unop(UnaryOp::F64Neg);
                }
                (Uo::Negate, ScalarTy::F32) => {
                    b.unop(UnaryOp::F32Neg);
                }
                (Uo::Negate, ScalarTy::I32) => {
                    let tmp = self.add_local(ValType::I32);
                    b.local_set(tmp);
                    b.i32_const(0);
                    b.local_get(tmp);
                    b.binop(BinaryOp::I32Sub);
                }
                (Uo::LogicalNot, ScalarTy::Bool) => {
                    b.unop(UnaryOp::I32Eqz);
                }
                (Uo::BitwiseNot, ScalarTy::I32 | ScalarTy::U32) => {
                    b.i32_const(-1);
                    b.binop(BinaryOp::I32Xor);
                }
                _ => {
                    return Err(CompileError::Unsupported(format!(
                        "unary {op:?} on {comp:?} is not supported"
                    )));
                }
            }
        }
        Ok(())
    }

    fn emit_binary(
        &self,
        b: &mut InstrSeqBuilder,
        op: naga::BinaryOperator,
        left: Handle<naga::Expression>,
        right: Handle<naga::Expression>,
    ) -> EResult<()> {
        let left_comps = self.expr_components(left)?;
        let right_comps = self.expr_components(right)?;
        let left_locals = self.operand_locals(b, left)?;
        let right_locals = self.operand_locals(b, right)?;
        let n = left_comps.len().max(right_comps.len());
        for i in 0..n {
            let li = left_locals[i % left_locals.len()];
            let ri = right_locals[i % right_locals.len()];
            let lty = left_comps[i % left_comps.len()];
            self.emit_binary_scalar(b, op, lty, li, ri)?;
        }
        Ok(())
    }

    fn emit_binary_scalar(
        &self,
        b: &mut InstrSeqBuilder,
        op: naga::BinaryOperator,
        ty: ScalarTy,
        l: LocalId,
        r: LocalId,
    ) -> EResult<()> {
        use naga::BinaryOperator as Bo;
        let simple = |b: &mut InstrSeqBuilder, op: BinaryOp| {
            b.local_get(l);
            b.local_get(r);
            b.binop(op);
        };
        match (ty, op) {
            (ScalarTy::F64, Bo::Add) => simple(b, BinaryOp::F64Add),
            (ScalarTy::F64, Bo::Subtract) => simple(b, BinaryOp::F64Sub),
            (ScalarTy::F64, Bo::Multiply) => simple(b, BinaryOp::F64Mul),
            (ScalarTy::F64, Bo::Divide) => simple(b, BinaryOp::F64Div),
            (ScalarTy::F64, Bo::Modulo) => {
                // WGSL float remainder: e1 - e2 * trunc(e1 / e2)
                b.local_get(l);
                b.local_get(r);
                b.local_get(l);
                b.local_get(r);
                b.binop(BinaryOp::F64Div);
                b.unop(UnaryOp::F64Trunc);
                b.binop(BinaryOp::F64Mul);
                b.binop(BinaryOp::F64Sub);
            }
            (ScalarTy::F64, Bo::Less) => simple(b, BinaryOp::F64Lt),
            (ScalarTy::F64, Bo::LessEqual) => simple(b, BinaryOp::F64Le),
            (ScalarTy::F64, Bo::Greater) => simple(b, BinaryOp::F64Gt),
            (ScalarTy::F64, Bo::GreaterEqual) => simple(b, BinaryOp::F64Ge),
            (ScalarTy::F64, Bo::Equal) => simple(b, BinaryOp::F64Eq),
            (ScalarTy::F64, Bo::NotEqual) => simple(b, BinaryOp::F64Ne),

            (ScalarTy::F32, Bo::Add) => simple(b, BinaryOp::F32Add),
            (ScalarTy::F32, Bo::Subtract) => simple(b, BinaryOp::F32Sub),
            (ScalarTy::F32, Bo::Multiply) => simple(b, BinaryOp::F32Mul),
            (ScalarTy::F32, Bo::Divide) => simple(b, BinaryOp::F32Div),
            (ScalarTy::F32, Bo::Modulo) => {
                b.local_get(l);
                b.local_get(r);
                b.local_get(l);
                b.local_get(r);
                b.binop(BinaryOp::F32Div);
                b.unop(UnaryOp::F32Trunc);
                b.binop(BinaryOp::F32Mul);
                b.binop(BinaryOp::F32Sub);
            }
            (ScalarTy::F32, Bo::Less) => simple(b, BinaryOp::F32Lt),
            (ScalarTy::F32, Bo::LessEqual) => simple(b, BinaryOp::F32Le),
            (ScalarTy::F32, Bo::Greater) => simple(b, BinaryOp::F32Gt),
            (ScalarTy::F32, Bo::GreaterEqual) => simple(b, BinaryOp::F32Ge),
            (ScalarTy::F32, Bo::Equal) => simple(b, BinaryOp::F32Eq),
            (ScalarTy::F32, Bo::NotEqual) => simple(b, BinaryOp::F32Ne),

            (ScalarTy::I32 | ScalarTy::U32, Bo::Add) => simple(b, BinaryOp::I32Add),
            (ScalarTy::I32 | ScalarTy::U32, Bo::Subtract) => simple(b, BinaryOp::I32Sub),
            (ScalarTy::I32 | ScalarTy::U32, Bo::Multiply) => simple(b, BinaryOp::I32Mul),
            (ScalarTy::I32, Bo::Divide | Bo::Modulo) | (ScalarTy::U32, Bo::Divide | Bo::Modulo) => {
                self.emit_guarded_int_divmod(b, ty, op, l, r);
            }
            (ScalarTy::I32, Bo::Less) => simple(b, BinaryOp::I32LtS),
            (ScalarTy::I32, Bo::LessEqual) => simple(b, BinaryOp::I32LeS),
            (ScalarTy::I32, Bo::Greater) => simple(b, BinaryOp::I32GtS),
            (ScalarTy::I32, Bo::GreaterEqual) => simple(b, BinaryOp::I32GeS),
            (ScalarTy::U32, Bo::Less) => simple(b, BinaryOp::I32LtU),
            (ScalarTy::U32, Bo::LessEqual) => simple(b, BinaryOp::I32LeU),
            (ScalarTy::U32, Bo::Greater) => simple(b, BinaryOp::I32GtU),
            (ScalarTy::U32, Bo::GreaterEqual) => simple(b, BinaryOp::I32GeU),
            (ScalarTy::I32 | ScalarTy::U32 | ScalarTy::Bool, Bo::Equal) => {
                simple(b, BinaryOp::I32Eq)
            }
            (ScalarTy::I32 | ScalarTy::U32 | ScalarTy::Bool, Bo::NotEqual) => {
                simple(b, BinaryOp::I32Ne)
            }
            (ScalarTy::I32 | ScalarTy::U32 | ScalarTy::Bool, Bo::And) => {
                simple(b, BinaryOp::I32And)
            }
            (ScalarTy::I32 | ScalarTy::U32 | ScalarTy::Bool, Bo::InclusiveOr) => {
                simple(b, BinaryOp::I32Or)
            }
            (ScalarTy::I32 | ScalarTy::U32, Bo::ExclusiveOr) => simple(b, BinaryOp::I32Xor),
            (ScalarTy::Bool, Bo::LogicalAnd) => simple(b, BinaryOp::I32And),
            (ScalarTy::Bool, Bo::LogicalOr) => simple(b, BinaryOp::I32Or),
            (ScalarTy::I32 | ScalarTy::U32, Bo::ShiftLeft) => simple(b, BinaryOp::I32Shl),
            (ScalarTy::I32, Bo::ShiftRight) => simple(b, BinaryOp::I32ShrS),
            (ScalarTy::U32, Bo::ShiftRight) => simple(b, BinaryOp::I32ShrU),
            _ => {
                return Err(CompileError::Unsupported(format!(
                    "binary {op:?} on {ty:?} is not supported"
                )));
            }
        }
        Ok(())
    }

    /// WGSL-defined integer division: `x/0 == x`, `MIN/-1 == MIN`,
    /// `x%0 == 0`, `MIN%-1 == 0` — the divisor is overridden to 1 in the
    /// trapping cases (mirrors naga's own backend polyfills).
    fn emit_guarded_int_divmod(
        &self,
        b: &mut InstrSeqBuilder,
        ty: ScalarTy,
        op: naga::BinaryOperator,
        l: LocalId,
        r: LocalId,
    ) {
        let safe = self.add_local(ValType::I32);
        // select picks the FIRST operand when the condition is true: divisor
        // 1 in the trapping cases, the real divisor otherwise.
        b.i32_const(1);
        b.local_get(r);
        // trap condition: rhs == 0 | (signed && lhs == MIN && rhs == -1)
        b.local_get(r);
        b.unop(UnaryOp::I32Eqz);
        if ty == ScalarTy::I32 {
            b.local_get(l);
            b.i32_const(i32::MIN);
            b.binop(BinaryOp::I32Eq);
            b.local_get(r);
            b.i32_const(-1);
            b.binop(BinaryOp::I32Eq);
            b.binop(BinaryOp::I32And);
            b.binop(BinaryOp::I32Or);
        }
        b.select(Some(ValType::I32));
        b.local_set(safe);

        let div = if ty == ScalarTy::I32 {
            BinaryOp::I32DivS
        } else {
            BinaryOp::I32DivU
        };
        match op {
            naga::BinaryOperator::Divide => {
                b.local_get(l);
                b.local_get(safe);
                b.binop(div);
            }
            _ => {
                // l - (l / safe) * safe
                b.local_get(l);
                b.local_get(l);
                b.local_get(safe);
                b.binop(div);
                b.local_get(safe);
                b.binop(BinaryOp::I32Mul);
                b.binop(BinaryOp::I32Sub);
            }
        }
    }

    fn emit_select(
        &self,
        b: &mut InstrSeqBuilder,
        condition: Handle<naga::Expression>,
        accept: Handle<naga::Expression>,
        reject: Handle<naga::Expression>,
    ) -> EResult<()> {
        let comps = self.expr_components(accept)?;
        let cond_comps = self.expr_components(condition)?;
        let accept_locals = self.operand_locals(b, accept)?;
        let reject_locals = self.operand_locals(b, reject)?;
        let cond_locals = self.operand_locals(b, condition)?;
        for i in 0..comps.len() {
            b.local_get(accept_locals[i]);
            b.local_get(reject_locals[i]);
            b.local_get(cond_locals[i % cond_comps.len()]);
            b.select(Some(comps[i].val_type()));
        }
        Ok(())
    }

    fn emit_relational(
        &self,
        b: &mut InstrSeqBuilder,
        fun: naga::RelationalFunction,
        argument: Handle<naga::Expression>,
    ) -> EResult<()> {
        use naga::RelationalFunction as Rf;
        let comps = self.expr_components(argument)?;
        let locals = self.operand_locals(b, argument)?;
        match fun {
            Rf::All | Rf::Any => {
                b.local_get(locals[0]);
                for local in &locals[1..] {
                    b.local_get(*local);
                    b.binop(if matches!(fun, Rf::All) {
                        BinaryOp::I32And
                    } else {
                        BinaryOp::I32Or
                    });
                }
                Ok(())
            }
            Rf::IsNan => {
                for (comp, local) in comps.iter().zip(&locals) {
                    b.local_get(*local);
                    b.local_get(*local);
                    b.binop(match comp {
                        ScalarTy::F64 => BinaryOp::F64Ne,
                        _ => BinaryOp::F32Ne,
                    });
                }
                Ok(())
            }
            Rf::IsInf => {
                for (comp, local) in comps.iter().zip(&locals) {
                    b.local_get(*local);
                    match comp {
                        ScalarTy::F64 => {
                            b.unop(UnaryOp::F64Abs);
                            b.f64_const(f64::INFINITY);
                            b.binop(BinaryOp::F64Eq);
                        }
                        _ => {
                            b.unop(UnaryOp::F32Abs);
                            b.f32_const(f32::INFINITY);
                            b.binop(BinaryOp::F32Eq);
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn emit_as(
        &self,
        b: &mut InstrSeqBuilder,
        expr: Handle<naga::Expression>,
        kind: naga::ScalarKind,
        convert: Option<u8>,
    ) -> EResult<()> {
        use naga::ScalarKind as Sk;
        let comps = self.expr_components(expr)?;
        let locals = self.operand_locals(b, expr)?;
        for (src, local) in comps.iter().zip(&locals) {
            b.local_get(*local);
            let Some(width) = convert else {
                // Bitcast.
                match (src, kind) {
                    (ScalarTy::F32, Sk::Sint | Sk::Uint) => {
                        b.unop(UnaryOp::I32ReinterpretF32);
                    }
                    (ScalarTy::I32 | ScalarTy::U32, Sk::Float) => {
                        b.unop(UnaryOp::F32ReinterpretI32);
                    }
                    (ScalarTy::I32 | ScalarTy::U32, Sk::Sint | Sk::Uint) => {}
                    _ => {
                        return Err(CompileError::Unsupported(format!(
                            "bitcast from {src:?} to {kind:?} is not supported"
                        )));
                    }
                }
                continue;
            };
            match (src, kind, width) {
                // Float-to-float.
                (ScalarTy::F64, Sk::Float, 8) | (ScalarTy::F32, Sk::Float, 4) => {}
                (ScalarTy::F64, Sk::Float, 4) => {
                    b.unop(UnaryOp::F32DemoteF64);
                }
                (ScalarTy::F32, Sk::Float, 8) => {
                    b.unop(UnaryOp::F64PromoteF32);
                }
                // Float-to-int (WGSL saturating semantics).
                (ScalarTy::F64, Sk::Sint, 4) => {
                    b.unop(UnaryOp::I32TruncSSatF64);
                }
                (ScalarTy::F64, Sk::Uint, 4) => {
                    b.unop(UnaryOp::I32TruncUSatF64);
                }
                (ScalarTy::F32, Sk::Sint, 4) => {
                    b.unop(UnaryOp::I32TruncSSatF32);
                }
                (ScalarTy::F32, Sk::Uint, 4) => {
                    b.unop(UnaryOp::I32TruncUSatF32);
                }
                // Int-to-float.
                (ScalarTy::I32, Sk::Float, 8) => {
                    b.unop(UnaryOp::F64ConvertSI32);
                }
                (ScalarTy::U32, Sk::Float, 8) => {
                    b.unop(UnaryOp::F64ConvertUI32);
                }
                (ScalarTy::I32, Sk::Float, 4) => {
                    b.unop(UnaryOp::F32ConvertSI32);
                }
                (ScalarTy::U32, Sk::Float, 4) => {
                    b.unop(UnaryOp::F32ConvertUI32);
                }
                // Bool sources are already 0/1 i32s.
                (ScalarTy::Bool, Sk::Sint | Sk::Uint, 4) => {}
                (ScalarTy::Bool, Sk::Float, 8) => {
                    b.unop(UnaryOp::F64ConvertSI32);
                }
                (ScalarTy::Bool, Sk::Float, 4) => {
                    b.unop(UnaryOp::F32ConvertSI32);
                }
                // Int-to-int.
                (ScalarTy::I32 | ScalarTy::U32, Sk::Sint | Sk::Uint, 4) => {}
                // To bool: nonzero.
                (ScalarTy::F64, Sk::Bool, _) => {
                    b.f64_const(0.0);
                    b.binop(BinaryOp::F64Ne);
                }
                (ScalarTy::F32, Sk::Bool, _) => {
                    b.f32_const(0.0);
                    b.binop(BinaryOp::F32Ne);
                }
                (ScalarTy::I32 | ScalarTy::U32 | ScalarTy::Bool, Sk::Bool, _) => {
                    b.i32_const(0);
                    b.binop(BinaryOp::I32Ne);
                }
                _ => {
                    return Err(CompileError::Unsupported(format!(
                        "conversion from {src:?} to {kind:?} width {width} is not supported"
                    )));
                }
            }
        }
        Ok(())
    }

    fn kernel(&self, name: &str) -> EResult<FunctionId> {
        self.kernels
            .get(name)
            .copied()
            .ok_or_else(|| CompileError::Internal(format!("missing template kernel `{name}`")))
    }

    /// Push one scalar through a f64 kernel, promoting/demoting f32.
    fn emit_kernel_1(
        &self,
        b: &mut InstrSeqBuilder,
        name: &str,
        ty: ScalarTy,
        local: LocalId,
    ) -> EResult<()> {
        b.local_get(local);
        if ty == ScalarTy::F32 {
            b.unop(UnaryOp::F64PromoteF32);
        }
        b.call(self.kernel(name)?);
        if ty == ScalarTy::F32 {
            b.unop(UnaryOp::F32DemoteF64);
        }
        Ok(())
    }

    fn emit_math(
        &self,
        b: &mut InstrSeqBuilder,
        fun: naga::MathFunction,
        arg: Handle<naga::Expression>,
        arg1: Option<Handle<naga::Expression>>,
        arg2: Option<Handle<naga::Expression>>,
        arg3: Option<Handle<naga::Expression>>,
    ) -> EResult<()> {
        use naga::MathFunction as Mf;
        let comps = self.expr_components(arg)?;
        let ty = comps[0];
        let n = comps.len();
        let a = self.operand_locals(b, arg)?;

        // Simple componentwise float natives.
        let float_unop = |op32: UnaryOp, op64: UnaryOp| -> Option<UnaryOp> {
            Some(match ty {
                ScalarTy::F64 => op64,
                ScalarTy::F32 => op32,
                _ => return None,
            })
        };
        let float_binop = |op32: BinaryOp, op64: BinaryOp| -> Option<BinaryOp> {
            Some(match ty {
                ScalarTy::F64 => op64,
                ScalarTy::F32 => op32,
                _ => return None,
            })
        };

        // Kernel-backed scalar transcendentals.
        let kernel_1 = |fun: naga::MathFunction| -> Option<&'static str> {
            Some(match fun {
                Mf::Sin => "wgsl_sin",
                Mf::Cos => "wgsl_cos",
                Mf::Tan => "wgsl_tan",
                Mf::Asin => "wgsl_asin",
                Mf::Acos => "wgsl_acos",
                Mf::Atan => "wgsl_atan",
                Mf::Sinh => "wgsl_sinh",
                Mf::Cosh => "wgsl_cosh",
                Mf::Tanh => "wgsl_tanh",
                Mf::Asinh => "wgsl_asinh",
                Mf::Acosh => "wgsl_acosh",
                Mf::Atanh => "wgsl_atanh",
                Mf::Exp => "wgsl_exp",
                Mf::Exp2 => "wgsl_exp2",
                Mf::Log => "wgsl_log",
                Mf::Log2 => "wgsl_log2",
                _ => return None,
            })
        };

        match fun {
            // -- componentwise, native --------------------------------------
            Mf::Abs => match ty {
                ScalarTy::F64 | ScalarTy::F32 => {
                    let op = float_unop(UnaryOp::F32Abs, UnaryOp::F64Abs).unwrap();
                    for local in &a {
                        b.local_get(*local);
                        b.unop(op);
                    }
                    Ok(())
                }
                ScalarTy::I32 => {
                    for local in &a {
                        // (x ^ (x >> 31)) - (x >> 31)
                        let sign = self.add_local(ValType::I32);
                        b.local_get(*local);
                        b.i32_const(31);
                        b.binop(BinaryOp::I32ShrS);
                        b.local_set(sign);
                        b.local_get(*local);
                        b.local_get(sign);
                        b.binop(BinaryOp::I32Xor);
                        b.local_get(sign);
                        b.binop(BinaryOp::I32Sub);
                    }
                    Ok(())
                }
                ScalarTy::U32 => {
                    for local in &a {
                        b.local_get(*local);
                    }
                    Ok(())
                }
                _ => Err(CompileError::Unsupported("abs on bool".to_string())),
            },
            Mf::Floor | Mf::Ceil | Mf::Trunc | Mf::Round | Mf::Sqrt => {
                let op = match fun {
                    Mf::Floor => float_unop(UnaryOp::F32Floor, UnaryOp::F64Floor),
                    Mf::Ceil => float_unop(UnaryOp::F32Ceil, UnaryOp::F64Ceil),
                    Mf::Trunc => float_unop(UnaryOp::F32Trunc, UnaryOp::F64Trunc),
                    Mf::Round => float_unop(UnaryOp::F32Nearest, UnaryOp::F64Nearest),
                    Mf::Sqrt => float_unop(UnaryOp::F32Sqrt, UnaryOp::F64Sqrt),
                    _ => unreachable!(),
                }
                .ok_or_else(|| {
                    CompileError::Unsupported(format!("{fun:?} requires a float operand"))
                })?;
                for local in &a {
                    b.local_get(*local);
                    b.unop(op);
                }
                Ok(())
            }
            Mf::Fract => {
                let (sub, floor) = match ty {
                    ScalarTy::F64 => (BinaryOp::F64Sub, UnaryOp::F64Floor),
                    ScalarTy::F32 => (BinaryOp::F32Sub, UnaryOp::F32Floor),
                    _ => {
                        return Err(CompileError::Unsupported(
                            "fract requires a float operand".to_string(),
                        ));
                    }
                };
                for local in &a {
                    b.local_get(*local);
                    b.local_get(*local);
                    b.unop(floor);
                    b.binop(sub);
                }
                Ok(())
            }
            Mf::InverseSqrt => {
                for local in &a {
                    match ty {
                        ScalarTy::F64 => {
                            b.f64_const(1.0);
                            b.local_get(*local);
                            b.unop(UnaryOp::F64Sqrt);
                            b.binop(BinaryOp::F64Div);
                        }
                        _ => {
                            b.f32_const(1.0);
                            b.local_get(*local);
                            b.unop(UnaryOp::F32Sqrt);
                            b.binop(BinaryOp::F32Div);
                        }
                    }
                }
                Ok(())
            }
            Mf::Min | Mf::Max => {
                let rhs = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                match ty {
                    ScalarTy::F64 | ScalarTy::F32 => {
                        let op = if matches!(fun, Mf::Min) {
                            float_binop(BinaryOp::F32Min, BinaryOp::F64Min).unwrap()
                        } else {
                            float_binop(BinaryOp::F32Max, BinaryOp::F64Max).unwrap()
                        };
                        for i in 0..n {
                            b.local_get(a[i]);
                            b.local_get(rhs[i % rhs.len()]);
                            b.binop(op);
                        }
                    }
                    ScalarTy::I32 | ScalarTy::U32 => {
                        let cmp = match (ty, fun) {
                            (ScalarTy::I32, Mf::Min) => BinaryOp::I32LtS,
                            (ScalarTy::I32, _) => BinaryOp::I32GtS,
                            (_, Mf::Min) => BinaryOp::I32LtU,
                            (_, _) => BinaryOp::I32GtU,
                        };
                        for i in 0..n {
                            let r = rhs[i % rhs.len()];
                            b.local_get(a[i]);
                            b.local_get(r);
                            b.local_get(a[i]);
                            b.local_get(r);
                            b.binop(cmp);
                            b.select(Some(ValType::I32));
                        }
                    }
                    _ => {
                        return Err(CompileError::Unsupported("min/max on bool".to_string()));
                    }
                }
                Ok(())
            }
            Mf::Clamp => {
                let lo = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                let hi = self.operand_locals(b, arg2.ok_or_else(missing_arg)?)?;
                match ty {
                    ScalarTy::F64 | ScalarTy::F32 => {
                        let (min, max) = match ty {
                            ScalarTy::F64 => (BinaryOp::F64Min, BinaryOp::F64Max),
                            _ => (BinaryOp::F32Min, BinaryOp::F32Max),
                        };
                        for i in 0..n {
                            b.local_get(a[i]);
                            b.local_get(lo[i % lo.len()]);
                            b.binop(max);
                            b.local_get(hi[i % hi.len()]);
                            b.binop(min);
                        }
                        Ok(())
                    }
                    _ => Err(CompileError::Unsupported(
                        "integer clamp is not supported yet".to_string(),
                    )),
                }
            }
            Mf::Saturate => {
                for local in &a {
                    match ty {
                        ScalarTy::F64 => {
                            b.local_get(*local);
                            b.f64_const(0.0);
                            b.binop(BinaryOp::F64Max);
                            b.f64_const(1.0);
                            b.binop(BinaryOp::F64Min);
                        }
                        _ => {
                            b.local_get(*local);
                            b.f32_const(0.0);
                            b.binop(BinaryOp::F32Max);
                            b.f32_const(1.0);
                            b.binop(BinaryOp::F32Min);
                        }
                    }
                }
                Ok(())
            }
            Mf::Sign => {
                for local in &a {
                    match ty {
                        ScalarTy::F64 => {
                            // (x > 0) - (x < 0), as f64
                            b.local_get(*local);
                            b.f64_const(0.0);
                            b.binop(BinaryOp::F64Gt);
                            b.local_get(*local);
                            b.f64_const(0.0);
                            b.binop(BinaryOp::F64Lt);
                            b.binop(BinaryOp::I32Sub);
                            b.unop(UnaryOp::F64ConvertSI32);
                        }
                        ScalarTy::F32 => {
                            b.local_get(*local);
                            b.f32_const(0.0);
                            b.binop(BinaryOp::F32Gt);
                            b.local_get(*local);
                            b.f32_const(0.0);
                            b.binop(BinaryOp::F32Lt);
                            b.binop(BinaryOp::I32Sub);
                            b.unop(UnaryOp::F32ConvertSI32);
                        }
                        ScalarTy::I32 => {
                            b.local_get(*local);
                            b.i32_const(0);
                            b.binop(BinaryOp::I32GtS);
                            b.local_get(*local);
                            b.i32_const(0);
                            b.binop(BinaryOp::I32LtS);
                            b.binop(BinaryOp::I32Sub);
                        }
                        _ => {
                            return Err(CompileError::Unsupported("sign on this type".to_string()));
                        }
                    }
                }
                Ok(())
            }
            Mf::Fma => {
                let b1 = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                let c = self.operand_locals(b, arg2.ok_or_else(missing_arg)?)?;
                let (mul, add) = match ty {
                    ScalarTy::F64 => (BinaryOp::F64Mul, BinaryOp::F64Add),
                    _ => (BinaryOp::F32Mul, BinaryOp::F32Add),
                };
                for i in 0..n {
                    b.local_get(a[i]);
                    b.local_get(b1[i % b1.len()]);
                    b.binop(mul);
                    b.local_get(c[i % c.len()]);
                    b.binop(add);
                }
                Ok(())
            }
            Mf::Mix => {
                // x*(1-a) + y*a, per WGSL.
                let y = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                let t = self.operand_locals(b, arg2.ok_or_else(missing_arg)?)?;
                for i in 0..n {
                    let ti = t[i % t.len()];
                    match ty {
                        ScalarTy::F64 => {
                            b.local_get(a[i]);
                            b.f64_const(1.0);
                            b.local_get(ti);
                            b.binop(BinaryOp::F64Sub);
                            b.binop(BinaryOp::F64Mul);
                            b.local_get(y[i % y.len()]);
                            b.local_get(ti);
                            b.binop(BinaryOp::F64Mul);
                            b.binop(BinaryOp::F64Add);
                        }
                        _ => {
                            b.local_get(a[i]);
                            b.f32_const(1.0);
                            b.local_get(ti);
                            b.binop(BinaryOp::F32Sub);
                            b.binop(BinaryOp::F32Mul);
                            b.local_get(y[i % y.len()]);
                            b.local_get(ti);
                            b.binop(BinaryOp::F32Mul);
                            b.binop(BinaryOp::F32Add);
                        }
                    }
                }
                Ok(())
            }
            Mf::Step => {
                // step(edge, x): x >= edge ? 1 : 0. `arg` is edge.
                let x = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                for i in 0..n {
                    match ty {
                        ScalarTy::F64 => {
                            b.local_get(x[i % x.len()]);
                            b.local_get(a[i]);
                            b.binop(BinaryOp::F64Ge);
                            b.unop(UnaryOp::F64ConvertSI32);
                        }
                        _ => {
                            b.local_get(x[i % x.len()]);
                            b.local_get(a[i]);
                            b.binop(BinaryOp::F32Ge);
                            b.unop(UnaryOp::F32ConvertSI32);
                        }
                    }
                }
                Ok(())
            }
            Mf::SmoothStep => {
                // t = clamp((x - e0) / (e1 - e0), 0, 1); t * t * (3 - 2t)
                let e1 = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                let x = self.operand_locals(b, arg2.ok_or_else(missing_arg)?)?;
                if ty != ScalarTy::F64 && ty != ScalarTy::F32 {
                    return Err(CompileError::Unsupported(
                        "smoothstep requires floats".to_string(),
                    ));
                }
                for i in 0..n {
                    let t = self.add_local(ty.val_type());
                    let (sub, div, min, max, mul) = match ty {
                        ScalarTy::F64 => (
                            BinaryOp::F64Sub,
                            BinaryOp::F64Div,
                            BinaryOp::F64Min,
                            BinaryOp::F64Max,
                            BinaryOp::F64Mul,
                        ),
                        _ => (
                            BinaryOp::F32Sub,
                            BinaryOp::F32Div,
                            BinaryOp::F32Min,
                            BinaryOp::F32Max,
                            BinaryOp::F32Mul,
                        ),
                    };
                    let konst = |b: &mut InstrSeqBuilder, v: f64| match ty {
                        ScalarTy::F64 => {
                            b.f64_const(v);
                        }
                        _ => {
                            b.f32_const(v as f32);
                        }
                    };
                    b.local_get(x[i % x.len()]);
                    b.local_get(a[i]);
                    b.binop(sub);
                    b.local_get(e1[i % e1.len()]);
                    b.local_get(a[i]);
                    b.binop(sub);
                    b.binop(div);
                    konst(b, 0.0);
                    b.binop(max);
                    konst(b, 1.0);
                    b.binop(min);
                    b.local_set(t);
                    b.local_get(t);
                    b.local_get(t);
                    b.binop(mul);
                    konst(b, 3.0);
                    konst(b, 2.0);
                    b.local_get(t);
                    b.binop(mul);
                    b.binop(sub);
                    b.binop(mul);
                }
                Ok(())
            }
            Mf::Radians | Mf::Degrees => {
                let factor = if matches!(fun, Mf::Radians) {
                    std::f64::consts::PI / 180.0
                } else {
                    180.0 / std::f64::consts::PI
                };
                for local in &a {
                    match ty {
                        ScalarTy::F64 => {
                            b.local_get(*local);
                            b.f64_const(factor);
                            b.binop(BinaryOp::F64Mul);
                        }
                        _ => {
                            b.local_get(*local);
                            b.f32_const(factor as f32);
                            b.binop(BinaryOp::F32Mul);
                        }
                    }
                }
                Ok(())
            }
            // -- kernel-backed transcendentals ------------------------------
            Mf::Sin
            | Mf::Cos
            | Mf::Tan
            | Mf::Asin
            | Mf::Acos
            | Mf::Atan
            | Mf::Sinh
            | Mf::Cosh
            | Mf::Tanh
            | Mf::Asinh
            | Mf::Acosh
            | Mf::Atanh
            | Mf::Exp
            | Mf::Exp2
            | Mf::Log
            | Mf::Log2 => {
                let name = kernel_1(fun).unwrap();
                for local in &a {
                    self.emit_kernel_1(b, name, ty, *local)?;
                }
                Ok(())
            }
            Mf::Atan2 => {
                let x = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                for i in 0..n {
                    b.local_get(a[i]);
                    if ty == ScalarTy::F32 {
                        b.unop(UnaryOp::F64PromoteF32);
                    }
                    b.local_get(x[i % x.len()]);
                    if ty == ScalarTy::F32 {
                        b.unop(UnaryOp::F64PromoteF32);
                    }
                    b.call(self.kernel("wgsl_atan2")?);
                    if ty == ScalarTy::F32 {
                        b.unop(UnaryOp::F32DemoteF64);
                    }
                }
                Ok(())
            }
            Mf::Pow => {
                let y = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                for i in 0..n {
                    b.local_get(a[i]);
                    if ty == ScalarTy::F32 {
                        b.unop(UnaryOp::F64PromoteF32);
                    }
                    b.local_get(y[i % y.len()]);
                    if ty == ScalarTy::F32 {
                        b.unop(UnaryOp::F64PromoteF32);
                    }
                    b.call(self.kernel("wgsl_pow")?);
                    if ty == ScalarTy::F32 {
                        b.unop(UnaryOp::F32DemoteF64);
                    }
                }
                Ok(())
            }
            Mf::Ldexp => {
                // x * 2^e, e is an integer.
                let e = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                for i in 0..n {
                    b.local_get(a[i]);
                    if ty == ScalarTy::F32 {
                        b.unop(UnaryOp::F64PromoteF32);
                    }
                    b.local_get(e[i % e.len()]);
                    b.unop(UnaryOp::F64ConvertSI32);
                    b.call(self.kernel("wgsl_exp2")?);
                    b.binop(BinaryOp::F64Mul);
                    if ty == ScalarTy::F32 {
                        b.unop(UnaryOp::F32DemoteF64);
                    }
                }
                Ok(())
            }
            // -- vector reductions ------------------------------------------
            Mf::Dot => {
                let rhs = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                self.emit_dot(b, ty, &a, &rhs)
            }
            Mf::Length => {
                self.emit_dot(b, ty, &a, &a)?;
                b.unop(match ty {
                    ScalarTy::F64 => UnaryOp::F64Sqrt,
                    _ => UnaryOp::F32Sqrt,
                });
                Ok(())
            }
            Mf::Distance => {
                let rhs = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                let (sub,) = match ty {
                    ScalarTy::F64 => (BinaryOp::F64Sub,),
                    _ => (BinaryOp::F32Sub,),
                };
                let diff: Vec<LocalId> = (0..n).map(|_| self.add_local(ty.val_type())).collect();
                for i in 0..n {
                    b.local_get(a[i]);
                    b.local_get(rhs[i % rhs.len()]);
                    b.binop(sub);
                    b.local_set(diff[i]);
                }
                self.emit_dot(b, ty, &diff, &diff)?;
                b.unop(match ty {
                    ScalarTy::F64 => UnaryOp::F64Sqrt,
                    _ => UnaryOp::F32Sqrt,
                });
                Ok(())
            }
            Mf::Normalize => {
                let len = self.add_local(ty.val_type());
                self.emit_dot(b, ty, &a, &a)?;
                b.unop(match ty {
                    ScalarTy::F64 => UnaryOp::F64Sqrt,
                    _ => UnaryOp::F32Sqrt,
                });
                b.local_set(len);
                let div = match ty {
                    ScalarTy::F64 => BinaryOp::F64Div,
                    _ => BinaryOp::F32Div,
                };
                for local in &a {
                    b.local_get(*local);
                    b.local_get(len);
                    b.binop(div);
                }
                Ok(())
            }
            Mf::Cross => {
                let rhs = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                if n != 3 {
                    return Err(CompileError::Type("cross requires vec3".to_string()));
                }
                let (mul, sub) = match ty {
                    ScalarTy::F64 => (BinaryOp::F64Mul, BinaryOp::F64Sub),
                    _ => (BinaryOp::F32Mul, BinaryOp::F32Sub),
                };
                for (i, j) in [(1, 2), (2, 0), (0, 1)] {
                    b.local_get(a[i]);
                    b.local_get(rhs[j]);
                    b.binop(mul);
                    b.local_get(a[j]);
                    b.local_get(rhs[i]);
                    b.binop(mul);
                    b.binop(sub);
                }
                Ok(())
            }
            Mf::FaceForward => {
                // faceForward(N, I, Nref): dot(Nref, I) < 0 ? N : -N
                let i_arg = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                let nref = self.operand_locals(b, arg2.ok_or_else(missing_arg)?)?;
                let cond = self.add_local(ValType::I32);
                self.emit_dot(b, ty, &nref, &i_arg)?;
                match ty {
                    ScalarTy::F64 => {
                        b.f64_const(0.0);
                        b.binop(BinaryOp::F64Lt);
                    }
                    _ => {
                        b.f32_const(0.0);
                        b.binop(BinaryOp::F32Lt);
                    }
                }
                b.local_set(cond);
                let neg = match ty {
                    ScalarTy::F64 => UnaryOp::F64Neg,
                    _ => UnaryOp::F32Neg,
                };
                for local in &a {
                    b.local_get(*local);
                    b.local_get(*local);
                    b.unop(neg);
                    b.local_get(cond);
                    b.select(Some(ty.val_type()));
                }
                Ok(())
            }
            Mf::Reflect => {
                // reflect(I, N) = I - 2 * dot(N, I) * N
                let normal = self.operand_locals(b, arg1.ok_or_else(missing_arg)?)?;
                let two_dot = self.add_local(ty.val_type());
                self.emit_dot(b, ty, &normal, &a)?;
                match ty {
                    ScalarTy::F64 => {
                        b.f64_const(2.0);
                        b.binop(BinaryOp::F64Mul);
                    }
                    _ => {
                        b.f32_const(2.0);
                        b.binop(BinaryOp::F32Mul);
                    }
                }
                b.local_set(two_dot);
                let (mul, sub) = match ty {
                    ScalarTy::F64 => (BinaryOp::F64Mul, BinaryOp::F64Sub),
                    _ => (BinaryOp::F32Mul, BinaryOp::F32Sub),
                };
                for i in 0..n {
                    b.local_get(a[i]);
                    b.local_get(two_dot);
                    b.local_get(normal[i % normal.len()]);
                    b.binop(mul);
                    b.binop(sub);
                }
                Ok(())
            }
            // -- integer bit ops --------------------------------------------
            Mf::CountOneBits => {
                for local in &a {
                    b.local_get(*local);
                    b.unop(UnaryOp::I32Popcnt);
                }
                Ok(())
            }
            Mf::CountLeadingZeros => {
                for local in &a {
                    b.local_get(*local);
                    b.unop(UnaryOp::I32Clz);
                }
                Ok(())
            }
            Mf::CountTrailingZeros => {
                for local in &a {
                    b.local_get(*local);
                    b.unop(UnaryOp::I32Ctz);
                }
                Ok(())
            }
            other => {
                let _ = arg3;
                Err(CompileError::Unsupported(format!(
                    "math function {other:?} is not supported by the model dialect"
                )))
            }
        }
    }

    fn emit_dot(
        &self,
        b: &mut InstrSeqBuilder,
        ty: ScalarTy,
        lhs: &[LocalId],
        rhs: &[LocalId],
    ) -> EResult<()> {
        let (mul, add) = match ty {
            ScalarTy::F64 => (BinaryOp::F64Mul, BinaryOp::F64Add),
            ScalarTy::F32 => (BinaryOp::F32Mul, BinaryOp::F32Add),
            _ => {
                return Err(CompileError::Unsupported(
                    "integer dot products are not supported".to_string(),
                ));
            }
        };
        for i in 0..lhs.len() {
            b.local_get(lhs[i]);
            b.local_get(rhs[i % rhs.len()]);
            b.binop(mul);
            if i > 0 {
                b.binop(add);
            }
        }
        Ok(())
    }

    // -- statements ---------------------------------------------------------

    fn emit_block(&self, b: &mut InstrSeqBuilder, block: &naga::Block) {
        use naga::Statement as St;
        for stmt in block.iter() {
            if self.failed() {
                return;
            }
            match stmt {
                St::Emit(range) => {
                    for handle in range.clone() {
                        if let Err(error) = self.cache_expr(b, handle) {
                            self.fail(error);
                            return;
                        }
                    }
                }
                St::Block(inner) => {
                    self.emit_block(b, inner);
                }
                St::Store { pointer, value } => {
                    let result = self
                        .resolve_place(b, *pointer)
                        .and_then(|place| self.emit_store(b, place, *value));
                    if let Err(error) = result {
                        self.fail(error);
                        return;
                    }
                }
                St::If {
                    condition,
                    accept,
                    reject,
                } => {
                    if let Err(error) = self.emit_expr(b, *condition) {
                        self.fail(error);
                        return;
                    }
                    b.if_else(
                        None,
                        |then_b| self.emit_block(then_b, accept),
                        |else_b| self.emit_block(else_b, reject),
                    );
                }
                St::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    b.block(None, |break_b| {
                        let break_to = break_b.id();
                        break_b.loop_(None, |head_b| {
                            let head = head_b.id();
                            head_b.block(None, |cont_b| {
                                let continue_to = cont_b.id();
                                self.frames.borrow_mut().push(Frame::Loop {
                                    break_to,
                                    continue_to,
                                });
                                self.emit_block(cont_b, body);
                                self.frames.borrow_mut().pop();
                            });
                            self.emit_block(head_b, continuing);
                            if let Some(condition) = break_if {
                                if let Err(error) = self.emit_expr(head_b, *condition) {
                                    self.fail(error);
                                    return;
                                }
                                head_b.br_if(break_to);
                            }
                            head_b.br(head);
                        });
                    });
                }
                St::Break => {
                    let frames = self.frames.borrow();
                    match frames.last() {
                        Some(Frame::Loop { break_to, .. }) | Some(Frame::Switch { break_to }) => {
                            b.br(*break_to);
                        }
                        None => {
                            drop(frames);
                            self.fail(CompileError::Internal(
                                "break outside loop/switch".to_string(),
                            ));
                        }
                    }
                    return; // statements after a break are unreachable
                }
                St::Continue => {
                    let target = self.frames.borrow().iter().rev().find_map(|f| match f {
                        Frame::Loop { continue_to, .. } => Some(*continue_to),
                        Frame::Switch { .. } => None,
                    });
                    match target {
                        Some(target) => {
                            b.br(target);
                        }
                        None => {
                            self.fail(CompileError::Internal("continue outside loop".to_string()))
                        }
                    }
                    return;
                }
                St::Return { value } => {
                    if let Some(value) = value
                        && let Err(error) = self.emit_expr(b, *value)
                    {
                        self.fail(error);
                        return;
                    }
                    b.return_();
                    return;
                }
                St::Call {
                    function,
                    arguments,
                    result,
                } => {
                    for argument in arguments {
                        if let Err(error) = self.emit_expr(b, *argument) {
                            self.fail(error);
                            return;
                        }
                    }
                    let Some(&fid) = self.func_ids.get(function) else {
                        self.fail(CompileError::Internal(
                            "call to un-lowered function".to_string(),
                        ));
                        return;
                    };
                    b.call(fid);
                    if let Some(result) = result {
                        let comps = match self.expr_components(*result) {
                            Ok(comps) => comps,
                            Err(error) => {
                                self.fail(error);
                                return;
                            }
                        };
                        let locals: Vec<LocalId> = comps
                            .iter()
                            .map(|ty| self.add_local(ty.val_type()))
                            .collect();
                        for local in locals.iter().rev() {
                            b.local_set(*local);
                        }
                        self.cache.borrow_mut().insert(*result, locals);
                    }
                }
                St::Switch { selector, cases } => {
                    self.emit_switch(b, *selector, cases);
                }
                other => {
                    self.fail(CompileError::Unsupported(format!(
                        "statement {other:?} is not part of the model dialect"
                    )));
                    return;
                }
            }
        }
    }

    fn emit_switch(
        &self,
        b: &mut InstrSeqBuilder,
        selector: Handle<naga::Expression>,
        cases: &[naga::SwitchCase],
    ) {
        // Group multi-selector cases (empty fall-through bodies attach their
        // values to the next real body), then lower to an if-else ladder
        // inside one block so `break` has a target.
        let sel = match self.operand_locals(b, selector) {
            Ok(locals) => locals[0],
            Err(error) => {
                self.fail(error);
                return;
            }
        };
        struct Group<'c> {
            values: Vec<i32>,
            is_default: bool,
            body: &'c naga::Block,
        }
        let mut groups: Vec<Group> = Vec::new();
        let mut pending_values: Vec<i32> = Vec::new();
        let mut pending_default = false;
        for case in cases {
            match case.value {
                naga::SwitchValue::I32(v) => pending_values.push(v),
                naga::SwitchValue::U32(v) => pending_values.push(v as i32),
                naga::SwitchValue::Default => pending_default = true,
            }
            if case.fall_through {
                if !case.body.is_empty() {
                    self.fail(CompileError::Unsupported(
                        "switch fallthrough with a non-empty body is not supported".to_string(),
                    ));
                    return;
                }
                continue;
            }
            groups.push(Group {
                values: std::mem::take(&mut pending_values),
                is_default: std::mem::replace(&mut pending_default, false),
                body: &case.body,
            });
        }

        fn emit_groups(cx: &FnCx, b: &mut InstrSeqBuilder, sel: LocalId, groups: &[Group]) {
            let Some((group, rest)) = groups.split_first() else {
                return;
            };
            if group.is_default {
                // The default group absorbs any remaining ladder position.
                cx.emit_block(b, group.body);
                return;
            }
            for (i, value) in group.values.iter().enumerate() {
                b.local_get(sel);
                b.i32_const(*value);
                b.binop(BinaryOp::I32Eq);
                if i > 0 {
                    b.binop(BinaryOp::I32Or);
                }
            }
            b.if_else(
                None,
                |then_b| cx.emit_block(then_b, group.body),
                |else_b| emit_groups(cx, else_b, sel, rest),
            );
        }

        // Order: value groups first, default last (WGSL default may appear
        // anywhere but matches only when nothing else does).
        let mut ordered: Vec<&Group> = groups.iter().filter(|g| !g.is_default).collect();
        ordered.extend(groups.iter().filter(|g| g.is_default));
        let owned: Vec<Group> = ordered
            .into_iter()
            .map(|g| Group {
                values: g.values.clone(),
                is_default: g.is_default,
                body: g.body,
            })
            .collect();

        b.block(None, |switch_b| {
            let break_to = switch_b.id();
            self.frames.borrow_mut().push(Frame::Switch { break_to });
            emit_groups(self, switch_b, sel, &owned);
            self.frames.borrow_mut().pop();
        });
    }
}

fn missing_arg() -> CompileError {
    CompileError::Internal("math builtin missing an argument".to_string())
}

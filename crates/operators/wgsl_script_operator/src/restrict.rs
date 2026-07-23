//! The model-dialect restriction pass.
//!
//! naga's validator accepts every legal *shader* construct; this pass cuts
//! the language down to the model dialect: a library-style module of pure
//! functions over scalars, vectors, and fixed-size arrays, with `const` and
//! `override` as the only module-scope state. Everything it rejects is
//! either statically meaningless in a model (bindings, entry points), a
//! violation of the stateless-model contract (`var<private>`), or simply
//! not lowered yet (matrices, structs).

use crate::CompileError;
use naga::{Handle, Module};

/// The script's required functions, discovered by signature.
pub struct ModelFunctions {
    pub scene: Handle<naga::Function>,
    pub bounds_min: Handle<naga::Function>,
    pub bounds_max: Handle<naga::Function>,
    /// 2 for a sketch, 3 for a volume (from `scene`'s parameter).
    pub dims: usize,
}

pub fn check_module(module: &Module) -> Result<(), CompileError> {
    if let Some(entry) = module.entry_points.first() {
        return Err(CompileError::Unsupported(format!(
            "entry point `{}`: the model dialect is a library of plain functions; \
             remove @compute/@vertex/@fragment entry points",
            entry.name
        )));
    }

    if let Some((_, var)) = module.global_variables.iter().next() {
        let name = var.name.as_deref().unwrap_or("<unnamed>");
        let reason = match var.space {
            naga::AddressSpace::Private => {
                "models are stateless: `var<private>` state would persist between \
                 samples; use `const`, `override`, or function-local `var`"
            }
            naga::AddressSpace::Handle => "textures and samplers are not part of the model dialect",
            _ => "module-scope `var` is not part of the model dialect; use `const` or `override`",
        };
        return Err(CompileError::Unsupported(format!(
            "global variable `{name}`: {reason}"
        )));
    }

    for (_, ty) in module.types.iter() {
        check_type(module, &ty.inner, ty.name.as_deref())?;
    }

    for (_, function) in module.functions.iter() {
        let fn_name = function.name.as_deref().unwrap_or("<unnamed>");
        for (_, expr) in function.expressions.iter() {
            check_expression(fn_name, expr)?;
        }
        check_block(fn_name, &function.body)?;
    }

    Ok(())
}

fn check_type(
    module: &Module,
    inner: &naga::TypeInner,
    name: Option<&str>,
) -> Result<(), CompileError> {
    use naga::TypeInner as Ti;
    let label = |what: &str| {
        CompileError::Unsupported(match name {
            Some(name) => format!("type `{name}`: {what}"),
            None => what.to_string(),
        })
    };
    match inner {
        Ti::Scalar(scalar) => check_scalar(scalar).map_err(|w| label(&w)),
        Ti::Vector { scalar, .. } => check_scalar(scalar).map_err(|w| label(&w)),
        Ti::Atomic(_) => Err(label("atomics are not part of the model dialect")),
        Ti::Pointer { .. } | Ti::ValuePointer { .. } => Ok(()), // internal to locals
        Ti::Array { base, size, .. } => {
            match size {
                naga::ArraySize::Constant(_) => {}
                _ => {
                    return Err(label(
                        "runtime-sized arrays are not part of the model dialect",
                    ));
                }
            }
            match &module.types[*base].inner {
                Ti::Scalar(scalar) | Ti::Vector { scalar, .. } => {
                    check_scalar(scalar).map_err(|w| label(&w))
                }
                _ => Err(label(
                    "only arrays of scalars or vectors are supported for now",
                )),
            }
        }
        Ti::Matrix { .. } => Err(label("matrices are not supported yet")),
        Ti::Struct { .. } => Err(label("structs are not supported yet")),
        Ti::Image { .. } | Ti::Sampler { .. } => Err(label(
            "textures and samplers are not part of the model dialect",
        )),
        _ => Err(label("this type is not part of the model dialect")),
    }
}

fn check_scalar(scalar: &naga::Scalar) -> Result<(), String> {
    use naga::ScalarKind as Sk;
    match (scalar.kind, scalar.width) {
        (Sk::Float, 8 | 4) | (Sk::Sint | Sk::Uint, 4) | (Sk::Bool, _) => Ok(()),
        (Sk::Float, 2) => Err("f16 is not part of the model dialect".to_string()),
        (Sk::Sint | Sk::Uint, width) => Err(format!(
            "{}-bit integers are not part of the model dialect",
            u32::from(width) * 8
        )),
        (kind, width) => Err(format!(
            "scalar {kind:?} of width {width} is not part of the model dialect"
        )),
    }
}

fn check_expression(fn_name: &str, expr: &naga::Expression) -> Result<(), CompileError> {
    use naga::Expression as Ex;
    let banned = match expr {
        Ex::ImageSample { .. } | Ex::ImageLoad { .. } | Ex::ImageQuery { .. } => {
            Some("texture operations")
        }
        Ex::Derivative { .. } => Some("derivatives (dpdx/dpdy/fwidth)"),
        Ex::AtomicResult { .. } => Some("atomics"),
        Ex::WorkGroupUniformLoadResult { .. } => Some("workgroup operations"),
        Ex::ArrayLength(_) => Some("runtime array lengths"),
        Ex::RayQueryVertexPositions { .. }
        | Ex::RayQueryProceedResult
        | Ex::RayQueryGetIntersection { .. } => Some("ray queries"),
        Ex::SubgroupBallotResult | Ex::SubgroupOperationResult { .. } => {
            Some("subgroup operations")
        }
        _ => None,
    };
    match banned {
        Some(what) => Err(CompileError::Unsupported(format!(
            "in `{fn_name}`: {what} are not part of the model dialect"
        ))),
        None => Ok(()),
    }
}

fn check_block(fn_name: &str, block: &naga::Block) -> Result<(), CompileError> {
    use naga::Statement as St;
    for stmt in block.iter() {
        let banned = match stmt {
            St::Kill => Some("discard"),
            St::ControlBarrier(_) | St::MemoryBarrier(_) => Some("barriers"),
            St::ImageStore { .. } => Some("texture stores"),
            St::Atomic { .. } | St::ImageAtomic { .. } => Some("atomics"),
            St::WorkGroupUniformLoad { .. } => Some("workgroup operations"),
            St::RayQuery { .. } | St::RayPipelineFunction(_) => Some("ray queries"),
            St::SubgroupBallot { .. }
            | St::SubgroupGather { .. }
            | St::SubgroupCollectiveOperation { .. } => Some("subgroup operations"),
            St::CooperativeStore { .. } => Some("cooperative matrix operations"),
            _ => None,
        };
        if let Some(what) = banned {
            return Err(CompileError::Unsupported(format!(
                "in `{fn_name}`: {what} are not part of the model dialect"
            )));
        }
        match stmt {
            St::Block(inner) => check_block(fn_name, inner)?,
            St::If { accept, reject, .. } => {
                check_block(fn_name, accept)?;
                check_block(fn_name, reject)?;
            }
            St::Switch { cases, .. } => {
                for case in cases {
                    // Empty fall-through bodies are how multi-selector cases
                    // (`case 0, 1:`) are encoded; real fallthrough is gone
                    // from WGSL and stays out of the dialect.
                    if case.fall_through && !case.body.is_empty() {
                        return Err(CompileError::Unsupported(format!(
                            "in `{fn_name}`: switch fallthrough is not supported"
                        )));
                    }
                    check_block(fn_name, &case.body)?;
                }
            }
            St::Loop {
                body, continuing, ..
            } => {
                check_block(fn_name, body)?;
                check_block(fn_name, continuing)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Find `scene`, `bounds_min`, and `bounds_max` and check their signatures.
pub fn find_model_functions(module: &Module) -> Result<ModelFunctions, CompileError> {
    let mut scene = None;
    let mut bounds_min = None;
    let mut bounds_max = None;
    for (handle, function) in module.functions.iter() {
        match function.name.as_deref() {
            Some("scene") => scene = Some(handle),
            Some("bounds_min") => bounds_min = Some(handle),
            Some("bounds_max") => bounds_max = Some(handle),
            _ => {}
        }
    }
    let scene = scene.ok_or(CompileError::MissingFunction(
        "fn scene(p: vec3<f64>) -> bool",
    ))?;
    let bounds_min = bounds_min.ok_or(CompileError::MissingFunction(
        "fn bounds_min() -> vec3<f64>",
    ))?;
    let bounds_max = bounds_max.ok_or(CompileError::MissingFunction(
        "fn bounds_max() -> vec3<f64>",
    ))?;

    let dims = check_scene_signature(module, scene)?;
    for (name, handle) in [("bounds_min", bounds_min), ("bounds_max", bounds_max)] {
        check_bounds_signature(module, name, handle, dims)?;
    }

    Ok(ModelFunctions {
        scene,
        bounds_min,
        bounds_max,
        dims,
    })
}

fn vector_f64_size(module: &Module, ty: Handle<naga::Type>) -> Option<usize> {
    match module.types[ty].inner {
        naga::TypeInner::Vector { size, scalar }
            if scalar.kind == naga::ScalarKind::Float && scalar.width == 8 =>
        {
            Some(size as usize)
        }
        _ => None,
    }
}

fn check_scene_signature(
    module: &Module,
    handle: Handle<naga::Function>,
) -> Result<usize, CompileError> {
    let function = &module.functions[handle];
    let signature = "`scene` must be `fn scene(p: vec3<f64>) -> bool` \
                     (or `vec2<f64>` for a 2D sketch)";
    let [argument] = function.arguments.as_slice() else {
        return Err(CompileError::Type(signature.to_string()));
    };
    let dims = vector_f64_size(module, argument.ty)
        .filter(|dims| matches!(dims, 2 | 3))
        .ok_or_else(|| CompileError::Type(signature.to_string()))?;
    let returns_bool = function.result.as_ref().is_some_and(|result| {
        matches!(
            module.types[result.ty].inner,
            naga::TypeInner::Scalar(naga::Scalar {
                kind: naga::ScalarKind::Bool,
                ..
            })
        )
    });
    if !returns_bool {
        return Err(CompileError::Type(format!(
            "{signature}; occupancy is the primary model form — express distance-style \
             logic as `return d <= 0.0;`"
        )));
    }
    Ok(dims)
}

fn check_bounds_signature(
    module: &Module,
    name: &str,
    handle: Handle<naga::Function>,
    dims: usize,
) -> Result<(), CompileError> {
    let function = &module.functions[handle];
    let ok = function.arguments.is_empty()
        && function
            .result
            .as_ref()
            .and_then(|result| vector_f64_size(module, result.ty))
            == Some(dims);
    if ok {
        Ok(())
    } else {
        Err(CompileError::Type(format!(
            "`{name}` must be `fn {name}() -> vec{dims}<f64>` to match scene's dimensionality"
        )))
    }
}

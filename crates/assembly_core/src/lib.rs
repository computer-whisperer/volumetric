//! Posing an assembly's parts as models: each part wrapped by its rigid
//! pose through `model_wrap_core` (query points get the inverse, bounds
//! the map) and the parts unioned through `model_merge_core`. Shared by
//! `assemble_operator` and `assembly_model_operator`.

use model_merge_core::{Combine, combine_models};
use model_wrap_core::{Affine, Wrapper};
use volumetric_abi::f64_map::F64Map;
use volumetric_abi::mechanism::{Assembly, Rigid};

/// The affine map of a pose.
pub fn affine(pose: &Rigid) -> Affine {
    Affine {
        linear: pose.linear,
        offset: pose.offset,
    }
}

/// `model` wrapped so that it sits at `pose`.
pub fn posed_model(model: &[u8], pose: &Rigid) -> Result<Vec<u8>, String> {
    let mut wrapper = Wrapper::parse(model)?;
    if wrapper.spatial() != 3 {
        return Err(format!(
            "a part must be a 3D model (this one has {} dimensions)",
            wrapper.dims
        ));
    }
    wrapper.apply_affine(&affine(pose))?;
    Ok(wrapper.finish())
}

/// Every part of `assembly` posed at `state` (the assembly's own state
/// when `None`), in part order.
pub fn posed_parts(assembly: &Assembly, state: Option<&F64Map>) -> Result<Vec<Vec<u8>>, String> {
    let poses = match state {
        Some(state) => assembly.mechanism.pose(state)?,
        None => assembly.poses(),
    };
    assembly
        .parts
        .iter()
        .zip(&poses)
        .map(|(part, pose)| {
            posed_model(&part.model, pose).map_err(|e| format!("part `{}`: {e}", part.name))
        })
        .collect()
}

/// One posed part of `assembly` by name.
pub fn posed_part(
    assembly: &Assembly,
    name: &str,
    state: Option<&F64Map>,
) -> Result<Vec<u8>, String> {
    let index = assembly
        .parts
        .iter()
        .position(|p| p.name == name)
        .ok_or_else(|| {
            format!(
                "`{name}` is not a part of the assembly (parts: {})",
                assembly.mechanism.parts.join(", ")
            )
        })?;
    let pose = match state {
        Some(state) => assembly.mechanism.pose(state)?[index],
        None => assembly.poses()[index],
    };
    posed_model(&assembly.parts[index].model, &pose).map_err(|e| format!("part `{name}`: {e}"))
}

/// The union of every posed part: the assembly as one model.
pub fn posed_union(assembly: &Assembly, state: Option<&F64Map>) -> Result<Vec<u8>, String> {
    combine_models(&posed_parts(assembly, state)?, Combine::Union)
}

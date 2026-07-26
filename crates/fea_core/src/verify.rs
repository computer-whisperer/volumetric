//! Certificate verification for solved frame meshes: is a claimed solution
//! (per-strut `stiffness_scale` plus the `displacement`/`rotation`/
//! `contact_force` fields of its final forward solve) actually the
//! equilibrium the solver would have accepted?
//!
//! This is the trust boundary of the external-solve pipeline: expensive
//! inverse searches run outside the engine on non-portable backends, and the
//! deterministic engine re-checks the result cheaply — one stiffness apply
//! plus one contact scan — before downstream steps consume it. The checks
//! mirror the solver's own acceptance criteria, sharing its code
//! ([`crate::contact_frame`], [`crate::rigid_contact_surface`], the same
//! release tolerance), so "verified" means precisely "a fixed point of
//! [`crate::solve`]'s contact iteration with a converged CG residual":
//!
//! - **Equilibrium**: `K u` vanishes on free dofs, measured relative to the
//!   residual of the prescribed-displacements-only start — the same
//!   reference CG's relative tolerance uses.
//! - **Settled contact**: re-running the activation scan changes nothing
//!   (no penetrating node outside the active set, no active node drifted
//!   off the rigid surface beyond the solver's slack).
//! - **Unilateral contact**: no active node pulls on the rigid body beyond
//!   the solver's own release tolerance.

use crate::frame::FrameModel;
use crate::{ContactFrame, RigidBody, SolveConfig, StiffnessModel, contact_frame};
use volumetric_abi::fea::{FeaElementKind, FeaMesh};

/// What verification measured. Thresholds are the caller's judgment call
/// (the import operator exposes them as config); the counts are absolute.
#[derive(Clone, Debug)]
pub struct VerifyReport {
    /// `‖K u‖` over free dofs, relative to the prescribed-only start's
    /// residual — the metric CG's own convergence test uses. A solution
    /// solved at `cg_tolerance` t verifies at roughly t or below.
    pub relative_residual: f64,
    /// Nodes the contact_force field marks as active.
    pub active_contacts: usize,
    /// Activation-scan violations: nodes that penetrate the rigid body (or
    /// active nodes drifted off its surface) beyond the solver's slack.
    /// A settled solution scores 0.
    pub contact_set_changes: usize,
    /// Active nodes pulling on the rigid body beyond the solver's release
    /// tolerance. A settled solution scores 0.
    pub contact_releases: usize,
}

fn node_field<'m>(mesh: &'m FeaMesh, name: &str, components: usize) -> Result<&'m [f64], String> {
    let field = mesh
        .node_fields
        .iter()
        .find(|f| f.name == name && f.components == components)
        .ok_or_else(|| {
            format!("solution mesh has no `{name}` node field ({components} components)")
        })?;
    if field.data.len() != mesh.node_count() * components {
        return Err(format!(
            "`{name}` field has {} values for {} nodes x {components}",
            field.data.len(),
            mesh.node_count()
        ));
    }
    Ok(&field.data)
}

/// Verify a solved Bar2 frame mesh against the rigid body it claims to rest
/// on. `config` supplies the material and fixed boundary (the physics the
/// solution must satisfy); its solver knobs are irrelevant here.
///
/// The mesh must carry the solved fields (`stiffness_scale` via the usual
/// element field, plus `displacement`, `rotation`, `contact_force` node
/// fields — exactly what `fea_bundle::apply_inverse_result` writes).
pub fn verify_frame_solution(
    mesh: &FeaMesh,
    rigid: &mut dyn RigidBody,
    config: &SolveConfig,
) -> Result<VerifyReport, String> {
    if mesh.element_kind != FeaElementKind::Bar2 {
        return Err(format!(
            "solution verification covers Bar2 frame meshes; got {:?}",
            mesh.element_kind
        ));
    }
    let node_count = mesh.node_count();
    let displacement = node_field(mesh, "displacement", 3)?;
    let rotation = node_field(mesh, "rotation", 3)?;
    let contact_force = node_field(mesh, "contact_force", 3)?;

    // The model under the claimed design: FrameModel reads `radius` and
    // `stiffness_scale` straight from the mesh fields.
    let model = FrameModel::new(mesh, config.material)?;
    let dpn = model.dofs_per_node();
    let n = node_count * dpn;
    let h = model.length_scale();
    let ContactFrame {
        contact_axis,
        contact_sign,
        scan_limit,
        slack,
        mut constrained,
        fixed_node,
    } = contact_frame(mesh, dpn, h, config.fixed_boundary);

    // Reassemble the 6-dof solution vector.
    let mut u = vec![0.0f64; n];
    for node in 0..node_count {
        let base = node * dpn;
        u[base..base + 3].copy_from_slice(&displacement[node * 3..node * 3 + 3]);
        u[base + 3..base + 6].copy_from_slice(&rotation[node * 3..node * 3 + 3]);
    }
    if u.iter().any(|v| !v.is_finite()) {
        return Err("solution displacement/rotation contains non-finite values".to_string());
    }

    // Active contact set: the nodes the solution says the body presses on.
    let active: Vec<usize> = (0..node_count)
        .filter(|&node| !fixed_node[node] && contact_force[node * 3 + contact_axis] != 0.0)
        .collect();
    for &node in &active {
        constrained[node * dpn + contact_axis] = true;
    }

    // Equilibrium: K u must vanish on free dofs. Reference scale: the
    // residual of the prescribed-only start, CG's own denominator.
    let mut forces = vec![0.0f64; n];
    model.apply(&u, &mut forces);
    let mut u0 = vec![0.0f64; n];
    for i in 0..n {
        if constrained[i] {
            u0[i] = u[i];
        }
    }
    let mut r0 = vec![0.0f64; n];
    model.apply(&u0, &mut r0);
    let free_norm = |v: &[f64]| {
        v.iter()
            .zip(&constrained)
            .filter(|(_, c)| !**c)
            .map(|(v, _)| v * v)
            .sum::<f64>()
            .sqrt()
    };
    let residual = free_norm(&forces);
    let reference = free_norm(&r0);
    let relative_residual = if reference > 0.0 {
        residual / reference
    } else if residual == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };

    // Unilateral contact: mirror the solver's release scan. Forces pressing
    // toward the glued face satisfy `-sign * f > 0`; pulls beyond the
    // relative release tolerance would have been evicted.
    let peak_compression = active
        .iter()
        .map(|&node| -contact_sign * forces[node * dpn + contact_axis])
        .fold(0.0f64, f64::max);
    let release_tol = peak_compression * 1e-3;
    let contact_releases = active
        .iter()
        .filter(|&&node| contact_sign * forces[node * dpn + contact_axis] > release_tol)
        .count();

    // Settled contact: re-run the activation scan; a genuine fixed point
    // changes nothing. Covers both penetration by non-active nodes and
    // active nodes drifted off the surface beyond the solver's slack.
    let mut contact_set_changes = 0usize;
    for node in 0..node_count {
        if fixed_node[node] {
            continue;
        }
        let p = mesh.node_position(node);
        let base = node * dpn;
        let deformed = [p[0] + u[base], p[1] + u[base + 1], p[2] + u[base + 2]];
        let mut probe = deformed;
        probe[contact_axis] -= contact_sign * slack;
        if !rigid.is_inside(probe) {
            continue;
        }
        let surface = crate::rigid_contact_surface(
            rigid,
            deformed,
            contact_axis,
            contact_sign,
            h,
            scan_limit,
        )?;
        let prescribed_u = surface - p[contact_axis];
        let is_active = contact_force[node * 3 + contact_axis] != 0.0;
        if !is_active || (u[base + contact_axis] - prescribed_u).abs() > slack {
            contact_set_changes += 1;
        }
    }

    Ok(VerifyReport {
        relative_residual,
        active_contacts: active.len(),
        contact_set_changes,
        contact_releases,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedBoundary, InverseConfig, Material, solve_inverse};
    use volumetric_abi::fea::{FeaField, FeaMesh};

    /// The lattice spans z in [0, 3]; the plate presses 0.2 cells deep.
    const PLATE_LEVEL: f64 = 2.8;

    /// A small cubic strut lattice pressed by a flat plate: run the real
    /// inverse solver, write its result onto the mesh the way the operator
    /// does, and the verifier must accept it; tamper with the design and it
    /// must not.
    fn solved_mesh() -> (FeaMesh, SolveConfig) {
        let mut mesh = crate::frame::tests::cubic_lattice(4);
        let config = SolveConfig {
            material: Material {
                youngs_modulus: 1.0,
                poissons_ratio: 0.3,
            },
            fixed_boundary: FixedBoundary::ZMin,
            ..Default::default()
        };
        let inverse = InverseConfig {
            solve: config,
            max_iterations: 3,
            ..Default::default()
        };
        let mut rigid = plate(PLATE_LEVEL);
        let mut target = |_: [f64; 2]| 1.0;
        let result = solve_inverse(&mesh, &mut rigid, &mut target, &inverse).expect("solve");

        // Mirror fea_bundle::apply_inverse_result (fea_core can't depend on
        // fea_bundle; keep the field names in sync).
        let push = |fields: &mut Vec<FeaField>, name: &str, components: usize, data: Vec<f64>| {
            fields.push(FeaField {
                name: name.to_string(),
                components,
                data,
            });
        };
        push(
            &mut mesh.element_fields,
            "stiffness_scale",
            1,
            result.stiffness_scale,
        );
        push(
            &mut mesh.node_fields,
            "displacement",
            3,
            result.solve.displacement,
        );
        push(
            &mut mesh.node_fields,
            "rotation",
            3,
            result.solve.rotation.expect("frame solve has rotations"),
        );
        push(
            &mut mesh.node_fields,
            "contact_force",
            3,
            result.solve.contact_force,
        );
        (mesh, config)
    }

    fn plate(level: f64) -> impl FnMut([f64; 3]) -> bool {
        move |p: [f64; 3]| p[2] >= level
    }

    #[test]
    fn genuine_solution_verifies() {
        let (mesh, config) = solved_mesh();
        let mut rigid = plate(PLATE_LEVEL);
        let report = verify_frame_solution(&mesh, &mut rigid, &config).expect("verify");
        assert!(
            report.relative_residual < 1e-6,
            "genuine solution residual {}",
            report.relative_residual
        );
        assert!(report.active_contacts > 0);
        assert_eq!(report.contact_set_changes, 0);
        assert_eq!(report.contact_releases, 0);
    }

    #[test]
    fn tampered_scales_fail_equilibrium() {
        let (mut mesh, config) = solved_mesh();
        let scales = mesh
            .element_fields
            .iter_mut()
            .find(|f| f.name == "stiffness_scale")
            .unwrap();
        // Soften every third strut 5x: scales stay in (0, 1] but the
        // claimed displacement is no longer this design's equilibrium.
        for (e, s) in scales.data.iter_mut().enumerate() {
            if e % 3 == 0 {
                *s *= 0.2;
            }
        }
        let mut rigid = plate(PLATE_LEVEL);
        let report = verify_frame_solution(&mesh, &mut rigid, &config).expect("verify");
        assert!(
            report.relative_residual > 1e-3,
            "tampered solution slipped through at residual {}",
            report.relative_residual
        );
    }

    #[test]
    fn tampered_displacement_fails_contact_scan() {
        let (mut mesh, config) = solved_mesh();
        // Push every node 10% of a cell deeper into the plate: penetration.
        let displacement = mesh
            .node_fields
            .iter_mut()
            .find(|f| f.name == "displacement")
            .unwrap();
        for v in displacement.data.iter_mut().skip(2).step_by(3) {
            *v += 0.1;
        }
        let mut rigid = plate(PLATE_LEVEL);
        let report = verify_frame_solution(&mesh, &mut rigid, &config).expect("verify");
        assert!(
            report.contact_set_changes > 0 || report.relative_residual > 1e-3,
            "penetrating solution slipped through: {report:?}"
        );
    }

    #[test]
    fn missing_fields_are_reported() {
        let (mut mesh, config) = solved_mesh();
        mesh.node_fields.retain(|f| f.name != "contact_force");
        let mut rigid = plate(PLATE_LEVEL);
        let err = verify_frame_solution(&mesh, &mut rigid, &config).unwrap_err();
        assert!(err.contains("contact_force"), "unexpected error: {err}");
    }
}

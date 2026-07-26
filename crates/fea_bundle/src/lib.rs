//! FEA problem/solution bundle format: the contract between the portable
//! engine and external (non-portable) solve backends.
//!
//! A **problem bundle** is everything an external solver needs to reproduce
//! an `fea_inverse` step outside the engine: the mesh, the load cases (each
//! a rigid-body model WASM plus a target pressure-map model WASM), and the
//! operator's CBOR config verbatim. `volumetric_cli fea-export` writes one;
//! `fea-solve` (or any future backend) consumes it.
//!
//! A **solution bundle** carries the answer back: an operator-identical
//! result mesh (the problem mesh plus the solved fields — see
//! [`apply_inverse_result`]) with solver provenance. The engine does not
//! trust it: `fea_solution_import_operator` re-verifies the claimed
//! equilibrium against the live timeline's mesh and rigid body before
//! letting downstream steps consume it (solve externally on fast untrusted
//! hardware, certify portably in-engine).
//!
//! Both bundles are CBOR. `load_cases` is a list from day one — print-drag
//! load cases at green-state material properties are the known next
//! requirement — but v1 producers emit exactly one case ("seating") and v1
//! consumers reject anything else, loudly.

use fea_core::InverseResult;
use serde::{Deserialize, Serialize};
use volumetric_abi::fea::{FeaField, FeaMesh};

/// Problem bundle format version this crate writes and accepts.
pub const PROBLEM_VERSION: u32 = 1;
/// Solution bundle format version this crate writes and accepts.
pub const SOLUTION_VERSION: u32 = 1;

/// What kind of solve the problem asks for. `Solve` (a plain forward solve)
/// is expected later; only `Inverse` exists today.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProblemKind {
    #[serde(rename = "inverse")]
    Inverse,
}

/// One load case: a rigid body pressing into the mesh and the target
/// interface pressure distribution to match.
#[derive(Clone, Serialize, Deserialize)]
pub struct LoadCase {
    /// Human-readable case name ("seating", "print_drag", ...).
    pub name: String,
    /// 3D occupancy model WASM for the rigid body.
    #[serde(with = "serde_bytes")]
    pub rigid_model: Vec<u8>,
    /// 2D target pressure-map model WASM.
    #[serde(with = "serde_bytes")]
    pub target_map: Vec<u8>,
}

/// An exported FEA problem: the full inputs of an `fea_inverse` step.
#[derive(Clone, Serialize, Deserialize)]
pub struct FeaProblem {
    pub version: u32,
    pub kind: ProblemKind,
    /// FeaMesh CBOR (`volumetric_abi::fea::encode_fea_mesh`).
    #[serde(with = "serde_bytes")]
    pub mesh: Vec<u8>,
    pub load_cases: Vec<LoadCase>,
    /// The operator's CBOR config, verbatim — solvers interpret it with
    /// [`decode_inverse_config`] so every backend reads the same knobs.
    #[serde(with = "serde_bytes")]
    pub config: Vec<u8>,
    /// Informational provenance ("project.vproj step 8"), never trusted.
    pub source: Option<String>,
}

/// Solver-reported statistics, carried for provenance and display; the
/// engine verifies the solution physically and never trusts these.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolutionStats {
    /// Which solver produced this ("fea_core-native", "cuda", ...).
    pub backend: String,
    pub wall_seconds: f64,
    pub inverse_iterations: u32,
    pub converged: bool,
    pub distribution_error: f64,
}

/// An external solve's answer to a [`FeaProblem`].
#[derive(Clone, Serialize, Deserialize)]
pub struct FeaSolution {
    pub version: u32,
    /// blake3 of the problem bundle file bytes this answers. Staleness
    /// breadcrumb for humans and CLIs; the in-engine import verifies
    /// physically and does not rely on it.
    #[serde(with = "serde_bytes")]
    pub problem_blake3: Vec<u8>,
    /// Operator-identical result mesh: the problem mesh with the solved
    /// fields upserted (see [`apply_inverse_result`]), FeaMesh CBOR.
    #[serde(with = "serde_bytes")]
    pub mesh: Vec<u8>,
    pub stats: SolutionStats,
}

fn encode<T: Serialize>(value: &T, what: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    ciborium::into_writer(value, &mut out).map_err(|e| format!("encoding {what}: {e}"))?;
    Ok(out)
}

pub fn encode_problem(problem: &FeaProblem) -> Result<Vec<u8>, String> {
    encode(problem, "FeaProblem")
}

pub fn decode_problem(bytes: &[u8]) -> Result<FeaProblem, String> {
    let problem: FeaProblem = ciborium::from_reader(bytes)
        .map_err(|e| format!("decoding FeaProblem (is this a problem bundle?): {e}"))?;
    if problem.version != PROBLEM_VERSION {
        return Err(format!(
            "problem bundle version {} (this build reads {PROBLEM_VERSION})",
            problem.version
        ));
    }
    Ok(problem)
}

pub fn encode_solution(solution: &FeaSolution) -> Result<Vec<u8>, String> {
    encode(solution, "FeaSolution")
}

pub fn decode_solution(bytes: &[u8]) -> Result<FeaSolution, String> {
    let solution: FeaSolution = ciborium::from_reader(bytes)
        .map_err(|e| format!("decoding FeaSolution (is this a solution bundle?): {e}"))?;
    if solution.version != SOLUTION_VERSION {
        return Err(format!(
            "solution bundle version {} (this build reads {SOLUTION_VERSION})",
            solution.version
        ));
    }
    Ok(solution)
}

/// The `fea_inverse` operator's CBOR config, decoded into `fea_core` terms.
/// One interpretation shared by the in-engine operator and every external
/// backend, so a bundle solves identically wherever it lands.
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct InverseOperatorConfig {
    pub youngs_modulus: f64,
    pub poissons_ratio: f64,
    pub fixed_boundary: String,
    pub max_iterations: u32,
    pub tolerance: f64,
    pub exponent: f64,
    pub min_scale: f64,
    pub column_size: f64,
    pub max_contact_iterations: u32,
    pub cg_tolerance: f64,
    pub preconditioner: String,
    pub schwarz_target_nodes: u32,
    pub stress_stiffening_passes: u32,
}

impl Default for InverseOperatorConfig {
    fn default() -> Self {
        Self {
            youngs_modulus: 1.0,
            poissons_ratio: 0.3,
            fixed_boundary: "zmin".to_string(),
            max_iterations: 20,
            tolerance: 0.02,
            exponent: 0.5,
            min_scale: 0.01,
            column_size: 0.0,
            max_contact_iterations: 64,
            cg_tolerance: 1e-8,
            preconditioner: "auto".to_string(),
            schwarz_target_nodes: 128,
            stress_stiffening_passes: 0,
        }
    }
}

impl InverseOperatorConfig {
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() {
            return Ok(Self::default());
        }
        ciborium::from_reader(bytes).map_err(|e| format!("decoding inverse config: {e}"))
    }

    /// Build the `fea_core` config this operator config describes.
    pub fn to_inverse_config(&self) -> Result<fea_core::InverseConfig, String> {
        Ok(fea_core::InverseConfig {
            solve: fea_core::SolveConfig {
                material: fea_core::Material {
                    youngs_modulus: self.youngs_modulus,
                    poissons_ratio: self.poissons_ratio,
                },
                fixed_boundary: fea_core::FixedBoundary::parse(&self.fixed_boundary)?,
                max_contact_iterations: self.max_contact_iterations as usize,
                cg_tolerance: self.cg_tolerance,
                preconditioner: fea_core::PrecondChoice::parse(
                    &self.preconditioner,
                    self.schwarz_target_nodes as usize,
                )?,
                stress_stiffening_passes: self.stress_stiffening_passes as usize,
                ..Default::default()
            },
            max_iterations: self.max_iterations as usize,
            tolerance: self.tolerance,
            exponent: self.exponent,
            min_scale: self.min_scale,
            column_size: self.column_size,
        })
    }
}

/// Replace-or-append a field, so re-running on an already-solved mesh
/// doesn't accumulate duplicates.
fn upsert(fields: &mut Vec<FeaField>, field: FeaField) {
    match fields.iter_mut().find(|f| f.name == field.name) {
        Some(existing) => *existing = field,
        None => fields.push(field),
    }
}

/// Write an [`InverseResult`] onto its mesh exactly the way
/// `fea_inverse_operator` does — element fields `stiffness_scale` and
/// `strain_energy_density`, node fields `target_force`, `displacement`,
/// `rotation` (frame meshes), `contact_force`. Every producer of a solved
/// mesh (the in-engine operator, external backends) goes through here so
/// downstream steps see one shape.
pub fn apply_inverse_result(mesh: &mut FeaMesh, result: InverseResult) {
    upsert(
        &mut mesh.element_fields,
        FeaField {
            name: "stiffness_scale".to_string(),
            components: 1,
            data: result.stiffness_scale,
        },
    );
    upsert(
        &mut mesh.node_fields,
        FeaField {
            name: "target_force".to_string(),
            components: 1,
            data: result.target_force,
        },
    );
    upsert(
        &mut mesh.node_fields,
        FeaField {
            name: "displacement".to_string(),
            components: 3,
            data: result.solve.displacement,
        },
    );
    if let Some(rotation) = result.solve.rotation {
        upsert(
            &mut mesh.node_fields,
            FeaField {
                name: "rotation".to_string(),
                components: 3,
                data: rotation,
            },
        );
    }
    upsert(
        &mut mesh.node_fields,
        FeaField {
            name: "contact_force".to_string(),
            components: 3,
            data: result.solve.contact_force,
        },
    );
    upsert(
        &mut mesh.element_fields,
        FeaField {
            name: "strain_energy_density".to_string(),
            components: 1,
            data: result.solve.strain_energy_density,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn problem_round_trips() {
        let problem = FeaProblem {
            version: PROBLEM_VERSION,
            kind: ProblemKind::Inverse,
            mesh: vec![1, 2, 3],
            load_cases: vec![LoadCase {
                name: "seating".to_string(),
                rigid_model: vec![4, 5],
                target_map: vec![6],
            }],
            config: vec![7, 8],
            source: Some("test.vproj step 3".to_string()),
        };
        let bytes = encode_problem(&problem).unwrap();
        let back = decode_problem(&bytes).unwrap();
        assert_eq!(back.kind, ProblemKind::Inverse);
        assert_eq!(back.mesh, problem.mesh);
        assert_eq!(back.load_cases.len(), 1);
        assert_eq!(back.load_cases[0].name, "seating");
        assert_eq!(back.config, problem.config);
    }

    #[test]
    fn solution_round_trips() {
        let solution = FeaSolution {
            version: SOLUTION_VERSION,
            problem_blake3: vec![0xab; 32],
            mesh: vec![9, 9, 9],
            stats: SolutionStats {
                backend: "test".to_string(),
                wall_seconds: 1.5,
                inverse_iterations: 4,
                converged: true,
                distribution_error: 0.03,
            },
        };
        let bytes = encode_solution(&solution).unwrap();
        let back = decode_solution(&bytes).unwrap();
        assert_eq!(back.problem_blake3, solution.problem_blake3);
        assert_eq!(back.mesh, solution.mesh);
        assert!(back.stats.converged);
    }

    #[test]
    fn version_mismatch_is_rejected() {
        let mut solution = FeaSolution {
            version: SOLUTION_VERSION + 1,
            problem_blake3: vec![0; 32],
            mesh: vec![],
            stats: SolutionStats {
                backend: String::new(),
                wall_seconds: 0.0,
                inverse_iterations: 0,
                converged: false,
                distribution_error: 0.0,
            },
        };
        let bytes = encode_solution(&solution).unwrap();
        assert!(decode_solution(&bytes).is_err());
        solution.version = SOLUTION_VERSION;
        let bytes = encode_solution(&solution).unwrap();
        assert!(decode_solution(&bytes).is_ok());
    }

    #[test]
    fn empty_config_decodes_to_defaults() {
        let config = InverseOperatorConfig::decode(&[]).unwrap();
        assert_eq!(config.max_iterations, 20);
        assert_eq!(config.preconditioner, "auto");
        let inverse = config.to_inverse_config().unwrap();
        assert_eq!(inverse.max_iterations, 20);
    }
}

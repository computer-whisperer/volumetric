//! Strut Model Operator.
//!
//! Realizes a Bar2 strut lattice (from `strut_pattern_operator`, usually
//! after `fea_solve_operator` / `fea_inverse_operator`) as printable
//! geometry: a ModelWASM sampling the union of one capsule per strut.
//! This is where dimensionless mechanical design becomes dimensions —
//! the solve/inverse loop drives per-strut `stiffness_scale` (a modulus
//! multiplier), and this operator maps it to a printed radius:
//!
//! `radius_e = radius_field_e * stiffness_scale_e ^ (1 / radius_exponent)`
//!
//! A single radius can't reproduce a modulus scale in both axial (r^2)
//! and bending (r^4) stiffness; `radius_exponent` picks the match — the
//! default 4 preserves bending response (foams are bending-dominated),
//! 2 would preserve axial. `min_radius` floors the result (printability);
//! `max_radius` caps it (0 = none).
//!
//! `displacement_scale` optionally realizes the deformed lattice instead
//! of the rest shape: node positions move by `displacement_scale *
//! displacement` (requires the solved 3-component `displacement` node
//! field when nonzero; 0 = rest shape, the printing default).
//!
//! Mechanism: `strut_model_core::build_payload` bakes the capsules and a
//! BVH; the payload is patched into the embedded `strut_model_template`
//! module as data segments (the `mesh_to_model_operator` pattern — no
//! synthesized instructions).
//!
//! Inputs:
//! - Input 0: FeaMesh — Bar2, with a scalar `radius` element field
//!   (optional scalar `stiffness_scale` element field, defaults to 1)
//! - Input 1: CBOR configuration:
//!   `{ radius_exponent: float .default 4.0, min_radius: float .default
//!   0.0, max_radius: float .default 0.0, displacement_scale: float
//!   .default 0.0 }`
//!
//! Output 0: ModelWASM (3D, occupancy-only; bounds = capsule extents).
//!
//! The embedded template binary is regenerated with:
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p strut_model_template
//! cp target/wasm32-unknown-unknown/release/strut_model_template.wasm \
//!    crates/operators/strut_model_operator/template/
//! ```

use strut_model_core::Capsule;
use volumetric_abi::fea::{FeaElementKind, FeaMesh, decode_fea_mesh};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};
use walrus::{FunctionId, Module, ModuleConfig};

/// The prebuilt template module (see the module docs for regeneration).
const TEMPLATE: &[u8] = include_bytes!("../template/strut_model_template.wasm");

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
struct StrutModelConfig {
    /// `radius = radius_field * scale^(1/radius_exponent)`: 4 matches
    /// bending stiffness, 2 axial.
    radius_exponent: f64,
    /// Printability floor on realized radii (0 = none).
    min_radius: f64,
    /// Cap on realized radii (0 = none).
    max_radius: f64,
    /// Realize node positions displaced by this multiple of the solved
    /// `displacement` field (0 = rest shape).
    displacement_scale: f64,
}

impl Default for StrutModelConfig {
    fn default() -> Self {
        Self {
            radius_exponent: 4.0,
            min_radius: 0.0,
            max_radius: 0.0,
            displacement_scale: 0.0,
        }
    }
}

/// Turn the mesh into capsules under the config's realization rules.
/// Pure — natively unit-testable.
fn realize_capsules(mesh: &FeaMesh, config: &StrutModelConfig) -> Result<Vec<Capsule>, String> {
    if mesh.element_kind != FeaElementKind::Bar2 {
        return Err(format!(
            "strut realization needs a Bar2 strut lattice, got {:?} elements",
            mesh.element_kind
        ));
    }
    if !(config.radius_exponent.is_finite() && config.radius_exponent >= 1.0) {
        return Err(format!(
            "radius_exponent must be >= 1, got {}",
            config.radius_exponent
        ));
    }
    if !(config.min_radius.is_finite() && config.min_radius >= 0.0) {
        return Err(format!("min_radius must be >= 0, got {}", config.min_radius));
    }
    if !(config.max_radius.is_finite() && config.max_radius >= 0.0) {
        return Err(format!("max_radius must be >= 0, got {}", config.max_radius));
    }
    if config.max_radius > 0.0 && config.max_radius < config.min_radius {
        return Err(format!(
            "max_radius {} is below min_radius {}",
            config.max_radius, config.min_radius
        ));
    }
    if !config.displacement_scale.is_finite() {
        return Err(format!(
            "displacement_scale must be finite, got {}",
            config.displacement_scale
        ));
    }

    let radius = mesh
        .element_fields
        .iter()
        .find(|f| f.name == "radius" && f.components == 1)
        .ok_or_else(|| "mesh has no scalar `radius` element field".to_string())?;
    let scale = mesh
        .element_fields
        .iter()
        .find(|f| f.name == "stiffness_scale" && f.components == 1);

    // Node positions, optionally displaced.
    let mut positions = mesh.node_positions.clone();
    if config.displacement_scale != 0.0 {
        let displacement = mesh
            .node_fields
            .iter()
            .find(|f| f.name == "displacement" && f.components == 3)
            .ok_or_else(|| {
                "displacement_scale is nonzero but the mesh has no 3-component \
                 `displacement` node field; run fea_solve_operator first"
                    .to_string()
            })?;
        for (p, u) in positions.iter_mut().zip(&displacement.data) {
            *p += config.displacement_scale * u;
        }
    }
    let position = |node: u32| -> [f64; 3] {
        std::array::from_fn(|i| positions[node as usize * 3 + i])
    };

    let mut capsules = Vec::with_capacity(mesh.element_count());
    for e in 0..mesh.element_count() {
        let base = radius.data[e];
        if !(base.is_finite() && base > 0.0) {
            return Err(format!("strut {e} has invalid radius {base}"));
        }
        let s = scale.map_or(1.0, |f| f.data[e]);
        if !(s.is_finite() && s >= 0.0) {
            return Err(format!("strut {e} has invalid stiffness_scale {s}"));
        }
        let mut r = base * s.powf(1.0 / config.radius_exponent);
        if config.min_radius > 0.0 {
            r = r.max(config.min_radius);
        }
        if config.max_radius > 0.0 {
            r = r.min(config.max_radius);
        }
        if r <= 0.0 {
            // scale 0 with no floor: the strut was designed away.
            continue;
        }
        let pair = mesh.element(e);
        capsules.push(Capsule {
            a: position(pair[0]),
            b: position(pair[1]),
            radius: r,
        });
    }
    if capsules.is_empty() {
        return Err(
            "every strut realized to zero radius (all stiffness_scale 0 and no \
             min_radius floor)"
                .to_string(),
        );
    }
    Ok(capsules)
}

/// Read the constant a trivial `() -> i32` function returns.
fn const_i32_return(module: &Module, func_id: FunctionId) -> Option<i32> {
    let local = match &module.funcs.get(func_id).kind {
        walrus::FunctionKind::Local(local) => local,
        _ => return None,
    };
    let block = local.block(local.entry_block());
    match block.instrs.as_slice() {
        [(walrus::ir::Instr::Const(c), _)] => match c.value {
            walrus::ir::Value::I32(v) => Some(v),
            _ => None,
        },
        _ => None,
    }
}

fn patch_template(payload: &[u8]) -> Result<Vec<u8>, String> {
    let config = ModuleConfig::new();
    let mut module = Module::from_buffer_with_config(TEMPLATE, &config)
        .map_err(|e| format!("failed to parse the embedded template: {e}"))?;

    let memory_id = module
        .exports
        .iter()
        .find(|e| e.name == "memory")
        .and_then(|e| match e.item {
            walrus::ExportItem::Memory(m) => Some(m),
            _ => None,
        })
        .ok_or("template missing memory export")?;

    // The patch slot's address, then drop the helper export — it is not
    // part of the Model ABI.
    let slot_export = module
        .exports
        .iter()
        .find(|e| e.name == "strut_payload_slot")
        .map(|e| (e.id(), e.item))
        .ok_or("template missing strut_payload_slot export")?;
    let slot_addr = match slot_export.1 {
        walrus::ExportItem::Function(f) => const_i32_return(&module, f)
            .ok_or("template strut_payload_slot is not a constant function")?,
        _ => return Err("template strut_payload_slot is not a function".to_string()),
    };
    module.exports.delete(slot_export.0);

    // Payload in freshly reserved pages; base address into the slot.
    let base = {
        let memory = module.memories.get_mut(memory_id);
        let base = memory.initial * 65536;
        memory.initial += (payload.len() as u64).div_ceil(65536);
        if let Some(max) = memory.maximum {
            memory.maximum = Some(max.max(memory.initial));
        }
        base
    };
    module.data.add(
        walrus::DataKind::Active {
            memory: memory_id,
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(base as i32)),
        },
        payload.to_vec(),
    );
    module.data.add(
        walrus::DataKind::Active {
            memory: memory_id,
            offset: walrus::ConstExpr::Value(walrus::ir::Value::I32(slot_addr)),
        },
        (base as u32).to_le_bytes().to_vec(),
    );

    Ok(module.emit_wasm())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let config = {
        let buf = read_input(1);
        if buf.is_empty() {
            StrutModelConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&buf)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };

    let result = decode_fea_mesh(&read_input(0))
        .and_then(|mesh| realize_capsules(&mesh, &config))
        .and_then(|capsules| strut_model_core::build_payload(&capsules))
        .and_then(|payload| patch_template(&payload));
    match result {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("strut model realization failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || OperatorMetadata {
        name: "strut_model_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        inputs: vec![
            OperatorMetadataInput::FeaMesh,
            OperatorMetadataInput::CBORConfiguration(
                "{ radius_exponent: float .default 4.0, min_radius: float .default 0.0, max_radius: float .default 0.0, displacement_scale: float .default 0.0 }"
                    .to_string(),
            ),
        ],
        input_names: vec!["Strut lattice".to_string(), "Config".to_string()],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::fea::FeaField;

    /// Two struts along x sharing a node, radii 0.1, plus fields.
    fn two_strut_mesh() -> FeaMesh {
        FeaMesh {
            element_kind: FeaElementKind::Bar2,
            node_positions: vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                2.0, 0.0, 0.0,
            ],
            connectivity: vec![0, 1, 1, 2],
            node_fields: vec![],
            element_fields: vec![FeaField {
                name: "radius".to_string(),
                components: 1,
                data: vec![0.1, 0.1],
            }],
        }
    }

    #[test]
    fn scale_maps_to_radius_with_the_exponent() {
        let mut mesh = two_strut_mesh();
        mesh.element_fields.push(FeaField {
            name: "stiffness_scale".to_string(),
            components: 1,
            data: vec![1.0, 0.0625], // 1/16
        });
        // Bending-matched (n=4): r * (1/16)^(1/4) = r/2.
        let capsules = realize_capsules(&mesh, &StrutModelConfig::default()).unwrap();
        assert_eq!(capsules[0].radius, 0.1);
        assert!((capsules[1].radius - 0.05).abs() < 1e-12);
        // Axial-matched (n=2): r * (1/16)^(1/2) = r/4.
        let axial = StrutModelConfig {
            radius_exponent: 2.0,
            ..Default::default()
        };
        let capsules = realize_capsules(&mesh, &axial).unwrap();
        assert!((capsules[1].radius - 0.025).abs() < 1e-12);
    }

    #[test]
    fn radius_floor_and_cap_apply() {
        let mut mesh = two_strut_mesh();
        mesh.element_fields.push(FeaField {
            name: "stiffness_scale".to_string(),
            components: 1,
            data: vec![1.0, 1e-8],
        });
        let config = StrutModelConfig {
            min_radius: 0.03,
            max_radius: 0.08,
            ..Default::default()
        };
        let capsules = realize_capsules(&mesh, &config).unwrap();
        assert_eq!(capsules[0].radius, 0.08, "cap");
        assert_eq!(capsules[1].radius, 0.03, "floor");
    }

    #[test]
    fn zero_scale_struts_vanish_without_a_floor() {
        let mut mesh = two_strut_mesh();
        mesh.element_fields.push(FeaField {
            name: "stiffness_scale".to_string(),
            components: 1,
            data: vec![1.0, 0.0],
        });
        let capsules = realize_capsules(&mesh, &StrutModelConfig::default()).unwrap();
        assert_eq!(capsules.len(), 1);

        mesh.element_fields.retain(|f| f.name != "stiffness_scale");
        mesh.element_fields.push(FeaField {
            name: "stiffness_scale".to_string(),
            components: 1,
            data: vec![0.0, 0.0],
        });
        let err = realize_capsules(&mesh, &StrutModelConfig::default()).unwrap_err();
        assert!(err.contains("zero radius"), "unexpected error: {err}");
    }

    #[test]
    fn displacement_realization_moves_the_capsules() {
        let mut mesh = two_strut_mesh();
        mesh.node_fields.push(FeaField {
            name: "displacement".to_string(),
            components: 3,
            data: vec![
                0.0, 0.0, -0.1, //
                0.0, 0.0, -0.2, //
                0.0, 0.0, -0.3,
            ],
        });
        let config = StrutModelConfig {
            displacement_scale: 1.0,
            ..Default::default()
        };
        let capsules = realize_capsules(&mesh, &config).unwrap();
        assert_eq!(capsules[0].a[2], -0.1);
        assert_eq!(capsules[0].b[2], -0.2);
        assert_eq!(capsules[1].b[2], -0.3);

        // Half-scale realization.
        let half = StrutModelConfig {
            displacement_scale: 0.5,
            ..Default::default()
        };
        let capsules = realize_capsules(&mesh, &half).unwrap();
        assert_eq!(capsules[0].a[2], -0.05);

        // Rest shape by default.
        let capsules = realize_capsules(&mesh, &StrutModelConfig::default()).unwrap();
        assert_eq!(capsules[0].a[2], 0.0);
    }

    #[test]
    fn missing_fields_and_wrong_kinds_are_errors() {
        let mut mesh = two_strut_mesh();
        mesh.element_fields.clear();
        let err = realize_capsules(&mesh, &StrutModelConfig::default()).unwrap_err();
        assert!(err.contains("radius"), "unexpected error: {err}");

        let mesh = two_strut_mesh();
        let config = StrutModelConfig {
            displacement_scale: 1.0,
            ..Default::default()
        };
        let err = realize_capsules(&mesh, &config).unwrap_err();
        assert!(err.contains("displacement"), "unexpected error: {err}");

        let mut hex = two_strut_mesh();
        hex.element_kind = FeaElementKind::Hex8;
        hex.connectivity = vec![0, 1, 2, 0, 1, 2, 0, 1];
        hex.element_fields[0].data = vec![0.1];
        let err = realize_capsules(&hex, &StrutModelConfig::default()).unwrap_err();
        assert!(err.contains("Bar2"), "unexpected error: {err}");
    }
}

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
//! 2 would preserve axial. Two clips shape the mapping: `scale_min`/
//! `scale_max` clamp the driving `stiffness_scale` *input* before the power
//! law (so extreme inverse-solve scales don't run the radius away), and
//! `min_radius`/`max_radius` clamp the realized radius *output* (printability
//! floor and cap; 0 = none on each).
//!
//! `displacement_scale` optionally realizes the deformed lattice instead
//! of the rest shape: node positions move by `displacement_scale *
//! displacement` (requires the solved 3-component `displacement` node
//! field when nonzero; 0 = rest shape, the printing default).
//!
//! ## Channel passthrough
//!
//! The realized model carries the mesh's per-strut FEA scalars as sample
//! channels: `sample` is unchanged occupancy, and `sample_channels` also
//! reports, for each declared channel, the raw value of the strut that owns
//! the sampled point. `channels` names the element fields to expose
//! (comma-separated; empty = every scalar element field present, e.g.
//! `radius`, `stiffness_scale`, `strain_energy_density`), each as a
//! `Custom("fea.<name>")` channel. At most five fit alongside occupancy (the
//! 3D IO buffer's output capacity). Values pass through raw — the transfer
//! clips act on the geometry, not the channels.
//!
//! Mechanism: `strut_model_core::build_payload` bakes the capsules, the BVH,
//! and the per-strut channel rows; the payload is patched into the embedded
//! `strut_model_template` module as data segments (the
//! `mesh_to_model_operator` pattern — no synthesized instructions), and the
//! template serves `get_sample_format`/`sample_channels` from it.
//!
//! Inputs:
//! - Input 0: FeaMesh — Bar2, with a scalar `radius` element field
//!   (optional scalar `stiffness_scale` element field, defaults to 1)
//! - Input 1: CBOR configuration:
//!   `{ radius_exponent: float .default 4.0, scale_min: float .default 0.0,
//!   scale_max: float .default 0.0, min_radius: float .default 0.0,
//!   max_radius: float .default 0.0, displacement_scale: float .default 0.0,
//!   channels: tstr .default "" }`
//!
//! Output 0: ModelWASM (3D; occupancy plus the passthrough channels; bounds =
//! capsule extents).
//!
//! The embedded template binary is regenerated with:
//! ```text
//! cargo build --release --target wasm32-unknown-unknown -p strut_model_template
//! cp target/wasm32-unknown-unknown/release/strut_model_template.wasm \
//!    crates/operators/strut_model_operator/template/
//! ```

use strut_model_core::{Capsule, ChannelPayload};
use volumetric_abi::fea::{FeaElementKind, FeaField, FeaMesh, decode_fea_mesh};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{
    ChannelKind, OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput, SampleChannel,
    SampleFormat, encode_sample_format,
};
use walrus::{FunctionId, Module, ModuleConfig};

/// Most passthrough channels the 3D model IO buffer can carry: it holds
/// `2 * dims` f64s, so the output half fits `dims * 2 = 6` f32s at `dims = 3`.
/// One is occupancy, leaving five for passthrough (the host rejects more).
const MAX_PASSTHROUGH_CHANNELS: usize = 5;

/// The prebuilt template module (see the module docs for regeneration).
const TEMPLATE: &[u8] = include_bytes!("../template/strut_model_template.wasm");

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct StrutModelConfig {
    /// `radius = radius_field * scale^(1/radius_exponent)`: 4 matches
    /// bending stiffness, 2 axial.
    radius_exponent: f64,
    /// Clamp `stiffness_scale` to at least this before the power law (input
    /// range clip; 0 = no floor). Distinct from `min_radius`, which clips the
    /// realized radius: this bounds the driving field, so extreme inverse-solve
    /// scales don't run the mapping away.
    scale_min: f64,
    /// Clamp `stiffness_scale` to at most this before the power law (0 = no
    /// cap).
    scale_max: f64,
    /// Printability floor on realized radii (0 = none).
    min_radius: f64,
    /// Cap on realized radii (0 = none).
    max_radius: f64,
    /// Realize node positions displaced by this multiple of the solved
    /// `displacement` field (0 = rest shape).
    displacement_scale: f64,
    /// Comma-separated element-field names to pass through as sample channels
    /// (empty = every scalar element field present). Each becomes a `Custom`
    /// channel carrying the strut's raw value.
    channels: String,
}

impl Default for StrutModelConfig {
    fn default() -> Self {
        Self {
            radius_exponent: 4.0,
            scale_min: 0.0,
            scale_max: 0.0,
            min_radius: 0.0,
            max_radius: 0.0,
            displacement_scale: 0.0,
            channels: String::new(),
        }
    }
}

/// Realized geometry plus the per-strut channel rows to pass through.
#[derive(Debug)]
struct Realized {
    capsules: Vec<Capsule>,
    /// Passthrough channel names (element-field names), in order.
    channel_names: Vec<String>,
    /// Per-kept-strut values, row-major and aligned with `capsules`: row `e`
    /// is `channel_values[e * n .. (e + 1) * n]`, `n = channel_names.len()`.
    channel_values: Vec<f32>,
}

/// The element-field scalars to pass through as channels: an explicit
/// comma-separated list, or (empty) every scalar element field present.
fn select_channels<'a>(mesh: &'a FeaMesh, spec: &str) -> Result<Vec<&'a FeaField>, String> {
    let requested: Vec<&str> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if requested.is_empty() {
        return Ok(mesh
            .element_fields
            .iter()
            .filter(|f| f.components == 1)
            .collect());
    }
    let mut out: Vec<&FeaField> = Vec::with_capacity(requested.len());
    for name in requested {
        if out.iter().any(|f| f.name == name) {
            return Err(format!("channel field `{name}` is listed more than once"));
        }
        let field = mesh
            .element_fields
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| format!("channel field `{name}` is not an element field on the mesh"))?;
        if field.components != 1 {
            return Err(format!(
                "channel field `{name}` has {} components; only scalar element fields \
                 can be passthrough channels",
                field.components
            ));
        }
        out.push(field);
    }
    Ok(out)
}

/// The declared sample format: occupancy plus one `Custom` channel per
/// passthrough field.
fn sample_format(names: &[String]) -> SampleFormat {
    let mut channels = Vec::with_capacity(names.len() + 1);
    channels.push(SampleChannel {
        name: "occupancy".to_string(),
        kind: ChannelKind::Occupancy,
    });
    for name in names {
        channels.push(SampleChannel {
            name: name.clone(),
            kind: ChannelKind::Custom(format!("fea.{name}")),
        });
    }
    SampleFormat { channels }
}

/// Turn the mesh into capsules and per-strut channel rows under the config's
/// realization rules. Pure — natively unit-testable.
fn realize(mesh: &FeaMesh, config: &StrutModelConfig) -> Result<Realized, String> {
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
    if !(config.scale_min.is_finite() && config.scale_min >= 0.0) {
        return Err(format!("scale_min must be >= 0, got {}", config.scale_min));
    }
    if !(config.scale_max.is_finite() && config.scale_max >= 0.0) {
        return Err(format!("scale_max must be >= 0, got {}", config.scale_max));
    }
    if config.scale_max > 0.0 && config.scale_max < config.scale_min {
        return Err(format!(
            "scale_max {} is below scale_min {}",
            config.scale_max, config.scale_min
        ));
    }
    if !(config.min_radius.is_finite() && config.min_radius >= 0.0) {
        return Err(format!(
            "min_radius must be >= 0, got {}",
            config.min_radius
        ));
    }
    if !(config.max_radius.is_finite() && config.max_radius >= 0.0) {
        return Err(format!(
            "max_radius must be >= 0, got {}",
            config.max_radius
        ));
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

    let channel_fields = select_channels(mesh, &config.channels)?;
    if channel_fields.len() > MAX_PASSTHROUGH_CHANNELS {
        return Err(format!(
            "{} passthrough channels exceed the {MAX_PASSTHROUGH_CHANNELS}-channel limit; \
             restrict them with the `channels` config",
            channel_fields.len()
        ));
    }
    let channel_names: Vec<String> = channel_fields.iter().map(|f| f.name.clone()).collect();

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
    let position =
        |node: u32| -> [f64; 3] { std::array::from_fn(|i| positions[node as usize * 3 + i]) };

    let mut capsules = Vec::with_capacity(mesh.element_count());
    let mut channel_values = Vec::with_capacity(mesh.element_count() * channel_fields.len());
    for e in 0..mesh.element_count() {
        let base = radius.data[e];
        if !(base.is_finite() && base > 0.0) {
            return Err(format!("strut {e} has invalid radius {base}"));
        }
        let mut s = scale.map_or(1.0, |f| f.data[e]);
        if !(s.is_finite() && s >= 0.0) {
            return Err(format!("strut {e} has invalid stiffness_scale {s}"));
        }
        // Input range clip on the driving field, before the power law.
        if config.scale_min > 0.0 {
            s = s.max(config.scale_min);
        }
        if config.scale_max > 0.0 {
            s = s.min(config.scale_max);
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
        // Channel row for this kept strut (raw passthrough values).
        for f in &channel_fields {
            channel_values.push(f.data[e] as f32);
        }
    }
    if capsules.is_empty() {
        return Err(
            "every strut realized to zero radius (all stiffness_scale 0 and no \
             min_radius floor)"
                .to_string(),
        );
    }
    Ok(Realized {
        capsules,
        channel_names,
        channel_values,
    })
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
        .and_then(|mesh| realize(&mesh, &config))
        .and_then(|realized| {
            let channels = ChannelPayload {
                count: realized.channel_names.len(),
                format_cbor: encode_sample_format(&sample_format(&realized.channel_names)),
                values: realized.channel_values,
            };
            strut_model_core::build_payload(&realized.capsules, &channels)
        })
        .and_then(|payload| patch_template(&payload));
    match result {
        Ok(wasm) => post_output(0, &wasm),
        Err(e) => report_error(&format!("strut model realization failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        OperatorMetadata {
        name: "strut_model_operator".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        display_name: "Strut Model".to_string(),
        description: "Realize a strut lattice as printable geometry, one capsule per strut.".to_string(),
        category: "Lattice".to_string(),
        icon_svg: volumetric_abi::icon_svg!(
            r##"<path d="M10.5 20.5 3.5 13.5a4.95 4.95 0 1 1 7-7l7 7a4.95 4.95 0 1 1-7 7Z"/>"##,
            r##"<path d="m8.5 8.5 7 7"/>"##,
        )
        .to_string(),
        inputs: vec![
            OperatorMetadataInput::FeaMesh,
            OperatorMetadataInput::CBORConfiguration(
                "{ radius_exponent: float .default 4.0, scale_min: float .default 0.0, scale_max: float .default 0.0, min_radius: float .default 0.0, max_radius: float .default 0.0, displacement_scale: float .default 0.0, channels: tstr .default \"\" }"
                    .to_string(),
            ),
        ],
        input_names: vec!["Strut lattice".to_string(), "Config".to_string()],
        outputs: vec![OperatorMetadataOutput::ModelWASM],
    }
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
        let capsules = realize(&mesh, &StrutModelConfig::default())
            .unwrap()
            .capsules;
        assert_eq!(capsules[0].radius, 0.1);
        assert!((capsules[1].radius - 0.05).abs() < 1e-12);
        // Axial-matched (n=2): r * (1/16)^(1/2) = r/4.
        let axial = StrutModelConfig {
            radius_exponent: 2.0,
            ..Default::default()
        };
        let capsules = realize(&mesh, &axial).unwrap().capsules;
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
        let capsules = realize(&mesh, &config).unwrap().capsules;
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
        let capsules = realize(&mesh, &StrutModelConfig::default())
            .unwrap()
            .capsules;
        assert_eq!(capsules.len(), 1);

        mesh.element_fields.retain(|f| f.name != "stiffness_scale");
        mesh.element_fields.push(FeaField {
            name: "stiffness_scale".to_string(),
            components: 1,
            data: vec![0.0, 0.0],
        });
        let err = realize(&mesh, &StrutModelConfig::default()).unwrap_err();
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
        let capsules = realize(&mesh, &config).unwrap().capsules;
        assert_eq!(capsules[0].a[2], -0.1);
        assert_eq!(capsules[0].b[2], -0.2);
        assert_eq!(capsules[1].b[2], -0.3);

        // Half-scale realization.
        let half = StrutModelConfig {
            displacement_scale: 0.5,
            ..Default::default()
        };
        let capsules = realize(&mesh, &half).unwrap().capsules;
        assert_eq!(capsules[0].a[2], -0.05);

        // Rest shape by default.
        let capsules = realize(&mesh, &StrutModelConfig::default())
            .unwrap()
            .capsules;
        assert_eq!(capsules[0].a[2], 0.0);
    }

    #[test]
    fn missing_fields_and_wrong_kinds_are_errors() {
        let mut mesh = two_strut_mesh();
        mesh.element_fields.clear();
        let err = realize(&mesh, &StrutModelConfig::default()).unwrap_err();
        assert!(err.contains("radius"), "unexpected error: {err}");

        let mesh = two_strut_mesh();
        let config = StrutModelConfig {
            displacement_scale: 1.0,
            ..Default::default()
        };
        let err = realize(&mesh, &config).unwrap_err();
        assert!(err.contains("displacement"), "unexpected error: {err}");

        let mut hex = two_strut_mesh();
        hex.element_kind = FeaElementKind::Hex8;
        hex.connectivity = vec![0, 1, 2, 0, 1, 2, 0, 1];
        hex.element_fields[0].data = vec![0.1];
        let err = realize(&hex, &StrutModelConfig::default()).unwrap_err();
        assert!(err.contains("Bar2"), "unexpected error: {err}");
    }

    #[test]
    fn scale_input_clip_clamps_before_the_power_law() {
        let mut mesh = two_strut_mesh();
        mesh.element_fields.push(FeaField {
            name: "stiffness_scale".to_string(),
            components: 1,
            data: vec![16.0, 0.0625],
        });
        // Clip the *input* scale to [0.5, 4]; the power law (n = 4) then acts
        // on the clamped value. This is not the radius output clip.
        let config = StrutModelConfig {
            scale_min: 0.5,
            scale_max: 4.0,
            ..Default::default()
        };
        let capsules = realize(&mesh, &config).unwrap().capsules;
        // strut 0: 16 -> 4, r = 0.1 * 4^(1/4) = 0.1 * sqrt(2)
        assert!((capsules[0].radius - 0.1 * 2f64.sqrt()).abs() < 1e-9);
        // strut 1: 0.0625 -> 0.5, r = 0.1 * 0.5^(1/4)
        assert!((capsules[1].radius - 0.1 * 0.5f64.powf(0.25)).abs() < 1e-9);
        // Without the input clip both would be 0.2 and 0.05.
        let unclipped = realize(&mesh, &StrutModelConfig::default())
            .unwrap()
            .capsules;
        assert!((unclipped[0].radius - 0.2).abs() < 1e-9);
        assert!((unclipped[1].radius - 0.05).abs() < 1e-9);
    }

    /// A mesh carrying the three scalar element fields a solved lattice has.
    fn solved_mesh() -> FeaMesh {
        let mut mesh = two_strut_mesh();
        mesh.element_fields.push(FeaField {
            name: "stiffness_scale".to_string(),
            components: 1,
            data: vec![1.0, 0.25],
        });
        mesh.element_fields.push(FeaField {
            name: "strain_energy_density".to_string(),
            components: 1,
            data: vec![5.0, 7.0],
        });
        mesh
    }

    #[test]
    fn default_channels_pass_through_all_scalar_element_fields() {
        let realized = realize(&solved_mesh(), &StrutModelConfig::default()).unwrap();
        assert_eq!(
            realized.channel_names,
            ["radius", "stiffness_scale", "strain_energy_density"]
        );
        // Row-major, one row per kept strut, in field-declaration order.
        assert_eq!(
            realized.channel_values,
            [0.1, 1.0, 5.0, 0.1, 0.25, 7.0].map(|v| v as f32)
        );
    }

    #[test]
    fn explicit_channels_select_and_order_fields() {
        let config = StrutModelConfig {
            channels: "strain_energy_density, radius".to_string(),
            ..Default::default()
        };
        let realized = realize(&solved_mesh(), &config).unwrap();
        assert_eq!(realized.channel_names, ["strain_energy_density", "radius"]);
        assert_eq!(
            realized.channel_values,
            [5.0, 0.1, 7.0, 0.1].map(|v| v as f32)
        );
    }

    #[test]
    fn dropped_struts_drop_their_channel_rows() {
        let mut mesh = solved_mesh();
        // Strut 1 is designed away (scale 0, no floor).
        mesh.element_fields[1].data = vec![1.0, 0.0];
        let realized = realize(&mesh, &StrutModelConfig::default()).unwrap();
        assert_eq!(realized.capsules.len(), 1);
        // Exactly one row survives, glued to the kept strut.
        assert_eq!(realized.channel_values, [0.1, 1.0, 5.0].map(|v| v as f32));
    }

    #[test]
    fn channel_selection_errors_are_clear() {
        let mesh = solved_mesh();

        let missing = StrutModelConfig {
            channels: "nope".to_string(),
            ..Default::default()
        };
        let err = realize(&mesh, &missing).unwrap_err();
        assert!(err.contains("nope"), "unexpected error: {err}");

        let dup = StrutModelConfig {
            channels: "radius, radius".to_string(),
            ..Default::default()
        };
        let err = realize(&mesh, &dup).unwrap_err();
        assert!(err.contains("more than once"), "unexpected error: {err}");

        // A vector element field can't be a scalar channel.
        let mut vec_mesh = solved_mesh();
        vec_mesh.element_fields.push(FeaField {
            name: "gradient".to_string(),
            components: 3,
            data: vec![0.0; 6],
        });
        let non_scalar = StrutModelConfig {
            channels: "gradient".to_string(),
            ..Default::default()
        };
        let err = realize(&vec_mesh, &non_scalar).unwrap_err();
        assert!(err.contains("components"), "unexpected error: {err}");
    }

    #[test]
    fn too_many_channels_is_an_error() {
        let mut mesh = two_strut_mesh();
        // radius + five more scalar element fields = six > the limit of five.
        for i in 0..5 {
            mesh.element_fields.push(FeaField {
                name: format!("field{i}"),
                components: 1,
                data: vec![0.0, 0.0],
            });
        }
        let err = realize(&mesh, &StrutModelConfig::default()).unwrap_err();
        assert!(err.contains("exceed"), "unexpected error: {err}");
    }

    #[test]
    fn sample_format_declares_occupancy_then_custom_channels() {
        let format = sample_format(&["stiffness_scale".to_string()]);
        assert_eq!(format.channels.len(), 2);
        assert_eq!(format.channels[0].kind, ChannelKind::Occupancy);
        assert_eq!(format.channels[1].name, "stiffness_scale");
        assert_eq!(
            format.channels[1].kind,
            ChannelKind::Custom("fea.stiffness_scale".to_string())
        );
        // The declared format must pass the ABI's structural rules.
        format.validate().unwrap();
    }

    #[test]
    fn occupancy_fallback_cbor_matches_the_default_format() {
        // The template's unpatched-payload fallback constant must stay in sync
        // with the ABI's default (occupancy-only) sample format.
        assert_eq!(
            encode_sample_format(&SampleFormat::default()),
            strut_model_core::OCCUPANCY_FORMAT_CBOR
        );
    }
}

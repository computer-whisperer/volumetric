//! Pattern operator: mirror, linear and circular instancing of one model.
//!
//! Host/operator ABI: see the `volumetric_abi` crate; wrapper mechanics:
//! `model_wrap_core`; semantics: README.md (surfaced as the operator's docs).
//!
//! The config's optional blocks compose into a list of instance placements
//! (affine maps, identity first). The wrapped `sample` tries each
//! instance's inverse map on the query point in turn and returns at the
//! first occupied one; `sample_channels` does the same and leaves that
//! instance's row in the output buffer; `get_bounds` encloses every
//! instance's mapped box. The instance list unrolls at generation time.

use model_wrap_core::emit::{self, Bounds, BoundsMapper};
use model_wrap_core::{Affine, Wrapper};
use volumetric_abi::host::{post_output, read_input, report_error};
use volumetric_abi::{OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput};
use walrus::ValType;
use walrus::ir::{BinaryOp, LoadKind, MemArg};

/// Unrolled instance ceiling: every instance adds a transform and a call to
/// the sample wrappers.
const MAX_INSTANCES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct Mirror {
    axis: Axis,
    offset: f64,
    keep_original: bool,
}

impl Default for Mirror {
    fn default() -> Self {
        Mirror {
            axis: Axis::X,
            offset: 0.0,
            keep_original: true,
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct Linear {
    count: u32,
    dx: f64,
    dy: f64,
    dz: f64,
}

impl Default for Linear {
    fn default() -> Self {
        Linear {
            count: 2,
            dx: 0.0,
            dy: 0.0,
            dz: 0.0,
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
struct Circular {
    count: u32,
    axis: Axis,
    cx: f64,
    cy: f64,
    cz: f64,
    sweep_deg: f64,
}

impl Default for Circular {
    fn default() -> Self {
        Circular {
            count: 6,
            axis: Axis::Z,
            cx: 0.0,
            cy: 0.0,
            cz: 0.0,
            sweep_deg: 360.0,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(default)]
struct PatternConfig {
    mirror: Option<Mirror>,
    linear: Option<Linear>,
    circular: Option<Circular>,
}

/// Every instance's placement (the map from the model's own frame to the
/// instance's), identity first, for a model with `spatial` transformable
/// axes. Each block patterns the placements the previous blocks produced.
fn instances(cfg: &PatternConfig, spatial: usize) -> Result<Vec<Affine>, String> {
    let mut maps = vec![Affine::IDENTITY];

    if let Some(mirror) = &cfg.mirror {
        if spatial == 2 && mirror.axis == Axis::Z {
            return Err("2D models mirror in-plane only: mirror axis must be x or y".to_string());
        }
        let reflect = Affine::mirror(mirror.axis.index(), mirror.offset);
        let images: Vec<Affine> = maps.iter().map(|t| t.then(&reflect)).collect();
        if !mirror.keep_original {
            maps.clear();
        }
        maps.extend(images);
    }

    if let Some(linear) = &cfg.linear {
        if linear.count < 1 {
            return Err("linear count must be at least 1".to_string());
        }
        let step = [linear.dx, linear.dy, linear.dz];
        maps = (0..linear.count)
            .flat_map(|k| {
                let shift = Affine::translation(step.map(|s| s * f64::from(k)));
                maps.iter().map(move |t| t.then(&shift)).collect::<Vec<_>>()
            })
            .collect();
    }

    if let Some(circular) = &cfg.circular {
        if spatial == 2 && circular.axis != Axis::Z {
            return Err("2D models rotate in-plane only: circular axis must be z".to_string());
        }
        if circular.count < 1 {
            return Err("circular count must be at least 1".to_string());
        }
        let full_turn = (circular.sweep_deg.abs() - 360.0).abs() < 1e-9;
        let step = if circular.count == 1 {
            0.0
        } else if full_turn {
            circular.sweep_deg / f64::from(circular.count)
        } else {
            circular.sweep_deg / f64::from(circular.count - 1)
        };
        let center = [circular.cx, circular.cy, circular.cz];
        maps = (0..circular.count)
            .flat_map(|k| {
                let turn =
                    Affine::rotation_about(circular.axis.index(), center, step * f64::from(k));
                maps.iter().map(move |t| t.then(&turn)).collect::<Vec<_>>()
            })
            .collect();
    }

    if maps.len() > MAX_INSTANCES {
        return Err(format!(
            "pattern has {} instances; at most {MAX_INSTANCES} are supported",
            maps.len()
        ));
    }
    Ok(maps)
}

fn transform_wasm(input_bytes: &[u8], cfg: &PatternConfig) -> Result<Vec<u8>, String> {
    let mut wrapper = Wrapper::parse(input_bytes)?;
    let n = wrapper.spatial();
    let maps: Vec<Affine> = instances(cfg, n)?
        .iter()
        .map(|map| map.restricted(n))
        .collect();
    let inverses = maps
        .iter()
        .map(|map| {
            map.inverse()
                .ok_or_else(|| "pattern placement is singular".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let occupancy_at = MemArg {
        align: 2,
        offset: 0,
    };

    // sample: the first instance containing the point decides.
    wrapper.wrap("sample", |body| {
        let pos = body.args[0];
        let (memory, original) = (body.memory, body.original);
        let result = body.locals.add(ValType::F32);
        let mut seq = body.builder.func_body();
        let src = emit::load_point(&mut seq, body.locals, memory, pos, n);
        seq.block(None, |block| {
            let done = block.id();
            for inverse in &inverses {
                emit::store_mapped_point(block, memory, pos, &src, inverse);
                block
                    .local_get(pos)
                    .call(original)
                    .local_tee(result)
                    .f32_const(0.5)
                    .binop(BinaryOp::F32Gt)
                    .br_if(done);
            }
        });
        seq.local_get(result);
    });

    // sample_channels: same search; the winning instance's row stays in the
    // output buffer (channel 0 is occupancy).
    wrapper.wrap("sample_channels", |body| {
        let (pos, out) = (body.args[0], body.args[1]);
        let (memory, original) = (body.memory, body.original);
        let mut seq = body.builder.func_body();
        let src = emit::load_point(&mut seq, body.locals, memory, pos, n);
        seq.block(None, |block| {
            let done = block.id();
            for inverse in &inverses {
                emit::store_mapped_point(block, memory, pos, &src, inverse);
                block.local_get(pos).local_get(out).call(original);
                block
                    .local_get(out)
                    .load(memory, LoadKind::F32, occupancy_at)
                    .f32_const(0.5)
                    .binop(BinaryOp::F32Gt)
                    .br_if(done);
            }
        });
    });

    // get_bounds: enclose every instance's box.
    wrapper.wrap_bounds(|seq, locals, memory, out| {
        let source = Bounds::new(locals, n);
        emit::load_bounds(seq, memory, out, &source);
        let enclosure = Bounds::new(locals, n);
        let instance = Bounds::new(locals, n);
        let mapper = BoundsMapper::new(locals, n);
        mapper.map(seq, &source, &maps[0], &enclosure);
        for map in &maps[1..] {
            mapper.map(seq, &source, map, &instance);
            emit::fold_bounds(seq, &enclosure, &instance);
        }
        emit::store_bounds(seq, memory, out, &enclosure);
    });

    Ok(wrapper.finish())
}

#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let buf = read_input(0);
    let cfg_buf = read_input(1);
    let cfg = if cfg_buf.is_empty() {
        PatternConfig::default()
    } else {
        match ciborium::de::from_reader(std::io::Cursor::new(&cfg_buf)) {
            Ok(cfg) => cfg,
            Err(e) => {
                report_error(&format!("invalid configuration: {e}"));
                return;
            }
        }
    };
    match transform_wasm(&buf, &cfg) {
        Ok(output) => post_output(0, &output),
        Err(e) => report_error(&format!("pattern failed: {e}")),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ ? mirror: { axis: "x" / "y" / "z" .default "x", offset: float .default 0.0, keep_original: bool .default true }, ? linear: { count: int .ge 1 .default 2, dx: float .default 0.0, dy: float .default 0.0, dz: float .default 0.0 }, ? circular: { count: int .ge 1 .default 6, axis: "x" / "y" / "z" .default "z", cx: float .default 0.0, cy: float .default 0.0, cz: float .default 0.0, sweep_deg: float .default 360.0 } }"#.to_string();
        OperatorMetadata {
            name: "pattern_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Pattern".to_string(),
            description: "Repeat a model as mirrored, linear and circular copies in one step."
                .to_string(),
            category: "Transforms".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<rect x="3" y="3" width="7" height="7" rx="1"/>"##,
                r##"<rect x="14" y="3" width="7" height="7" rx="1"/>"##,
                r##"<rect x="3" y="14" width="7" height="7" rx="1"/>"##,
                r##"<rect x="14" y="14" width="7" height="7" rx="1"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::ModelWASM,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec!["Model".to_string(), "Config".to_string()],
            outputs: vec![OperatorMetadataOutput::ModelWASM],
            output_names: vec![],
        }
    })
}

#![doc = include_str!("../README.md")]

use ply_core::convert::place;
use ply_core::{Element, is_ply, read_ply};
use volumetric_abi::splat::{Provenance, Splat, SplatKind, sh_rest_per_point};
use volumetric_abi::viewset::ViewSet;
#[cfg(target_arch = "wasm32")]
use volumetric_abi::{
    OperatorMetadata, OperatorMetadataInput, OperatorMetadataOutput,
    host::{post_output, post_warning, read_input, report_error},
    splat::encode_splat,
    viewset::decode_viewset,
};

/// The view set the splat was trained from, with the bytes it travels as
/// (their blake3 hash is the engine's content hash of the asset).
pub struct TrainingViews<'a> {
    pub set: &'a ViewSet,
    pub bytes: &'a [u8],
}

/// The log-scale a surfel's third axis gets when the file has none: one
/// nanometre.
pub const SURFEL_THIRD_LOG_SCALE: f32 = -20.72;

/// Below this ratio of the third scale to the other two, for the median
/// primitive, `auto` reads surfels.
const SURFEL_RATIO: f64 = 0.01;

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct SplatImportConfig {
    pub kind: String,
    pub scale: f64,
    pub center: bool,
    pub session: String,
    pub field: String,
    pub setup: String,
    pub views_hash: String,
    pub training: String,
}

impl Default for SplatImportConfig {
    fn default() -> Self {
        Self {
            kind: "auto".to_string(),
            scale: 1.0,
            center: false,
            session: String::new(),
            field: String::new(),
            setup: String::new(),
            views_hash: String::new(),
            training: String::new(),
        }
    }
}

/// Reads the file as a splat, with the warnings a host should relay. With
/// `views` the splat takes the set's world frame and provenance and the
/// hash of its bytes; the config's strings override where non-empty.
pub fn import(
    bytes: &[u8],
    views: Option<TrainingViews<'_>>,
    config: &SplatImportConfig,
) -> Result<(Splat, Vec<String>), String> {
    if !is_ply(bytes) {
        return Err(
            "unrecognised splat format: the 3DGS PLY layout is the supported format, and this \
             file does not start with its `ply` magic line"
                .to_string(),
        );
    }
    if !(config.scale.is_finite() && config.scale > 0.0) {
        return Err(format!("scale must be positive, got {}", config.scale));
    }
    let wanted = match config.kind.as_str() {
        "auto" => None,
        "gaussian" => Some(SplatKind::Gaussian3d),
        "surfel" => Some(SplatKind::Surfel2d),
        other => {
            return Err(format!(
                "kind must be auto, gaussian or surfel, not {other:?}"
            ));
        }
    };
    let mut warnings = Vec::new();
    let file = read_ply(bytes)?;
    let faces = file.element("face").map_or(0, |f| f.count);
    if faces > 0 {
        warnings.push(format!("the file also carries {faces} face(s), ignored"));
    }
    let vertex = file
        .element("vertex")
        .ok_or_else(|| "the file has no `vertex` element".to_string())?;
    let n = vertex.count;
    if n == 0 {
        return Err("the file has no primitives".to_string());
    }
    let column = |name: &str| -> Result<&[f64], String> {
        vertex.scalar(name).ok_or_else(|| {
            format!(
                "the vertex element has no scalar property {name:?}; a splat in the 3DGS \
                 layout carries x y z, f_dc_0..2, opacity, scale_0..2 and rot_0..3"
            )
        })
    };

    let mut means = interleave(&[column("x")?, column("y")?, column("z")?]);
    place(&mut means, config.center, config.scale)?;
    let sh0 = interleave(&[column("f_dc_0")?, column("f_dc_1")?, column("f_dc_2")?]);
    let (sh_degree, sh_rest) = sh_rest_columns(vertex, n)?;
    let opacities = column("opacity")?.to_vec();
    let scale_0 = column("scale_0")?;
    let scale_1 = column("scale_1")?;
    let scale_2 = vertex.scalar("scale_2");
    let quats = interleave(&[
        column("rot_0")?,
        column("rot_1")?,
        column("rot_2")?,
        column("rot_3")?,
    ]);

    let kind = match wanted {
        Some(kind) => kind,
        None => {
            let detected = detect_kind(scale_0, scale_1, scale_2);
            warnings.push(format!(
                "kind auto: read as {}s ({})",
                detected.name(),
                match scale_2 {
                    None => "the file has two scales".to_string(),
                    Some(_) => format!(
                        "the median third scale is {:.3} of the others",
                        median_third_ratio(scale_0, scale_1, scale_2.unwrap())
                    ),
                }
            ));
            detected
        }
    };
    let log_scale = config.scale.ln();
    let mut scales = Vec::with_capacity(3 * n);
    for i in 0..n {
        scales.push(scale_0[i] + log_scale);
        scales.push(scale_1[i] + log_scale);
        scales.push(match scale_2 {
            Some(s2) => s2[i] + log_scale,
            None => f64::from(SURFEL_THIRD_LOG_SCALE),
        });
    }

    let normals = match (
        vertex.scalar("nx"),
        vertex.scalar("ny"),
        vertex.scalar("nz"),
    ) {
        (Some(nx), Some(ny), Some(nz)) => {
            let normals = interleave(&[nx, ny, nz]);
            if normals.iter().all(|v| *v == 0.0) {
                warnings.push("normals are all zero, dropped".to_string());
                Vec::new()
            } else {
                normals
            }
        }
        _ => Vec::new(),
    };

    let to_f32 = |v: Vec<f64>| -> Vec<f32> { v.into_iter().map(|x| x as f32).collect() };
    let (world, mut provenance, mut views_hash) = match &views {
        Some(views) => (
            views.set.world.clone(),
            views.set.provenance.clone(),
            blake3::hash(views.bytes).to_hex().to_string(),
        ),
        None => (Default::default(), Provenance::default(), String::new()),
    };
    let override_with = |field: &mut String, value: &str| {
        if !value.is_empty() {
            *field = value.to_string();
        }
    };
    override_with(&mut provenance.session, &config.session);
    override_with(&mut provenance.field, &config.field);
    override_with(&mut provenance.setup, &config.setup);
    override_with(&mut views_hash, &config.views_hash);
    provenance.origin = String::new();
    provenance.tools.push(format!(
        "splat_import_operator {} kind {}",
        env!("CARGO_PKG_VERSION"),
        config.kind
    ));
    let splat = Splat {
        world,
        provenance,
        views_hash,
        training: config.training.clone(),
        count: n as u32,
        means: to_f32(means),
        scales: to_f32(scales),
        quats: to_f32(quats),
        opacities: to_f32(opacities),
        sh0: to_f32(sh0),
        sh_rest: to_f32(sh_rest),
        normals: to_f32(normals),
        ..Splat::empty(kind, sh_degree)
    };
    splat.validate()?;
    Ok((splat, warnings))
}

/// The `f_rest_k` columns in order, and the SH degree they imply.
fn sh_rest_columns(vertex: &Element, n: usize) -> Result<(u32, Vec<f64>), String> {
    let mut columns: Vec<&[f64]> = Vec::new();
    while let Some(c) = vertex.scalar(&format!("f_rest_{}", columns.len())) {
        columns.push(c);
    }
    let degree = (0..=3u32)
        .find(|d| sh_rest_per_point(*d) == columns.len())
        .ok_or_else(|| {
            format!(
                "{} f_rest properties do not match an SH degree (0, 9, 24 or 45 expected)",
                columns.len()
            )
        })?;
    let mut rest = Vec::with_capacity(n * columns.len());
    for i in 0..n {
        for c in &columns {
            rest.push(c[i]);
        }
    }
    Ok((degree, rest))
}

/// The median over primitives of `exp(scale_2 - min(scale_0, scale_1))`.
fn median_third_ratio(scale_0: &[f64], scale_1: &[f64], scale_2: &[f64]) -> f64 {
    let mut ratios: Vec<f64> = (0..scale_2.len())
        .map(|i| (scale_2[i] - scale_0[i].min(scale_1[i])).exp())
        .collect();
    ratios.sort_by(f64::total_cmp);
    ratios[ratios.len() / 2]
}

fn detect_kind(scale_0: &[f64], scale_1: &[f64], scale_2: Option<&[f64]>) -> SplatKind {
    match scale_2 {
        None => SplatKind::Surfel2d,
        Some(s2) if median_third_ratio(scale_0, scale_1, s2) < SURFEL_RATIO => SplatKind::Surfel2d,
        Some(_) => SplatKind::Gaussian3d,
    }
}

fn interleave(columns: &[&[f64]]) -> Vec<f64> {
    let n = columns[0].len();
    let mut out = Vec::with_capacity(n * columns.len());
    for i in 0..n {
        for c in columns {
            out.push(c[i]);
        }
    }
    out
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn run() {
    let bytes = read_input(0);
    let views_bytes = read_input(1);
    let views = if views_bytes.is_empty() {
        None
    } else {
        match decode_viewset(&views_bytes) {
            Ok(set) => Some(set),
            Err(e) => {
                report_error(&format!("input 1 is not a view set: {e}"));
                return;
            }
        }
    };
    let config = {
        let cfg = read_input(2);
        if cfg.is_empty() {
            SplatImportConfig::default()
        } else {
            match ciborium::de::from_reader(std::io::Cursor::new(&cfg)) {
                Ok(config) => config,
                Err(e) => {
                    report_error(&format!("invalid configuration: {e}"));
                    return;
                }
            }
        }
    };
    let training = views.as_ref().map(|set| TrainingViews {
        set,
        bytes: &views_bytes,
    });
    match import(&bytes, training, &config) {
        Ok((splat, warnings)) => {
            for warning in &warnings {
                post_warning(warning);
            }
            post_output(0, &encode_splat(&splat));
        }
        Err(e) => report_error(&format!("splat import failed: {e}")),
    }
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn get_metadata() -> i64 {
    static METADATA: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    volumetric_abi::metadata_reply(&METADATA, || {
        let schema = r#"{ kind: "auto" / "gaussian" / "surfel" .default "auto", scale: float .default 1.0, center: bool .default false, session: tstr .default "", field: tstr .default "", setup: tstr .default "", views_hash: tstr .default "", training: tstr .default "" }"#
            .to_string();
        OperatorMetadata {
            name: "splat_import_operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            docs: include_str!("../README.md").to_string(),
            display_name: "Splat Import".to_string(),
            description: "Read a trained Gaussian splat (3DGS PLY) as a Splat value.".to_string(),
            category: "Import".to_string(),
            icon_svg: volumetric_abi::icon_svg!(
                r##"<ellipse cx="9" cy="9" rx="5" ry="3" transform="rotate(-30 9 9)"/>"##,
                r##"<ellipse cx="15" cy="14" rx="5" ry="2.5" transform="rotate(20 15 14)"/>"##,
                r##"<ellipse cx="8" cy="17" rx="3" ry="2"/>"##,
            )
            .to_string(),
            inputs: vec![
                OperatorMetadataInput::Blob,
                OperatorMetadataInput::ViewSet,
                OperatorMetadataInput::CBORConfiguration(schema),
            ],
            variadic_input: None,
            input_names: vec![
                "Splat file".to_string(),
                "Views".to_string(),
                "Config".to_string(),
            ],
            outputs: vec![OperatorMetadataOutput::Splat],
            output_names: vec!["Splat".to_string()],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ply_core::{Element, Format, PlyFile, Property, PropertyData, PropertyKind, ScalarType};

    fn scalar(name: &str, values: Vec<f64>) -> Property {
        Property {
            name: name.to_string(),
            kind: PropertyKind::Scalar(ScalarType::F32),
            data: PropertyData::Scalar(values),
        }
    }

    /// Three primitives along x at SH degree 1, with the third scale
    /// `third` (log) and normals as given.
    fn splat_ply(third: Option<f64>, normals: Option<[f64; 3]>) -> Vec<u8> {
        let n = 3;
        let mut properties = vec![
            scalar("x", vec![0.0, 1.0, 2.0]),
            scalar("y", vec![0.0; n]),
            scalar("z", vec![1.0; n]),
        ];
        if let Some(nrm) = normals {
            properties.push(scalar("nx", vec![nrm[0]; n]));
            properties.push(scalar("ny", vec![nrm[1]; n]));
            properties.push(scalar("nz", vec![nrm[2]; n]));
        }
        properties.push(scalar("f_dc_0", vec![1.0; n]));
        properties.push(scalar("f_dc_1", vec![0.0; n]));
        properties.push(scalar("f_dc_2", vec![-1.0; n]));
        for k in 0..9 {
            properties.push(scalar(&format!("f_rest_{k}"), vec![k as f64; n]));
        }
        properties.push(scalar("opacity", vec![0.0, 3.0, -3.0]));
        properties.push(scalar("scale_0", vec![-3.0; n]));
        properties.push(scalar("scale_1", vec![-4.0; n]));
        if let Some(third) = third {
            properties.push(scalar("scale_2", vec![third; n]));
        }
        properties.push(scalar("rot_0", vec![2.0; n]));
        properties.push(scalar("rot_1", vec![0.0; n]));
        properties.push(scalar("rot_2", vec![0.0; n]));
        properties.push(scalar("rot_3", vec![0.0; n]));
        ply_core::write_ply(&PlyFile {
            format: Format::BinaryLittleEndian,
            comments: vec![],
            obj_info: vec![],
            elements: vec![Element {
                name: "vertex".to_string(),
                count: n,
                properties,
            }],
        })
        .unwrap()
    }

    #[test]
    fn reads_the_3dgs_layout_as_gaussians() {
        let (splat, warnings) = import(
            &splat_ply(Some(-3.5), Some([0.0; 3])),
            None,
            &SplatImportConfig::default(),
        )
        .unwrap();
        assert_eq!(splat.kind, SplatKind::Gaussian3d);
        assert_eq!(splat.sh_degree, 1);
        assert_eq!(splat.count, 3);
        assert_eq!(splat.mean(2), [2.0, 0.0, 1.0]);
        assert_eq!(
            &splat.sh_rest[9..18],
            &(0..9).map(|k| k as f32).collect::<Vec<_>>()[..]
        );
        assert!((splat.opacity(1) - 0.9526).abs() < 1e-3);
        assert_eq!(splat.quat(0), [1.0, 0.0, 0.0, 0.0]);
        assert!(splat.normals.is_empty(), "all-zero normals are dropped");
        assert!(warnings.iter().any(|w| w.contains("normals are all zero")));
        assert!(
            warnings.iter().any(|w| w.contains("read as gaussians")),
            "{warnings:?}"
        );
        assert_eq!(splat.provenance.tools.len(), 1);
    }

    #[test]
    fn a_thin_third_scale_or_a_missing_one_reads_as_surfels() {
        let (thin, _) = import(
            &splat_ply(Some(-12.0), None),
            None,
            &SplatImportConfig::default(),
        )
        .unwrap();
        assert_eq!(thin.kind, SplatKind::Surfel2d);
        let (two, warnings) =
            import(&splat_ply(None, None), None, &SplatImportConfig::default()).unwrap();
        assert_eq!(two.kind, SplatKind::Surfel2d);
        assert_eq!(two.scales[2], SURFEL_THIRD_LOG_SCALE);
        assert!(warnings.iter().any(|w| w.contains("two scales")));
        let forced = SplatImportConfig {
            kind: "gaussian".to_string(),
            ..Default::default()
        };
        let (g, warnings) = import(&splat_ply(Some(-12.0), None), None, &forced).unwrap();
        assert_eq!(g.kind, SplatKind::Gaussian3d);
        assert!(warnings.is_empty());
    }

    #[test]
    fn scale_and_center_move_the_centres_and_the_scales() {
        let config = SplatImportConfig {
            scale: 2.0,
            center: true,
            session: "s".to_string(),
            views_hash: "abc".to_string(),
            ..Default::default()
        };
        let (splat, _) =
            import(&splat_ply(Some(-3.5), Some([0.0, 0.0, 1.0])), None, &config).unwrap();
        // Centred on x = 1, z = 1, then doubled.
        assert_eq!(splat.mean(0), [-2.0, 0.0, 0.0]);
        assert_eq!(splat.mean(2), [2.0, 0.0, 0.0]);
        assert!((splat.scales[0] - (-3.0 + 2f32.ln())).abs() < 1e-6);
        assert_eq!(splat.normals.len(), 9);
        assert_eq!(splat.provenance.session, "s");
        assert_eq!(splat.views_hash, "abc");
    }

    #[test]
    fn refuses_clouds_bad_kinds_and_bad_scales() {
        let cloud = ply_core::write_ply(&PlyFile {
            format: Format::Ascii,
            comments: vec![],
            obj_info: vec![],
            elements: vec![Element {
                name: "vertex".to_string(),
                count: 1,
                properties: vec![
                    scalar("x", vec![0.0]),
                    scalar("y", vec![0.0]),
                    scalar("z", vec![0.0]),
                ],
            }],
        })
        .unwrap();
        let err = import(&cloud, None, &SplatImportConfig::default()).unwrap_err();
        assert!(err.contains("f_dc_0"), "{err}");
        let err = import(b"solid x\n", None, &SplatImportConfig::default()).unwrap_err();
        assert!(err.contains("3DGS PLY layout"), "{err}");
        let bad = SplatImportConfig {
            kind: "flat".to_string(),
            ..Default::default()
        };
        assert!(
            import(&splat_ply(None, None), None, &bad)
                .unwrap_err()
                .contains("kind")
        );
        let bad = SplatImportConfig {
            scale: -1.0,
            ..Default::default()
        };
        assert!(
            import(&splat_ply(None, None), None, &bad)
                .unwrap_err()
                .contains("scale")
        );
    }

    #[test]
    fn the_view_set_gives_the_world_the_provenance_and_the_hash() {
        use volumetric_abi::viewset::{Provenance, VIEWSET_SCHEMA, WorldFrame, encode_viewset};
        let set = ViewSet {
            schema: VIEWSET_SCHEMA,
            world: WorldFrame {
                up: [0.0, 0.0, 1.0],
            },
            provenance: Provenance {
                session: "chair".to_string(),
                field: "cards".to_string(),
                tools: vec!["survey".to_string()],
                origin: "/somewhere".to_string(),
                ..Provenance::default()
            },
            cameras: vec![],
            views: vec![],
            markers: vec![],
            board: None,
        };
        let bytes = encode_viewset(&set);
        let config = SplatImportConfig {
            setup: "on the card".to_string(),
            ..Default::default()
        };
        let (splat, _) = import(
            &splat_ply(Some(-3.5), None),
            Some(TrainingViews {
                set: &set,
                bytes: &bytes,
            }),
            &config,
        )
        .unwrap();
        assert_eq!(splat.world.up, [0.0, 0.0, 1.0]);
        assert_eq!(splat.provenance.session, "chair");
        assert_eq!(splat.provenance.field, "cards");
        assert_eq!(splat.provenance.setup, "on the card");
        assert_eq!(splat.provenance.origin, "");
        assert_eq!(splat.provenance.tools.len(), 2);
        assert_eq!(splat.views_hash, blake3::hash(&bytes).to_hex().to_string());
        assert_eq!(splat.views_hash.len(), 64);
    }
}

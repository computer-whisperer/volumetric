//! Fit a feature to a PLY point cloud natively, exactly as the operator
//! would, and print the feature and its statistics.
//!
//!   fit_cloud <cloud.ply> [--kind plane|line|point|sphere|cylinder]
//!             [--seed x,y,z | --seed-line x,y,z,dx,dy,dz] [--radius r]
//!             [--tolerance t] [--stride n] [--normals]
//!
//! `--normals` estimates normals first (the Cloud Normals kernel, defaults),
//! which a cylinder fit without a seed line needs.

use cloud_fit_operator::{Cloud, CloudFitConfig, Feature, Kind, fit, fit_map};
use volumetric_abi::subspace::Subspace;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: fit_cloud <cloud.ply> [options]");
    let mut kind = Kind::Plane;
    let mut seed: Option<Subspace> = None;
    let mut config = CloudFitConfig::default();
    let mut stride = 1u32;
    let mut normals = false;
    let mut i = 2;
    let numbers =
        |s: &str| -> Vec<f64> { s.split(',').map(|v| v.parse().expect("number")).collect() };
    while i < args.len() {
        match args[i].as_str() {
            "--kind" => {
                kind = match args[i + 1].as_str() {
                    "plane" => Kind::Plane,
                    "line" => Kind::Line,
                    "point" => Kind::Point,
                    "sphere" => Kind::Sphere,
                    "cylinder" => Kind::Cylinder,
                    other => panic!("unknown kind {other}"),
                };
                i += 2;
            }
            "--seed" => {
                seed = Some(Subspace::point(numbers(&args[i + 1])));
                i += 2;
            }
            "--seed-line" => {
                let v = numbers(&args[i + 1]);
                let len = (v[3] * v[3] + v[4] * v[4] + v[5] * v[5]).sqrt();
                seed = Some(Subspace {
                    dimensions: 3,
                    origin: v[..3].to_vec(),
                    basis: vec![v[3] / len, v[4] / len, v[5] / len],
                });
                i += 2;
            }
            "--radius" => {
                config.search_radius = args[i + 1].parse().expect("radius");
                i += 2;
            }
            "--tolerance" => {
                config.tolerance = args[i + 1].parse().expect("tolerance");
                i += 2;
            }
            "--stride" => {
                stride = args[i + 1].parse().expect("stride");
                i += 2;
            }
            "--normals" => {
                normals = true;
                i += 1;
            }
            other => panic!("unknown option {other}"),
        }
    }
    config.kind = kind;

    let bytes = std::fs::read(path).expect("read cloud");
    let import = point_cloud_import_operator::PointCloudImportConfig {
        stride,
        ..point_cloud_import_operator::PointCloudImportConfig::default()
    };
    let start = std::time::Instant::now();
    let (mut mesh, _) = point_cloud_import_operator::import(&bytes, &import).expect("import");
    println!(
        "{} points imported in {:.2?}",
        mesh.node_count(),
        start.elapsed()
    );
    if normals {
        let start = std::time::Instant::now();
        let points: Vec<[f64; 3]> = (0..mesh.node_count())
            .map(|i| mesh.node_position(i))
            .collect();
        let (estimated, degenerate) = cloud_core::normals::estimate(
            &points,
            &cloud_core::normals::CloudNormalsConfig::default(),
        )
        .expect("normals");
        mesh.node_fields.push(volumetric_abi::fea::FeaField {
            name: volumetric_abi::fea::NORMAL_FIELD_NAME.to_string(),
            components: 3,
            data: estimated.iter().flat_map(|n| n.iter().copied()).collect(),
        });
        println!(
            "normals estimated in {:.2?} ({degenerate} degenerate)",
            start.elapsed()
        );
    }
    let cloud = Cloud::from_mesh(&mesh);
    let start = std::time::Instant::now();
    match fit(&cloud, seed.as_ref(), &config) {
        Ok(result) => {
            println!("fit in {:.2?}", start.elapsed());
            match result.feature {
                Feature::Point { center } => println!("point {center:?}"),
                Feature::Line { origin, direction } => {
                    println!("line through {origin:?} along {direction:?}")
                }
                Feature::Plane { origin, normal } => {
                    println!("plane through {origin:?} with normal {normal:?}")
                }
                Feature::Sphere { center, radius } => println!("sphere at {center:?} r {radius}"),
                Feature::Cylinder {
                    origin,
                    direction,
                    radius,
                } => println!("cylinder axis through {origin:?} along {direction:?} r {radius}"),
            }
            println!(
                "subspace origin {:?} basis {:?}",
                result.subspace.origin, result.subspace.basis
            );
            for (key, value) in fit_map(&result) {
                println!("  {key} = {value}");
            }
        }
        Err(e) => {
            println!("fit failed: {e}");
            std::process::exit(1);
        }
    }
}

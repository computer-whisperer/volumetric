use std::path::PathBuf;

use volumetric::adaptive_surface_nets_2::stage4::research::attempt_runner::dump_attempt_sample_cloud;
use volumetric::adaptive_surface_nets_2::stage4::research::experiments::ml_policy::{
    dump_ml_policy_sample_cloud, MlPolicyDumpKind,
};

fn main() {
    let mut attempt: Option<u8> = None;
    let mut output: Option<PathBuf> = None;
    let mut ml_policy: Option<MlPolicyDumpKind> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--attempt" => {
                if let Some(value) = args.next() {
                    attempt = value.parse::<u8>().ok();
                }
            }
            "--ml-policy" => {
                if let Some(value) = args.next() {
                    ml_policy = match value.as_str() {
                        "directional" => Some(MlPolicyDumpKind::Directional),
                        "octant-argmax" => Some(MlPolicyDumpKind::OctantArgmax),
                        "octant-lerp" => Some(MlPolicyDumpKind::OctantLerp),
                        _ => None,
                    };
                }
            }
            "--out" => {
                if let Some(value) = args.next() {
                    output = Some(PathBuf::from(value));
                }
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {}
        }
    }

    if let Some(kind) = ml_policy {
        let output = output.unwrap_or_else(|| match kind {
            MlPolicyDumpKind::Directional => PathBuf::from("sample_cloud_ml_directional.cbor"),
            MlPolicyDumpKind::OctantArgmax => PathBuf::from("sample_cloud_ml_octant_argmax.cbor"),
            MlPolicyDumpKind::OctantLerp => PathBuf::from("sample_cloud_ml_octant_lerp.cbor"),
        });
        dump_ml_policy_sample_cloud(kind, &output);
        return;
    }

    let Some(attempt) = attempt else {
        print_help();
        return;
    };
    let output = output.unwrap_or_else(|| PathBuf::from(format!("sample_cloud_attempt{attempt}.cbor")));

    dump_attempt_sample_cloud(attempt, &output);
}

fn print_help() {
    eprintln!("Usage:");
    eprintln!("  sample_cloud_dump --attempt <0|1|2> [--out <file.cbor>]");
    eprintln!("  sample_cloud_dump --ml-policy <directional|octant-argmax|octant-lerp> [--out <file.cbor>]");
}

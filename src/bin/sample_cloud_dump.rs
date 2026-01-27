use std::path::PathBuf;

use volumetric::adaptive_surface_nets_2::stage4::research::attempt_runner::dump_attempt_sample_cloud;
use volumetric::adaptive_surface_nets_2::stage4::research::experiments::ml_policy::{
    dump_ml_policy_sample_cloud, MlPolicyDumpKind,
};
use volumetric::adaptive_surface_nets_2::stage4::research::experiments::rnn_policy::{
    load_and_dump_rnn_policy, run_corner_diagnostic, run_reward_sweep, train_and_save_rnn_policy, DEFAULT_MODEL_PATH,
};

#[derive(Clone, Copy)]
enum RnnCommand {
    Train,
    Dump,
    Sweep,
    Diagnostic,
}

fn main() {
    let mut attempt: Option<u8> = None;
    let mut output: Option<PathBuf> = None;
    let mut ml_policy: Option<MlPolicyDumpKind> = None;
    let mut rnn_command: Option<RnnCommand> = None;
    let mut model_path: Option<PathBuf> = None;
    let mut use_discrete = false;

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
            "--rnn-policy" => {
                if let Some(value) = args.next() {
                    rnn_command = match value.as_str() {
                        "train" => Some(RnnCommand::Train),
                        "dump" => Some(RnnCommand::Dump),
                        "sweep" => Some(RnnCommand::Sweep),
                        "diag" => Some(RnnCommand::Diagnostic),
                        _ => None,
                    };
                }
            }
            "--model" => {
                if let Some(value) = args.next() {
                    model_path = Some(PathBuf::from(value));
                }
            }
            "--out" => {
                if let Some(value) = args.next() {
                    output = Some(PathBuf::from(value));
                }
            }
            "--discrete" => {
                use_discrete = true;
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            _ => {}
        }
    }

    // Handle RNN policy commands
    if let Some(cmd) = rnn_command {
        let model = model_path.unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));
        match cmd {
            RnnCommand::Train => {
                train_and_save_rnn_policy(&model);
            }
            RnnCommand::Dump => {
                let output = output.unwrap_or_else(|| PathBuf::from("sample_cloud_rnn_trained.cbor"));
                load_and_dump_rnn_policy(&model, &output, use_discrete);
            }
            RnnCommand::Sweep => {
                run_reward_sweep();
            }
            RnnCommand::Diagnostic => {
                run_corner_diagnostic();
            }
        }
        return;
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
    eprintln!("  sample_cloud_dump --rnn-policy train [--model <file.bin>]");
    eprintln!("  sample_cloud_dump --rnn-policy dump [--model <file.bin>] [--out <file.cbor>] [--discrete]");
    eprintln!("  sample_cloud_dump --rnn-policy sweep    # Run reward configuration sweep");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --discrete    Use discrete corner sampling instead of weighted positions");
}

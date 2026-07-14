//! One-command wasm artifact build: the baseline modules for every
//! operator and model (`cargo build-wasm`), the wasm32-wasip1-threads
//! variants of the heavy fea solvers, and the dual-blob packing that
//! embeds each variant into its baseline artifact (see
//! `volumetric::wasm::variant`).
//!
//! Run as `cargo wasm-dist`. Prefer it over bare `cargo build-wasm`
//! whenever the fea operators matter: the packed section lives inside the
//! baseline artifact file, so a bare baseline rebuild that relinks drops
//! the threaded variants until the next pack.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Operators additionally built for wasm32-wasip1-threads (with their
/// `threaded` feature) and packed as dual blobs.
const THREADED_OPERATORS: &[&str] = &["fea_solve_operator", "fea_inverse_operator"];

/// The threaded link needs an explicit shared-memory cap and the target's
/// default (1 GiB) is too small for large FEA meshes. The maximum is
/// reserved address space, not committed memory — 4 GiB costs nothing up
/// front.
const THREADED_RUSTFLAGS: &str = "-C link-arg=--max-memory=4294967296";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/wasm_dist sits two levels below the workspace root")
        .to_path_buf()
}

fn run(mut cmd: Command) -> anyhow::Result<()> {
    let rendered = format!("{cmd:?}");
    let status = cmd.status()?;
    anyhow::ensure!(status.success(), "command failed: {rendered}");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let root = workspace_root();

    let mut baseline = Command::new("cargo");
    baseline.current_dir(&root).arg("build-wasm");
    run(baseline)?;

    let mut threaded = Command::new("cargo");
    threaded
        .current_dir(&root)
        .args(["build", "--release", "--target", "wasm32-wasip1-threads"]);
    for op in THREADED_OPERATORS {
        threaded.args(["-p", op, "--features", &format!("{op}/threaded")]);
    }
    // Extend ambient RUSTFLAGS rather than clobbering them. Build scripts
    // don't see RUSTFLAGS when --target is set, so the wasm link flag can't
    // leak into host builds.
    let mut rustflags = std::env::var("RUSTFLAGS").unwrap_or_default();
    if !rustflags.is_empty() {
        rustflags.push(' ');
    }
    rustflags.push_str(THREADED_RUSTFLAGS);
    threaded.env("RUSTFLAGS", rustflags);
    run(threaded)?;

    for op in THREADED_OPERATORS {
        let baseline_path = root.join(format!("target/wasm32-unknown-unknown/release/{op}.wasm"));
        let variant_path = root.join(format!("target/wasm32-wasip1-threads/release/{op}.wasm"));
        let baseline_bytes = std::fs::read(&baseline_path)?;
        let variant_bytes = std::fs::read(&variant_path)?;
        let packed =
            volumetric::wasm::variant::embed_threaded_variant(&baseline_bytes, &variant_bytes)
                .map_err(|e| anyhow::anyhow!("packing {op}: {e}"))?;
        // Skip identical rewrites so artifact mtimes (and everything
        // fingerprinting them, e.g. volumetric_assets) stay quiet.
        let changed = packed != baseline_bytes;
        if changed {
            std::fs::write(&baseline_path, &packed)?;
        }
        println!(
            "{op}: {:.2} MB packed ({:.2} MB threaded variant){}",
            packed.len() as f64 / 1e6,
            variant_bytes.len() as f64 / 1e6,
            if changed { "" } else { " [unchanged]" },
        );
    }
    Ok(())
}

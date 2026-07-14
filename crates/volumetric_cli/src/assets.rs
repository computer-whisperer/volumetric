//! Bundled asset access: the `assets` subcommand and name-based resolution
//! of model/operator specs against the assets compiled into the binary.
//!
//! Anywhere the CLI takes a model or operator, the spec is resolved as a
//! filesystem path first; if no file exists there, it is looked up as a
//! bundled asset name (the same set the GUI offers). This keeps CLI scripts
//! portable — no target/wasm32-unknown-unknown paths required.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::path::Path;

/// Magic bytes every WASM module starts with (`\0asm`).
const WASM_MAGIC: &[u8; 4] = b"\0asm";

/// Returns an error if `bytes` is not a WASM module. `what` names the input
/// in the message (e.g. "model", "operator").
pub fn ensure_wasm(bytes: &[u8], what: &str, origin: &str) -> Result<()> {
    if bytes.starts_with(WASM_MAGIC) {
        Ok(())
    } else {
        anyhow::bail!(
            "{origin} is not a WASM module (missing \\0asm magic); expected {what} bytes. \
             Use project-add-asset for non-WASM assets such as Lua source."
        )
    }
}

/// Resolve a model spec: a path to a `.wasm` file, or the name of a bundled
/// model. Returns (asset name, bytes).
pub fn resolve_model_spec(spec: &str) -> Result<(String, Vec<u8>)> {
    resolve_spec(spec, "model", volumetric_assets::get_model, || {
        volumetric_assets::models()
    })
}

/// Resolve an operator spec: a path to a `.wasm` file, or the name of a
/// bundled operator (e.g. `extrude_operator`). Returns (asset name, bytes).
pub fn resolve_operator_spec(spec: &str) -> Result<(String, Vec<u8>)> {
    resolve_spec(spec, "operator", volumetric_assets::get_operator, || {
        volumetric_assets::operators()
    })
}

fn resolve_spec(
    spec: &str,
    what: &str,
    lookup: fn(&str) -> Option<&'static volumetric_assets::BundledAsset>,
    all: fn() -> &'static [volumetric_assets::BundledAsset],
) -> Result<(String, Vec<u8>)> {
    let path = Path::new(spec);
    if path.exists() {
        let bytes =
            std::fs::read(path).with_context(|| format!("Failed to read {what} file {spec}"))?;
        ensure_wasm(&bytes, what, spec)?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(what)
            .to_string();
        return Ok((name, bytes));
    }

    if let Some(asset) = lookup(spec) {
        return Ok((asset.name.to_string(), asset.bytes.to_vec()));
    }

    let available: Vec<&str> = all().iter().map(|a| a.name).collect();
    anyhow::bail!(
        "no file at '{spec}' and no bundled {what} with that name. \
         Bundled {what}s: {}",
        available.join(", ")
    )
}

// === Assets Subcommand ===

#[derive(Parser, Debug)]
pub struct AssetsArgs {
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Serialize)]
struct AssetInfo {
    name: &'static str,
    version: &'static str,
    size_bytes: usize,
}

#[derive(Debug, Serialize)]
struct AssetsOutput {
    models: Vec<AssetInfo>,
    operators: Vec<AssetInfo>,
}

pub fn run_assets(args: AssetsArgs) -> Result<()> {
    // Name/version/size only: display metadata lives in each module's own
    // `get_metadata()` (see `info --input <module.wasm>` for one module).
    let to_info = |a: &volumetric_assets::BundledAsset| AssetInfo {
        name: a.name,
        version: a.version,
        size_bytes: a.bytes.len(),
    };
    let output = AssetsOutput {
        models: volumetric_assets::models().iter().map(to_info).collect(),
        operators: volumetric_assets::operators().iter().map(to_info).collect(),
    };

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&output).context("Failed to serialize JSON")?
        );
    } else {
        println!("Bundled models ({}):", output.models.len());
        for m in &output.models {
            println!("  {} (v{}, {} bytes)", m.name, m.version, m.size_bytes);
        }
        println!();
        println!("Bundled operators ({}):", output.operators.len());
        for o in &output.operators {
            println!("  {} (v{}, {} bytes)", o.name, o.version, o.size_bytes);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_magic_is_enforced() {
        assert!(ensure_wasm(b"\0asm\x01\0\0\0", "model", "x").is_ok());
        assert!(ensure_wasm(b"function is_inside(x, y)", "model", "x").is_err());
        assert!(ensure_wasm(b"", "model", "x").is_err());
    }

    #[test]
    fn bundled_operators_resolve_by_name() {
        let (name, bytes) = resolve_operator_spec("translate_operator").unwrap();
        assert_eq!(name, "translate_operator");
        assert!(bytes.starts_with(b"\0asm"));
    }

    #[test]
    fn unknown_spec_lists_bundled_names() {
        let err = resolve_operator_spec("no_such_operator").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("translate_operator"), "got: {msg}");
    }
}

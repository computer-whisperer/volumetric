//! Bundled WASM assets for volumetric.
//!
//! This crate provides pre-compiled WASM modules for models and operators,
//! embedded at build time. This allows both native and web builds to have
//! access to demo models without requiring filesystem access.
//!
//! # Features
//!
//! - `models` (default): Include model WASM files (~27KB total)
//! - `operators` (default): Include operator WASM files (~7.2MB total)
//!
//! # Example
//!
//! ```ignore
//! use volumetric_assets::{get_model, get_operator, models, operators};
//!
//! // Get all available models
//! for model in models() {
//!     println!("{}: {} bytes", model.name, model.bytes.len());
//! }
//!
//! // Get a specific model by name
//! if let Some(sphere) = get_model("simple_sphere_model") {
//!     // Use sphere.bytes to instantiate the WASM module
//! }
//! ```

/// Category of a bundled asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetCategory {
    /// A model that generates SDF data
    Model,
    /// An operator that transforms SDF data
    Operator,
}

/// A bundled WASM asset: a byte source plus the identity fields hosts need
/// without compiling the module. Display metadata (names, descriptions,
/// categories, icons) deliberately lives in the module itself — read it
/// via `get_metadata()`, not from this registry.
#[derive(Debug, Clone, Copy)]
pub struct BundledAsset {
    /// The crate/module name (e.g., "simple_sphere_model")
    pub name: &'static str,
    /// The asset crate's Cargo.toml version, read at build time. Bundled
    /// crates declare `version: env!("CARGO_PKG_VERSION")` in their runtime
    /// metadata, so this matches the declared metadata version without
    /// compiling the module.
    pub version: &'static str,
    /// Hex SHA-256 of `bytes`, computed at build time. Projects embed operator
    /// bytes verbatim, so a host can detect a stale embedded copy — even one
    /// that kept the same `version` across a rebuild — by hashing the import
    /// and comparing against this. The UI's
    /// `bundled_asset_registry_matches_declared_metadata` test keeps it honest.
    pub hash: &'static str,
    /// The raw WASM bytes
    pub bytes: &'static [u8],
    /// The category of this asset
    pub category: AssetCategory,
}

// Include the generated asset registry
include!(concat!(env!("OUT_DIR"), "/asset_registry.rs"));

/// Get all bundled model assets.
pub fn models() -> &'static [BundledAsset] {
    BUNDLED_MODELS
}

/// Get all bundled operator assets.
pub fn operators() -> &'static [BundledAsset] {
    BUNDLED_OPERATORS
}

/// Get a bundled model by name.
pub fn get_model(name: &str) -> Option<&'static BundledAsset> {
    BUNDLED_MODELS.iter().find(|a| a.name == name)
}

/// Get a bundled operator by name.
pub fn get_operator(name: &str) -> Option<&'static BundledAsset> {
    BUNDLED_OPERATORS.iter().find(|a| a.name == name)
}

/// Get any bundled asset by name (searches both models and operators).
pub fn get_asset(name: &str) -> Option<&'static BundledAsset> {
    get_model(name).or_else(|| get_operator(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_models_available() {
        // This test will fail if WASM files haven't been built
        // That's expected - we print warnings during build
        let models = models();
        println!("Found {} bundled models", models.len());
        for model in models {
            println!("  - {}: {} bytes", model.name, model.bytes.len());
            assert!(!model.bytes.is_empty());
            assert_eq!(model.category, AssetCategory::Model);
        }
    }

    #[test]
    fn test_operators_available() {
        let operators = operators();
        println!("Found {} bundled operators", operators.len());
        for op in operators {
            println!("  - {}: {} bytes", op.name, op.bytes.len());
            assert!(!op.bytes.is_empty());
            assert_eq!(op.category, AssetCategory::Operator);
        }
    }

    #[test]
    fn test_get_by_name() {
        // These may return None if WASM files haven't been built
        if let Some(sphere) = get_model("simple_sphere_model") {
            assert_eq!(sphere.category, AssetCategory::Model);
            assert!(!sphere.version.is_empty());
        }
        if let Some(boolean) = get_operator("boolean_operator") {
            assert_eq!(boolean.category, AssetCategory::Operator);
            assert!(!boolean.version.is_empty());
        }
    }
}

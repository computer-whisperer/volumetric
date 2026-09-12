//! Questions hosts ask of a project's assets: its imports as loaded
//! assets, and the view set among a list of assets.

use crate::{AssetTypeHint, LoadedAsset, Project};
use volumetric_abi::viewset::{ViewSet, decode_viewset};

/// A project's imports as loaded assets (no run needed).
pub fn imports_as_assets(project: &Project) -> Vec<LoadedAsset> {
    project
        .imports()
        .iter()
        .map(|import| {
            LoadedAsset::from_parts(
                import.id.clone(),
                import.data.clone(),
                import.type_hint,
                vec![],
            )
        })
        .collect()
}

/// The view set asset named by `wanted` among `assets`, or the only one.
/// The error for several sets lists them so a host can name one.
pub fn viewset_asset<'a>(
    assets: &'a [LoadedAsset],
    wanted: Option<&str>,
) -> Result<&'a LoadedAsset, String> {
    let sets: Vec<&LoadedAsset> = assets
        .iter()
        .filter(|a| a.type_hint() == Some(AssetTypeHint::ViewSet))
        .collect();
    let ids = || {
        sets.iter()
            .map(|a| a.id().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    match wanted {
        Some(id) => sets
            .iter()
            .find(|a| a.id() == id)
            .copied()
            .ok_or_else(|| format!("no view set asset '{id}'. Available: {}", ids())),
        None => match sets.as_slice() {
            [] => Err("no view set asset in the project".to_string()),
            [only] => Ok(only),
            _ => Err(format!("several view sets; name one. Available: {}", ids())),
        },
    }
}

/// The view set named by `wanted` among `assets` (or the only one), decoded.
pub fn viewset(assets: &[LoadedAsset], wanted: Option<&str>) -> Result<ViewSet, String> {
    let asset = viewset_asset(assets, wanted)?;
    decode_viewset(asset.data()).map_err(|err| format!("asset '{}': {err}", asset.id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(id: &str, hint: AssetTypeHint, data: Vec<u8>) -> LoadedAsset {
        LoadedAsset::from_parts(id.to_string(), data, Some(hint), vec![])
    }

    #[test]
    fn the_view_set_is_found_by_name_or_alone() {
        let mut set = ViewSet {
            cameras: vec![volumetric_abi::viewset::CameraModel::pinhole(
                4, 4, 2.0, 2.0, 2.0, 2.0,
            )],
            ..ViewSet::default()
        };
        set.views
            .push(volumetric_abi::viewset::View::unposed("v1", 0));
        let bytes = volumetric_abi::viewset::encode_viewset(&set);
        let assets = vec![
            asset("blob", AssetTypeHint::Binary, vec![1]),
            asset("a", AssetTypeHint::ViewSet, bytes.clone()),
        ];
        assert_eq!(viewset(&assets, None).unwrap().views[0].id, "v1");
        assert!(
            viewset(&assets, Some("blob"))
                .unwrap_err()
                .contains("no view set asset 'blob'")
        );
        let two = [
            assets.clone(),
            vec![asset("b", AssetTypeHint::ViewSet, bytes)],
        ]
        .concat();
        assert!(viewset(&two, None).unwrap_err().contains("a, b"));
        assert_eq!(viewset_asset(&two, Some("b")).unwrap().id(), "b");
        assert!(viewset(&[], None).is_err());
        assert!(viewset(&[asset("bad", AssetTypeHint::ViewSet, vec![0xff])], None).is_err());
    }

    #[test]
    fn imports_become_assets_with_their_hints() {
        let mut project = Project::new();
        project.insert_model("m", b"\0asm\x01\0\0\0".to_vec());
        let imports = imports_as_assets(&project);
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].type_hint(), Some(AssetTypeHint::Model));
        assert_eq!(imports[0].data(), b"\0asm\x01\0\0\0");
    }
}

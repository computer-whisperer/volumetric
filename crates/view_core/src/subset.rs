//! A subset of a view set's views with their pictures re-embedded: the
//! views a project carries for look-through and audit. The CLI's
//! `view-select` and the Python bindings are this call.

use anyhow::{Result, anyhow};
use volumetric_abi::viewset::ViewSet;

use crate::manifest::{Selection, select_views};
use crate::stills::{Embed, embed_pictures};

/// What each kept view carries afterwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reembed {
    /// Whatever the view carries now.
    Keep,
    /// The original file, read through the view's source.
    Full,
    /// A reduced JPEG of the original.
    Preview,
    /// Only the reference to the original.
    None,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SubsetOptions {
    pub selection: Selection,
    /// Keep only posed views.
    pub posed: bool,
    /// Keep only views carrying every one of these tags.
    pub tags: Vec<String>,
    pub embed: Reembed,
    pub preview_px: u32,
}

impl Default for SubsetOptions {
    fn default() -> Self {
        Self {
            selection: Selection::default(),
            posed: false,
            tags: Vec::new(),
            embed: Reembed::Keep,
            preview_px: 1600,
        }
    }
}

/// The subset as a set of its own (cameras, markers and board shared),
/// with a provenance line, and the bytes of pictures it embeds.
pub fn subset(set: &ViewSet, options: &SubsetOptions) -> Result<(ViewSet, usize)> {
    let candidates: Vec<&volumetric_abi::viewset::View> = set
        .views
        .iter()
        .filter(|view| !options.posed || view.camera_to_world.is_some())
        .filter(|view| options.tags.iter().all(|tag| view.tags.contains(tag)))
        .collect();
    let chosen = select_views(&candidates, &options.selection)?;
    let mut selected = ViewSet {
        views: chosen.into_iter().cloned().collect(),
        ..set.clone()
    };
    let embedded = match options.embed {
        Reembed::Keep => selected
            .views
            .iter()
            .map(|v| v.image.as_ref().map_or(0, Vec::len))
            .sum(),
        Reembed::Full => embed_pictures(&mut selected, Embed::Full, options.preview_px, 88)?,
        Reembed::Preview => embed_pictures(&mut selected, Embed::Preview, options.preview_px, 88)?,
        Reembed::None => embed_pictures(&mut selected, Embed::None, options.preview_px, 88)?,
    };
    selected.provenance.tools.push(format!(
        "volumetric view-select ({} of {} views)",
        selected.views.len(),
        set.views.len()
    ));
    selected
        .validate()
        .map_err(|err| anyhow!("selected set is invalid: {err}"))?;
    Ok((selected, embedded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric_abi::viewset::{CameraModel, View};

    #[test]
    fn views_are_kept_by_id_pose_and_tag() {
        let mut set = ViewSet {
            cameras: vec![CameraModel::pinhole(4, 4, 2.0, 2.0, 2.0, 2.0)],
            ..ViewSet::default()
        };
        let pose = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        for (id, posed, tag) in [
            ("a", true, "keep"),
            ("b", false, "keep"),
            ("c", true, "drop"),
        ] {
            let mut view = if posed {
                View::posed(id, 0, pose)
            } else {
                View::unposed(id, 0)
            };
            view.tags.push(tag.to_string());
            view.image = Some(vec![0; 10]);
            set.views.push(view);
        }
        let ids = |s: &ViewSet| s.views.iter().map(|v| v.id.clone()).collect::<Vec<_>>();
        let (posed, bytes) = subset(
            &set,
            &SubsetOptions {
                posed: true,
                ..SubsetOptions::default()
            },
        )
        .unwrap();
        assert_eq!(ids(&posed), ["a", "c"]);
        assert_eq!(bytes, 20);
        assert!(posed.provenance.tools.last().unwrap().contains("2 of 3"));
        let (tagged, _) = subset(
            &set,
            &SubsetOptions {
                tags: vec!["keep".to_string()],
                ..SubsetOptions::default()
            },
        )
        .unwrap();
        assert_eq!(ids(&tagged), ["a", "b"]);
        let (named, bytes) = subset(
            &set,
            &SubsetOptions {
                selection: Selection {
                    ids: vec!["c".to_string()],
                    ..Selection::default()
                },
                embed: Reembed::None,
                ..SubsetOptions::default()
            },
        )
        .unwrap();
        assert_eq!(ids(&named), ["c"]);
        assert_eq!(bytes, 0);
        assert!(named.views[0].image.is_none());
        assert!(
            subset(
                &set,
                &SubsetOptions {
                    tags: vec!["nothing".to_string()],
                    ..SubsetOptions::default()
                }
            )
            .is_err()
        );
    }
}

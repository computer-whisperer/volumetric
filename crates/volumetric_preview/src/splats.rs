//! A Gaussian splat in the scene: the renderer's [`renderer::SplatData`]
//! built from the splat value, with bounds from the centres' percentiles
//! and the counts the UI shows.

use std::sync::Arc;

use glam::Mat4;
use volumetric::splat::{Splat, SplatKind, decode_splat};
use volumetric_renderer as renderer;

use crate::{OutputStats, PreviewBounds, PreviewEntity, PreviewRequest};

/// The renderer's form of a splat: scaled world axes from the scales and
/// rotations, activated opacities, and the colour coefficients in the
/// renderer's layout.
pub fn splat_data(splat: &Splat) -> renderer::SplatData {
    let n = splat.len();
    let mut positions = Vec::with_capacity(n);
    let mut axes = Vec::with_capacity(n);
    let mut opacities = Vec::with_capacity(n);
    let rest = volumetric::splat::sh_rest_per_point(splat.sh_degree);
    let mut sh = Vec::with_capacity(n * (3 + rest));
    for i in 0..n {
        positions.push(splat.mean(i));
        let r = splat.rotation(i);
        let mut s = splat.scale(i);
        // A surfel is flat: whatever the file holds as a third scale (an
        // untrained leftover in gsplat's 2DGS export) plays no part.
        if splat.kind == SplatKind::Surfel2d {
            s[2] = 0.0;
        }
        // Column k of R, scaled by s[k].
        let mut a = [0.0f32; 9];
        for k in 0..3 {
            for row in 0..3 {
                a[3 * k + row] = r[row][k] * s[k];
            }
        }
        axes.push(a);
        opacities.push(splat.opacity(i));
        sh.extend_from_slice(&splat.sh0[3 * i..3 * i + 3]);
        sh.extend_from_slice(&splat.sh_rest[i * rest..(i + 1) * rest]);
    }
    renderer::SplatData {
        surfels: splat.kind == SplatKind::Surfel2d,
        positions,
        axes,
        opacities,
        sh_degree: splat.sh_degree,
        sh,
    }
}

/// The lines the viewport's statistics show for a splat.
pub fn splat_detail(splat: &Splat) -> Vec<String> {
    let opaque = (0..splat.len())
        .filter(|&i| splat.opacity(i) >= 0.5)
        .count();
    let mut detail = vec![
        format!(
            "{} {}s, SH degree {}, {} at or above half opacity",
            splat.len(),
            splat.kind.name(),
            splat.sh_degree,
            opaque
        ),
        format!(
            "up ({}, {}, {}); normals {}",
            splat.world.up[0],
            splat.world.up[1],
            splat.world.up[2],
            if splat.normals.is_empty() {
                "from the orientation"
            } else {
                "stored"
            }
        ),
    ];
    let p = &splat.provenance;
    let labels: Vec<String> = [
        ("session", &p.session),
        ("field", &p.field),
        ("setup", &p.setup),
    ]
    .into_iter()
    .filter(|(_, value)| !value.is_empty())
    .map(|(name, value)| format!("{name} {value}"))
    .collect();
    if !labels.is_empty() {
        detail.push(labels.join(" · "));
    }
    if !splat.views_hash.is_empty() {
        detail.push(format!(
            "trained from view set {}",
            &splat.views_hash[..16.min(splat.views_hash.len())]
        ));
    }
    detail
}

/// Preview for a Splat value. The bounds are the centres' 1st to 99th
/// percentiles, so the stray primitives a trainer leaves far out do not
/// set the framing.
pub(crate) fn build_splat_preview(
    request: &PreviewRequest,
    build_start: web_time::Instant,
) -> Result<PreviewEntity, String> {
    let splat = decode_splat(request.data.as_slice())?;
    let data = splat_data(&splat);
    let (lo, hi) = splat
        .bounds_quantile(0.01, 0.99)
        .unwrap_or(([-1.0; 3], [1.0; 3]));
    let mut scene = renderer::SceneData::new();
    scene.add_splat(
        Arc::new(data),
        Mat4::IDENTITY,
        renderer::SplatStyle::default(),
    );
    let stats = OutputStats {
        points: splat.len(),
        detail: splat_detail(&splat),
        mesh_ms: build_start.elapsed().as_secs_f64() * 1000.0,
        ..Default::default()
    };
    Ok(PreviewEntity {
        scene,
        bounds: PreviewBounds {
            min: (lo[0] as f32, lo[1] as f32, lo[2] as f32),
            max: (hi[0] as f32, hi[1] as f32, hi[2] as f32),
        },
        stats,
        wireframe_lines: None,
        mesh_keys: Vec::new(),
        subspace: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric::AssetTypeHint;
    use volumetric::splat::{SplatKind, encode_splat, sh_rest_per_point};

    fn sample() -> Splat {
        let mut s = Splat::empty(SplatKind::Gaussian3d, 1);
        s.count = 2;
        s.means = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        let h = std::f32::consts::FRAC_1_SQRT_2;
        s.scales = vec![(0.1f32).ln(), (0.2f32).ln(), (0.3f32).ln(), 0.0, 0.0, 0.0];
        s.quats = vec![1.0, 0.0, 0.0, 0.0, h, 0.0, 0.0, h];
        s.opacities = vec![0.0, 10.0];
        s.sh0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        s.sh_rest = (0..2 * sh_rest_per_point(1)).map(|k| k as f32).collect();
        s
    }

    #[test]
    fn covariances_follow_the_scales_and_rotation() {
        let data = splat_data(&sample());
        assert!(!data.surfels);
        let c = data.covariance(0);
        assert!(
            (c[0] - 0.01).abs() < 1e-6 && (c[3] - 0.04).abs() < 1e-6 && (c[5] - 0.09).abs() < 1e-6
        );
        assert_eq!(c[1], 0.0);
        // The second is a unit sphere turned 90° about z: still the identity.
        let c = data.covariance(1);
        assert!((c[0] - 1.0).abs() < 1e-5 && (c[3] - 1.0).abs() < 1e-5 && c[1].abs() < 1e-5);
        assert!((data.opacities[0] - 0.5).abs() < 1e-6);
        // Surfels drop the third axis.
        let mut surfel = sample();
        surfel.kind = SplatKind::Surfel2d;
        let surfel_data = splat_data(&surfel);
        assert!(surfel_data.surfels);
        let c = surfel_data.covariance(0);
        assert_eq!(c[5], 0.0);
        assert!((c[0] - 0.01).abs() < 1e-6);
        assert_eq!(data.sh_per_point(), 12);
        assert_eq!(&data.sh[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&data.sh[12..15], &[4.0, 5.0, 6.0]);
        assert_eq!(
            data.sh[15], 9.0,
            "the second point's first higher coefficient"
        );
    }

    #[test]
    fn the_preview_carries_the_splat_and_its_bounds() {
        let splat = sample();
        let request = PreviewRequest {
            asset_id: "splat".to_string(),
            source_hash: [0; 32],
            data: Arc::new(encode_splat(&splat)),
            type_hint: Some(AssetTypeHint::Splat),
            precursor_ids: vec![],
            plan: crate::PreviewPlan::Splat,
            wireframe: false,
            show_grid: false,
            show_bounds: false,
            ssao: false,
            ssao_radius: 0.5,
            ssao_bias: 0.02,
            ssao_strength: 1.0,
            stale: false,
        };
        let entity = crate::build_preview_scene(&request).unwrap();
        assert_eq!(entity.scene.splats.len(), 1);
        assert_eq!(entity.scene.splats[0].0.len(), 2);
        assert_eq!(entity.bounds.min, (0.0, 0.0, 0.0));
        assert_eq!(entity.bounds.max, (1.0, 2.0, 3.0));
        assert_eq!(entity.stats.points, 2);
        assert!(entity.stats.detail[0].contains("2 gaussians"));
    }
}

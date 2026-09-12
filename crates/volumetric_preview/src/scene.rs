//! The per-kind preview builders behind [`build_preview_scene`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use glam::Vec3;
use volumetric::AssetTypeHint;
use volumetric::wasm::ModelExecutor as _;
use volumetric_renderer as renderer;

use crate::gizmo::pad3;
use crate::{
    ExecutionBackend, LocalBackend, OutputStats, PreviewBounds, PreviewEntity, PreviewMeshPlan,
    PreviewPlan, PreviewRequest,
};

pub fn build_preview_scene(request: &PreviewRequest) -> Result<PreviewEntity, String> {
    static NEVER: AtomicBool = AtomicBool::new(false);
    build_preview_scene_cancellable(request, &NEVER)
        .map(|entity| entity.expect("a never-set cancel flag cannot cancel the build"))
}

/// [`build_preview_scene`] with cooperative cancellation. `Ok(None)` means
/// the flag was observed set and the build was abandoned. Only the ASN2
/// meshing path (by far the longest-running plan) checks mid-build; the
/// other plans check once up front.
pub fn build_preview_scene_cancellable(
    request: &PreviewRequest,
    cancel: &AtomicBool,
) -> Result<Option<PreviewEntity>, String> {
    build_preview_scene_with(request, cancel, &LocalBackend)
}

/// [`build_preview_scene_cancellable`] with the ASN2 meshing pass routed
/// through `backend` (scene assembly, colormapping, and the cheap plans stay
/// in-process — they need the local model executor or are not worth a round
/// trip).
pub fn build_preview_scene_with(
    request: &PreviewRequest,
    cancel: &AtomicBool,
    backend: &dyn ExecutionBackend,
) -> Result<Option<PreviewEntity>, String> {
    build_preview_scene_monitored(request, cancel, backend, &|_| {})
}

/// [`build_preview_scene_with`] forwarding meshing progress to `progress`
/// (only the ASN2 plan reports; the cheap plans finish too fast to matter).
pub fn build_preview_scene_monitored(
    request: &PreviewRequest,
    cancel: &AtomicBool,
    backend: &dyn ExecutionBackend,
    progress: &dyn Fn(volumetric::BuildProgress),
) -> Result<Option<PreviewEntity>, String> {
    match preview_prelude(request, cancel)? {
        None => Ok(None),
        Some(PreviewStage::Done(entity)) => Ok(Some(*entity)),
        Some(PreviewStage::NeedsMesh(pending)) => {
            let Some(mesh) =
                backend.mesh_model(request.data.as_slice(), &pending.config, cancel, progress)?
            else {
                return Ok(None);
            };
            Ok(Some(preview_postlude(request, pending, mesh)))
        }
    }
}

/// Output of [`preview_prelude`]: either a finished preview, or an ASN2
/// plan whose meshing pass the caller must run — through
/// [`ExecutionBackend::mesh_model`] on a worker thread natively, or an
/// awaited daemon fetch on the web — before finishing with
/// [`preview_postlude`]. Split this way so the single-threaded web shell
/// can suspend at the one point that crosses the wire.
pub enum PreviewStage {
    Done(Box<PreviewEntity>),
    NeedsMesh(PendingMesh),
}

/// The ASN2 meshing request plus everything the postlude needs to finish
/// the preview once a mesh exists.
pub struct PendingMesh {
    pub config: volumetric::adaptive_surface_nets_2::AdaptiveMeshConfig2,
    color_channel: Option<String>,
    stats: OutputStats,
    build_start: web_time::Instant,
}

/// Everything of a preview build except the ASN2 meshing pass. `Ok(None)`
/// means the cancel flag was observed.
pub fn preview_prelude(
    request: &PreviewRequest,
    cancel: &AtomicBool,
) -> Result<Option<PreviewStage>, String> {
    let build_start = web_time::Instant::now();
    let mut stats = OutputStats::default();

    if cancel.load(Ordering::Relaxed) {
        return Ok(None);
    }

    let done = |entity: PreviewEntity| Some(PreviewStage::Done(Box::new(entity)));

    // Explicit mesh values are data, not sampleable models: draw them
    // directly, ignoring the model mesh plans.
    if request.type_hint == Some(AssetTypeHint::FeaMesh) {
        return build_fea_mesh_preview(request, build_start).map(done);
    }
    if request.type_hint == Some(AssetTypeHint::TriMesh) {
        return build_tri_mesh_preview(request, build_start).map(done);
    }
    if request.type_hint == Some(AssetTypeHint::Subspace) {
        return build_subspace_preview(request, build_start).map(done);
    }
    if request.type_hint == Some(AssetTypeHint::ViewSet) {
        return crate::views::build_viewset_preview(request, build_start).map(done);
    }
    if request.type_hint == Some(AssetTypeHint::Splat) {
        return crate::splats::build_splat_preview(request, build_start).map(done);
    }

    // 2D sketches get a flat raster preview; the 3D mesh plans don't apply.
    let dims = volumetric::model_dimensions_from_bytes(request.data.as_slice())
        .map_err(format_error_chain)?;
    if dims == 2 {
        return build_sketch_preview(request, build_start).map(done);
    }

    // The plan normally matches the runtime dimensionality (the UI probes
    // it statically); if a model defeats the static probe, fall back to a
    // cheap default plan rather than failing.
    let fallback_plan;
    let (mesh_plan, color_channel) = match &request.plan {
        PreviewPlan::Model3d {
            mesh,
            color_channel,
            ..
        } => (mesh, color_channel.as_deref()),
        _ => {
            fallback_plan = PreviewMeshPlan::PointCloud { resolution: 24 };
            (&fallback_plan, None)
        }
    };
    let (scene, wireframe_lines, bounds_min, bounds_max) = match mesh_plan {
        PreviewMeshPlan::PointCloud { resolution } => {
            let (points, bounds_min, bounds_max) =
                volumetric::sample_model_from_bytes(request.data.as_slice(), *resolution)
                    .map_err(format_error_chain)?;
            stats.points = points.len();
            let mut scene = renderer::SceneData::new();
            scene.add_points(
                renderer::convert_points_to_point_data(&points),
                glam::Mat4::IDENTITY,
                renderer::PointStyle {
                    size: 4.0,
                    size_mode: renderer::WidthMode::ScreenSpace,
                    shape: renderer::PointShape::Circle,
                    depth_mode: renderer::DepthMode::Normal,
                },
            );
            (scene, None, bounds_min, bounds_max)
        }
        PreviewMeshPlan::MarchingCubes { resolution } => {
            let (triangles, bounds_min, bounds_max) =
                volumetric::generate_marching_cubes_mesh_from_bytes(
                    request.data.as_slice(),
                    *resolution,
                )
                .map_err(format_error_chain)?;
            stats.triangles = triangles.len();
            let vertices = triangles_to_mesh_vertices(&triangles);
            let wireframe = mesh_edge_lines(&vertices, None);
            let mut scene = renderer::SceneData::new();
            scene.add_mesh(
                renderer::MeshData {
                    vertices,
                    indices: None,
                },
                glam::Mat4::IDENTITY,
                renderer::MaterialId(0),
            );
            (scene, Some(wireframe), bounds_min, bounds_max)
        }
        PreviewMeshPlan::AdaptiveSurfaceNets2 { .. } => {
            let config = mesh_plan
                .adaptive_surface_nets_config()
                .ok_or_else(|| "missing adaptive surface nets config".to_string())?;
            return Ok(Some(PreviewStage::NeedsMesh(PendingMesh {
                config,
                color_channel: color_channel.map(str::to_string),
                stats,
                build_start,
            })));
        }
    };

    Ok(Some(PreviewStage::Done(Box::new(finish_preview_scene(
        request,
        scene,
        wireframe_lines,
        (bounds_min, bounds_max),
        color_channel.map(str::to_string),
        stats,
        build_start,
    )))))
}

/// Finishes a preview once the ASN2 mesh exists: scene assembly plus the
/// shared colormap/stats tail. Purely local — the pending state carries
/// everything the prelude gathered.
pub fn preview_postlude(
    request: &PreviewRequest,
    pending: PendingMesh,
    mesh: Arc<volumetric::AdaptiveMeshV2Result>,
) -> PreviewEntity {
    let PendingMesh {
        config: _,
        color_channel,
        mut stats,
        build_start,
    } = pending;
    stats.triangles = mesh.indices.len() / 3;
    stats.samples = mesh.stats.total_samples;
    stats.detail = asn2_stage_lines(&mesh.stats);
    // Tripwire: build results are supposed to carry one finite unit
    // normal per vertex (local builds do; a stale remote daemon once
    // shipped all-degenerate normals that rendered as uniform white).
    // Substitute +Z for anything degenerate and surface the counts.
    let mut degenerate_normals = 0usize;
    let mut nonfinite_positions = 0usize;
    if mesh.normals.len() != mesh.vertices.len() {
        stats.detail.push(format!(
            "preview guard: normals/vertices length mismatch {} vs {}",
            mesh.normals.len(),
            mesh.vertices.len()
        ));
    }
    let vertices: Vec<renderer::MeshVertex> = mesh
        .vertices
        .iter()
        .zip(mesh.normals.iter())
        .map(|(position, normal)| {
            if !(position.0.is_finite() && position.1.is_finite() && position.2.is_finite()) {
                nonfinite_positions += 1;
            }
            let len2 = normal.0 * normal.0 + normal.1 * normal.1 + normal.2 * normal.2;
            let normal = if len2.is_finite() && len2 > 1e-20 {
                *normal
            } else {
                degenerate_normals += 1;
                (0.0, 0.0, 1.0)
            };
            renderer::MeshVertex::new((*position).into(), normal.into())
        })
        .collect();
    if degenerate_normals > 0 || nonfinite_positions > 0 {
        stats.detail.push(format!(
            "preview guard: {degenerate_normals} degenerate normals substituted,              {nonfinite_positions} non-finite positions"
        ));
    }
    let wireframe = mesh_edge_lines(&vertices, Some(&mesh.indices));
    let mut scene = renderer::SceneData::new();
    scene.add_mesh(
        renderer::MeshData {
            vertices,
            indices: Some(mesh.indices.clone()),
        },
        glam::Mat4::IDENTITY,
        renderer::MaterialId(0),
    );
    finish_preview_scene(
        request,
        scene,
        Some(wireframe),
        (mesh.bounds_min, mesh.bounds_max),
        color_channel,
        stats,
        build_start,
    )
}

/// The tail every 3D preview shares: channel discovery + colormap, timing,
/// and entity assembly.
fn finish_preview_scene(
    request: &PreviewRequest,
    mut scene: renderer::SceneData,
    wireframe_lines: Option<renderer::LineData>,
    (bounds_min, bounds_max): ((f32, f32, f32), (f32, f32, f32)),
    color_channel: Option<String>,
    mut stats: OutputStats,
    build_start: web_time::Instant,
) -> PreviewEntity {
    // Channel discovery + colormap: mirror the declared channels into the
    // stats (feeds the "Color by" picker and the slice lightbox), and when
    // a channel is selected, colormap the built points/vertices by sampling
    // it. The module is already in the executor cache from the meshing pass,
    // so this executor is cheap to create.
    match volumetric::wasm::create_model_executor(request.data.as_slice()) {
        Ok(mut executor) => {
            stats.model_channels = executor
                .sample_format()
                .map(|format| format.channels.iter().map(|c| c.name.clone()).collect())
                .unwrap_or_default();
            if let Some(channel) = &color_channel {
                match colormap_scene_by_channel(&mut scene, &mut executor, channel) {
                    Ok((value_min, value_max)) => stats.detail.push(format!(
                        "Color: {channel} in [{value_min:.4}, {value_max:.4}]"
                    )),
                    Err(err) => stats.detail.push(format!("Color: {err}")),
                }
            } else if let Some(trio) = executor
                .sample_format()
                .ok()
                .and_then(|format| format.color_trio())
            {
                // No scalar channel picked, but the model carries true
                // surface colors (e.g. a styled STEP import): render them.
                match truecolor_scene_by_trio(&mut scene, &mut executor, trio) {
                    Ok(()) => stats.detail.push("Color: model surface colors".to_string()),
                    Err(err) => stats.detail.push(format!("Color: {err}")),
                }
            } else if matches!(
                &request.plan,
                PreviewPlan::Model3d {
                    tint_uncolored: true,
                    ..
                }
            ) {
                // Uncolored model with part tinting on: a muted tint keyed
                // by the output id keeps flush-fitting parts apart when
                // several are pinned into one viewport.
                let tint = part_tint(&request.asset_id);
                tint_scene(&mut scene, tint);
                stats
                    .detail
                    .push(format!("Color: part tint ({})", request.asset_id));
            }
        }
        Err(err) => {
            if color_channel.is_some() {
                stats.detail.push(format!("Color: {err}"));
            }
        }
    }

    stats.mesh_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    PreviewEntity {
        scene,
        bounds: PreviewBounds {
            min: bounds_min,
            max: bounds_max,
        },
        stats,
        wireframe_lines,
        subspace: None,
    }
}

/// Colormaps every point instance and mesh vertex of a built preview scene
/// by the named sample channel: viridis over the sampled range, magenta for
/// non-finite samples. Returns the value range the colormap spans.
fn colormap_scene_by_channel(
    scene: &mut renderer::SceneData,
    executor: &mut impl volumetric::wasm::ModelExecutor,
    channel: &str,
) -> Result<(f32, f32), String> {
    let channel_idx = executor
        .sample_format()
        .map_err(|err| err.to_string())?
        .channels
        .iter()
        .position(|c| c.name == channel)
        .ok_or_else(|| format!("channel {channel:?} not declared"))?;

    let mut sample = |position: [f32; 3]| -> Result<f32, String> {
        let row = executor
            .sample_channels_nd(&[
                f64::from(position[0]),
                f64::from(position[1]),
                f64::from(position[2]),
            ])
            .map_err(|err| err.to_string())?;
        Ok(row.get(channel_idx).copied().unwrap_or(f32::NAN))
    };

    // Sample everything first: the colormap needs the whole range.
    let mut point_values: Vec<Vec<f32>> = Vec::new();
    for (points, _, _) in &scene.points {
        let mut values = Vec::with_capacity(points.points.len());
        for point in &points.points {
            values.push(sample(point.position)?);
        }
        point_values.push(values);
    }
    let mut vertex_values: Vec<Vec<f32>> = Vec::new();
    for (mesh, _, _) in &scene.meshes {
        let mut values = Vec::with_capacity(mesh.vertices.len());
        for vertex in &mesh.vertices {
            values.push(sample(vertex.position)?);
        }
        vertex_values.push(values);
    }

    let mut value_min = f32::INFINITY;
    let mut value_max = f32::NEG_INFINITY;
    for &v in point_values.iter().chain(vertex_values.iter()).flatten() {
        if v.is_finite() {
            value_min = value_min.min(v);
            value_max = value_max.max(v);
        }
    }
    if value_min > value_max {
        (value_min, value_max) = (0.0, 0.0);
    }
    let span = (value_max - value_min).max(f32::EPSILON);
    let color_of = |v: f32| -> [f32; 4] {
        if !v.is_finite() {
            return [1.0, 0.0, 1.0, 1.0];
        }
        let c = volumetric::viridis((v - value_min) / span);
        [c[0], c[1], c[2], 1.0]
    };

    for ((points, _, _), values) in scene.points.iter_mut().zip(&point_values) {
        for (point, &v) in points.points.iter_mut().zip(values) {
            point.color = color_of(v);
        }
    }
    for ((mesh, _, _), values) in scene.meshes.iter_mut().zip(&vertex_values) {
        for (vertex, &v) in mesh.vertices.iter_mut().zip(values) {
            vertex.color = color_of(v);
        }
    }
    Ok((value_min, value_max))
}

/// Colors every point instance and mesh vertex by the model's declared
/// sRGB surface-color trio, sampled at each position and converted to
/// linear RGB for the renderer. Non-finite components fall back to white
/// (the unstyled-surface convention).
fn truecolor_scene_by_trio(
    scene: &mut renderer::SceneData,
    executor: &mut impl volumetric::wasm::ModelExecutor,
    trio: [usize; 3],
) -> Result<(), String> {
    let mut color_at = |position: [f32; 3]| -> Result<[f32; 4], String> {
        let row = executor
            .sample_channels_nd(&[
                f64::from(position[0]),
                f64::from(position[1]),
                f64::from(position[2]),
            ])
            .map_err(|err| err.to_string())?;
        let mut rgba = [1.0f32; 4];
        for (out, &idx) in rgba.iter_mut().zip(&trio) {
            let v = row.get(idx).copied().unwrap_or(f32::NAN);
            if v.is_finite() {
                *out = srgb_to_linear(v.clamp(0.0, 1.0));
            }
        }
        Ok(rgba)
    };
    for (points, _, _) in &mut scene.points {
        for point in &mut points.points {
            point.color = color_at(point.position)?;
        }
    }
    for (mesh, _, _) in &mut scene.meshes {
        for vertex in &mut mesh.vertices {
            vertex.color = color_at(vertex.position)?;
        }
    }
    Ok(())
}

/// One sRGB component to linear, the renderer's working space.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// One channel of a mesh or cloud `color` field (sRGB in `[0, 1]`, see
/// `volumetric::fea::COLOR_FIELD_NAME`) as the linear value the renderer
/// expects; out-of-range values clamp.
pub fn field_channel_to_linear(c: f64) -> f32 {
    srgb_to_linear(c.clamp(0.0, 1.0) as f32)
}

/// A muted, deterministic tint for an uncolored part: FNV-1a over the
/// output id (stable across sessions and platforms — pin-set changes
/// never recolor a part) picks one of twelve pastel hues, converted to
/// linear RGBA for the renderer.
pub fn part_tint(id: &str) -> [f32; 4] {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in id.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    let hue = (hash % 12) as f32 * 30.0;

    // HSL at fixed saturation/lightness: distinct but never garish.
    let (saturation, lightness) = (0.45, 0.72);
    let chroma = (1.0 - (2.0 * lightness - 1.0f32).abs()) * saturation;
    let hue_prime = hue / 60.0;
    let x = chroma * (1.0 - (hue_prime % 2.0 - 1.0).abs());
    let (r, g, b) = match hue_prime as u32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let m = lightness - chroma / 2.0;
    [
        srgb_to_linear(r + m),
        srgb_to_linear(g + m),
        srgb_to_linear(b + m),
        1.0,
    ]
}

/// Sets every point instance and mesh vertex of a built scene to one flat
/// color (the part-tint path; per-face colors go through the channel
/// machinery instead).
fn tint_scene(scene: &mut renderer::SceneData, color: [f32; 4]) {
    for (points, _, _) in &mut scene.points {
        for point in &mut points.points {
            point.color = color;
        }
    }
    for (mesh, _, _) in &mut scene.meshes {
        for vertex in &mut mesh.vertices {
            vertex.color = color;
        }
    }
}

/// Style for the wireframe overlay: thin dark depth-tested lines. The line
/// pipeline uses `LessEqual` depth compare, so lines coincident with mesh
/// edges win over the faces they border.
pub fn wireframe_style() -> renderer::LineStyle {
    renderer::LineStyle {
        width: 1.0,
        width_mode: renderer::WidthMode::ScreenSpace,
        pattern: renderer::LinePattern::Solid,
        depth_mode: renderer::DepthMode::Normal,
    }
}

/// Unique edges of a mesh as line segments. With an index buffer, edges are
/// deduplicated by index pair; for triangle soup, by quantized endpoint
/// positions.
fn mesh_edge_lines(
    vertices: &[renderer::MeshVertex],
    indices: Option<&[u32]>,
) -> renderer::LineData {
    const COLOR: [f32; 4] = [0.05, 0.06, 0.08, 0.9];
    let mut segments = Vec::new();
    match indices {
        Some(indices) => {
            let mut seen = std::collections::HashSet::new();
            for tri in indices.chunks_exact(3) {
                for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                    if seen.insert((a.min(b), a.max(b))) {
                        segments.push(renderer::LineSegment {
                            start: vertices[a as usize].position,
                            end: vertices[b as usize].position,
                            color: COLOR,
                        });
                    }
                }
            }
        }
        None => {
            let key = |p: [f32; 3]| (p[0].to_bits(), p[1].to_bits(), p[2].to_bits());
            let mut seen = std::collections::HashSet::new();
            for tri in vertices.chunks_exact(3) {
                for (a, b) in [(0usize, 1usize), (1, 2), (2, 0)] {
                    let (pa, pb) = (tri[a].position, tri[b].position);
                    let (ka, kb) = (key(pa), key(pb));
                    let edge_key = if ka <= kb { (ka, kb) } else { (kb, ka) };
                    if seen.insert(edge_key) {
                        segments.push(renderer::LineSegment {
                            start: pa,
                            end: pb,
                            color: COLOR,
                        });
                    }
                }
            }
        }
    }
    renderer::LineData { segments }
}

/// Per-stage profiling lines for the ASN2 mesher, shown in the output's
/// settings popover (v1 had these in a collapsible "Profiling Details").
fn asn2_stage_lines(stats: &volumetric::adaptive_surface_nets_2::MeshingStats2) -> Vec<String> {
    let ms = |secs: f64| secs * 1000.0;
    let mut lines = vec![
        format!(
            "S1 discovery {:.1} ms · {} samples · {} cells{}",
            ms(stats.stage1_time_secs),
            stats.stage1_samples,
            stats.stage1_mixed_cells,
            if stats.stage1_probe_seeds > 0 {
                format!(" · {} probe seeds", stats.stage1_probe_seeds)
            } else {
                String::new()
            }
        ),
        format!(
            "S2 subdivide {:.1} ms · {} tris",
            ms(stats.stage2_time_secs),
            stats.stage2_triangles_emitted
        ),
        format!(
            "S3 topology {:.1} ms · {} verts",
            ms(stats.stage3_time_secs),
            stats.stage3_unique_vertices
        ),
        format!(
            "S4 refine {:.1} ms · {} samples",
            ms(stats.stage4_time_secs),
            stats.stage4_samples
        ),
    ];
    if stats.sharp_regions > 0 || stats.sharp_candidates > 0 {
        lines.push(format!(
            "S4.5 sharp {:.1} ms · {} regions · {} snapped · {} welded",
            ms(stats.stage4_5_time_secs),
            stats.sharp_regions,
            stats.sharp_snapped_edges + stats.sharp_snapped_corners,
            stats.sharp_welded_vertices
        ));
    }
    lines
}

/// Flat z=0 preview of a 2D sketch: run-length spans of occupied raster
/// cells become double-sided quads (one +z face, one -z face).
fn build_sketch_preview(
    request: &PreviewRequest,
    build_start: web_time::Instant,
) -> Result<PreviewEntity, String> {
    let (resolution, color_channel) = match &request.plan {
        PreviewPlan::Sketch {
            resolution,
            color_channel,
        } => (*resolution, color_channel.as_deref()),
        // Kind/dims mismatch (static probe defeated): default raster.
        _ => (256, None),
    };
    let raster = volumetric::rasterize_sketch_channel_from_bytes(
        request.data.as_slice(),
        resolution,
        color_channel,
    )
    .map_err(format_error_chain)?;

    let cell_w = (raster.bounds_max.0 - raster.bounds_min.0) / raster.width as f32;
    let cell_h = (raster.bounds_max.1 - raster.bounds_min.1) / raster.height as f32;

    let mut vertices: Vec<renderer::MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut emit_quad = |x0: f32, x1: f32, y0: f32, y1: f32, color: [f32; 4]| {
        let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
        for (normal, winding) in [
            ([0.0f32, 0.0, 1.0], [0u32, 1, 2, 0, 2, 3]),
            ([0.0f32, 0.0, -1.0], [0u32, 2, 1, 0, 3, 2]),
        ] {
            let base = vertices.len() as u32;
            for (x, y) in corners {
                vertices.push(renderer::MeshVertex::colored([x, y, 0.0], normal, color));
            }
            indices.extend(winding.iter().map(|i| base + i));
        }
    };

    // Cell classification for per-row run-length merging: None = empty,
    // Some(level) = draw with that level's color. Occupancy sketches keep
    // the mask look (untinted, holes where empty); scalar fields draw
    // every finite cell colormapped over the sampled value range.
    let binary = raster.is_binary();
    const LEVELS: usize = 48;
    let value_span = (raster.value_max - raster.value_min).max(f32::MIN_POSITIVE);
    let classify = |xi: usize, yi: usize| -> Option<usize> {
        if binary {
            raster.cell(xi, yi).then_some(LEVELS)
        } else {
            let v = raster.value(xi, yi);
            if !v.is_finite() {
                return None;
            }
            let t = (v - raster.value_min) / value_span;
            Some(((t * LEVELS as f32) as usize).min(LEVELS - 1))
        }
    };
    let level_color = |level: usize| -> [f32; 4] {
        if binary {
            [1.0; 4]
        } else {
            let [r, g, b] = volumetric::viridis((level as f32 + 0.5) / LEVELS as f32);
            [r, g, b, 1.0]
        }
    };

    for yi in 0..raster.height {
        let y0 = raster.bounds_min.1 + cell_h * yi as f32;
        let y1 = y0 + cell_h;
        let mut run: Option<(usize, usize)> = None; // (start, level)
        for xi in 0..=raster.width {
            let class = (xi < raster.width).then(|| classify(xi, yi)).flatten();
            if let Some((start, level)) = run
                && class != Some(level)
            {
                let x0 = raster.bounds_min.0 + cell_w * start as f32;
                let x1 = raster.bounds_min.0 + cell_w * xi as f32;
                emit_quad(x0, x1, y0, y1, level_color(level));
                run = None;
            }
            if run.is_none()
                && let Some(level) = class
            {
                run = Some((xi, level));
            }
        }
    }

    let triangles = indices.len() / 3;
    let mut scene = renderer::SceneData::new();
    scene.add_mesh(
        renderer::MeshData {
            vertices,
            indices: Some(indices),
        },
        glam::Mat4::IDENTITY,
        renderer::MaterialId(0),
    );

    let mut detail = vec![format!(
        "2D sketch raster {}x{}",
        raster.width, raster.height
    )];
    if !binary {
        let field = color_channel.unwrap_or("field");
        detail.push(format!(
            "{field} {:.4} .. {:.4} (viridis)",
            raster.value_min, raster.value_max
        ));
    }
    // Mirror the declared channels into the stats, like the 3D path —
    // this feeds the "Color by" picker for channeled 2D models (e.g. a
    // planar slice of a density model). The module is already in the
    // executor cache from the raster pass.
    let model_channels = volumetric::wasm::create_model_executor(request.data.as_slice())
        .ok()
        .and_then(|mut executor| {
            use volumetric::wasm::ModelExecutor;
            executor
                .sample_format()
                .map(|format| format.channels.iter().map(|c| c.name.clone()).collect())
                .ok()
        })
        .unwrap_or_default();
    let stats = OutputStats {
        triangles,
        samples: (raster.width * raster.height) as u64,
        detail,
        model_channels,
        mesh_ms: build_start.elapsed().as_secs_f64() * 1000.0,
        ..Default::default()
    };
    Ok(PreviewEntity {
        scene,
        bounds: PreviewBounds {
            min: (raster.bounds_min.0, raster.bounds_min.1, 0.0),
            max: (raster.bounds_max.0, raster.bounds_max.1, 0.0),
        },
        stats,
        wireframe_lines: None,
        subspace: None,
    })
}

/// Preview of an FEA mesh output: the mesh's boundary faces, flat-shaded
/// or colormapped by a chosen field (viridis over the field's range),
/// optionally in the deformed configuration (displacement x exaggeration),
/// with the wireframe overlay tracing element edges (quad edges only — no
/// triangulation diagonals).
fn build_fea_mesh_preview(
    request: &PreviewRequest,
    build_start: web_time::Instant,
) -> Result<PreviewEntity, String> {
    let mut mesh = volumetric::fea::decode_fea_mesh(request.data.as_slice())?;
    let faces = mesh.boundary_faces();

    let (want_deformed, exaggeration, color_field, fixed_range) = match &request.plan {
        PreviewPlan::FeaMesh {
            deformed,
            exaggeration_tenths,
            color_field,
            color_range,
        } => (
            *deformed,
            f64::from(*exaggeration_tenths) / 10.0,
            color_field.clone(),
            *color_range,
        ),
        // Plan/kind mismatch (shouldn't happen): the default view.
        _ => (true, 1.0, None, None),
    };

    // Every colormappable field, mirrored to the settings popover's picker
    // through the stats.
    // (The `color` and `normal` fields are rendered directly, not
    // colormapped, so they stay out of the picker.)
    let fea_fields: Vec<String> = mesh
        .node_fields
        .iter()
        .filter(|f| f.components == 1 || f.components == 3)
        .filter(|f| {
            f.name != volumetric::fea::COLOR_FIELD_NAME
                && f.name != volumetric::fea::NORMAL_FIELD_NAME
        })
        .map(|f| format!("node:{}", f.name))
        .chain(
            mesh.element_fields
                .iter()
                .filter(|f| f.components == 1)
                .map(|f| format!("element:{}", f.name)),
        )
        .collect();

    // Resolve the colormapped field to one scalar per node or per element
    // (3-component node fields color by magnitude).
    enum ColorSource {
        Node(Vec<f64>),
        Element(Vec<f64>),
    }
    let mut extra_detail = Vec::new();
    let color_source = color_field.as_deref().and_then(|qualified| {
        let (container, name) = qualified.split_once(':')?;
        let scalars = |field: &volumetric::fea::FeaField| -> Option<Vec<f64>> {
            match field.components {
                1 => Some(field.data.clone()),
                3 => Some(
                    field
                        .data
                        .chunks_exact(3)
                        .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
                        .collect(),
                ),
                _ => None,
            }
        };
        let source = match container {
            "node" => mesh
                .node_fields
                .iter()
                .find(|f| f.name == name)
                .and_then(scalars)
                .map(ColorSource::Node),
            "element" => mesh
                .element_fields
                .iter()
                .find(|f| f.name == name && f.components == 1)
                .map(|f| ColorSource::Element(f.data.clone())),
            _ => None,
        };
        if source.is_none() {
            extra_detail.push(format!("colormap field {qualified} not in this mesh"));
        }
        source
    });
    let color_range = color_source.as_ref().map(|source| {
        let values = match source {
            ColorSource::Node(v) | ColorSource::Element(v) => v,
        };
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in values {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        match fixed_range {
            Some(range) => (range.lo(), range.hi()),
            None if lo > hi => (0.0, 0.0),
            None => (lo, hi),
        }
    });
    if let (Some(field), Some((lo, hi))) = (&color_field, color_range) {
        extra_detail.push(format!(
            "colormap {field} {lo:.4} .. {hi:.4} (viridis{})",
            if fixed_range.is_some() {
                ", clamped"
            } else {
                ""
            }
        ));
    }
    let color_for = |value: f64| -> [f32; 4] {
        let (lo, hi) = color_range.unwrap_or((0.0, 0.0));
        let t = if hi > lo {
            (((value - lo) / (hi - lo)) as f32).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let [r, g, b] = volumetric::viridis(t);
        [r, g, b, 1.0]
    };

    // Deformed configuration: positions + displacement x exaggeration;
    // connectivity (and thus the boundary) is unchanged. Garbage-magnitude
    // fields (a solve run with qualitative unit parameters, a diverged
    // contact loop) would scatter the nodes over a cloud thousands of
    // times the part size, so a displacement that dwarfs the part draws
    // rescaled-to-fit instead — the deformation *shape* stays readable and
    // the detail line names the substitution.
    if want_deformed
        && let Some(displacement) = mesh
            .node_fields
            .iter()
            .find(|f| f.name == "displacement" && f.components == 3)
    {
        let max_u = displacement
            .data
            .chunks_exact(3)
            .map(|u| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt())
            .fold(0.0f64, f64::max);
        let diag = {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for p in mesh.node_positions.chunks_exact(3) {
                for c in 0..3 {
                    lo[c] = lo[c].min(p[c]);
                    hi[c] = hi[c].max(p[c]);
                }
            }
            (0..3).map(|c| (hi[c] - lo[c]).powi(2)).sum::<f64>().sqrt()
        };
        // Clamp when the deformed cloud would span more than twice the
        // part; draw it at a quarter-diagonal instead.
        let requested = max_u * exaggeration;
        let scale = if diag > 0.0 && requested > 2.0 * diag {
            0.25 * diag / max_u
        } else {
            exaggeration
        };
        if scale != exaggeration {
            extra_detail.push(format!(
                "max |u| = {max_u:.4} dwarfs the part (diagonal {diag:.4}) — \
                 deformation drawn rescaled x{scale:.3e}, not to scale"
            ));
        } else if (exaggeration - 1.0).abs() > 1e-9 {
            extra_detail.push(format!("deformed x{exaggeration} · max |u| = {max_u:.4}"));
        } else {
            extra_detail.push(format!("deformed view · max |u| = {max_u:.4}"));
        }
        let data = displacement.data.clone();
        for (p, u) in mesh.node_positions.iter_mut().zip(&data) {
            *p += u * scale;
        }
    }
    if let Some(contact) = mesh
        .node_fields
        .iter()
        .find(|f| f.name == "contact_force" && f.components == 3)
    {
        let total_fz: f64 = contact.data.chunks_exact(3).map(|f| f[2]).sum();
        let touching = contact
            .data
            .chunks_exact(3)
            .filter(|f| f[0] != 0.0 || f[1] != 0.0 || f[2] != 0.0)
            .count();
        if touching > 0 {
            extra_detail.push(format!("contact Fz = {total_fz:.4} over {touching} nodes"));
        }
    }

    let position = |node: u32| -> [f32; 3] {
        let p = mesh.node_position(node as usize);
        [p[0] as f32, p[1] as f32, p[2] as f32]
    };

    // Flat-shaded triangle soup: two triangles per boundary quad, each with
    // its own face normal (deformed meshes can have non-planar quads),
    // corners carrying the colormap color (white when no field is chosen).
    let mut vertices: Vec<renderer::MeshVertex> = Vec::with_capacity(faces.len() * 6);
    let mut emit_triangle = |corners: [([f32; 3], [f32; 4]); 3]| {
        let [(a, _), (b, _), (c, _)] = corners;
        let (ab, ac) = (Vec3::from(b) - Vec3::from(a), Vec3::from(c) - Vec3::from(a));
        let normal = ab.cross(ac).normalize_or_zero().to_array();
        for (p, color) in corners {
            vertices.push(renderer::MeshVertex::colored(p, normal, color));
        }
    };
    for (element, quad) in &faces {
        let corner = |slot: usize| -> ([f32; 3], [f32; 4]) {
            let node = quad[slot];
            let color = match &color_source {
                Some(ColorSource::Node(values)) => color_for(values[node as usize]),
                Some(ColorSource::Element(values)) => color_for(values[*element as usize]),
                None => [1.0; 4],
            };
            (position(node), color)
        };
        let [a, b, c, d] = [corner(0), corner(1), corner(2), corner(3)];
        emit_triangle([a, b, c]);
        emit_triangle([a, c, d]);
    }

    // Wireframe from the quads' perimeter edges, deduplicated by node pair.
    let mut seen = std::collections::HashSet::new();
    let mut segments = Vec::new();
    for (_, quad) in &faces {
        for (a, b) in [
            (quad[0], quad[1]),
            (quad[1], quad[2]),
            (quad[2], quad[3]),
            (quad[3], quad[0]),
        ] {
            if seen.insert((a.min(b), a.max(b))) {
                segments.push(renderer::LineSegment {
                    start: position(a),
                    end: position(b),
                    color: [0.05, 0.06, 0.08, 0.9],
                });
            }
        }
    }

    // Bar2 strut meshes have no boundary faces: draw every strut as a
    // hexagonal capped prism at the mesh's own `radius` field (the base
    // radius — the scale-to-radius exponent belongs to the realization
    // operator's config, so the designed radii are what the strut_model
    // output's preview shows; colormap element:stiffness_scale to see the
    // design here). Radial side normals shade the prisms as round tubes;
    // joints rely on overlap at shared nodes rather than sphere blending.
    if mesh.element_kind == volumetric::fea::FeaElementKind::Bar2 {
        let radius_field = mesh
            .element_fields
            .iter()
            .find(|f| f.name == "radius" && f.components == 1);
        // Fallback for meshes without radii: a tenth of the mean strut
        // length, so the structure still reads.
        let mean_length = {
            let total: f32 = (0..mesh.element_count())
                .map(|e| {
                    let p = position(mesh.element(e)[0]);
                    let q = position(mesh.element(e)[1]);
                    (Vec3::from(q) - Vec3::from(p)).length()
                })
                .sum();
            total / mesh.element_count().max(1) as f32
        };
        let fallback_radius = (mean_length / 10.0).max(1e-6);

        const SIDES: usize = 6;
        for e in 0..mesh.element_count() {
            let [na, nb] = [mesh.element(e)[0], mesh.element(e)[1]];
            let a = Vec3::from(position(na));
            let b = Vec3::from(position(nb));
            let axis = b - a;
            let length = axis.length();
            if !(length.is_finite() && length > 0.0) {
                continue;
            }
            let axis = axis / length;
            let radius = radius_field
                .map(|f| f.data[e] as f32)
                .filter(|r| r.is_finite() && *r > 0.0)
                .unwrap_or(fallback_radius);
            let (color_a, color_b) = match &color_source {
                Some(ColorSource::Element(values)) => {
                    let c = color_for(values[e]);
                    (c, c)
                }
                Some(ColorSource::Node(values)) => (
                    color_for(values[na as usize]),
                    color_for(values[nb as usize]),
                ),
                None => {
                    let c = [0.82, 0.85, 0.9, 1.0];
                    (c, c)
                }
            };

            // An orthonormal ring basis perpendicular to the strut.
            let seed = if axis.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            let u = axis.cross(seed).normalize();
            let w = axis.cross(u);
            let ring_dir = |k: usize| -> Vec3 {
                let theta = std::f32::consts::TAU * k as f32 / SIDES as f32;
                u * theta.cos() + w * theta.sin()
            };

            // Sides: radial (smooth) normals, colors interpolating the
            // strut's ends.
            for k in 0..SIDES {
                let (d0, d1) = (ring_dir(k), ring_dir((k + 1) % SIDES));
                let quad = [
                    (a + d0 * radius, d0, color_a),
                    (b + d0 * radius, d0, color_b),
                    (b + d1 * radius, d1, color_b),
                    (a + d1 * radius, d1, color_a),
                ];
                for idx in [0, 1, 2, 0, 2, 3] {
                    let (p, n, c) = quad[idx];
                    vertices.push(renderer::MeshVertex::colored(p.to_array(), n.to_array(), c));
                }
            }
            // Flat end caps (fans anchored at ring vertex 0).
            for k in 1..SIDES - 1 {
                for (p0, p1, p2, normal, color) in [
                    (
                        a + ring_dir(0) * radius,
                        a + ring_dir(k + 1) * radius,
                        a + ring_dir(k) * radius,
                        -axis,
                        color_a,
                    ),
                    (
                        b + ring_dir(0) * radius,
                        b + ring_dir(k) * radius,
                        b + ring_dir(k + 1) * radius,
                        axis,
                        color_b,
                    ),
                ] {
                    for p in [p0, p1, p2] {
                        vertices.push(renderer::MeshVertex::colored(
                            p.to_array(),
                            normal.to_array(),
                            color,
                        ));
                    }
                }
            }
        }
    }

    // Point1 clouds go through the point pipeline: one screen-space dot
    // per point, coloured by the chosen colormap field, else by the
    // cloud's own `color` field (a scanned cloud's RGB), else neutral.
    // Every point is submitted; the retained upload reports any dropped
    // at the GPU buffer limit.
    let mut points: Vec<renderer::PointInstance> = Vec::new();
    let mut point_detail = None;
    if mesh.element_kind == volumetric::fea::FeaElementKind::Point1 {
        let count = mesh.element_count();
        let rgb = mesh
            .node_fields
            .iter()
            .find(|f| f.name == volumetric::fea::COLOR_FIELD_NAME && f.components == 3);
        points.reserve(count);
        for e in 0..count {
            let node = mesh.element(e)[0];
            let color = match (&color_source, rgb) {
                (Some(ColorSource::Node(values)), _) => color_for(values[node as usize]),
                (Some(ColorSource::Element(values)), _) => color_for(values[e]),
                (None, Some(field)) => {
                    let c = &field.data[node as usize * 3..node as usize * 3 + 3];
                    [
                        field_channel_to_linear(c[0]),
                        field_channel_to_linear(c[1]),
                        field_channel_to_linear(c[2]),
                        1.0,
                    ]
                }
                (None, None) => [0.82, 0.85, 0.9, 1.0],
            };
            points.push(renderer::PointInstance {
                position: position(node),
                color,
            });
        }
        if rgb.is_some() {
            point_detail = Some(if color_source.is_some() {
                "the cloud's colours are hidden by the colormap".to_string()
            } else {
                "coloured by the cloud's color field".to_string()
            });
        }
    }

    let mut bounds = PreviewBounds {
        min: (f32::INFINITY, f32::INFINITY, f32::INFINITY),
        max: (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
    };
    for n in 0..mesh.node_count() {
        let p = position(n as u32);
        bounds.min = (
            bounds.min.0.min(p[0]),
            bounds.min.1.min(p[1]),
            bounds.min.2.min(p[2]),
        );
        bounds.max = (
            bounds.max.0.max(p[0]),
            bounds.max.1.max(p[1]),
            bounds.max.2.max(p[2]),
        );
    }
    if mesh.node_count() == 0 {
        bounds = PreviewBounds {
            min: (-1.0, -1.0, -1.0),
            max: (1.0, 1.0, 1.0),
        };
    }

    let mut detail = vec![match mesh.element_kind {
        volumetric::fea::FeaElementKind::Bar2 => format!(
            "FEA strut mesh: {} nodes · {} struts",
            mesh.node_count(),
            mesh.element_count()
        ),
        volumetric::fea::FeaElementKind::Point1 => {
            format!("Point cloud: {} points", mesh.element_count())
        }
        volumetric::fea::FeaElementKind::Hex8 => format!(
            "FEA mesh: {} nodes · {} elements · {} boundary faces",
            mesh.node_count(),
            mesh.element_count(),
            faces.len()
        ),
    }];
    detail.extend(point_detail);
    detail.extend(extra_detail);
    let stats = OutputStats {
        triangles: vertices.len() / 3,
        points: points.len(),
        detail,
        fea_fields,
        mesh_ms: build_start.elapsed().as_secs_f64() * 1000.0,
        ..Default::default()
    };

    let mut scene = renderer::SceneData::new();
    if !vertices.is_empty() {
        scene.add_mesh(
            renderer::MeshData {
                vertices,
                indices: None,
            },
            glam::Mat4::IDENTITY,
            renderer::MaterialId(0),
        );
    }
    let is_cloud = !points.is_empty();
    if is_cloud {
        // Dense scans read better as fine dots; sparse site sets as
        // markers you can see.
        let size = if points.len() > 200_000 { 2.5 } else { 5.0 };
        scene.add_points(
            renderer::PointData { points },
            glam::Mat4::IDENTITY,
            renderer::PointStyle {
                size,
                size_mode: renderer::WidthMode::ScreenSpace,
                shape: renderer::PointShape::Circle,
                depth_mode: renderer::DepthMode::Normal,
            },
        );
    }

    Ok(PreviewEntity {
        scene,
        bounds,
        stats,
        wireframe_lines: (!is_cloud).then_some(renderer::LineData { segments }),
        subspace: None,
    })
}

/// Preview of a general triangle mesh: the triangles exactly as they are,
/// drawn double-sided so open and non-manifold meshes (a scan with holes, a
/// single free triangle) render cleanly from every angle, with a wireframe
/// of the unique edges.
fn build_tri_mesh_preview(
    request: &PreviewRequest,
    build_start: web_time::Instant,
) -> Result<PreviewEntity, String> {
    let mesh = volumetric::trimesh::decode_tri_mesh(request.data.as_slice())?;

    let position = |vertex: u32| -> [f32; 3] {
        let p = mesh.position(vertex as usize);
        [p[0] as f32, p[1] as f32, p[2] as f32]
    };

    // Per-vertex colours (a scanned or painted mesh's `color` field) tint
    // the corners; otherwise the plain material shows.
    let rgb = mesh
        .vertex_fields
        .iter()
        .find(|f| f.name == volumetric::trimesh::COLOR_FIELD_NAME && f.components == 3);
    let color_of = |vertex: u32| -> [f32; 4] {
        match rgb {
            Some(field) => {
                let c = &field.data[vertex as usize * 3..vertex as usize * 3 + 3];
                [
                    field_channel_to_linear(c[0]),
                    field_channel_to_linear(c[1]),
                    field_channel_to_linear(c[2]),
                    1.0,
                ]
            }
            None => [1.0; 4],
        }
    };

    let mut vertices: Vec<renderer::MeshVertex> = Vec::with_capacity(mesh.triangle_count() * 6);
    let mut emit = |corners: [u32; 3]| {
        let [a, b, c] = corners.map(position);
        let (ab, ac) = (Vec3::from(b) - Vec3::from(a), Vec3::from(c) - Vec3::from(a));
        let normal = ab.cross(ac).normalize_or_zero().to_array();
        for (v, p) in corners.into_iter().zip([a, b, c]) {
            vertices.push(renderer::MeshVertex::colored(p, normal, color_of(v)));
        }
    };
    for t in 0..mesh.triangle_count() {
        let [i, j, k] = mesh.triangle(t);
        emit([i, j, k]);
        emit([i, k, j]); // back face, so open meshes show from both sides
    }

    let mut seen = std::collections::HashSet::new();
    let mut segments = Vec::new();
    for t in 0..mesh.triangle_count() {
        let [i, j, k] = mesh.triangle(t);
        for (a, b) in [(i, j), (j, k), (k, i)] {
            if seen.insert((a.min(b), a.max(b))) {
                segments.push(renderer::LineSegment {
                    start: position(a),
                    end: position(b),
                    color: [0.05, 0.06, 0.08, 0.9],
                });
            }
        }
    }

    let bounds = match mesh.bounds() {
        Some(b) => PreviewBounds {
            min: (b[0] as f32, b[2] as f32, b[4] as f32),
            max: (b[1] as f32, b[3] as f32, b[5] as f32),
        },
        None => PreviewBounds {
            min: (-1.0, -1.0, -1.0),
            max: (1.0, 1.0, 1.0),
        },
    };

    let mut detail = vec![format!(
        "triangle mesh: {} vertices · {} triangles",
        mesh.vertex_count(),
        mesh.triangle_count()
    )];
    if rgb.is_some() {
        detail.push("coloured by the mesh's color field".to_string());
    }
    let stats = OutputStats {
        triangles: vertices.len() / 3,
        detail,
        mesh_ms: build_start.elapsed().as_secs_f64() * 1000.0,
        ..Default::default()
    };

    let mut scene = renderer::SceneData::new();
    scene.add_mesh(
        renderer::MeshData {
            vertices,
            indices: None,
        },
        glam::Mat4::IDENTITY,
        renderer::MaterialId(0),
    );

    Ok(PreviewEntity {
        scene,
        bounds,
        stats,
        wireframe_lines: Some(renderer::LineData { segments }),
        subspace: None,
    })
}

/// Preview for a Subspace value: decode and validate now, but bake no
/// geometry — the gizmo is generated per frame in `submit_scene`, sized
/// to the whole scene. The placeholder bounds around the chart origin
/// give the camera something to frame when the gizmo is alone.
fn build_subspace_preview(
    request: &PreviewRequest,
    build_start: web_time::Instant,
) -> Result<PreviewEntity, String> {
    let subspace = volumetric::subspace::decode_subspace(request.data.as_slice())?;
    if subspace.ambient() > 3 {
        return Err(format!(
            "cannot draw a subspace of {}-space in the 3D viewport",
            subspace.ambient()
        ));
    }
    let kind = match subspace.rank() {
        0 => "point",
        1 => "line",
        2 => "plane",
        _ => "frame",
    };
    let stats = OutputStats {
        detail: vec![format!(
            "subspace: {kind} (rank {}) in {}-space",
            subspace.rank(),
            subspace.ambient()
        )],
        mesh_ms: build_start.elapsed().as_secs_f64() * 1000.0,
        ..Default::default()
    };
    let origin = pad3(&subspace.origin);
    Ok(PreviewEntity {
        scene: renderer::SceneData::new(),
        bounds: PreviewBounds {
            min: (origin.x - 1.0, origin.y - 1.0, origin.z - 1.0),
            max: (origin.x + 1.0, origin.y + 1.0, origin.z + 1.0),
        },
        stats,
        wireframe_lines: None,
        subspace: Some(subspace),
    })
}

fn triangles_to_mesh_vertices(triangles: &[volumetric::Triangle]) -> Vec<renderer::MeshVertex> {
    let mut out = Vec::with_capacity(triangles.len() * 3);

    for tri in triangles {
        let a = tri.vertices[0];
        let b = tri.vertices[1];
        let c = tri.vertices[2];
        let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
        let ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2);
        let face_n = (
            ab.1 * ac.2 - ab.2 * ac.1,
            ab.2 * ac.0 - ab.0 * ac.2,
            ab.0 * ac.1 - ab.1 * ac.0,
        );
        let avg_n = (
            tri.normals[0].0 + tri.normals[1].0 + tri.normals[2].0,
            tri.normals[0].1 + tri.normals[1].1 + tri.normals[2].1,
            tri.normals[0].2 + tri.normals[1].2 + tri.normals[2].2,
        );
        let dot = face_n.0 * avg_n.0 + face_n.1 * avg_n.1 + face_n.2 * avg_n.2;
        let idxs: [usize; 3] = if dot < 0.0 { [0, 2, 1] } else { [0, 1, 2] };

        for i in idxs {
            let v = tri.vertices[i];
            let n = tri.normals[i];
            let normal = if n.0 == 0.0 && n.1 == 0.0 && n.2 == 0.0 {
                [0.0, 1.0, 0.0]
            } else {
                [n.0, n.1, n.2]
            };
            out.push(renderer::MeshVertex::new([v.0, v.1, v.2], normal));
        }
    }

    out
}

pub fn format_error_chain(error: anyhow::Error) -> String {
    error
        .chain()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

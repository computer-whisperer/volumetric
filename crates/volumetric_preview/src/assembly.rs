//! Assemblies and mechanisms in the scene: each part meshed on its own
//! (its unposed bytes are its mesh-cache key, so a state change re-meshes
//! nothing) and drawn under its pose in the part's tint, with every moving
//! joint's axis as a line through its posed position.

use std::sync::Arc;

use glam::{Mat4, Vec3};
use volumetric::mechanism::{Assembly, JointKind, Mechanism, Rigid, WORLD, decode_assembly};
use volumetric_renderer as renderer;

use crate::scene::{
    MeshJob, PreviewStage, format_error_chain, mesh_edge_lines, mesh_vertices, part_tint,
    triangles_to_mesh_vertices,
};
use crate::{
    OutputStats, PreviewBounds, PreviewEntity, PreviewMeshPlan, PreviewPlan, PreviewRequest,
};

/// Colour of a revolute joint's axis line.
const REVOLUTE_COLOR: [f32; 4] = [1.0, 0.55, 0.15, 1.0];
/// Colour of a prismatic joint's axis line.
const PRISMATIC_COLOR: [f32; 4] = [0.3, 0.75, 1.0, 1.0];
/// Axis line half-length as a fraction of the scene's diagonal.
const AXIS_HALF_FRACTION: f32 = 0.35;

/// The half-length the joint axes are drawn with for a scene of `bounds`.
pub fn joint_axis_half(bounds: &PreviewBounds) -> f32 {
    let diagonal = (bounds.max_vec3() - bounds.min_vec3()).length();
    if diagonal.is_finite() && diagonal > 0.0 {
        diagonal * AXIS_HALF_FRACTION
    } else {
        1.0
    }
}

/// What the postlude needs to finish an assembly once its parts' meshes
/// exist.
pub struct PendingAssembly {
    assembly: Arc<Assembly>,
    poses: Vec<Mat4>,
    /// Each part's mesh identity: its mesh-cache key folded with its name
    /// (the tint is baked into the vertices).
    keys: Vec<[u8; 32]>,
}

/// The identity of a part's ASN2 mesh as uploaded: the mesh-cache key of
/// its bytes and recipe, folded with the part's name.
fn part_mesh_key(
    name: &str,
    model: &[u8],
    config: &volumetric::adaptive_surface_nets_2::AdaptiveMeshConfig2,
) -> [u8; 32] {
    let key = volumetric::MeshCacheKey::new(model, config);
    let mut bytes = key.as_bytes().to_vec();
    bytes.extend_from_slice(name.as_bytes());
    volumetric::content_fingerprint(&bytes)
}

/// A part's pose as the renderer's matrix.
fn pose_matrix(pose: &Rigid) -> Mat4 {
    Mat4::from_cols_array(&pose.to_cols_array_f32())
}

/// The lines the viewport's statistics show for a mechanism at `state`.
fn mechanism_detail(mechanism: &Mechanism, state: &volumetric::f64_map::F64Map) -> Vec<String> {
    let mut detail = vec![format!(
        "mechanism: {} parts · {} joints · {} states",
        mechanism.parts.len(),
        mechanism.joints.len(),
        mechanism.state_keys().len()
    )];
    let states: Vec<String> = mechanism
        .joints
        .iter()
        .filter(|j| j.is_state())
        .map(|j| {
            let value = state.get(&j.name).copied().unwrap_or(j.default);
            match j.kind {
                JointKind::Revolute => format!("{} {value:.1}°", j.name),
                _ => format!("{} {:.1} mm", j.name, value * 1000.0),
            }
        })
        .collect();
    if !states.is_empty() {
        detail.push(format!("state: {}", states.join(", ")));
    }
    detail
}

/// Every moving joint's axis under `poses` (one per part, in part order):
/// a segment of `half` either side of the axis origin, orange for a
/// revolute joint and blue for a prismatic one.
pub fn joint_axis_lines(mechanism: &Mechanism, poses: &[Rigid], half: f32) -> renderer::LineData {
    let mut segments = Vec::new();
    for joint in &mechanism.joints {
        let Some(axis) = joint.axis.and_then(|a| a.unit()) else {
            continue;
        };
        let color = match joint.kind {
            JointKind::Fixed => continue,
            JointKind::Revolute => REVOLUTE_COLOR,
            JointKind::Prismatic => PRISMATIC_COLOR,
        };
        let parent = if joint.parent == WORLD {
            Rigid::IDENTITY
        } else {
            mechanism
                .parts
                .iter()
                .position(|p| *p == joint.parent)
                .map(|i| poses[i])
                .unwrap_or(Rigid::IDENTITY)
        };
        let origin = parent.apply(axis.origin);
        let direction = parent.apply_vector(axis.direction);
        let o = Vec3::new(origin[0] as f32, origin[1] as f32, origin[2] as f32);
        let d = Vec3::new(
            direction[0] as f32,
            direction[1] as f32,
            direction[2] as f32,
        );
        segments.push(renderer::LineSegment {
            start: (o - d * half).to_array(),
            end: (o + d * half).to_array(),
            color,
        });
    }
    renderer::LineData { segments }
}

/// The style the joint axes are drawn in.
pub fn joint_axis_style() -> renderer::LineStyle {
    renderer::LineStyle {
        width: 2.0,
        width_mode: renderer::WidthMode::ScreenSpace,
        pattern: renderer::LinePattern::Solid,
        depth_mode: renderer::DepthMode::Normal,
    }
}

/// The world-space box of `bounds` transformed by `pose`.
fn posed_bounds(pose: &Mat4, min: (f32, f32, f32), max: (f32, f32, f32)) -> PreviewBounds {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for corner in 0..8 {
        let p = Vec3::new(
            if corner & 1 == 0 { min.0 } else { max.0 },
            if corner & 2 == 0 { min.1 } else { max.1 },
            if corner & 4 == 0 { min.2 } else { max.2 },
        );
        let q = pose.transform_point3(p);
        lo = lo.min(q);
        hi = hi.max(q);
    }
    PreviewBounds {
        min: (lo.x, lo.y, lo.z),
        max: (hi.x, hi.y, hi.z),
    }
}

/// A part's built geometry before it is placed: its mesh (or point
/// cloud), the edge lines for the wireframe toggle, and its own bounds.
struct BuiltPart {
    mesh: Option<renderer::MeshData>,
    points: Option<renderer::PointData>,
    wireframe: Option<renderer::LineData>,
    bounds: ((f32, f32, f32), (f32, f32, f32)),
    /// The mesh's identity for GPU residency, when it has a stable one.
    key: Option<[u8; 32]>,
}

/// Places every built part under its pose, tinted by name, and draws the
/// joints' axes; the wireframe overlay carries every part's edges, posed.
fn place(
    request: &PreviewRequest,
    assembly: &Arc<Assembly>,
    poses: &[Mat4],
    built: Vec<BuiltPart>,
    mut stats: OutputStats,
    build_start: web_time::Instant,
) -> PreviewEntity {
    let mut scene = renderer::SceneData::new();
    let mut mesh_keys = Vec::new();
    let mut part_of_mesh = Vec::new();
    let mut wireframe = renderer::LineData {
        segments: Vec::new(),
    };
    let mut bounds: Option<PreviewBounds> = None;
    for (part_index, ((part, pose), mut built)) in
        assembly.parts.iter().zip(poses).zip(built).enumerate()
    {
        let tint = part_tint(&part.name);
        if let Some(mut mesh) = built.mesh.take() {
            for vertex in &mut mesh.vertices {
                vertex.color = tint;
            }
            scene.add_mesh(mesh, *pose, renderer::MaterialId(0));
            mesh_keys.push(built.key);
            part_of_mesh.push(part_index);
        }
        if let Some(mut points) = built.points.take() {
            for point in &mut points.points {
                point.color = tint;
            }
            scene.add_points(
                points,
                *pose,
                renderer::PointStyle {
                    size: 4.0,
                    size_mode: renderer::WidthMode::ScreenSpace,
                    shape: renderer::PointShape::Circle,
                    depth_mode: renderer::DepthMode::Normal,
                },
            );
        }
        if let Some(lines) = built.wireframe.take() {
            wireframe
                .segments
                .extend(lines.segments.into_iter().map(|s| renderer::LineSegment {
                    start: pose.transform_point3(Vec3::from(s.start)).to_array(),
                    end: pose.transform_point3(Vec3::from(s.end)).to_array(),
                    color: s.color,
                }));
        }
        let part_bounds = posed_bounds(pose, built.bounds.0, built.bounds.1);
        bounds = Some(match bounds {
            Some(b) => b.union(part_bounds),
            None => part_bounds,
        });
    }
    let bounds = bounds.unwrap_or(PreviewBounds {
        min: (-1.0, -1.0, -1.0),
        max: (1.0, 1.0, 1.0),
    });
    let half = joint_axis_half(&bounds);
    let rigid_poses = assembly.poses();
    scene.add_lines(
        joint_axis_lines(&assembly.mechanism, &rigid_poses, half),
        Mat4::IDENTITY,
        joint_axis_style(),
    );
    stats
        .detail
        .splice(0..0, mechanism_detail(&assembly.mechanism, &assembly.state));
    stats
        .detail
        .push(format!("Color: part tints ({})", request.asset_id));
    stats.mesh_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    PreviewEntity {
        scene,
        bounds,
        stats,
        wireframe_lines: (!wireframe.segments.is_empty()).then_some(wireframe),
        mesh_keys,
        articulated: Some(Arc::new(crate::Articulated {
            assembly: assembly.clone(),
            part_of_mesh,
        })),
        subspace: None,
    }
}

/// Starts an assembly preview: the cheap plans build every part here; the
/// ASN2 plan hands back one meshing job per part for the caller to run.
pub fn build_assembly_preview(
    request: &PreviewRequest,
    build_start: web_time::Instant,
) -> Result<PreviewStage, String> {
    let assembly = Arc::new(decode_assembly(request.data.as_slice())?);
    let poses: Vec<Mat4> = assembly.poses().iter().map(pose_matrix).collect();
    let fallback;
    let mesh_plan = match &request.plan {
        PreviewPlan::Assembly { mesh } => mesh,
        _ => {
            fallback = PreviewMeshPlan::PointCloud { resolution: 24 };
            &fallback
        }
    };
    let mut stats = OutputStats::default();
    let built: Vec<BuiltPart> = match mesh_plan {
        PreviewMeshPlan::PointCloud { resolution } => assembly
            .parts
            .iter()
            .map(|part| {
                let (points, min, max) =
                    volumetric::sample_model_from_bytes(&part.model, *resolution)
                        .map_err(|e| format!("part `{}`: {}", part.name, format_error_chain(e)))?;
                stats.points += points.len();
                Ok(BuiltPart {
                    mesh: None,
                    points: Some(renderer::convert_points_to_point_data(&points)),
                    wireframe: None,
                    bounds: (min, max),
                    key: None,
                })
            })
            .collect::<Result<_, String>>()?,
        PreviewMeshPlan::MarchingCubes { resolution } => assembly
            .parts
            .iter()
            .map(|part| {
                let (triangles, min, max) =
                    volumetric::generate_marching_cubes_mesh_from_bytes(&part.model, *resolution)
                        .map_err(|e| format!("part `{}`: {}", part.name, format_error_chain(e)))?;
                stats.triangles += triangles.len();
                let vertices = triangles_to_mesh_vertices(&triangles);
                let wireframe = mesh_edge_lines(&vertices, None);
                Ok(BuiltPart {
                    mesh: Some(renderer::MeshData {
                        vertices,
                        indices: None,
                    }),
                    points: None,
                    wireframe: Some(wireframe),
                    bounds: (min, max),
                    key: None,
                })
            })
            .collect::<Result<_, String>>()?,
        PreviewMeshPlan::AdaptiveSurfaceNets2 { .. } => {
            let config = mesh_plan
                .adaptive_surface_nets_config()
                .ok_or_else(|| "missing adaptive surface nets config".to_string())?;
            let jobs = assembly
                .parts
                .iter()
                .map(|part| MeshJob {
                    data: Arc::new(part.model.clone()),
                    config: config.clone(),
                })
                .collect();
            let keys = assembly
                .parts
                .iter()
                .map(|part| part_mesh_key(&part.name, &part.model, &config))
                .collect();
            return Ok(PreviewStage::NeedsMesh(
                crate::scene::PendingMesh::assembly(
                    jobs,
                    PendingAssembly {
                        assembly,
                        poses,
                        keys,
                    },
                    stats,
                    build_start,
                ),
            ));
        }
    };
    Ok(PreviewStage::Done(Box::new(place(
        request,
        &assembly,
        &poses,
        built,
        stats,
        build_start,
    ))))
}

/// Finishes an ASN2 assembly preview from its parts' meshes, in part order.
pub fn finish_assembly_preview(
    request: &PreviewRequest,
    pending: PendingAssembly,
    meshes: &[Arc<volumetric::AdaptiveMeshV2Result>],
    mut stats: OutputStats,
    build_start: web_time::Instant,
) -> PreviewEntity {
    let built = meshes
        .iter()
        .zip(&pending.keys)
        .map(|(mesh, key)| {
            stats.triangles += mesh.indices.len() / 3;
            stats.samples += mesh.stats.total_samples;
            let vertices = mesh_vertices(mesh, &mut stats);
            let wireframe = mesh_edge_lines(&vertices, Some(&mesh.indices));
            BuiltPart {
                mesh: Some(renderer::MeshData {
                    vertices,
                    indices: Some(mesh.indices.clone()),
                }),
                points: None,
                wireframe: Some(wireframe),
                bounds: (mesh.bounds_min, mesh.bounds_max),
                key: Some(*key),
            }
        })
        .collect();
    place(
        request,
        &pending.assembly,
        &pending.poses,
        built,
        stats,
        build_start,
    )
}

/// A mechanism on its own: its joints' axes at the rest state, sized to
/// the axes' spread.
pub fn build_mechanism_preview(
    request: &PreviewRequest,
    build_start: web_time::Instant,
) -> Result<PreviewEntity, String> {
    let mechanism = volumetric::mechanism::decode_mechanism(request.data.as_slice())?;
    let poses = vec![Rigid::IDENTITY; mechanism.parts.len()];
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for axis in mechanism.joints.iter().filter_map(|j| j.axis) {
        let o = Vec3::new(
            axis.origin[0] as f32,
            axis.origin[1] as f32,
            axis.origin[2] as f32,
        );
        lo = lo.min(o);
        hi = hi.max(o);
    }
    if !(lo.x.is_finite() && hi.x.is_finite()) {
        lo = Vec3::splat(-1.0);
        hi = Vec3::splat(1.0);
    }
    let spread = (hi - lo).length().max(0.2);
    let half = spread * AXIS_HALF_FRACTION;
    let mut scene = renderer::SceneData::new();
    scene.add_lines(
        joint_axis_lines(&mechanism, &poses, half),
        Mat4::IDENTITY,
        joint_axis_style(),
    );
    let stats = OutputStats {
        detail: mechanism_detail(&mechanism, &mechanism.default_state()),
        mesh_ms: build_start.elapsed().as_secs_f64() * 1000.0,
        ..Default::default()
    };
    Ok(PreviewEntity {
        scene,
        bounds: PreviewBounds {
            min: (lo.x - half, lo.y - half, lo.z - half),
            max: (hi.x + half, hi.y + half, hi.z + half),
        },
        stats,
        wireframe_lines: None,
        mesh_keys: Vec::new(),
        articulated: None,
        subspace: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use volumetric::mechanism::{Axis, Joint};

    #[test]
    fn joint_axes_follow_their_parents_pose() {
        // `a` fixed to the world; `b` swivels about z through (1, 0, 0) on
        // `a`; `c` slides along x through (2, 0, 0) on `b`.
        let joint = |name: &str, kind, parent: &str, child: &str, axis| Joint {
            name: name.to_string(),
            kind,
            parent: parent.to_string(),
            child: child.to_string(),
            axis,
            min: -180.0,
            max: 180.0,
            default: 0.0,
            drive: None,
            continuous: false,
        };
        let mechanism = Mechanism::new(
            ["a", "b", "c"].map(str::to_string).to_vec(),
            vec![
                joint("mount", JointKind::Fixed, WORLD, "a", None),
                joint(
                    "swivel",
                    JointKind::Revolute,
                    "a",
                    "b",
                    Some(Axis {
                        origin: [1.0, 0.0, 0.0],
                        direction: [0.0, 0.0, 1.0],
                    }),
                ),
                joint(
                    "slide",
                    JointKind::Prismatic,
                    "b",
                    "c",
                    Some(Axis {
                        origin: [2.0, 0.0, 0.0],
                        direction: [1.0, 0.0, 0.0],
                    }),
                ),
            ],
        );
        mechanism.validate().unwrap();
        let state = volumetric::f64_map::F64Map::from([("swivel".to_string(), 90.0)]);
        let poses = mechanism.pose(&state).unwrap();
        let lines = joint_axis_lines(&mechanism, &poses, 0.5);
        // The fixed joint draws nothing; the swivel's axis is unmoved; the
        // slide's axis rides on the swivelled `b`: through (1, 1, 0) along y.
        assert_eq!(lines.segments.len(), 2);
        let swivel = &lines.segments[0];
        assert_eq!(swivel.color, REVOLUTE_COLOR);
        assert_eq!(swivel.start, [1.0, 0.0, -0.5]);
        assert_eq!(swivel.end, [1.0, 0.0, 0.5]);
        let slide = &lines.segments[1];
        assert_eq!(slide.color, PRISMATIC_COLOR);
        let close = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5);
        assert!(close(slide.start, [1.0, 0.5, 0.0]), "{:?}", slide.start);
        assert!(close(slide.end, [1.0, 1.5, 0.0]), "{:?}", slide.end);
    }
}

//! The frame, measured: headless renders whose pixels and picks are
//! checked against what the scene says they must be. Each scenario runs
//! on the default adapter and again on a GLES adapter under WebGL2's
//! limits, the nearest thing to the web fallback that runs here. A
//! missing adapter skips that run and says so.

use glam::{Mat4, Vec3};

use crate::offscreen::Offscreen;
use crate::{
    CameraView, DepthMode, GridPlanes, LineData, LinePattern, LineSegment, LineStyle, MaterialId,
    MeshData, MeshVertex, ObjectId, RenderSettings, Renderer, WidthMode,
};

const W: u32 = 128;
const H: u32 = 96;
const BACKGROUND: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// Runs `scenario` on every adapter kind available.
fn on_each_backend(scenario: impl Fn(&Offscreen)) {
    for (name, offscreen) in [
        ("default", Offscreen::new()),
        ("webgl2-like", Offscreen::new_webgl2_like()),
    ] {
        match offscreen {
            Ok(offscreen) => {
                eprintln!("{name}: {}", offscreen.adapter_name());
                scenario(&offscreen);
            }
            Err(err) => eprintln!("{name}: skipped ({err})"),
        }
    }
}

/// A rectangle centred on `centre` spanning `±u`, `±v`, facing `u × v`.
fn quad(centre: Vec3, u: Vec3, v: Vec3, color: [f32; 4]) -> MeshData {
    let normal = u.cross(v).normalize().to_array();
    let vertex = |p: Vec3| MeshVertex::colored(p.to_array(), normal, color);
    MeshData {
        vertices: vec![
            vertex(centre - u - v),
            vertex(centre + u - v),
            vertex(centre + u + v),
            vertex(centre - u + v),
        ],
        indices: Some(vec![0, 1, 2, 0, 2, 3]),
    }
}

/// From 3 m up the Z axis, looking down at the origin; +Y is up the frame.
fn view() -> CameraView {
    CameraView::look_at(
        Vec3::new(0.0, 0.0, 3.0),
        Vec3::ZERO,
        Vec3::Y,
        0.8,
        W as f32 / H as f32,
        0.1,
        10.0,
    )
}

fn settings(ssao: bool) -> RenderSettings {
    let mut settings = RenderSettings {
        background_color: BACKGROUND,
        ssao_enabled: ssao,
        ..RenderSettings::default()
    };
    settings.grid.planes = GridPlanes::NONE;
    settings
}

fn pixel(rgba: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * W + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2]]
}

fn srgb_byte(linear: f32) -> f32 {
    let encoded = if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    encoded * 255.0
}

fn close(actual: [u8; 3], expected: [f32; 3], tolerance: f32) -> bool {
    (0..3).all(|i| (actual[i] as f32 - expected[i]).abs() <= tolerance)
}

/// A flat surface resolves to the lighting model's value for its normal
/// and vertex colour, and pixels no geometry wrote are the background.
#[test]
fn a_lit_surface_resolves_to_the_lighting_models_value() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let grey = [0.25, 0.25, 0.25, 1.0];
        let mesh = renderer.create_retained_mesh(
            offscreen.device(),
            &quad(Vec3::ZERO, Vec3::X, Vec3::Y, grey),
        );
        renderer.submit_retained_mesh(&mesh, Mat4::IDENTITY, ObjectId(1), MaterialId(0));
        let (rgba, info) = offscreen
            .render_rgba(&mut renderer, &view(), &settings(false))
            .unwrap();
        assert_eq!(info.overflow, None);

        // Facing +Z under the fixed light: ambient plus the diffuse term.
        let n_dot_l = Vec3::new(0.4, 0.7, 0.2).normalize().z;
        let shade = 0.22 + 0.78 * n_dot_l;
        let expected = [0.85, 0.9, 1.0].map(|tint: f32| srgb_byte(0.25 * tint * shade));
        let centre = pixel(&rgba, W / 2, H / 2);
        assert!(close(centre, expected, 3.0), "{centre:?} vs {expected:?}");
        assert_eq!(pixel(&rgba, 2, 2), [0, 0, 255], "background corner");
    });
}

/// A pick reports the nearest mesh's object id and the world point on
/// its surface; over background it reports neither.
#[test]
fn a_pick_reports_the_object_and_world_point_under_a_pixel() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let white = [1.0; 4];
        // A wide quad at z = -0.5 with a small one in front at z = 0.3.
        let far = renderer.create_retained_mesh(
            offscreen.device(),
            &quad(Vec3::ZERO, Vec3::X * 0.8, Vec3::Y * 0.8, white),
        );
        let near = renderer.create_retained_mesh(
            offscreen.device(),
            &quad(Vec3::ZERO, Vec3::X * 0.2, Vec3::Y * 0.2, white),
        );
        assert_eq!(
            offscreen.pick(&mut renderer, (W / 2, H / 2)),
            None,
            "no frame yet"
        );
        renderer.submit_retained_mesh(
            &far,
            Mat4::from_translation(Vec3::new(0.0, 0.0, -0.5)),
            ObjectId(7),
            MaterialId(0),
        );
        renderer.submit_retained_mesh(
            &near,
            Mat4::from_translation(Vec3::new(0.0, 0.0, 0.3)),
            ObjectId(9),
            MaterialId(0),
        );
        offscreen
            .render_rgba(&mut renderer, &view(), &settings(false))
            .unwrap();

        let centre = offscreen
            .pick(&mut renderer, (W / 2, H / 2))
            .expect("a pick");
        assert_eq!(centre.object, ObjectId(9));
        let world = centre.world.expect("a surface");
        assert!((world.z - 0.3).abs() < 1e-3, "{world}");
        assert!(world.x.abs() < 0.03 && world.y.abs() < 0.03, "{world}");

        // Off the small quad but on the wide one: 20 px right of centre is
        // x = 20/64 * tan(0.4) * (4/3) * 3.5 m at the far quad's depth.
        let side = offscreen
            .pick(&mut renderer, (W / 2 + 20, H / 2))
            .expect("a pick");
        assert_eq!(side.object, ObjectId(7));
        let world = side.world.expect("a surface");
        assert!((world.z + 0.5).abs() < 1e-3, "{world}");
        let x = 20.5 / 64.0 * 0.4f32.tan() * (W as f32 / H as f32) * 3.5;
        assert!((world.x - x).abs() < 1e-3, "{world} vs x = {x}");

        let corner = offscreen.pick(&mut renderer, (1, 1)).expect("a pick");
        assert_eq!(corner.object, ObjectId::NONE);
        assert_eq!(corner.world, None);
        assert_eq!(
            offscreen.pick(&mut renderer, (W, 0)),
            None,
            "outside the frame"
        );
    });
}

fn line(y: f32, z: f32, color: [f32; 4]) -> LineData {
    LineData {
        segments: vec![LineSegment {
            start: [-1.0, y, z],
            end: [1.0, y, z],
            color,
        }],
    }
}

fn style(width: f32, depth_mode: DepthMode) -> LineStyle {
    LineStyle {
        width,
        width_mode: WidthMode::ScreenSpace,
        pattern: LinePattern::Solid,
        depth_mode,
    }
}

/// How many pixels of column `x` are mostly `channel`.
fn rows_of(rgba: &[u8], x: u32, channel: usize) -> u32 {
    (0..H)
        .filter(|&y| {
            let p = pixel(rgba, x, y);
            p[channel] > 128 && p[(channel + 1) % 3] < 100 && p[(channel + 2) % 3] < 100
        })
        .count() as u32
}

/// Two immediate line batches in one frame are each drawn with their own
/// width: the frame used to merge them under the last batch's style.
#[test]
fn immediate_line_batches_keep_their_own_styles() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.background_color = [0.0, 0.0, 0.0, 1.0];
        renderer.submit_lines(
            &line(0.5, 0.0, [1.0, 0.0, 0.0, 1.0]),
            Mat4::IDENTITY,
            style(2.0, DepthMode::Overlay),
        );
        renderer.submit_lines(
            &line(-0.5, 0.0, [0.0, 1.0, 0.0, 1.0]),
            Mat4::IDENTITY,
            style(12.0, DepthMode::Overlay),
        );
        let (rgba, _) = offscreen
            .render_rgba(&mut renderer, &view(), &settings)
            .unwrap();
        let (thin, thick) = (rows_of(&rgba, W / 2, 0), rows_of(&rgba, W / 2, 1));
        assert!((1..=3).contains(&thin), "2 px line covers {thin} rows");
        assert!((10..=13).contains(&thick), "12 px line covers {thick} rows");
    });
}

/// A depth-tested line behind a mesh is hidden by it; the same line as an
/// overlay is drawn over it.
#[test]
fn meshes_occlude_depth_tested_lines_but_not_overlays() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mesh = renderer.create_retained_mesh(
            offscreen.device(),
            &quad(Vec3::ZERO, Vec3::X * 0.5, Vec3::Y * 0.5, [1.0; 4]),
        );
        let red = [1.0, 0.0, 0.0, 1.0];
        let frame = |renderer: &mut Renderer, depth_mode| {
            renderer.submit_retained_mesh(&mesh, Mat4::IDENTITY, ObjectId(1), MaterialId(0));
            renderer.submit_lines(
                &line(0.0, -0.5, red),
                Mat4::IDENTITY,
                style(6.0, depth_mode),
            );
            offscreen
                .render_rgba(renderer, &view(), &settings(false))
                .unwrap()
                .0
        };
        let hidden = frame(&mut renderer, DepthMode::Normal);
        assert_eq!(rows_of(&hidden, W / 2, 0), 0, "line shows through the mesh");
        // Past the mesh's edge the line is in the open.
        assert!(rows_of(&hidden, 36, 0) >= 4, "line missing beside the mesh");
        let over = frame(&mut renderer, DepthMode::Overlay);
        assert!(
            rows_of(&over, W / 2, 0) >= 4,
            "overlay line hidden by the mesh"
        );
    });
}

/// Ambient occlusion darkens a floor where a wall stands on it and leaves
/// open floor alone.
#[test]
fn ambient_occlusion_darkens_the_foot_of_a_wall() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let white = [1.0; 4];
        let floor = renderer.create_retained_mesh(
            offscreen.device(),
            &quad(Vec3::ZERO, Vec3::X, Vec3::Y, white),
        );
        // A wall along x = 0, facing -X, standing 1 m off the floor.
        let wall = renderer.create_retained_mesh(
            offscreen.device(),
            &quad(Vec3::new(0.0, 0.0, 0.5), Vec3::Z * 0.5, Vec3::Y, white),
        );
        // From the wall's facing side, 45 degrees up: the wall's foot is
        // the frame's centre and the floor runs down the frame from it.
        let oblique = CameraView::look_at(
            Vec3::new(-2.0, 0.0, 2.0),
            Vec3::ZERO,
            Vec3::Z,
            0.8,
            W as f32 / H as f32,
            0.1,
            10.0,
        );
        // The bias is in non-linear depth, where this view's whole AO
        // radius spans 0.006: the default 0.025 can never be exceeded.
        let frame = |renderer: &mut Renderer, ssao| {
            renderer.submit_retained_mesh(&floor, Mat4::IDENTITY, ObjectId(1), MaterialId(0));
            renderer.submit_retained_mesh(&wall, Mat4::IDENTITY, ObjectId(2), MaterialId(0));
            let settings = RenderSettings {
                ssao_bias: 0.0005,
                ..settings(ssao)
            };
            offscreen
                .render_rgba(renderer, &oblique, &settings)
                .unwrap()
                .0
        };
        // Floor rows below the centre: 1 px down is 0.03 m from the wall,
        // 28 px down is 0.8 m, beyond the 0.5 m AO radius. Averaged over a
        // few columns, since the AO is noisy.
        let mean = |rgba: &[u8], y: u32| {
            (W / 2 - 4..W / 2 + 4)
                .map(|x| pixel(rgba, x, y)[1] as f32)
                .sum::<f32>()
                / 8.0
        };
        let (foot, open) = (H / 2 + 1, H / 2 + 28);
        let plain = frame(&mut renderer, false);
        assert!((mean(&plain, foot) - mean(&plain, open)).abs() < 1.0);
        let occluded = frame(&mut renderer, true);
        assert!(
            mean(&occluded, foot) < mean(&plain, foot) - 5.0,
            "foot {} vs unoccluded {}",
            mean(&occluded, foot),
            mean(&plain, foot)
        );
        assert!(
            (mean(&occluded, open) - mean(&plain, open)).abs() < 1.0,
            "open floor {} vs unoccluded {}",
            mean(&occluded, open),
            mean(&plain, open)
        );
    });
}

/// Writes the placeholder test scene's frame as a binary PPM to the path
/// in `VOLUMETRIC_FRAME_DUMP`, for looking at what the frame draws:
/// `VOLUMETRIC_FRAME_DUMP=/tmp/frame.ppm cargo test -p volumetric_renderer dump_the_test_scene -- --ignored`.
#[test]
#[ignore]
fn dump_the_test_scene() {
    let Ok(path) = std::env::var("VOLUMETRIC_FRAME_DUMP") else {
        return;
    };
    let offscreen = Offscreen::new().unwrap();
    let (w, h) = (960u32, 640u32);
    let mut renderer = offscreen.renderer(w, h);
    let scene = renderer
        .create_retained_scene(offscreen.device(), &crate::test_scenes::create_test_scene());
    for (mesh, transform) in &scene.meshes {
        renderer.submit_retained_mesh(mesh, *transform, ObjectId(1), MaterialId(0));
    }
    for lines in &scene.lines {
        renderer.submit_retained_lines(lines);
    }
    for points in &scene.points {
        renderer.submit_retained_points(points);
    }
    let mut camera = crate::test_scenes::create_test_camera();
    camera.fit_clip_planes();
    let view = CameraView::from_camera(&camera, w as f32 / h as f32);
    let (rgba, _) = offscreen
        .render_rgba(&mut renderer, &view, &RenderSettings::default())
        .unwrap();
    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
    ppm.extend(rgba.chunks_exact(4).flat_map(|px| [px[0], px[1], px[2]]));
    std::fs::write(path, ppm).unwrap();
}

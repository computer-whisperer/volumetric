//! The frame, measured: headless renders whose pixels and picks are
//! checked against what the scene says they must be. Each scenario runs
//! on the default adapter and again on a GLES adapter under WebGL2's
//! limits, the nearest thing to the web fallback that runs here. A
//! missing adapter skips that run and says so.

use glam::{Mat4, Vec2, Vec3};

use crate::offscreen::Offscreen;
use crate::{
    CameraView, DepthMode, GizmoPart, GridSpacing, LineData, LinePattern, LineSegment, LineStyle,
    MaterialId, MeshData, MeshVertex, ObjectId, RenderSettings, Renderer, SurfelData, SurfelVertex,
    ViewGizmo, WidthMode,
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
        texture: None,
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

/// The frame the scenarios start from: no grid, no edge lines, and
/// ambient occlusion as asked.
fn settings(ao: bool) -> RenderSettings {
    let mut settings = RenderSettings {
        background_color: BACKGROUND,
        ..RenderSettings::default()
    };
    settings.ao.enabled = ao;
    settings.edges.enabled = false;
    settings.grid.visible = false;
    settings
}

/// The lighting model, as the resolve pass computes it: the colour of a
/// surface of `albedo` whose normal, and the direction to the eye, are
/// given in the camera's frame. No occlusion, no edge.
fn lit(settings: &RenderSettings, albedo: f32, normal: Vec3, to_eye: Vec3) -> [f32; 3] {
    let material = settings.materials[0];
    let rig = &settings.lighting;
    let exponent = 2.0 / material.roughness.powi(4) - 2.0;
    std::array::from_fn(|c| {
        // The scenarios look straight down, so the world's up is the
        // camera's toward-the-viewer axis.
        let ambient = rig.ground[c] + (rig.sky[c] - rig.ground[c]) * (normal.z * 0.5 + 0.5);
        let mut diffuse = 0.0;
        let mut highlight = 0.0;
        for light in rig.lights {
            let l = Vec3::from(light.direction).normalize();
            diffuse += light.color[c] * normal.dot(l).max(0.0);
            if normal.dot(l) >= 0.0 {
                let h = (l + to_eye).normalize();
                highlight += light.color[c] * normal.dot(h).max(0.0).powf(exponent);
            }
        }
        let color =
            albedo * material.base_tint[c] * (ambient + diffuse) + highlight * material.specular;
        assert!(color < 0.8, "the scenario stays under the shoulder");
        color
    })
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
            offscreen.queue(),
            &quad(Vec3::ZERO, Vec3::X, Vec3::Y, grey),
        );
        renderer.submit_retained_mesh(&mesh, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
        let (rgba, info) = offscreen
            .render_rgba(&mut renderer, &view(), &settings(false))
            .unwrap();
        assert_eq!(info.overflow, None);

        // Facing the viewer at the frame's centre.
        let expected = lit(&settings(false), 0.25, Vec3::Z, Vec3::Z).map(srgb_byte);
        let centre = pixel(&rgba, W / 2, H / 2);
        assert!(close(centre, expected, 3.0), "{centre:?} vs {expected:?}");
        assert_eq!(pixel(&rgba, 2, 2), [0, 0, 255], "background corner");
    });
}

/// A 2×2 checker, top row red then green, bottom row blue then white,
/// with a grey 1×1 level under it.
fn checker() -> crate::TextureData {
    crate::TextureData::from_levels(
        2,
        2,
        vec![
            vec![
                255, 0, 0, 255, 0, 255, 0, 255, //
                0, 0, 255, 255, 255, 255, 255, 255,
            ],
            vec![128, 128, 128, 255],
        ],
    )
}

/// A quad over `±1` in x and y that maps the whole of `texture`, its top
/// row toward +y.
fn textured_quad(texture: &std::sync::Arc<crate::SceneTexture>) -> MeshData {
    let normal = [0.0, 0.0, 1.0];
    let vertex =
        |x: f32, y: f32, uv: [f32; 2]| MeshVertex::textured([x, y, 0.0], normal, [1.0; 4], uv);
    MeshData {
        vertices: vec![
            vertex(-1.0, -1.0, [0.0, 1.0]),
            vertex(1.0, -1.0, [1.0, 1.0]),
            vertex(1.0, 1.0, [1.0, 0.0]),
            vertex(-1.0, 1.0, [0.0, 0.0]),
        ],
        indices: Some(vec![0, 1, 2, 0, 2, 3]),
        texture: Some(texture.clone()),
    }
}

/// Whether the frame shows the checker's red texel, at the quad's top
/// left.
fn shows_red_top_left(offscreen: &Offscreen, renderer: &mut Renderer) -> bool {
    let (rgba, _) = offscreen
        .render_rgba(renderer, &view(), &settings(false))
        .unwrap();
    let (px, py) = at(&view(), Vec3::new(-0.5, 0.5, 0.0));
    let [r, g, b] = pixel(&rgba, px, py).map(i32::from);
    r > g + 60 && r > b + 60
}

/// A texture that can make its pixels again gives them up at the first
/// upload, and every later mesh that takes the copy made then from the
/// same set shares it: nothing is decoded or sent twice. One that cannot
/// keeps them.
#[test]
fn a_reloadable_texture_keeps_only_the_devices_copy() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    on_each_backend(|offscreen| {
        let reloads = Arc::new(AtomicUsize::new(0));
        let counter = reloads.clone();
        let texture = Arc::new(crate::SceneTexture::reloadable(checker(), move || {
            counter.fetch_add(1, Ordering::Relaxed);
            Some(checker())
        }));
        assert!(texture.holds_pixels());
        assert_eq!(texture.size(), (2, 2));

        let mut renderer = offscreen.renderer(W, H);
        let (device, queue) = (offscreen.device(), offscreen.queue());
        let mut uploaded = crate::GpuTextureSet::default();
        let first = renderer.create_retained_mesh_with(
            device,
            queue,
            &textured_quad(&texture),
            &mut uploaded,
        );
        assert!(!texture.holds_pixels(), "dropped once the device has them");
        assert_eq!(uploaded.len(), 1);
        assert_eq!(
            Arc::strong_count(&texture),
            1,
            "the device's copy does not keep the scene texture alive"
        );
        // The mesh goes (its output hidden); the set keeps the copy, so
        // showing it again takes neither the pixels nor a reload.
        drop(first);
        let again = renderer.create_retained_mesh_with(
            device,
            queue,
            &textured_quad(&texture),
            &mut uploaded,
        );
        assert_eq!(reloads.load(Ordering::Relaxed), 0);
        assert_eq!(uploaded.len(), 1);
        renderer.submit_retained_mesh(&again, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
        assert!(shows_red_top_left(offscreen, &mut renderer));

        // Without a set the copy lives only as long as its mesh: the next
        // mesh uploads again, from a reload.
        let _alone = renderer.create_retained_mesh(device, queue, &textured_quad(&texture));
        assert_eq!(reloads.load(Ordering::Relaxed), 1);

        let kept = Arc::new(crate::SceneTexture::new(checker()));
        let _mesh = renderer.create_retained_mesh(device, queue, &textured_quad(&kept));
        assert!(kept.holds_pixels(), "nothing could make them again");
    });
}

/// Meshes of one scene that use the same texture share one upload, and a
/// scene dropped and made again from the set it was made with uploads
/// nothing: no reload, nothing added to the set. A second renderer given
/// the same set makes its own copy, from a reload.
#[test]
fn a_kept_set_outlives_the_retained_scene() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    on_each_backend(|offscreen| {
        let reloads = Arc::new(AtomicUsize::new(0));
        let counter = reloads.clone();
        let texture = Arc::new(crate::SceneTexture::reloadable(checker(), move || {
            counter.fetch_add(1, Ordering::Relaxed);
            Some(checker())
        }));
        let quad = textured_quad(&texture);
        let scene = crate::SceneData {
            meshes: vec![
                (quad.clone(), Mat4::IDENTITY, MaterialId(0)),
                (quad, Mat4::from_translation(Vec3::Z), MaterialId(0)),
            ],
            ..Default::default()
        };
        let (device, queue) = (offscreen.device(), offscreen.queue());

        // Within one call, without a set: one upload for both meshes.
        let renderer = offscreen.renderer(W, H);
        let retained = renderer.create_retained_scene(device, queue, &scene);
        assert_eq!(retained.meshes.len(), 2);
        assert_eq!(reloads.load(Ordering::Relaxed), 0);
        assert!(!texture.holds_pixels());
        drop(retained);

        let mut renderer = offscreen.renderer(W, H);
        let mut uploaded = crate::GpuTextureSet::default();
        let retained = renderer.create_retained_scene_with(device, queue, &scene, &mut uploaded);
        assert_eq!(
            reloads.load(Ordering::Relaxed),
            1,
            "the first scene's copy went with it, so this is a reload"
        );
        assert_eq!(uploaded.len(), 1);
        drop(retained);
        let retained = renderer.create_retained_scene_with(device, queue, &scene, &mut uploaded);
        assert_eq!(
            reloads.load(Ordering::Relaxed),
            1,
            "the set's copy is reused"
        );
        assert_eq!(uploaded.len(), 1);
        renderer.submit_retained_mesh(
            &retained.meshes[0].0,
            Mat4::IDENTITY,
            ObjectId(1),
            MaterialId(0),
            1.0,
        );
        assert!(shows_red_top_left(offscreen, &mut renderer));

        let other = offscreen.renderer(W, H);
        let _theirs = other.create_retained_scene_with(device, queue, &scene, &mut uploaded);
        assert_eq!(reloads.load(Ordering::Relaxed), 2, "its own copy, reloaded");
        assert_eq!(uploaded.len(), 2);
    });
}

/// A set kept across reloads lets go of the copies whose texture is gone:
/// at the next upload into it, or when asked. A mesh still drawing such a
/// copy keeps it.
#[test]
fn a_set_drops_copies_whose_texture_is_gone() {
    use std::sync::Arc;
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let (device, queue) = (offscreen.device(), offscreen.queue());
        let mut uploaded = crate::GpuTextureSet::default();

        // Three loads of one scene, each with a texture of its own that is
        // gone before the next.
        for _ in 0..3 {
            let texture = Arc::new(crate::SceneTexture::new(checker()));
            let _mesh = renderer.create_retained_mesh_with(
                device,
                queue,
                &textured_quad(&texture),
                &mut uploaded,
            );
        }
        assert_eq!(
            uploaded.len(),
            1,
            "each upload let go of the copy before it"
        );
        assert_eq!(uploaded.prune(), 1);
        assert!(uploaded.is_empty());

        // A mesh outlives both its texture and the set's hold on the copy.
        let texture = Arc::new(crate::SceneTexture::new(checker()));
        let mesh = renderer.create_retained_mesh_with(
            device,
            queue,
            &textured_quad(&texture),
            &mut uploaded,
        );
        drop(texture);
        assert_eq!(uploaded.prune(), 1);
        assert!(uploaded.is_empty());
        renderer.submit_retained_mesh(&mesh, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
        assert!(shows_red_top_left(offscreen, &mut renderer));
    });
}

/// A texture larger than the device's largest side is uploaded from the
/// first mip level that fits, and every mesh that shares the copy reports
/// the size it went up at.
#[test]
fn a_shared_texture_reports_its_downscale_to_every_mesh() {
    use std::sync::Arc;
    on_each_backend(|offscreen| {
        // A row twice the largest side, with a level that fits under it.
        let side = offscreen.max_frame_side();
        let wide = crate::TextureData::from_levels(
            side * 2,
            1,
            vec![vec![255; side as usize * 8], vec![255; side as usize * 4]],
        );
        let texture = Arc::new(crate::SceneTexture::new(wide));
        let expected = Some((side, 1));
        let renderer = offscreen.renderer(W, H);
        let (device, queue) = (offscreen.device(), offscreen.queue());
        let mut uploaded = crate::GpuTextureSet::default();
        let first = renderer.create_retained_mesh_with(
            device,
            queue,
            &textured_quad(&texture),
            &mut uploaded,
        );
        let second = renderer.create_retained_mesh_with(
            device,
            queue,
            &textured_quad(&texture),
            &mut uploaded,
        );
        assert_eq!(uploaded.len(), 1);
        assert_eq!(first.texture_downscaled, expected);
        assert_eq!(second.texture_downscaled, expected);
    });
}

/// A renderer on a second device gets its own copy: the pixels are made
/// again for it, and a texture that cannot make them draws its mesh
/// untextured.
#[test]
fn another_device_reloads_the_texture() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let (Ok(one), Ok(two)) = (Offscreen::new(), Offscreen::new()) else {
        eprintln!("skipped: no adapter");
        return;
    };
    let reloads = Arc::new(AtomicUsize::new(0));
    let counter = reloads.clone();
    let texture = Arc::new(crate::SceneTexture::reloadable(checker(), move || {
        counter.fetch_add(1, Ordering::Relaxed);
        Some(checker())
    }));
    let lost = Arc::new(crate::SceneTexture::reloadable(checker(), || None));
    for (offscreen, expected_reloads) in [(&one, 0), (&two, 1)] {
        let mut renderer = offscreen.renderer(W, H);
        let (device, queue) = (offscreen.device(), offscreen.queue());
        let mesh = renderer.create_retained_mesh(device, queue, &textured_quad(&texture));
        assert_eq!(reloads.load(Ordering::Relaxed), expected_reloads);
        renderer.submit_retained_mesh(&mesh, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
        assert!(shows_red_top_left(offscreen, &mut renderer));

        // The first renderer to draw `lost` takes its pixels; the next has
        // none to take and the quad is the plain white of its vertices.
        let mut renderer = offscreen.renderer(W, H);
        let mesh = renderer.create_retained_mesh(device, queue, &textured_quad(&lost));
        renderer.submit_retained_mesh(&mesh, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
        assert_eq!(
            shows_red_top_left(offscreen, &mut renderer),
            expected_reloads == 0
        );
    }
}

/// A textured quad shows its texture: a 2×2 checker of red, green, blue
/// and white lands one colour per quadrant, with the image's top row at
/// the top of the frame (uv (0, 0) is the image's top-left texel).
#[test]
fn a_textured_quad_shows_its_texture_the_right_way_up() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let texture = checker();
        let normal = [0.0, 0.0, 1.0];
        let vertex =
            |x: f32, y: f32, uv: [f32; 2]| MeshVertex::textured([x, y, 0.0], normal, [1.0; 4], uv);
        let mesh = renderer.create_retained_mesh(
            offscreen.device(),
            offscreen.queue(),
            &MeshData {
                vertices: vec![
                    vertex(-1.0, -1.0, [0.0, 1.0]),
                    vertex(1.0, -1.0, [1.0, 1.0]),
                    vertex(1.0, 1.0, [1.0, 0.0]),
                    vertex(-1.0, 1.0, [0.0, 0.0]),
                ],
                indices: Some(vec![0, 1, 2, 0, 2, 3]),
                texture: Some(std::sync::Arc::new(crate::SceneTexture::new(texture))),
            },
        );
        assert_eq!(mesh.texture_downscaled, None);
        renderer.submit_retained_mesh(&mesh, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
        let (rgba, info) = offscreen
            .render_rgba(&mut renderer, &view(), &settings(false))
            .unwrap();
        assert_eq!(info.overflow, None);

        let view = view();
        let sample = |x: f32, y: f32| {
            let (px, py) = at(&view, Vec3::new(x, y, 0.0));
            pixel(&rgba, px, py)
        };
        let top_left = sample(-0.5, 0.5);
        let top_right = sample(0.5, 0.5);
        let bottom_left = sample(-0.5, -0.5);
        let bottom_right = sample(0.5, -0.5);
        assert!(
            top_left[0] > top_left[1] + 60 && top_left[0] > top_left[2] + 60,
            "top-left red: {top_left:?}"
        );
        assert!(
            top_right[1] > top_right[0] + 60 && top_right[1] > top_right[2] + 60,
            "top-right green: {top_right:?}"
        );
        assert!(
            bottom_left[2] > bottom_left[0] + 60 && bottom_left[2] > bottom_left[1] + 60,
            "bottom-left blue: {bottom_left:?}"
        );
        // The white quadrant is what the same quad draws untextured: the
        // white texel and the shared white binding light the same.
        let plain = renderer.create_retained_mesh(
            offscreen.device(),
            offscreen.queue(),
            &quad(Vec3::ZERO, Vec3::X, Vec3::Y, [1.0; 4]),
        );
        renderer.submit_retained_mesh(&plain, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
        let (untextured, _) = offscreen
            .render_rgba(&mut renderer, &view, &settings(false))
            .unwrap();
        let (px, py) = at(&view, Vec3::new(0.5, -0.5, 0.0));
        let expected = pixel(&untextured, px, py);
        assert!(
            (0..3).all(|c| (bottom_right[c] as i32 - expected[c] as i32).abs() <= 2),
            "bottom-right white: {bottom_right:?} vs untextured {expected:?}"
        );
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
            offscreen.queue(),
            &quad(Vec3::ZERO, Vec3::X * 0.8, Vec3::Y * 0.8, white),
        );
        let near = renderer.create_retained_mesh(
            offscreen.device(),
            offscreen.queue(),
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
            1.0,
        );
        renderer.submit_retained_mesh(
            &near,
            Mat4::from_translation(Vec3::new(0.0, 0.0, 0.3)),
            ObjectId(9),
            MaterialId(0),
            1.0,
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

/// Uploads `mesh` and submits it `z` up the view axis at `opacity`.
fn submit_at(
    offscreen: &Offscreen,
    renderer: &mut Renderer,
    mesh: &MeshData,
    z: f32,
    object: u32,
    opacity: f32,
) {
    let mesh = renderer.create_retained_mesh(offscreen.device(), offscreen.queue(), mesh);
    renderer.submit_retained_mesh(
        &mesh,
        Mat4::from_translation(Vec3::new(0.0, 0.0, z)),
        ObjectId(object),
        MaterialId(0),
        opacity,
    );
}

/// `front` at `opacity` over `behind`, channel by channel in linear light.
fn over(front: [f32; 3], opacity: f32, behind: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|c| front[c] * opacity + behind[c] * (1.0 - opacity))
}

/// How many pixels of two frames differ by more than `tolerance` in a
/// channel, alpha included.
fn differing(a: &[u8], b: &[u8], tolerance: u8) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(a, b)| (0..4).any(|c| a[c].abs_diff(b[c]) > tolerance))
        .count()
}

/// A transparent mesh is lit as an opaque one and blended at its opacity
/// over what is behind it, the opaque scene or the background, once
/// however many layers the frame peels. Behind an opaque mesh, or at no
/// opacity, it leaves the frame as it was.
#[test]
fn a_transparent_surface_blends_over_what_is_behind_it() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.antialiasing = false;
        let wide = quad(
            Vec3::ZERO,
            Vec3::X * 0.8,
            Vec3::Y * 0.8,
            [0.25, 0.25, 0.25, 1.0],
        );
        let small = quad(
            Vec3::ZERO,
            Vec3::X * 0.2,
            Vec3::Y * 1.2,
            [0.6, 0.6, 0.6, 1.0],
        );
        let dark = lit(&settings, 0.25, Vec3::Z, Vec3::Z);
        let pale = lit(&settings, 0.6, Vec3::Z, Vec3::Z);
        let background = [BACKGROUND[0], BACKGROUND[1], BACKGROUND[2]];

        // The small quad in front, taller than the wide one: over it at
        // the centre, over the background near the top.
        submit_at(offscreen, &mut renderer, &wide, -0.5, 1, 1.0);
        submit_at(offscreen, &mut renderer, &small, 0.3, 2, 0.5);
        let (rgba, _) = offscreen
            .render_rgba(&mut renderer, &view(), &settings)
            .unwrap();
        let centre = pixel(&rgba, W / 2, H / 2);
        let expected = over(pale, 0.5, dark).map(srgb_byte);
        assert!(close(centre, expected, 3.0), "{centre:?} vs {expected:?}");
        let top = pixel(&rgba, W / 2, 4);
        let expected = over(pale, 0.5, background).map(srgb_byte);
        assert!(close(top, expected, 3.0), "{top:?} vs {expected:?}");
        let side = pixel(&rgba, W / 2 + 20, H / 2);
        assert!(close(side, dark.map(srgb_byte), 3.0), "{side:?}");
        assert_eq!(pixel(&rgba, 2, 2), [0, 0, 255], "background corner");

        // The opaque mesh alone, and the same with the transparent one
        // behind it or at no opacity at all.
        let mut frame = |behind: Option<(f32, f32)>| {
            submit_at(offscreen, &mut renderer, &wide, -0.5, 1, 1.0);
            if let Some((z, opacity)) = behind {
                submit_at(offscreen, &mut renderer, &small, z, 2, opacity);
            }
            offscreen
                .render_rgba(&mut renderer, &view(), &settings)
                .unwrap()
                .0
        };
        let alone = frame(None);
        let hidden = frame(Some((-0.8, 0.5)));
        let centre = (((H / 2) * W + W / 2) * 4) as usize;
        assert_eq!(hidden[centre..centre + 4], alone[centre..centre + 4]);
        assert!(frame(Some((0.3, 0.0))) == alone, "no opacity draws nothing");

        // Over no background at all the frame's alpha is the coverage,
        // and its colour is premultiplied by it.
        settings.background_color = [0.0; 4];
        submit_at(offscreen, &mut renderer, &small, 0.3, 2, 0.5);
        let (rgba, _) = offscreen
            .render_rgba(&mut renderer, &view(), &settings)
            .unwrap();
        assert!(rgba[centre + 3].abs_diff(128) <= 1, "{}", rgba[centre + 3]);
        let expected = over(pale, 0.5, [0.0; 3]).map(srgb_byte);
        let centre = pixel(&rgba, W / 2, H / 2);
        assert!(close(centre, expected, 3.0), "{centre:?} vs {expected:?}");
    });
}

/// Transparent meshes that pass through each other blend in depth order
/// at every pixel, whichever is submitted first.
#[test]
fn transparent_surfaces_blend_in_depth_order_however_they_are_submitted() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.antialiasing = false;
        settings.materials[0].specular = 0.0;
        // Two sheets crossing along the Y axis: the first is the nearer
        // one on the left of the frame, the second on the right.
        let sheets = [
            (Vec3::new(0.8, 0.0, -0.3), 0.6, 0.7),
            (Vec3::new(0.8, 0.0, 0.3), 0.2, 0.4),
        ];
        let meshes = sheets.map(|(u, albedo, _)| {
            quad(Vec3::ZERO, u, Vec3::Y * 0.5, [albedo, albedo, albedo, 1.0])
        });
        let colours = sheets.map(|(u, albedo, _)| {
            let normal = u.cross(Vec3::Y).normalize();
            lit(&settings, albedo, normal, Vec3::Z)
        });
        let opacity = sheets.map(|(_, _, opacity)| opacity);
        let background = [BACKGROUND[0], BACKGROUND[1], BACKGROUND[2]];

        let mut frame = |order: [usize; 2]| {
            for i in order {
                submit_at(
                    offscreen,
                    &mut renderer,
                    &meshes[i],
                    0.0,
                    i as u32 + 1,
                    opacity[i],
                );
            }
            offscreen
                .render_rgba(&mut renderer, &view(), &settings)
                .unwrap()
                .0
        };
        let forward = frame([0, 1]);
        assert!(frame([1, 0]) == forward, "the order of submission shows");

        for (x, [near, far]) in [(W / 2 - 20, [0, 1]), (W / 2 + 20, [1, 0])] {
            let behind = over(colours[far], opacity[far], background);
            let expected = over(colours[near], opacity[near], behind).map(srgb_byte);
            let actual = pixel(&forward, x, H / 2);
            assert!(close(actual, expected, 3.0), "{actual:?} vs {expected:?}");
        }
    });
}

/// As a mesh's opacity reaches 1 the frame arrives at the opaque one's:
/// lighting, edge lines and anti-aliasing alike. (Ambient occlusion is
/// the exception: a transparent mesh takes none.)
#[test]
fn a_mesh_just_short_of_opaque_draws_as_the_opaque_one_does() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.edges.enabled = true;
        let floor = quad(
            Vec3::ZERO,
            Vec3::X * 0.9,
            Vec3::Y * 0.6,
            [0.5, 0.5, 0.5, 1.0],
        );
        let ball = sphere(Vec3::new(-0.2, 0.0, 0.0), 0.35);
        let tilted = quad(
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.25, 0.0, 0.2),
            Vec3::Y * 0.3,
            [1.0; 4],
        );
        let mut frame = |opacity: f32| {
            submit_at(offscreen, &mut renderer, &floor, -0.4, 1, opacity);
            submit_at(offscreen, &mut renderer, &ball, 0.0, 2, opacity);
            submit_at(offscreen, &mut renderer, &tilted, 0.2, 3, opacity);
            offscreen
                .render_rgba(&mut renderer, &view(), &settings)
                .unwrap()
                .0
        };
        let opaque = frame(1.0);
        let nearly = frame(0.999);
        assert_eq!(differing(&nearly, &opaque, 1), 0);
        // Not for want of anything to compare: the edge lines are there.
        settings.edges.enabled = false;
        let plain = {
            submit_at(offscreen, &mut renderer, &floor, -0.4, 1, 1.0);
            submit_at(offscreen, &mut renderer, &ball, 0.0, 2, 1.0);
            submit_at(offscreen, &mut renderer, &tilted, 0.2, 3, 1.0);
            offscreen
                .render_rgba(&mut renderer, &view(), &settings)
                .unwrap()
                .0
        };
        assert!(differing(&plain, &opaque, 8) > 100);
    });
}

/// A surface seen through a transparent mesh gets no edge line where
/// that mesh ends: it is one surface on both sides, though it changes
/// layer there, by as many layers as the mesh has walls. Drawn just short
/// of opaque behind the transparent mesh, it shows what it shows drawn
/// opaque, where the G-buffer holds it whole.
#[test]
fn a_surface_seen_through_a_transparent_mesh_takes_no_edge_from_it() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.edges.enabled = true;
        settings.antialiasing = false;
        // The sheet is the fourth surface deep behind three walls.
        settings.transparency_layers = 4;
        // A sheet across the whole frame, and in front of it walls over
        // half of the frame, ending together down the middle.
        let behind = quad(
            Vec3::ZERO,
            Vec3::X * 3.0,
            Vec3::Y * 3.0,
            [0.6, 0.6, 0.6, 1.0],
        );
        for side in [-1.0, 1.0] {
            let half = quad(
                Vec3::new(side * 1.5, 0.0, 0.0),
                Vec3::X * 1.5,
                Vec3::Y * 3.0,
                [0.3, 0.3, 0.3, 1.0],
            );
            for walls in 1..=3 {
                let mut frame = |opacity: f32| {
                    let mut items = vec![(&behind, -0.5, 1, opacity)];
                    items.extend((0..walls).map(|wall| (&half, 0.3 - 0.2 * wall as f32, 2, 0.5)));
                    frame_of(offscreen, &mut renderer, &settings, &items)
                };
                let opaque = frame(1.0);
                let nearly = frame(0.999);
                assert_eq!(
                    differing(&nearly, &opaque, 1),
                    0,
                    "{walls} walls on side {side}"
                );
                if walls > 1 {
                    continue;
                }
                // The transparent mesh's own edge is drawn on the row
                // through the middle of the frame, beside the middle.
                let row = H / 2;
                let flat = pixel(&opaque, if side < 0.0 { 4 } else { W - 4 }, row)[1];
                let dark: Vec<u32> = (0..W)
                    .filter(|&x| {
                        pixel(&opaque, x, row)[1] + 5 < flat.min(pixel(&opaque, W - 4, row)[1])
                            || pixel(&opaque, x, row)[1] + 5 < flat.min(pixel(&opaque, 4, row)[1])
                    })
                    .collect();
                assert!(
                    !dark.is_empty() && dark.iter().all(|x| x.abs_diff(W / 2) <= 1),
                    "{dark:?} on side {side}"
                );
            }
        }
    });
}

/// The frame of `items`, each a mesh, how far up the view axis it is
/// moved, its object and its opacity.
fn frame_of(
    offscreen: &Offscreen,
    renderer: &mut Renderer,
    settings: &RenderSettings,
    items: &[(&MeshData, f32, u32, f32)],
) -> Vec<u8> {
    for (mesh, z, object, opacity) in items {
        submit_at(offscreen, renderer, mesh, *z, *object, *opacity);
    }
    offscreen
        .render_rgba(renderer, &view(), settings)
        .unwrap()
        .0
}

/// A transparent surface lying just in front of another object's, as a
/// model does over the model it was made from, is edged along its edges
/// and nowhere else: the other surface is beside it in depth, not in the
/// picture.
#[test]
fn a_transparent_surface_close_to_another_is_edged_at_its_edges_only() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut edged = settings(false);
        edged.edges.enabled = true;
        edged.antialiasing = false;
        let mut plain = edged.clone();
        plain.edges.enabled = false;
        let background = {
            let empty = frame_of(offscreen, &mut renderer, &plain, &[]);
            pixel(&empty, 2, 2)
        };
        let centre = |rgba: &[u8]| pixel(rgba, W / 2, H / 2);

        // A shell a hundredth of a metre outside a ball, three metres
        // from the eye.
        let ball = sphere(Vec3::ZERO, 0.35);
        let shell = sphere(Vec3::ZERO, 0.36);
        for ball_opacity in [1.0, 0.5] {
            let items = [(&ball, 0.0, 1, ball_opacity), (&shell, 0.0, 2, 0.5)];
            let with = frame_of(offscreen, &mut renderer, &edged, &items);
            let without = frame_of(offscreen, &mut renderer, &plain, &items);
            assert_eq!(centre(&with), centre(&without), "ball at {ball_opacity}");
            let covered = (0..W * H)
                .filter(|i| pixel(&without, i % W, i / W) != background)
                .count();
            // Every edged pixel is on the rim: within three pixels of
            // the farthest covered pixel from the centre of the frame.
            let from_centre = |i: u32| {
                let (x, y) = (
                    (i % W) as f32 - W as f32 / 2.0,
                    (i / W) as f32 - H as f32 / 2.0,
                );
                x.hypot(y)
            };
            let rim = (0..W * H)
                .filter(|i| pixel(&without, i % W, i / W) != background)
                .map(from_centre)
                .fold(0.0, f32::max);
            let lined: Vec<u32> = (0..W * H)
                .filter(|&i| {
                    let (a, b) = (pixel(&with, i % W, i / W), pixel(&without, i % W, i / W));
                    (0..3).any(|c| a[c].abs_diff(b[c]) > 3)
                })
                .collect();
            assert!(!lined.is_empty() && lined.len() * 2 < covered);
            assert!(
                lined.iter().all(|&i| from_centre(i) > rim - 3.0),
                "ball at {ball_opacity}: edged away from the rim"
            );
        }

        // A sheet two hundredths above a floor, either of them the
        // transparent one or both.
        let floor = quad(
            Vec3::ZERO,
            Vec3::X * 0.8,
            Vec3::Y * 0.6,
            [0.6, 0.6, 0.6, 1.0],
        );
        let sheet = quad(
            Vec3::ZERO,
            Vec3::X * 0.4,
            Vec3::Y * 0.3,
            [0.3, 0.3, 0.3, 1.0],
        );
        for (floor_opacity, sheet_opacity) in [(1.0, 0.5), (0.999, 0.5), (0.5, 0.5)] {
            let items = [
                (&floor, -0.5, 1, floor_opacity),
                (&sheet, -0.48, 2, sheet_opacity),
            ];
            let with = frame_of(offscreen, &mut renderer, &edged, &items);
            let without = frame_of(offscreen, &mut renderer, &plain, &items);
            assert_eq!(
                centre(&with),
                centre(&without),
                "floor at {floor_opacity}, sheet at {sheet_opacity}"
            );
        }

        // The sheet just under a transparent floor instead: its edges
        // are still its own, and show through the floor.
        let items = [(&floor, -0.5, 1, 0.5), (&sheet, -0.52, 2, 0.5)];
        let with = frame_of(offscreen, &mut renderer, &edged, &items);
        let without = frame_of(offscreen, &mut renderer, &plain, &items);
        assert_eq!(centre(&with), centre(&without));
        let (left, row) = at(&view(), Vec3::new(-0.4, 0.0, -0.52));
        let lined = (left - 2..=left + 2)
            .filter(|&x| pixel(&with, x, row) != pixel(&without, x, row))
            .count();
        assert!(lined >= 1, "the sheet's edge under the floor");
    });
}

/// Where one object at a time is just short of opaque among opaque ones,
/// the frame is the all-opaque one: in front of them, behind them and
/// passing through them.
#[test]
fn a_mesh_just_short_of_opaque_among_opaque_ones_draws_as_they_do() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.edges.enabled = true;
        let floor = quad(
            Vec3::ZERO,
            Vec3::X * 0.9,
            Vec3::Y * 0.6,
            [0.5, 0.5, 0.5, 1.0],
        );
        // Sunk into the floor.
        let ball = sphere(Vec3::new(-0.2, 0.0, 0.0), 0.35);
        let tilted = quad(
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.25, 0.0, 0.2),
            Vec3::Y * 0.3,
            [1.0; 4],
        );
        // Passing through the floor along the Y axis.
        let crossing = quad(
            Vec3::new(0.2, -0.3, 0.0),
            Vec3::new(0.3, 0.0, 0.25),
            Vec3::Y * 0.15,
            [0.8, 0.8, 0.8, 1.0],
        );
        let scene = [
            (&floor, -0.4),
            (&ball, -0.2),
            (&tilted, 0.2),
            (&crossing, -0.4),
        ];
        let mut frame = |short: Option<usize>| {
            let items: Vec<_> = scene
                .iter()
                .enumerate()
                .map(|(i, (mesh, z))| {
                    let opacity = if short == Some(i) { 0.999 } else { 1.0 };
                    (*mesh, *z, i as u32 + 1, opacity)
                })
                .collect();
            frame_of(offscreen, &mut renderer, &settings, &items)
        };
        let opaque = frame(None);
        for short in 0..scene.len() {
            let count = differing(&frame(Some(short)), &opaque, 1);
            assert_eq!(count, 0, "object {}", short + 1);
        }
    });
}

/// A mesh too faint to see leaves nothing in the frame: no line on a
/// mesh it passes through, lies close to, or ends flush against, and no
/// gap in that mesh's own outline. The frame is the one without it.
#[test]
fn a_faint_mesh_leaves_no_mark_on_the_meshes_around_it() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.edges.enabled = true;
        settings.antialiasing = false;
        let black = [0.0, 0.0, 0.0, 1.0];
        let grey = [0.6, 0.6, 0.6, 1.0];
        // Two sheets crossing along the Y axis; a ball with another, a
        // little larger and to one side, so that each is the outer one
        // along part of the rim; two sheets side by side in one plane.
        let one = quad(Vec3::ZERO, Vec3::new(0.6, 0.0, -0.3), Vec3::Y * 0.4, grey);
        let other = quad(Vec3::ZERO, Vec3::new(0.6, 0.0, 0.3), Vec3::Y * 0.4, black);
        let ball = sphere(Vec3::ZERO, 0.35);
        let beside = MeshData {
            vertices: sphere(Vec3::new(0.03, 0.02, 0.0), 0.36)
                .vertices
                .iter()
                .map(|v| MeshVertex::colored(v.position, v.normal, black))
                .collect(),
            ..sphere(Vec3::ZERO, 0.36)
        };
        let left = quad(
            Vec3::new(-0.3, 0.0, 0.0),
            Vec3::X * 0.3,
            Vec3::Y * 0.4,
            grey,
        );
        let right = quad(
            Vec3::new(0.3, 0.0, 0.0),
            Vec3::X * 0.3,
            Vec3::Y * 0.4,
            black,
        );
        // The last number is how many pixels may differ: one, on the
        // ball's rim, where the surface is too steep across a pixel for
        // the edge test to follow it from one layer to the next.
        for (name, shown, faint, stray) in [
            ("crossing", &one, &other, 0),
            ("close", &ball, &beside, 1),
            ("flush on the right", &left, &right, 0),
            ("flush on the left", &right, &left, 0),
        ] {
            for opacity in [0.999, 0.5] {
                let with = frame_of(
                    offscreen,
                    &mut renderer,
                    &settings,
                    &[(shown, 0.0, 2, opacity), (faint, 0.0, 3, 0.002)],
                );
                let without = frame_of(
                    offscreen,
                    &mut renderer,
                    &settings,
                    &[(shown, 0.0, 2, opacity)],
                );
                // The faint mesh's own pixels, outside the other, change
                // by its two thousandths and no more.
                let count = differing(&with, &without, 3);
                assert!(count <= stray, "{name} at {opacity}: {count}");
                // And it is a frame with edge lines in it.
                settings.edges.enabled = false;
                let plain = frame_of(
                    offscreen,
                    &mut renderer,
                    &settings,
                    &[(shown, 0.0, 2, opacity)],
                );
                settings.edges.enabled = true;
                assert!(differing(&without, &plain, 3) > 20, "{name}");
            }
        }
    });
}

/// A fade's faint end is smooth: opacity is kept finely enough that the
/// first steps a mesh takes over a dark background are small ones, and
/// below the least that could show it is not drawn, nor picked even by
/// a pick that lands on every mesh drawn.
#[test]
fn the_faint_end_of_a_fade_is_smooth() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.antialiasing = false;
        settings.background_color = [0.0, 0.0, 0.0, 1.0];
        settings.pick_opacity = 0.0;
        let sheet = quad(Vec3::ZERO, Vec3::X * 0.5, Vec3::Y * 0.5, [1.0; 4]);
        let mut level = |opacity: f32| {
            let rgba = frame_of(
                offscreen,
                &mut renderer,
                &settings,
                &[(&sheet, 0.0, 1, opacity)],
            );
            let picked = offscreen
                .pick(&mut renderer, (W / 2, H / 2))
                .expect("a pick");
            (pixel(&rgba, W / 2, H / 2)[1], picked.object)
        };
        assert_eq!(level(0.0001), (0, ObjectId::NONE));
        let mut last = 0;
        for step in 1..=40 {
            let (now, object) = level(step as f32 * 0.0005);
            assert_eq!(object, ObjectId(1));
            assert!(
                now >= last && now - last <= 3,
                "from {last} to {now} at {}",
                step as f32 * 0.0005
            );
            last = now;
        }
        assert!(last > 20, "{last}");
    });
}

/// Where two surfaces cross, the line is the mark of the one that goes
/// behind, made once: a mesh just short of opaque that a transparent
/// one passes through shows what it shows drawn opaque, whichever of the
/// two is in front on the left and whichever has the higher object. (The
/// surface that comes out in front at the crossing takes no mark of its
/// own there, from behind the other.)
#[test]
fn surfaces_that_cross_are_marked_along_the_crossing_once() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.edges.enabled = true;
        settings.antialiasing = false;
        // Two sheets crossing down the middle of the frame: the first is
        // the nearer one on the left, the second on the right.
        let sheets = [Vec3::new(0.8, 0.0, -0.3), Vec3::new(0.8, 0.0, 0.3)]
            .map(|u| quad(Vec3::ZERO, u, Vec3::Y * 0.5, [0.6, 0.6, 0.6, 1.0]));
        for solid in [0, 1] {
            for objects in [[1, 2], [2, 1]] {
                let mut frame = |opacity: f32| {
                    let items = [0, 1].map(|i| {
                        let opacity = if i == solid { opacity } else { 0.5 };
                        (&sheets[i], 0.0, objects[i], opacity)
                    });
                    frame_of(offscreen, &mut renderer, &settings, &items)
                };
                let opaque = frame(1.0);
                assert_eq!(
                    differing(&frame(0.999), &opaque, 1),
                    0,
                    "sheet {solid} solid, objects {objects:?}"
                );
                // The crossing is drawn: the middle of the row through
                // the middle of the frame is darker than either side.
                let row = H / 2;
                let level = |x: u32| pixel(&opaque, x, row)[1];
                let darkest = (W / 2 - 2..=W / 2 + 2).map(level).min().unwrap();
                assert!(
                    darkest + 10 < level(W / 2 - 8).min(level(W / 2 + 8)),
                    "sheet {solid} solid, objects {objects:?}: no line at the crossing"
                );
            }
        }
    });
}

/// An object's two surfaces close together (the walls of a thin shell)
/// are not taken for one another. Where the far wall ends against an
/// opaque part between the walls it is marked, though the near wall
/// goes on over the part at the depth the far wall would nearly have:
/// the frame is the one drawn with the near wall another object, when
/// nothing can be taken for anything. And where the near wall ends and
/// the far one goes on, neither is marked: so small a step within one
/// object is no edge, in the opaque scene either.
#[test]
fn the_walls_of_a_thin_shell_are_told_apart() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.edges.enabled = true;
        settings.antialiasing = false;
        let wall = quad(
            Vec3::ZERO,
            Vec3::X * 3.0,
            Vec3::Y * 3.0,
            [0.6, 0.6, 0.6, 1.0],
        );
        let half = quad(
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::X * 1.5,
            Vec3::Y * 3.0,
            [0.6, 0.6, 0.6, 1.0],
        );
        // Walls 0.02 m apart, 3 m off: well inside the edge test's
        // tolerance of 1.2 % of the distance.
        let row = H / 2;
        let middle = W / 2 - 3..=W / 2 + 3;

        // A part over the right half of the frame, between the walls.
        let mut frame = |near_object: u32| {
            let items = [
                (&wall, 0.02, near_object, 0.5),
                (&wall, 0.0, 1, 0.5),
                (&half, 0.01, 2, 1.0),
            ];
            frame_of(offscreen, &mut renderer, &settings, &items)
        };
        let apart = frame(3);
        assert_eq!(differing(&frame(1), &apart, 1), 0);
        let level = |x: u32| pixel(&apart, x, row)[1];
        let darkest = middle.clone().map(level).min().unwrap();
        assert!(
            darkest + 10 < level(W / 2 - 8).min(level(W / 2 + 8)),
            "no line at the part's edge"
        );

        // The near wall over the right half only, and no part.
        let items = [(&half, 0.02, 1, 0.5), (&wall, 0.0, 1, 0.5)];
        let ended = frame_of(offscreen, &mut renderer, &settings, &items);
        let level = |x: u32| pixel(&ended, x, row)[1];
        let (left, right) = (level(W / 2 - 8), level(W / 2 + 8));
        assert!(left.abs_diff(right) > 10, "the near wall does not show");
        for x in middle {
            assert!(
                level(x).abs_diff(left) <= 2 || level(x).abs_diff(right) <= 2,
                "a line where the near wall ends: {} at {x}, between {left} and {right}",
                level(x)
            );
        }
    });
}

/// A surface that slopes steeply across the frame is edged the same just
/// short of opaque as opaque, where a transparent mesh in front of it
/// ends, and so is an opaque one behind a mesh just short of opaque: the
/// edge test extrapolates a surface from the pixel a step back, which is
/// where the mesh in front begins.
#[test]
fn a_steep_surface_is_edged_alike_behind_a_transparent_mesh() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.edges.enabled = true;
        settings.antialiasing = false;
        let cover = quad(
            Vec3::new(-0.35, 0.0, 0.9),
            Vec3::X * 0.3,
            Vec3::Y * 0.25,
            [0.3, 0.3, 0.3, 1.0],
        );
        for slope in [-2.0, -1.4, -0.5, 0.5, 1.4, 2.0] {
            let steep = quad(
                Vec3::ZERO,
                Vec3::new(0.3, 0.0, -0.3 * slope),
                Vec3::Y * 0.5,
                [0.6, 0.6, 0.6, 1.0],
            );
            let mut frame = |steep_opacity: f32, cover_opacity: f32| {
                frame_of(
                    offscreen,
                    &mut renderer,
                    &settings,
                    &[
                        (&steep, 0.0, 1, steep_opacity),
                        (&cover, 0.0, 2, cover_opacity),
                    ],
                )
            };
            for cover_opacity in [0.5, 0.999] {
                let count = differing(&frame(0.999, cover_opacity), &frame(1.0, cover_opacity), 1);
                assert_eq!(count, 0, "slope {slope}, cover at {cover_opacity}");
            }
            let count = differing(&frame(1.0, 0.999), &frame(1.0, 1.0), 1);
            assert_eq!(
                count, 0,
                "slope {slope}, opaque behind a cover short of opaque"
            );
        }
    });
}

/// Two transparent surfaces in exactly the same place are both drawn,
/// the higher object number in front, whichever is submitted first.
#[test]
fn transparent_surfaces_in_the_same_place_are_both_drawn() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.antialiasing = false;
        let dark = quad(
            Vec3::ZERO,
            Vec3::X * 0.5,
            Vec3::Y * 0.5,
            [0.2, 0.2, 0.2, 1.0],
        );
        let pale = quad(
            Vec3::ZERO,
            Vec3::X * 0.5,
            Vec3::Y * 0.5,
            [0.6, 0.6, 0.6, 1.0],
        );
        let forward = frame_of(
            offscreen,
            &mut renderer,
            &settings,
            &[(&dark, 0.0, 1, 0.5), (&pale, 0.0, 2, 0.5)],
        );
        let backward = frame_of(
            offscreen,
            &mut renderer,
            &settings,
            &[(&pale, 0.0, 2, 0.5), (&dark, 0.0, 1, 0.5)],
        );
        assert!(forward == backward, "the order of submission shows");
        let background = [BACKGROUND[0], BACKGROUND[1], BACKGROUND[2]];
        let behind = over(lit(&settings, 0.2, Vec3::Z, Vec3::Z), 0.5, background);
        let expected = over(lit(&settings, 0.6, Vec3::Z, Vec3::Z), 0.5, behind).map(srgb_byte);
        let centre = pixel(&forward, W / 2, H / 2);
        assert!(close(centre, expected, 3.0), "{centre:?} vs {expected:?}");

        // Submitted in the order of their numbers, the same one is in
        // front whether both are opaque, both just short of it, or one
        // of each: the opaque frame's rule is the last one submitted.
        let mut frame = |dark_opacity: f32, pale_opacity: f32| {
            frame_of(
                offscreen,
                &mut renderer,
                &settings,
                &[(&dark, 0.0, 1, dark_opacity), (&pale, 0.0, 2, pale_opacity)],
            )
        };
        let opaque = frame(1.0, 1.0);
        for (dark_opacity, pale_opacity) in [(0.999, 0.999), (0.999, 1.0), (1.0, 0.999)] {
            assert_eq!(
                differing(&frame(dark_opacity, pale_opacity), &opaque, 1),
                0,
                "dark at {dark_opacity}, pale at {pale_opacity}"
            );
        }
    });
}

/// Two tilted surfaces in exactly the same place, one opaque and one
/// just short of it, draw as both opaque do: the same one in front at
/// every pixel. The opaque one's depth comes from the G-buffer's vertex
/// entry and the other's from the layers' (the same arithmetic with an
/// invariant position), and the rule for surfaces at one depth holds
/// only where the two come to the same bits.
#[test]
fn tilted_surfaces_in_the_same_place_keep_one_in_front() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.antialiasing = false;
        let mut seen = 0;
        for i in 0..40 {
            // Any directions and places will do, so long as they are not
            // round numbers.
            let (a, b) = (0.37 + i as f32 * 0.173, 0.11 + i as f32 * 0.291);
            let u = Vec3::new(a.cos(), a.sin() * b.cos(), a.sin() * b.sin()) * 0.61;
            let v = Vec3::new(-a.sin() * 0.3, b.cos(), b.sin()).normalize() * 0.47;
            let centre = Vec3::new(0.013 * i as f32 - 0.2, 0.07, 0.031 * i as f32 - 0.4);
            let dark = quad(centre, u, v, [0.2, 0.2, 0.2, 1.0]);
            let pale = quad(centre, u, v, [0.6, 0.6, 0.6, 1.0]);
            let z = 0.123 + 0.017 * i as f32;
            settings.edges.enabled = i % 2 == 0;
            let mut frame = |dark_opacity: f32, pale_opacity: f32| {
                frame_of(
                    offscreen,
                    &mut renderer,
                    &settings,
                    &[(&dark, z, 1, dark_opacity), (&pale, z, 2, pale_opacity)],
                )
            };
            let opaque = frame(1.0, 1.0);
            // Facing away, the sheets are not drawn at all.
            let background = pixel(&opaque, 0, 0);
            let covered = (0..W * H)
                .filter(|i| pixel(&opaque, i % W, i / W) != background)
                .count();
            if covered < 200 {
                continue;
            }
            seen += 1;
            for (dark_opacity, pale_opacity) in [(0.999, 0.999), (0.999, 1.0), (1.0, 0.999)] {
                assert_eq!(
                    differing(&frame(dark_opacity, pale_opacity), &opaque, 1),
                    0,
                    "sheet {i}: dark at {dark_opacity}, pale at {pale_opacity}"
                );
            }
        }
        assert!(seen >= 10, "{seen} sheets faced the eye");
    });
}

/// A pixel is drawn `transparency_layers` transparent surfaces deep,
/// counted from the eye: the ones behind are left out.
#[test]
fn transparent_surfaces_past_the_layer_limit_are_left_out_from_the_back() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.antialiasing = false;
        let sheets = [
            (0.3, 0.6),
            (0.0, 0.2),
            (-0.3, 0.45),
            (-0.6, 0.1),
            (-0.9, 0.3),
        ];
        let mut frame = |count: usize, layers: u32| {
            settings.transparency_layers = layers;
            // Farthest first, so the depth order is not the drawing order.
            for (i, (z, albedo)) in sheets[..count].iter().enumerate().rev() {
                let sheet = quad(Vec3::ZERO, Vec3::X * 0.6, Vec3::Y * 0.6, [*albedo; 4]);
                submit_at(offscreen, &mut renderer, &sheet, *z, i as u32 + 1, 0.5);
            }
            offscreen
                .render_rgba(&mut renderer, &view(), &settings)
                .unwrap()
                .0
        };
        for layers in [1, 2, 3, 4, 5] {
            let limited = frame(5, layers);
            assert!(limited == frame(layers as usize, 8), "{layers} layers");
            if layers < 5 {
                assert!(limited != frame(5, 8), "{layers} layers");
            }
        }
    });
}

/// A pick lands on the nearest surface that is opaque, or transparent
/// and at least `pick_opacity` opaque: it passes through the fainter
/// ones in front, however many, to that surface or to the opaque scene.
/// A frame without transparent meshes is not picked through an earlier
/// one's.
#[test]
fn a_pick_lands_on_the_nearest_surface_opaque_enough() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let far = quad(Vec3::ZERO, Vec3::X * 0.8, Vec3::Y * 0.8, [1.0; 4]);
        let near = quad(Vec3::ZERO, Vec3::X * 0.2, Vec3::Y * 0.2, [1.0; 4]);
        // `sheets`: how far up the view axis, the object and the opacity
        // of small sheets over the middle of an opaque one, object 7.
        let mut pick = |sheets: &[(f32, u32, f32)], pick_opacity: f32, pixel: (u32, u32)| {
            let mut settings = settings(false);
            settings.pick_opacity = pick_opacity;
            settings.transparency_layers = 3;
            submit_at(offscreen, &mut renderer, &far, -0.5, 7, 1.0);
            // Too faint to show in the frame, so not there to be picked.
            submit_at(offscreen, &mut renderer, &near, 0.9, 11, 0.0001);
            for (z, object, opacity) in sheets {
                submit_at(offscreen, &mut renderer, &near, *z, *object, *opacity);
            }
            offscreen
                .render_rgba(&mut renderer, &view(), &settings)
                .unwrap();
            let pick = offscreen.pick(&mut renderer, pixel).expect("a pick");
            (
                pick.object.0,
                (pick.world.expect("a surface").z * 1e3).round() / 1e3,
            )
        };
        let centre = (W / 2, H / 2);
        let by_default = RenderSettings::default().pick_opacity;
        assert_eq!(pick(&[(0.3, 9, 0.3)], by_default, centre), (9, 0.3));
        // Beside the transparent mesh, and behind the opaque one.
        assert_eq!(
            pick(&[(0.3, 9, 0.3)], by_default, (W / 2 + 20, H / 2)),
            (7, -0.5)
        );
        assert_eq!(pick(&[(-0.8, 9, 0.3)], by_default, centre), (7, -0.5));
        // Through one fainter sheet in front of it, and through two.
        let one = [(0.3, 9, 0.3), (0.6, 12, 0.1)];
        let two = [(0.3, 9, 0.3), (0.6, 12, 0.1), (0.7, 13, 0.2)];
        assert_eq!(pick(&one, by_default, centre), (9, 0.3));
        assert_eq!(pick(&two, by_default, centre), (9, 0.3));
        // Through them all to the opaque scene, when none is opaque
        // enough.
        assert_eq!(
            pick(&[(0.3, 9, 0.2), (0.6, 12, 0.1)], by_default, centre),
            (7, -0.5)
        );
        // What is opaque enough is the frame's setting.
        assert_eq!(pick(&one, 0.05, centre), (12, 0.6));
        assert_eq!(pick(&one, 0.3, centre), (9, 0.3));
        assert_eq!(pick(&one, 0.5, centre), (7, -0.5));
        // A mesh too faint to be drawn is not picked at any setting.
        assert_eq!(pick(&[], 0.0, centre), (7, -0.5));
        // The frame after the transparent meshes are gone.
        assert_eq!(pick(&[(0.3, 9, 0.3)], by_default, centre), (9, 0.3));
        assert_eq!(pick(&[], by_default, centre), (7, -0.5));
    });
}

/// Two transparent surfaces a hair apart are both drawn, wherever the
/// pair lies among the depth buffer's steps: the nearer in front, or
/// the higher object where the two come to one depth. The layers' depth
/// test must rank two fragments as the peel's comparison of their
/// 32-bit depths does: where the test called a tie that the comparison
/// did not, and kept the farther, the nearer was peeled away as drawn
/// already and never was.
///
/// This fails with a 24-bit depth target for the layers only on an
/// adapter whose 24-bit buffer is one: of those tried, Mesa's software
/// rasteriser (`LIBGL_ALWAYS_SOFTWARE=1`), and neither GPU.
#[test]
fn transparent_surfaces_a_hair_apart_are_both_drawn() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.antialiasing = false;
        let dark = quad(
            Vec3::ZERO,
            Vec3::X * 0.3,
            Vec3::Y * 0.3,
            [0.2, 0.2, 0.2, 1.0],
        );
        let pale = quad(
            Vec3::ZERO,
            Vec3::X * 0.3,
            Vec3::Y * 0.3,
            [0.6, 0.6, 0.6, 1.0],
        );
        // Close behind the near plane, where depths are small numbers
        // and 32-bit floats finer than a 24-bit buffer's steps (one of
        // which is about 1e-7 m here).
        let view = CameraView::look_at(
            Vec3::new(0.0, 0.0, 2.1),
            Vec3::ZERO,
            Vec3::Y,
            0.8,
            W as f32 / H as f32,
            2.0,
            10.0,
        );
        let background = [BACKGROUND[0], BACKGROUND[1], BACKGROUND[2]];
        let (d, p) = (
            lit(&settings, 0.2, Vec3::Z, Vec3::Z),
            lit(&settings, 0.6, Vec3::Z, Vec3::Z),
        );
        let dark_in_front = over(d, 0.5, over(p, 0.5, background)).map(srgb_byte);
        let pale_in_front = over(p, 0.5, over(d, 0.5, background)).map(srgb_byte);
        let mut centre = |base: f32, gap: f32| {
            // The dark sheet is the nearer, and the lower object: drawn
            // first, and behind if the two tie.
            submit_at(offscreen, &mut renderer, &dark, base + gap, 1, 0.5);
            submit_at(offscreen, &mut renderer, &pale, base, 2, 0.5);
            let frame = offscreen
                .render_rgba(&mut renderer, &view, &settings)
                .unwrap()
                .0;
            pixel(&frame, W / 2, H / 2)
        };
        let mut in_order = 0;
        for step in 0..64 {
            let base = step as f32 * 1.3e-8;
            for gap in [2e-8, 4e-8, 8e-8] {
                let centre = centre(base, gap);
                let right = close(centre, dark_in_front, 3.0);
                in_order += right as u32;
                assert!(
                    right || close(centre, pale_in_front, 3.0),
                    "{gap} apart at {base}: {centre:?}"
                );
            }
        }
        // Most of them are apart in 32 bits.
        assert!(in_order > 64, "{in_order} in order");
        let apart = centre(0.0, 1e-5);
        assert!(close(apart, dark_in_front, 3.0), "{apart:?}");
    });
}

/// The layers' targets are made by the first frame with a transparent
/// mesh, kept through a short run of frames without one, and given back
/// by the `LAYERS_KEPT_FRAMES`th of them; the next transparent frame
/// makes them again and draws what the first did.
#[test]
fn the_layers_targets_are_given_back() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let settings = settings(false);
        let sheet = quad(Vec3::ZERO, Vec3::X * 0.5, Vec3::Y * 0.5, [1.0; 4]);
        // An opaque sheet, object 1, with or without a transparent one,
        // object 2, in front of it.
        let frame = |renderer: &mut Renderer, transparent: bool| {
            let items = [(&sheet, 0.0, 1, 1.0), (&sheet, 0.3, 2, 0.5)];
            let items = &items[..1 + transparent as usize];
            frame_of(offscreen, renderer, &settings, items)
        };
        frame(&mut renderer, false);
        assert!(renderer.layers.is_none(), "before any transparent mesh");
        let first = frame(&mut renderer, true);
        assert!(renderer.layers.is_some());
        let picked = offscreen.pick(&mut renderer, (W / 2, H / 2));
        assert_eq!(picked.map(|pick| pick.object), Some(ObjectId(2)));
        for _ in 1..crate::LAYERS_KEPT_FRAMES {
            frame(&mut renderer, false);
        }
        assert!(renderer.layers.is_some(), "kept through a short run");
        // The layers still stand, and hold the transparent sheet: a pick
        // of a frame that did not draw it does not read them.
        let picked = offscreen.pick(&mut renderer, (W / 2, H / 2));
        assert_eq!(picked.map(|pick| pick.object), Some(ObjectId(1)));
        // A transparent frame starts the count again.
        frame(&mut renderer, true);
        for _ in 1..crate::LAYERS_KEPT_FRAMES {
            frame(&mut renderer, false);
        }
        assert!(renderer.layers.is_some(), "counted from the last one");
        frame(&mut renderer, false);
        assert!(renderer.layers.is_none(), "given back");
        let picked = offscreen.pick(&mut renderer, (W / 2, H / 2));
        assert_eq!(picked.map(|pick| pick.object), Some(ObjectId(1)));
        assert!(frame(&mut renderer, true) == first);
        assert!(renderer.layers.is_some());
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
            offscreen.queue(),
            &quad(Vec3::ZERO, Vec3::X * 0.5, Vec3::Y * 0.5, [1.0; 4]),
        );
        let red = [1.0, 0.0, 0.0, 1.0];
        let frame = |renderer: &mut Renderer, depth_mode| {
            renderer.submit_retained_mesh(&mesh, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
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
            offscreen.queue(),
            &quad(Vec3::ZERO, Vec3::X, Vec3::Y, white),
        );
        // A wall along x = 0, facing -X, standing 1 m off the floor.
        let wall = renderer.create_retained_mesh(
            offscreen.device(),
            offscreen.queue(),
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
        // The two meshes span a box 3 m corner to corner: a 0.5 m radius.
        let frame = |renderer: &mut Renderer, ao| {
            renderer.submit_retained_mesh(&floor, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
            renderer.submit_retained_mesh(&wall, Mat4::IDENTITY, ObjectId(2), MaterialId(0), 1.0);
            let mut settings = settings(ao);
            settings.ao.radius = 0.5 / 3.0;
            offscreen
                .render_rgba(renderer, &oblique, &settings)
                .unwrap()
                .0
        };
        // Floor rows below the centre: 1 px down is 0.03 m from the wall,
        // 28 px down is 0.8 m, beyond the 0.5 m AO radius. Averaged over a
        // few columns.
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

/// The lights are fixed to the camera: a face turned toward the viewer
/// is lit the same whichever way the camera looks at the world, apart
/// from the sky, which is the world's.
#[test]
fn the_lights_turn_with_the_view() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        // No sky gradient: only the camera's lights are left.
        settings.lighting.ground = settings.lighting.sky;
        let grey = [0.25, 0.25, 0.25, 1.0];
        let mut centre = |eye: Vec3, up: Vec3| {
            // A square facing the eye, through the origin.
            let toward = eye.normalize();
            let u = up.cross(toward).normalize() * 0.5;
            let v = toward.cross(u);
            let mesh = renderer.create_retained_mesh(
                offscreen.device(),
                offscreen.queue(),
                &quad(Vec3::ZERO, u, v, grey),
            );
            renderer.submit_retained_mesh(&mesh, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
            let view =
                CameraView::look_at(eye, Vec3::ZERO, up, 0.8, W as f32 / H as f32, 0.1, 10.0);
            let (rgba, _) = offscreen
                .render_rgba(&mut renderer, &view, &settings)
                .unwrap();
            pixel(&rgba, W / 2, H / 2)
        };
        let from_above = centre(Vec3::new(0.0, 0.0, 3.0), Vec3::Y);
        let from_the_side = centre(Vec3::new(3.0, 0.0, 0.0), Vec3::Z);
        let from_below = centre(Vec3::new(1.0, -2.0, -2.0), Vec3::Z);
        let expected = lit(&settings, 0.25, Vec3::Z, Vec3::Z).map(srgb_byte);
        for seen in [from_above, from_the_side, from_below] {
            assert!(close(seen, expected, 3.0), "{seen:?} vs {expected:?}");
        }
    });
}

/// A material's highlight shows where the surface mirrors a light toward
/// the eye, and a matte material has none.
#[test]
fn a_material_sets_the_highlight() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        // Tilted so its normal is halfway between the eye and the key
        // light: the mirror direction for the pixel at the centre.
        let mut settings = settings(false);
        let key = Vec3::from(settings.lighting.lights[0].direction).normalize();
        let normal = (key + Vec3::Z).normalize();
        let u = Vec3::Y.cross(normal).normalize() * 0.6;
        let v = normal.cross(u);
        let mesh = renderer.create_retained_mesh(
            offscreen.device(),
            offscreen.queue(),
            &quad(Vec3::ZERO, u, v, [0.25, 0.25, 0.25, 1.0]),
        );
        settings.materials = vec![
            crate::Material {
                specular: 0.0,
                ..Default::default()
            },
            crate::Material {
                specular: 0.4,
                roughness: 0.3,
                ..Default::default()
            },
        ];
        let mut centre = |material: u32| {
            renderer.submit_retained_mesh(
                &mesh,
                Mat4::IDENTITY,
                ObjectId(1),
                MaterialId(material),
                1.0,
            );
            let (rgba, _) = offscreen
                .render_rgba(&mut renderer, &view(), &settings)
                .unwrap();
            pixel(&rgba, W / 2, H / 2)
        };
        let matte = centre(0);
        let glossy = centre(1);
        let expected = lit(&settings, 0.25, normal, Vec3::Z).map(srgb_byte);
        assert!(close(matte, expected, 3.0), "{matte:?} vs {expected:?}");
        assert!(glossy[1] > matte[1] + 40, "{glossy:?} vs matte {matte:?}");
        // An id past the table's end takes the first material.
        assert_eq!(centre(9), matte);
    });
}

/// Edge lines mark a silhouette, a crease and the boundary between two
/// objects, each one pixel wide, and leave flat faces alone.
#[test]
fn edge_lines_mark_silhouettes_creases_and_object_boundaries() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let white = [1.0; 4];
        let device = offscreen.device();
        let queue = offscreen.queue();
        // Seen from above: a floor square with a ridge along x = 0 (two
        // faces meeting at 90 degrees), and a second object lying flush
        // in the floor's plane beside it.
        let left = renderer.create_retained_mesh(
            device,
            queue,
            &quad(
                Vec3::new(-0.3, 0.0, -0.3),
                Vec3::new(0.3, 0.0, 0.3),
                Vec3::Y * 0.5,
                white,
            ),
        );
        let right = renderer.create_retained_mesh(
            device,
            queue,
            &quad(
                Vec3::new(0.3, 0.0, -0.3),
                Vec3::new(0.3, 0.0, -0.3),
                Vec3::Y * 0.5,
                white,
            ),
        );
        let flush = renderer.create_retained_mesh(
            device,
            queue,
            &quad(
                Vec3::new(0.8, 0.0, -0.6),
                Vec3::X * 0.2,
                Vec3::Y * 0.5,
                white,
            ),
        );
        let view = view();
        let mut frame = |edges: bool| {
            let mut settings = settings(false);
            settings.edges.enabled = edges;
            settings.materials[0].specular = 0.0;
            settings.antialiasing = false;
            renderer.submit_retained_mesh(&left, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
            renderer.submit_retained_mesh(&right, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
            renderer.submit_retained_mesh(&flush, Mat4::IDENTITY, ObjectId(2), MaterialId(0), 1.0);
            offscreen
                .render_rgba(&mut renderer, &view, &settings)
                .unwrap()
                .0
        };
        let plain = frame(false);
        let edged = frame(true);
        // The pixels the lines darkened, along the row through the
        // middle of the frame.
        let (_, row) = at(&view, Vec3::ZERO);
        let darkened: Vec<u32> = (0..W)
            .filter(|&x| {
                let (a, b) = (pixel(&edged, x, row), pixel(&plain, x, row));
                a != b && a[1] < b[1]
            })
            .collect();
        let near = |world_x: f32, z: f32| {
            let (x, _) = at(&view, Vec3::new(world_x, 0.0, z));
            darkened.iter().filter(|&&d| d.abs_diff(x) <= 1).count()
        };
        assert_eq!(near(-0.6, -0.6), 1, "silhouette, left: {darkened:?}");
        assert_eq!(near(0.0, 0.0), 1, "crease: {darkened:?}");
        assert_eq!(near(0.6, -0.6), 1, "object boundary: {darkened:?}");
        assert_eq!(near(1.0, -0.6), 1, "silhouette, right: {darkened:?}");
        assert_eq!(darkened.len(), 4, "and nothing else: {darkened:?}");
    });
}

/// Ambient occlusion is sized by the scene, not by its units or the clip
/// planes: the same scene a thousand times smaller is shaded the same.
#[test]
fn ambient_occlusion_is_the_same_at_any_scale() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let white = [1.0; 4];
        let mut frame = |scale: f32| {
            let floor = renderer.create_retained_mesh(
                offscreen.device(),
                offscreen.queue(),
                &quad(Vec3::ZERO, Vec3::X * scale, Vec3::Y * scale, white),
            );
            let wall = renderer.create_retained_mesh(
                offscreen.device(),
                offscreen.queue(),
                &quad(
                    Vec3::new(0.0, 0.0, 0.5) * scale,
                    Vec3::Z * 0.5 * scale,
                    Vec3::Y * scale,
                    white,
                ),
            );
            renderer.submit_retained_mesh(&floor, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
            renderer.submit_retained_mesh(&wall, Mat4::IDENTITY, ObjectId(2), MaterialId(0), 1.0);
            let view = CameraView::look_at(
                Vec3::new(-2.0, 0.0, 2.0) * scale,
                Vec3::ZERO,
                Vec3::Z,
                0.8,
                W as f32 / H as f32,
                0.1 * scale,
                // A far plane well past the scene, differently so each time.
                (10.0 + scale) * scale.sqrt(),
            );
            offscreen
                .render_rgba(&mut renderer, &view, &settings(true))
                .unwrap()
                .0
        };
        let metres = frame(1.0);
        let millimetres = frame(0.001);
        let differing = (0..W * H)
            .filter(|i| {
                let (a, b) = (
                    pixel(&metres, i % W, i / W),
                    pixel(&millimetres, i % W, i / W),
                );
                (0..3).any(|c| a[c].abs_diff(b[c]) > 6)
            })
            .count();
        assert!(differing < 30, "{differing} pixels differ");
    });
}

/// How far a pixel is from the background colour, summed over channels.
fn off_background(p: [u8; 3]) -> u32 {
    p[0] as u32 + p[1] as u32 + (255 - p[2]) as u32
}

/// The pixel `world` lands on in `view`.
fn at(view: &CameraView, world: Vec3) -> (u32, u32) {
    let px = view.project(world, W, H).expect("in front of the camera");
    (px.x as u32, px.y as u32)
}

fn grid_settings(spacing: GridSpacing) -> RenderSettings {
    let mut settings = settings(false);
    settings.grid.visible = true;
    settings.grid.spacing = spacing;
    settings
}

/// Seen from above, the grid draws its lines where the plane's
/// coordinates are whole spacings and leaves the cells empty, the world X
/// and Y axes are red and green, and the frame reports the spacing.
#[test]
fn the_grid_draws_lines_at_its_spacing_and_coloured_axes() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let view = view();
        let (rgba, info) = offscreen
            .render_rgba(
                &mut renderer,
                &view,
                &grid_settings(GridSpacing::Fixed(0.05)),
            )
            .unwrap();
        assert_eq!(info.grid_spacing, Some(0.05));
        let sample = |x: f32, y: f32| {
            let (px, py) = at(&view, Vec3::new(x, y, 0.0));
            pixel(&rgba, px, py)
        };

        // The view is 3 m up, so a 0.5 m major cell is some 25 pixels.
        let cell = sample(0.25, 0.25);
        assert!(off_background(cell) < 12, "inside a cell: {cell:?}");
        let major = sample(0.5, 0.25);
        assert!(off_background(major) > 60, "on a major line: {major:?}");
        assert!(major[0] == major[1], "grid lines are grey: {major:?}");

        let x_axis = sample(0.25, 0.0);
        assert!(x_axis[0] > x_axis[1] + 50, "X axis: {x_axis:?}");
        let y_axis = sample(0.0, 0.25);
        assert!(y_axis[1] > y_axis[0] + 30, "Y axis: {y_axis:?}");

        // Hidden: nothing is drawn, and nothing is reported.
        let (rgba, info) = offscreen
            .render_rgba(&mut renderer, &view, &settings(false))
            .unwrap();
        assert_eq!(info.grid_spacing, None);
        let (px, py) = at(&view, Vec3::new(0.5, 0.25, 0.0));
        assert_eq!(pixel(&rgba, px, py), [0, 0, 255]);
    });
}

/// Geometry hides the grid behind it, and a surface lying in the grid's
/// own plane wins over it rather than fighting.
#[test]
fn geometry_occludes_the_grid() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let view = view();
        let white = [1.0; 4];
        for z in [0.3, 0.0] {
            let mesh = renderer.create_retained_mesh(
                offscreen.device(),
                offscreen.queue(),
                &quad(Vec3::new(0.0, 0.0, z), Vec3::X * 0.4, Vec3::Y * 0.4, white),
            );
            let frame = |renderer: &mut Renderer, settings: &RenderSettings| {
                renderer.submit_retained_mesh(
                    &mesh,
                    Mat4::IDENTITY,
                    ObjectId(1),
                    MaterialId(0),
                    1.0,
                );
                offscreen.render_rgba(renderer, &view, settings).unwrap().0
            };
            let plain = frame(&mut renderer, &settings(false));
            let gridded = frame(&mut renderer, &grid_settings(GridSpacing::Fixed(0.05)));
            // Over the quad the two frames agree, axis lines included.
            for (x, y) in [(0.0, 0.0), (0.25, 0.0), (0.0, 0.25), (0.1, 0.3)] {
                let (px, py) = at(&view, Vec3::new(x, y, z));
                assert_eq!(
                    pixel(&gridded, px, py),
                    pixel(&plain, px, py),
                    "quad at z = {z}, over ({x}, {y})"
                );
            }
            // Beside it the grid is there.
            let (px, py) = at(&view, Vec3::new(0.5, 0.25, 0.0));
            assert_ne!(pixel(&gridded, px, py), pixel(&plain, px, py));
        }
    });
}

/// The world Z axis is a blue line through the origin, of no extent: it
/// crosses the whole frame of a side view, behind the camera's near
/// plane and beyond the scene alike.
#[test]
fn the_z_axis_is_a_line_through_the_origin() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let view = CameraView::look_at(
            Vec3::new(0.0, -3.0, 0.4),
            Vec3::new(0.0, 0.0, 0.4),
            Vec3::Z,
            0.8,
            W as f32 / H as f32,
            0.1,
            10.0,
        );
        let (rgba, _) = offscreen
            .render_rgba(
                &mut renderer,
                &view,
                &grid_settings(GridSpacing::Fixed(0.05)),
            )
            .unwrap();
        let blue = |x: u32, y: u32| {
            let p = pixel(&rgba, x, y);
            // The axis colour over the background: the background has
            // no green at all.
            p[2] > 150 && p[1] > 80 && p[1] > p[0]
        };
        for y in [2, H / 4, H / 2, H - 3] {
            assert!(
                blue(W / 2, y) || blue(W / 2 - 1, y),
                "row {y}: {:?}",
                pixel(&rgba, W / 2, y)
            );
        }
        // Beside the line there is only background.
        assert_eq!(pixel(&rgba, W / 2 + 6, 4), [0, 0, 255]);

        // Without axes the plane is still drawn and the line is not.
        let mut settings = grid_settings(GridSpacing::Fixed(0.05));
        settings.grid.axes = false;
        let (rgba, _) = offscreen
            .render_rgba(&mut renderer, &view, &settings)
            .unwrap();
        assert_eq!(pixel(&rgba, W / 2, 4), [0, 0, 255]);
    });
}

/// Automatic spacing is the power of ten that keeps the minor cells a
/// readable size at the focus depth, and it follows the zoom.
#[test]
fn automatic_grid_spacing_follows_the_zoom() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut camera = crate::Camera {
            distance: 2.0,
            ..crate::Camera::default()
        };
        let mut spacing = |camera: &crate::Camera| {
            let settings = grid_settings(GridSpacing::Auto {
                focus_depth: camera.distance,
                min_cell_px: 8.0,
            });
            let view = camera.view(W as f32 / H as f32, None);
            let (_, info) = offscreen
                .render_rgba(&mut renderer, &view, &settings)
                .unwrap();
            let spacing = info.grid_spacing.expect("the grid is drawn");
            let cell_px = spacing / view.pixel_size(camera.distance, H);
            assert!((8.0..80.0).contains(&cell_px), "{cell_px} px cells");
            spacing
        };
        let near = spacing(&camera);
        camera.distance = 200.0;
        let far = spacing(&camera);
        assert!((far / near - 100.0).abs() < 0.01, "{near} then {far}");
        camera.projection = crate::Projection::Orthographic;
        assert_eq!(spacing(&camera), far, "the projection toggle keeps it");
    });
}

/// The gizmo is drawn where its layout says: each end's disc is its
/// axis's colour, the pointer's end is lit, and nothing is drawn outside
/// its circle.
#[test]
fn the_gizmo_is_drawn_as_laid_out() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let camera = crate::Camera::default();
        let mut settings = settings(false);
        let mut gizmo = ViewGizmo {
            orientation: camera.orientation,
            center: Vec2::new(64.0, 48.0),
            radius: 40.0,
            hovered: None,
        };
        settings.gizmo = Some(gizmo);
        let view = camera.view(W as f32 / H as f32, None);
        let (rgba, _) = offscreen
            .render_rgba(&mut renderer, &view, &settings)
            .unwrap();

        // Each negative end's disc is its axis's colour over the
        // background: sampled inside the ring, off the centre.
        for end in gizmo.ends() {
            let inside = end.center + Vec2::new(0.0, end.radius * 0.6);
            let p = pixel(&rgba, inside.x as u32, inside.y as u32);
            assert_eq!(
                gizmo.hit_test(inside),
                Some(GizmoPart::End {
                    axis: end.axis,
                    positive: end.positive
                })
            );
            assert_ne!(p, [0, 0, 255], "{end:?} drew nothing");
            if end.positive {
                let strongest = (0..3).max_by_key(|&c| p[c]).unwrap();
                assert_eq!(strongest, end.axis, "{end:?} is {p:?}");
            }
        }
        assert_eq!(pixel(&rgba, 64 + 44, 48), [0, 0, 255], "outside the circle");
        assert_eq!(pixel(&rgba, 4, 4), [0, 0, 255]);

        // Hovering an end lights it and puts a backdrop behind the gizmo.
        let x_end = *gizmo
            .ends()
            .iter()
            .find(|end| end.axis == 0 && end.positive)
            .unwrap();
        gizmo.hovered = Some(GizmoPart::End {
            axis: 0,
            positive: true,
        });
        settings.gizmo = Some(gizmo);
        let (lit, _) = offscreen
            .render_rgba(&mut renderer, &view, &settings)
            .unwrap();
        let inside = x_end.center + Vec2::new(0.0, x_end.radius * 0.6);
        let (x, y) = (inside.x as u32, inside.y as u32);
        assert!(pixel(&lit, x, y)[1] > pixel(&rgba, x, y)[1] + 20);
        let body = gizmo.center + Vec2::new(-0.5, 0.75) * gizmo.radius;
        assert_eq!(gizmo.hit_test(body), Some(GizmoPart::Body));
        assert_ne!(pixel(&lit, body.x as u32, body.y as u32), [0, 0, 255]);
        assert_eq!(pixel(&rgba, body.x as u32, body.y as u32), [0, 0, 255]);
    });
}

/// Anti-aliasing blends the pixels along a slanted silhouette between the
/// surface and the background, and changes nothing away from an edge.
#[test]
fn antialiasing_smooths_silhouettes_only() {
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mesh = renderer.create_retained_mesh(
            offscreen.device(),
            offscreen.queue(),
            &quad(
                Vec3::ZERO,
                Vec3::new(0.6, 0.15, 0.0),
                Vec3::new(-0.1, 0.4, 0.0),
                [1.0; 4],
            ),
        );
        let mut frame = |antialiasing: bool| {
            // Matte, so the surface is one colour all over.
            let mut settings = settings(false);
            settings.materials[0].specular = 0.0;
            settings.antialiasing = antialiasing;
            renderer.submit_retained_mesh(&mesh, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
            offscreen
                .render_rgba(&mut renderer, &view(), &settings)
                .unwrap()
                .0
        };
        let stepped = frame(false);
        let smooth = frame(true);

        let surface = pixel(&stepped, W / 2, H / 2);
        // Pixels that are neither the surface nor the background.
        let blended = |rgba: &[u8]| {
            (0..W * H)
                .filter(|i| {
                    let p = pixel(rgba, i % W, i / W);
                    p != surface && p != [0, 0, 255]
                })
                .count()
        };
        assert_eq!(blended(&stepped), 0, "without it there are two colours");
        let edge_pixels = blended(&smooth);
        assert!(edge_pixels > 60, "{edge_pixels} blended pixels");
        // Only near the silhouette: every blended pixel has both colours
        // within two pixels of it in the stepped frame.
        for i in 0..W * H {
            let (x, y) = (i % W, i / W);
            if pixel(&smooth, x, y) == pixel(&stepped, x, y) {
                continue;
            }
            let mut seen = [false; 2];
            for ny in y.saturating_sub(2)..(y + 3).min(H) {
                for nx in x.saturating_sub(2)..(x + 3).min(W) {
                    seen[(pixel(&stepped, nx, ny) == surface) as usize] = true;
                }
            }
            assert_eq!(seen, [true; 2], "({x}, {y}) changed away from an edge");
        }
    });
}

/// A frame drawn twice as large with a pixel scale of 2 and scaled down
/// is the same picture, smoother: a line keeps its width, an edge line
/// stays one pixel, and a slanted silhouette gains in-between values
/// without any other anti-aliasing.
#[test]
fn a_supersampled_frame_is_the_same_picture_smoother() {
    on_each_backend(|offscreen| {
        let mesh_data = quad(
            Vec3::new(0.0, 0.3, 0.0),
            Vec3::new(0.6, 0.15, 0.0),
            Vec3::new(-0.1, 0.4, 0.0),
            [1.0; 4],
        );
        let frame = |factor: u32| {
            let mut renderer = offscreen.renderer(W * factor, H * factor);
            let mesh =
                renderer.create_retained_mesh(offscreen.device(), offscreen.queue(), &mesh_data);
            renderer.submit_retained_mesh(&mesh, Mat4::IDENTITY, ObjectId(1), MaterialId(0), 1.0);
            renderer.submit_lines(
                &line(-0.5, 0.2, [1.0, 0.0, 0.0, 1.0]),
                Mat4::IDENTITY,
                style(4.0, DepthMode::Normal),
            );
            let mut settings = settings(false);
            settings.antialiasing = false;
            settings.edges.enabled = true;
            settings.materials[0].specular = 0.0;
            settings.pixel_scale = factor as f32;
            let (rgba, _) = offscreen
                .render_rgba(&mut renderer, &view(), &settings)
                .unwrap();
            crate::offscreen::downsample_rgba(&rgba, W * factor, H * factor, factor)
        };
        let once = frame(1);
        let twice = frame(2);

        // The 4 px line is as thick in both.
        let thick = |rgba: &[u8]| rows_of(rgba, W / 2, 0);
        assert!((3..=5).contains(&thick(&once)), "{}", thick(&once));
        assert!((3..=5).contains(&thick(&twice)), "{}", thick(&twice));

        // The pictures agree away from edges, and nearly everywhere.
        let far_apart = (0..W * H)
            .filter(|i| {
                let (a, b) = (pixel(&once, i % W, i / W), pixel(&twice, i % W, i / W));
                (0..3).any(|c| a[c].abs_diff(b[c]) > 40)
            })
            .count();
        assert!(far_apart < 250, "{far_apart} pixels far apart");
        let (cx, cy) = at(&view(), Vec3::new(0.0, 0.3, 0.0));
        assert_eq!(pixel(&once, cx, cy), pixel(&twice, cx, cy));

        // Along the quad's upper silhouette the single-sample frame has
        // the surface, the edge line and the background; the supersampled
        // one has values in between.
        let colours = |rgba: &[u8]| {
            (0..W)
                .flat_map(|x| (0..cy.saturating_sub(4)).map(move |y| (x, y)))
                .map(|(x, y)| pixel(rgba, x, y))
                .collect::<std::collections::HashSet<_>>()
                .len()
        };
        assert!(colours(&once) <= 3, "{} colours", colours(&once));
        assert!(colours(&twice) > 6, "{} colours", colours(&twice));
    });
}

/// A sphere with smooth normals.
fn sphere(centre: Vec3, radius: f32) -> MeshData {
    let (rings, sectors) = (32u32, 64u32);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for ring in 0..=rings {
        let polar = std::f32::consts::PI * ring as f32 / rings as f32;
        for sector in 0..=sectors {
            let turn = std::f32::consts::TAU * sector as f32 / sectors as f32;
            let n = Vec3::new(
                polar.sin() * turn.cos(),
                polar.sin() * turn.sin(),
                polar.cos(),
            );
            vertices.push(MeshVertex::new(
                (centre + n * radius).to_array(),
                n.to_array(),
            ));
        }
    }
    for ring in 0..rings {
        for sector in 0..sectors {
            let a = ring * (sectors + 1) + sector;
            let b = a + sectors + 1;
            indices.extend([a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    MeshData {
        vertices,
        indices: Some(indices),
        texture: None,
    }
}

/// Surfels over a sphere, `pitch` apart, each with the disc radius a
/// direct cast would give it.
fn sphere_surfels(centre: Vec3, radius: f32, pitch: f32) -> SurfelData {
    let count = (4.0 * std::f32::consts::PI * radius * radius / (pitch * pitch)) as usize;
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    let surfels = (0..count)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / count as f32;
            let ring = (1.0 - z * z).sqrt();
            let turn = golden * i as f32;
            let n = Vec3::new(ring * turn.cos(), ring * turn.sin(), z);
            SurfelVertex::new((centre + n * radius).to_array(), n.to_array(), 0.75 * pitch)
        })
        .collect();
    SurfelData { surfels }
}

/// Surfels drawn as discs give the picture the same surface gives as a
/// mesh: the same pixels lit the same, the same silhouette, and a pick
/// through them reports their object and a point on the surface.
#[test]
fn surfel_discs_draw_like_the_surface_they_sample() {
    on_each_backend(|offscreen| {
        let mut settings = settings(false);
        settings.edges.enabled = true;
        let centre = Vec3::new(0.0, 0.0, 0.5);
        let radius = 0.8;
        let pixel_at_sphere = view().pixel_size(2.5 - radius, H);
        let offset = Vec3::new(0.3, -0.2, 0.0);

        // Both are tinted the same way: the mesh by its vertices, the
        // discs by their draw.
        let tint = [0.9, 0.5, 0.3, 1.0];
        let mut renderer = offscreen.renderer(W, H);
        let mut tinted = sphere(centre, radius);
        for vertex in &mut tinted.vertices {
            vertex.color = tint;
        }
        let mesh = renderer.create_retained_mesh(offscreen.device(), offscreen.queue(), &tinted);
        renderer.submit_retained_mesh(
            &mesh,
            Mat4::from_translation(offset),
            ObjectId(3),
            MaterialId(0),
            1.0,
        );
        let (as_mesh, _) = offscreen
            .render_rgba(&mut renderer, &view(), &settings)
            .unwrap();

        let surfels = renderer.create_retained_surfels(
            offscreen.device(),
            &sphere_surfels(centre, radius, pixel_at_sphere),
        );
        renderer.submit_retained_surfels(
            &surfels,
            Mat4::from_translation(offset),
            ObjectId(3),
            MaterialId(0),
            tint,
        );
        let (as_discs, info) = offscreen
            .render_rgba(&mut renderer, &view(), &settings)
            .unwrap();
        assert_eq!(info.overflow, None);

        let (mut differ, mut covered_mesh, mut covered_discs) = (0, 0, 0);
        for y in 0..H {
            for x in 0..W {
                let (a, b) = (pixel(&as_mesh, x, y), pixel(&as_discs, x, y));
                covered_mesh += off_background(a) as usize;
                covered_discs += off_background(b) as usize;
                differ += ((0..3).any(|c| (a[c] as i32 - b[c] as i32).abs() > 24)) as usize;
            }
        }
        assert!(covered_mesh > 2000, "{covered_mesh} sphere pixels");
        // The tint reached the discs: the sphere's centre pixel is the
        // tint's hue (red over green over blue) in both renders.
        let middle = at(&view(), centre + offset + Vec3::Z * radius);
        for image in [&as_mesh, &as_discs] {
            let p = pixel(image, middle.0, middle.1);
            assert!(p[0] > p[1] && p[1] > p[2], "untinted pixel {p:?}");
        }
        // No holes: inside the mesh's silhouette (2 px in from any
        // background pixel) every pixel is covered by a disc. Discs seen
        // at a slant used to leave pinholes near the rim.
        let mut holes = 0;
        for y in 2..H - 2 {
            for x in 2..W - 2 {
                let interior = (-2i32..=2).all(|dy| {
                    (-2i32..=2).all(|dx| {
                        off_background(pixel(
                            &as_mesh,
                            (x as i32 + dx) as u32,
                            (y as i32 + dy) as u32,
                        )) == 1
                    })
                });
                holes += (interior && off_background(pixel(&as_discs, x, y)) == 0) as usize;
            }
        }
        assert_eq!(holes, 0, "{holes} pinholes inside the silhouette");
        assert!(
            (covered_discs as i64 - covered_mesh as i64).abs() * 50 < covered_mesh as i64,
            "{covered_discs} disc pixels against {covered_mesh} mesh pixels"
        );
        // The sphere's faceting and the discs' edges disagree on some
        // pixels; most agree to well within a shade.
        assert!(
            differ * 20 < covered_mesh,
            "{differ} of {covered_mesh} pixels differ"
        );

        let pick = offscreen
            .pick(
                &mut renderer,
                at(&view(), centre + offset + Vec3::Z * radius),
            )
            .expect("a pick");
        assert_eq!(pick.object, ObjectId(3));
        let world = pick.world.expect("a surface");
        let off = (world - centre - offset).length() - radius;
        assert!(off.abs() < pixel_at_sphere, "{off} m off the sphere");
    });
}

/// Writes an RGBA frame as a binary PPM.
fn write_ppm(path: impl AsRef<std::path::Path>, rgba: &[u8], w: u32, h: u32) {
    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
    ppm.extend(rgba.chunks_exact(4).flat_map(|px| [px[0], px[1], px[2]]));
    std::fs::write(path, ppm).unwrap();
}

/// A reproducible stream of numbers (xorshift), so that a randomised
/// scene is the same scene on every run and every machine.
struct Random(u32);

impl Random {
    /// The next number, in 0..1.
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }

    fn between(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next()
    }

    /// A direction, each component between -1 and 1 before it is scaled
    /// to length 1.
    fn direction(&mut self) -> Vec3 {
        loop {
            let v = Vec3::new(
                self.between(-1.0, 1.0),
                self.between(-1.0, 1.0),
                self.between(-1.0, 1.0),
            );
            if v.length() > 0.1 {
                return v.normalize();
            }
        }
    }
}

/// A flat-shaded box about the origin.
fn cuboid(half: Vec3, color: [f32; 4]) -> MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    for axis in 0..3 {
        for sign in [1.0, -1.0] {
            // Tangents in the order that faces the side outward.
            let (u, v) = match sign > 0.0 {
                true => ((axis + 1) % 3, (axis + 2) % 3),
                false => ((axis + 2) % 3, (axis + 1) % 3),
            };
            let side = quad(
                axes[axis] * half[axis] * sign,
                axes[u] * half[u],
                axes[v] * half[v],
                color,
            );
            let first = vertices.len() as u32;
            indices.extend(side.indices.unwrap().iter().map(|i| first + i));
            vertices.extend(side.vertices);
        }
    }
    MeshData {
        vertices,
        indices: Some(indices),
        texture: None,
    }
}

/// `mesh` and its inside: every triangle a second time, facing the other
/// way. Back faces are not drawn, so this is a mesh whose far walls
/// show.
fn double_sided(mesh: &MeshData) -> MeshData {
    let count = mesh.vertices.len() as u32;
    let mut vertices = mesh.vertices.clone();
    vertices.extend(mesh.vertices.iter().map(|vertex| MeshVertex {
        normal: vertex.normal.map(|n| -n),
        ..*vertex
    }));
    let outside = mesh.indices.clone().unwrap();
    let mut indices = outside.clone();
    indices.extend(
        outside
            .chunks_exact(3)
            .flat_map(|t| [t[0] + count, t[2] + count, t[1] + count]),
    );
    MeshData {
        vertices,
        indices: Some(indices),
        texture: None,
    }
}

/// How far the randomised scenes are from what transparency promises of
/// its edge lines, counted in scenes and pixels; the figures
/// RENDERING_ARCHITECTURE.md quotes ("Transparent meshes", "Edge
/// lines"). The frame tests above hold each promise on a scene built to
/// show it; this measures them on scenes nobody chose: two or three
/// boxes and spheres about the origin, turned and placed at random so
/// that most pass through each other, one in three with its inside
/// drawn, seen from a random direction. Three comparisons, each of two
/// frames that should be the same picture:
///
/// - **opaque among opaque**: one mesh of three just short of opaque
///   (0.999, which is stored as opacity 1), against all three opaque;
/// - **behind a transparent mesh**: a mesh just short of opaque behind
///   or through a half-transparent one, against the same mesh opaque;
/// - **a faint mesh**: two meshes with a third too faint to see (the
///   least opacity drawn) among them, against the two alone.
///
/// The first is asserted, since it has no exceptions. The other two are
/// printed, a line an adapter:
/// `cargo test -p volumetric_renderer randomised -- --ignored --nocapture`.
/// With `VOLUMETRIC_RANDOM_DUMP` naming a folder, the two frames of
/// every case that differs are written there, and a third with the
/// pixels that differ in red.
#[test]
#[ignore]
fn randomised_scenes_keep_their_edge_lines() {
    const SCENES: u32 = 60;
    on_each_backend(|offscreen| {
        let mut renderer = offscreen.renderer(W, H);
        let mut settings = settings(false);
        settings.edges.enabled = true;
        settings.antialiasing = false;
        // Deep enough that no surface is left out: the layer limit is
        // not what is measured.
        settings.transparency_layers = 8;
        // Just over the least opacity drawn: a surface of it changes a
        // pixel by half a level at most, so what differs by more than a
        // level is a mark and not the mesh.
        let faint = crate::MIN_OPACITY * 1.01;
        let dump = std::env::var_os("VOLUMETRIC_RANDOM_DUMP").map(std::path::PathBuf::from);
        let adapter = offscreen
            .adapter_name()
            .replace(|c: char| !c.is_alphanumeric(), "_");
        let adapter = &adapter[..adapter.len().min(12)];

        // Per comparison: cases compared, cases that differ, pixels that
        // differ in all, and in the worst case.
        let mut tally = [[0usize; 4]; 6];
        let mut worst_scene = [0u32; 6];
        for scene in 0..SCENES {
            let mut random = Random(0x9E37_79B9 ^ (scene + 1).wrapping_mul(0x85EB_CA6B));
            let solids: Vec<_> = (0..3)
                .map(|_| {
                    let mesh = if random.next() < 0.5 {
                        cuboid(
                            Vec3::new(
                                random.between(0.15, 0.6),
                                random.between(0.15, 0.6),
                                random.between(0.15, 0.6),
                            ),
                            [random.between(0.3, 1.0); 4],
                        )
                    } else {
                        sphere(Vec3::ZERO, random.between(0.2, 0.55))
                    };
                    let inside = random.next() < 1.0 / 3.0;
                    let mesh = if inside { double_sided(&mesh) } else { mesh };
                    let transform = Mat4::from_rotation_translation(
                        glam::Quat::from_axis_angle(
                            random.direction(),
                            random.between(0.0, std::f32::consts::TAU),
                        ),
                        random.direction() * random.between(0.0, 0.6),
                    );
                    let mesh =
                        renderer.create_retained_mesh(offscreen.device(), offscreen.queue(), &mesh);
                    (mesh, transform, inside)
                })
                .collect();
            let eye = random.direction() * 3.0;
            let up = if eye.normalize().z.abs() > 0.9 {
                Vec3::Y
            } else {
                Vec3::Z
            };
            let view =
                CameraView::look_at(eye, Vec3::ZERO, up, 0.8, W as f32 / H as f32, 0.1, 10.0);
            // The frame of some of the solids: which, as what object,
            // how opaque.
            let mut frame = |items: &[(usize, u32, f32)]| {
                for (solid, object, opacity) in items {
                    let (mesh, transform, _) = &solids[*solid];
                    renderer.submit_retained_mesh(
                        mesh,
                        *transform,
                        ObjectId(*object),
                        MaterialId(0),
                        *opacity,
                    );
                }
                offscreen
                    .render_rgba(&mut renderer, &view, &settings)
                    .unwrap()
                    .0
            };
            let mut compare = |which: usize, of: &[usize], case: &str, a: &[u8], b: &[u8]| {
                let pixels = differing(a, b, 1);
                let which = which + 3 * of.iter().any(|&solid| solids[solid].2) as usize;
                if let (Some(dump), true) = (&dump, pixels > 0) {
                    let marked: Vec<u8> = a
                        .chunks_exact(4)
                        .zip(b.chunks_exact(4))
                        .flat_map(|(a, b)| {
                            if (0..4).any(|c| a[c].abs_diff(b[c]) > 1) {
                                [255, 0, 0, 255]
                            } else {
                                [a[0] / 2 + 64, a[1] / 2 + 64, a[2] / 2 + 64, 255]
                            }
                        })
                        .collect();
                    let name = format!("{adapter}_{which}_{scene:02}_{case}_{pixels}px");
                    for (frame, kind) in [(a, "a"), (b, "b"), (&marked[..], "differ")] {
                        write_ppm(dump.join(format!("{name}_{kind}.ppm")), frame, W, H);
                    }
                }
                let tally = &mut tally[which];
                tally[0] += 1;
                tally[1] += (pixels > 0) as usize;
                tally[2] += pixels;
                if pixels > tally[3] {
                    tally[3] = pixels;
                    worst_scene[which] = scene;
                }
            };

            let all_opaque = frame(&[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
            for nearly in 0..3 {
                let items: Vec<_> = (0..3)
                    .map(|i| (i, i as u32 + 1, if i == nearly { 0.999 } else { 1.0 }))
                    .collect();
                compare(
                    0,
                    &[0, 1, 2],
                    &format!("mesh{nearly}"),
                    &frame(&items),
                    &all_opaque,
                );
            }

            for (cover, behind) in [(0, 1), (1, 0)] {
                let with = |opacity: f32| [(cover, 1, 0.5), (behind, 2, opacity)];
                compare(
                    1,
                    &[0, 1],
                    &format!("cover{cover}"),
                    &frame(&with(0.999)),
                    &frame(&with(1.0)),
                );
            }

            for other in [1.0, 0.5] {
                let alone = frame(&[(0, 2, 1.0), (1, 3, other)]);
                // The faint mesh as the highest object and as the lowest.
                for object in [4, 1] {
                    let with = frame(&[(0, 2, 1.0), (1, 3, other), (2, object, faint)]);
                    compare(
                        2,
                        &[0, 1, 2],
                        &format!("other{other}_faint{object}"),
                        &with,
                        &alone,
                    );
                }
            }
        }
        for (name, (tally, scene)) in [
            "opaque among opaque",
            "behind a transparent mesh",
            "a faint mesh",
            "opaque among opaque, insides drawn",
            "behind a transparent mesh, insides drawn",
            "a faint mesh, insides drawn",
        ]
        .iter()
        .zip(tally.iter().zip(worst_scene))
        {
            eprintln!(
                "  {name}: {} of {} cases differ, {} pixels in all, {} at most (scene {scene}), of {} a frame",
                tally[1],
                tally[0],
                tally[2],
                tally[3],
                W * H,
            );
        }
        assert_eq!(
            tally[0][1] + tally[3][1],
            0,
            "a mesh just short of opaque shows"
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
    // `VOLUMETRIC_FRAME_GLES=1` dumps the WebGL2-like adapter's frame;
    // `VOLUMETRIC_FRAME_STEPPED=1` turns anti-aliasing off.
    let offscreen = if std::env::var_os("VOLUMETRIC_FRAME_GLES").is_some() {
        Offscreen::new_webgl2_like()
    } else {
        Offscreen::new()
    }
    .unwrap();
    let (w, h) = (960u32, 640u32);
    let mut renderer = offscreen.renderer(w, h);
    let scene = renderer.create_retained_scene(
        offscreen.device(),
        offscreen.queue(),
        &crate::test_scenes::create_test_scene(),
    );
    // `VOLUMETRIC_FRAME_OPACITY=0.5` draws the cube, and every added
    // shape but the plate they stand on, at that opacity.
    let opacity = std::env::var("VOLUMETRIC_FRAME_OPACITY")
        .map_or(1.0, |opacity| opacity.parse::<f32>().unwrap());
    for (mesh, transform) in &scene.meshes {
        renderer.submit_retained_mesh(mesh, *transform, ObjectId(1), MaterialId(0), opacity);
    }
    // `VOLUMETRIC_FRAME_SHAPES=1` adds a plate, a step and two spheres
    // around the cube: curved surfaces and contact, to judge shading by.
    let shapes: Vec<_> = if std::env::var_os("VOLUMETRIC_FRAME_SHAPES").is_some() {
        let cube = crate::test_scenes::create_test_cube(1.0);
        let block = |centre: Vec3, size: Vec3| {
            (
                cube.clone(),
                Mat4::from_translation(centre) * Mat4::from_scale(size),
            )
        };
        vec![
            block(Vec3::new(0.0, 0.0, -0.05), Vec3::new(4.0, 4.0, 0.1)),
            block(Vec3::new(-1.2, 0.6, 0.15), Vec3::new(0.8, 1.6, 0.3)),
            block(Vec3::new(-1.3, 0.6, 0.45), Vec3::new(0.4, 1.2, 0.3)),
            (sphere(Vec3::new(0.9, -0.3, 0.4), 0.4), Mat4::IDENTITY),
            (sphere(Vec3::new(0.2, -1.1, 0.25), 0.25), Mat4::IDENTITY),
        ]
        .into_iter()
        .map(|(mesh, transform)| {
            (
                renderer.create_retained_mesh(offscreen.device(), offscreen.queue(), &mesh),
                transform,
            )
        })
        .collect()
    } else {
        Vec::new()
    };
    for (i, (mesh, transform)) in shapes.iter().enumerate() {
        let opacity = if i == 0 { 1.0 } else { opacity };
        renderer.submit_retained_mesh(
            mesh,
            *transform,
            ObjectId(2 + i as u32),
            MaterialId(0),
            opacity,
        );
    }
    for lines in &scene.lines {
        renderer.submit_retained_lines(lines);
    }
    for points in &scene.points {
        renderer.submit_retained_points(points);
    }
    let mut camera = crate::test_scenes::create_test_camera();
    // `VOLUMETRIC_FRAME_VIEW=yaw,pitch,distance[,o]` turns the camera
    // about its focus (degrees), sets its distance, and `o` makes it
    // orthographic.
    if let Ok(spec) = std::env::var("VOLUMETRIC_FRAME_VIEW") {
        let fields: Vec<&str> = spec.split(',').collect();
        let number = |i: usize| fields[i].parse::<f32>().unwrap();
        camera.orbit(
            camera.focus,
            number(0).to_radians(),
            number(1).to_radians(),
            crate::OrbitMode::Turntable,
        );
        camera.distance = number(2);
        if fields.get(3) == Some(&"o") {
            camera.projection = crate::Projection::Orthographic;
        }
    }
    let view = camera.view(w as f32 / h as f32, None);
    let mut settings = RenderSettings {
        antialiasing: std::env::var_os("VOLUMETRIC_FRAME_STEPPED").is_none(),
        ..RenderSettings::default()
    };
    settings.grid.spacing = GridSpacing::Auto {
        focus_depth: camera.distance,
        min_cell_px: GridSpacing::MIN_CELL_PX,
    };
    settings.gizmo = Some(ViewGizmo {
        orientation: camera.orientation,
        center: Vec2::new(w as f32 - 90.0, 90.0),
        radius: 70.0,
        hovered: None,
    });
    let (rgba, info) = offscreen
        .render_rgba(&mut renderer, &view, &settings)
        .unwrap();
    eprintln!("grid spacing: {:?}", info.grid_spacing);
    write_ppm(&path, &rgba, w, h);
}

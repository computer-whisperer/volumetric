//! Render a 2D model through the real sketch-preview path, headlessly.
//!
//! Reproduces GUI-reported 2D display defects without a window: rasterize
//! and build the flat preview exactly like the session does
//! (`build_preview_entity` on a 2D model), submit through the real wgpu
//! pipelines, and save the frame as a PNG, viewed face-on from +z.
//!
//! Usage: headless_sketch_preview <model.wasm> <out.png> [--resolution N]

use std::sync::Arc;

use volumetric_renderer::{Camera, RenderSettings, Renderer};
use volumetric_ui_v2::session::build_preview_scene;
use volumetric_ui_v2::{PreviewMeshPlan, PreviewRenderMode, PreviewRequest};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let res_value_pos = args.iter().position(|a| a == "--resolution").map(|i| i + 1);
    let positional: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|(i, a)| !a.starts_with("--") && Some(*i) != res_value_pos)
        .map(|(_, a)| a)
        .collect();
    if positional.len() != 2 {
        eprintln!("usage: headless_sketch_preview <model.wasm> <out.png> [--resolution N]");
        std::process::exit(1);
    }
    let resolution = res_value_pos
        .and_then(|i| args.get(i))
        .map(|v| v.parse().expect("--resolution takes an integer"))
        .unwrap_or(256);

    let wasm_bytes = std::fs::read(positional[0]).expect("read wasm");
    let request = PreviewRequest {
        asset_id: "sketch_debug".to_string(),
        data: Arc::new(wasm_bytes),
        type_hint: None,
        precursor_ids: vec![],
        render_mode: PreviewRenderMode::AdaptiveSurfaceNets2,
        mesh_plan: PreviewMeshPlan::PointCloud { resolution },
        wireframe: false,
        show_grid: false,
        ssao: false,
        ssao_radius: 0.1,
        ssao_bias: 0.02,
        ssao_strength: 1.0,
        stale: false,
    };
    let entity = build_preview_scene(&request).expect("preview build failed");
    println!(
        "preview: {} tris, stats: {:?}",
        entity.stats.triangles, entity.stats.detail
    );
    for (mesh, _, _) in &entity.scene.meshes {
        let (mut cmin, mut cmax) = ([f32::INFINITY; 4], [f32::NEG_INFINITY; 4]);
        for v in &mesh.vertices {
            for i in 0..4 {
                cmin[i] = cmin[i].min(v.color[i]);
                cmax[i] = cmax[i].max(v.color[i]);
            }
        }
        println!(
            "mesh: {} verts, vertex color min {cmin:?} max {cmax:?}",
            mesh.vertices.len()
        );
    }

    let (bmin, bmax) = (entity.bounds.min, entity.bounds.max);
    let center = glam::Vec3::new(
        (bmin.0 + bmax.0) * 0.5,
        (bmin.1 + bmax.1) * 0.5,
        (bmin.2 + bmax.2) * 0.5,
    );
    let extent = (bmax.0 - bmin.0).max(bmax.1 - bmin.1).max(0.01);

    let (w, h) = (1024u32, 1024u32);
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("no adapter");
        println!("adapter: {:?}", adapter.get_info().name);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("no device");

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut renderer = Renderer::new(format);
        renderer.set_viewport_size(&device, w, h);
        renderer.initialize(&device, &queue, Some(&adapter));

        for (mesh, transform, material) in &entity.scene.meshes {
            renderer.submit_mesh(mesh, *transform, *material);
        }

        // Face-on from +z (camera on the +z axis: phi = 90deg from +Y,
        // theta = 0), pulled back to frame the sketch.
        let mut camera = Camera::new(center, extent * 1.6);
        camera.theta = 0.0;
        camera.phi = std::f32::consts::FRAC_PI_2;

        let mut settings = RenderSettings::default();
        settings.grid.planes = volumetric_renderer::GridPlanes::NONE;
        settings.show_axis_indicator = false;
        settings.ssao_enabled = false;

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("headless_target"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());

        let mut encoder = device.create_command_encoder(&Default::default());
        renderer.render(&device, &queue, &mut encoder, &camera, &settings, &view);

        let bytes_per_row = w * 4;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bytes_per_row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = slice.get_mapped_range();
        let mut img = image::RgbaImage::new(w, h);
        img.copy_from_slice(&data);
        img.save(positional[1]).expect("save png");
        println!("wrote {}", positional[1]);
    });
}

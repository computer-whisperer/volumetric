//! Render a point-cloud file through the real preview pipeline, headlessly:
//! import it exactly as `point_cloud_import_operator` does, build the
//! session's preview entity (the point pipeline, colours from the cloud's
//! `color` field), draw it through the retained path the viewport uses,
//! and save the frame as a PNG.
//!
//! Usage: headless_cloud_preview <cloud.ply> <out.png> [--stride N] [--field node:NAME]
//!
//! `--field` colormaps by a node field instead of showing the cloud's own
//! colours, the way the settings popover's picker does.

use std::sync::Arc;

use volumetric::AssetTypeHint;
use volumetric_renderer::{Camera, RenderSettings, Renderer};
use volumetric_ui_v2::session::build_preview_scene;
use volumetric_ui_v2::{PreviewPlan, PreviewRequest};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value_of = |flag: &str| -> Option<&String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
    };
    let value_positions: Vec<usize> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| *a == "--stride" || *a == "--field")
        .map(|(i, _)| i + 1)
        .collect();
    let positional: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|(i, a)| !a.starts_with("--") && !value_positions.contains(i))
        .map(|(_, a)| a)
        .collect();
    if positional.len() != 2 {
        eprintln!(
            "usage: headless_cloud_preview <cloud.ply> <out.png> [--stride N] [--field node:NAME]"
        );
        std::process::exit(1);
    }
    let stride: u32 = value_of("--stride")
        .map(|v| v.parse().expect("--stride takes an integer"))
        .unwrap_or(1);
    let color_field = value_of("--field").cloned();

    let bytes = std::fs::read(positional[0]).expect("read cloud");
    let start = std::time::Instant::now();
    let config = point_cloud_import_operator::PointCloudImportConfig {
        stride,
        ..Default::default()
    };
    let (mesh, _) = point_cloud_import_operator::import(&bytes, &config).expect("import failed");
    println!(
        "imported {} points ({} fields) in {:.2}s",
        mesh.element_count(),
        mesh.node_fields.len(),
        start.elapsed().as_secs_f64()
    );
    let data = volumetric::fea::encode_fea_mesh(&mesh);

    let start = std::time::Instant::now();
    let request = PreviewRequest {
        asset_id: "cloud".to_string(),
        source_hash: volumetric::content_fingerprint(&data),
        data: Arc::new(data),
        type_hint: Some(AssetTypeHint::FeaMesh),
        precursor_ids: Vec::new(),
        plan: PreviewPlan::FeaMesh {
            deformed: false,
            exaggeration_tenths: 10,
            color_field,
        },
        wireframe: false,
        show_bounds: false,
        show_grid: false,
        ssao: false,
        ssao_radius: 0.5,
        ssao_bias: 0.025,
        ssao_strength: 1.0,
        stale: false,
    };
    let entity = build_preview_scene(&request).expect("preview build failed");
    println!(
        "preview: {} points, bounds {:?}..{:?}, {:.2}s; {:?}",
        entity.stats.points,
        entity.bounds.min,
        entity.bounds.max,
        start.elapsed().as_secs_f64(),
        entity.stats.detail
    );
    let (lo, hi) = (entity.bounds.min, entity.bounds.max);
    let lo = glam::Vec3::new(lo.0, lo.1, lo.2);
    let hi = glam::Vec3::new(hi.0, hi.1, hi.2);
    let center = (lo + hi) * 0.5;
    let radius = (hi - lo).length() * 1.1;

    let (w, h) = (1280u32, 1024u32);
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

        let resident = renderer.create_retained_scene(&device, &entity.scene);
        for points in &resident.points {
            println!(
                "retained batch: dropped {} at the buffer limit",
                points.dropped
            );
            renderer.submit_retained_points(points);
        }

        // From above and in front, the way a scan on a table is looked at.
        let mut camera = Camera::new(center, radius);
        camera.theta = 0.6_f32;
        camera.phi = 0.9_f32;

        let mut settings = RenderSettings::default();
        settings.grid.planes = volumetric_renderer::GridPlanes::NONE;
        settings.show_axis_indicator = false;

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

        let bytes_per_row = w * 4; // 5120, 256-aligned
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
        let data = slice.get_mapped_range().expect("map readback buffer");
        let mut img = image::RgbaImage::new(w, h);
        img.copy_from_slice(&data);
        img.save(positional[1]).expect("save png");
        println!("wrote {}", positional[1]);
    });
}

//! Render a WASM model through the real GPU preview pipeline, headlessly.
//!
//! Reproduces GUI-reported rendering defects without a window: mesh with the
//! production mesher (GUI-default settings), convert exactly like the session
//! preview build, submit through `VolumetricRenderer`'s actual wgpu pipelines
//! (G-buffer, SSAO, composite), and save the frame as a PNG.
//!
//! Usage: headless_preview <model.wasm> <out.png> [--depth N] [--no-sharp]

use volumetric::adaptive_surface_nets_2::AdaptiveMeshConfig2;
use volumetric::sharp_features::SharpFeatureConfig;
use volumetric_renderer::{Camera, MaterialId, MeshData, MeshVertex, RenderSettings, Renderer};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let depth_value_pos = args.iter().position(|a| a == "--depth").map(|i| i + 1);
    let positional: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|(i, a)| !a.starts_with("--") && Some(*i) != depth_value_pos)
        .map(|(_, a)| a)
        .collect();
    if positional.len() != 2 {
        eprintln!("usage: headless_preview <model.wasm> <out.png> [--depth N] [--no-sharp]");
        std::process::exit(1);
    }
    let sharp = !args.iter().any(|a| a == "--no-sharp");
    let max_depth = args
        .iter()
        .position(|a| a == "--depth")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.parse().expect("--depth takes an integer"))
        .unwrap_or(3);

    let wasm_bytes = std::fs::read(positional[0]).expect("read wasm");
    let config = AdaptiveMeshConfig2 {
        base_resolution: 8,
        max_depth,
        vertex_refinement_iterations: 8,
        normal_sample_iterations: 0,
        normal_epsilon_frac: 0.1,
        num_threads: 0,
        sharp_features: sharp.then(SharpFeatureConfig::default),
    };
    let mesh = volumetric::generate_adaptive_mesh_v2_from_bytes(&wasm_bytes, &config)
        .expect("meshing failed");
    println!(
        "mesh: {} verts, {} tris",
        mesh.vertices.len(),
        mesh.indices.len() / 3
    );

    // Exactly the session.rs preview conversion.
    let vertices: Vec<MeshVertex> = mesh
        .vertices
        .iter()
        .zip(mesh.normals.iter())
        .map(|(position, normal)| MeshVertex::new((*position).into(), (*normal).into()))
        .collect();
    let mesh_data = MeshData {
        vertices,
        indices: Some(mesh.indices),
    };

    let center = glam::Vec3::new(
        (mesh.bounds_min.0 + mesh.bounds_max.0) * 0.5,
        (mesh.bounds_min.1 + mesh.bounds_max.1) * 0.5,
        (mesh.bounds_min.2 + mesh.bounds_max.2) * 0.5,
    );

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

        renderer.submit_mesh(&mesh_data, glam::Mat4::IDENTITY, MaterialId(0));

        // Look at the near cap from front-left-above, like the screenshot.
        let mut camera = Camera::new(center, 4.5);
        camera.theta = 0.9_f32;
        camera.phi = 1.2_f32;

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

        // Read back.
        let bytes_per_row = w * 4; // 4096, already 256-aligned
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

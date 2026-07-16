//! Render a 3D model together with a Subspace gizmo through the real
//! preview path, headlessly.
//!
//! Builds the model preview exactly like the session does
//! (`build_preview_scene`), builds a Subspace preview from CBOR bytes the
//! same way, submits both through the real wgpu pipelines (the gizmo via
//! `submit_subspace_gizmo`, sized by the union bounds like the viewport),
//! and saves the frame as a PNG from the default orbit angle.
//!
//! Usage: headless_subspace_preview <model.wasm> <out.png> [--subspace <s.cbor>]
//!
//! Without `--subspace`, the gizmo defaults to the model's bottom-face
//! plane (`Subspace::from_bounds` with span/span/min) — the print-bed
//! plane that `model_bound_operator` produces by default.

use std::sync::Arc;

use volumetric::AssetTypeHint;
use volumetric::subspace::{BoundSelector, Subspace, encode_subspace};
use volumetric_renderer::{Camera, RenderSettings, Renderer};
use volumetric_ui_v2::session::{build_preview_scene, submit_subspace_gizmo};
use volumetric_ui_v2::{PreviewMeshPlan, PreviewPlan, PreviewRequest};

fn request(
    id: &str,
    data: Vec<u8>,
    type_hint: Option<AssetTypeHint>,
    plan: PreviewPlan,
) -> PreviewRequest {
    let source_hash = volumetric::content_fingerprint(&data);
    PreviewRequest {
        asset_id: id.to_string(),
        source_hash,
        data: Arc::new(data),
        type_hint,
        precursor_ids: vec![],
        plan,
        wireframe: false,
        show_bounds: false,
        show_grid: false,
        ssao: false,
        ssao_radius: 0.1,
        ssao_bias: 0.02,
        ssao_strength: 1.0,
        stale: false,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let subspace_path = args
        .iter()
        .position(|a| a == "--subspace")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let positional: Vec<&String> = {
        let value_positions: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == "--subspace")
            .map(|(i, _)| i + 1)
            .collect();
        args.iter()
            .enumerate()
            .filter(|(i, a)| !a.starts_with("--") && !value_positions.contains(i))
            .map(|(_, a)| a)
            .collect()
    };
    if positional.len() != 2 {
        eprintln!("usage: headless_subspace_preview <model.wasm> <out.png> [--subspace <s.cbor>]");
        std::process::exit(1);
    }

    let model_bytes = std::fs::read(positional[0]).expect("read model");
    let model_entity = build_preview_scene(&request(
        "model",
        model_bytes,
        None,
        PreviewPlan::Model3d {
            mesh: PreviewMeshPlan::MarchingCubes { resolution: 64 },
            color_channel: None,
        },
    ))
    .expect("model preview build failed");
    println!(
        "model: {} tris, bounds {:?}..{:?}",
        model_entity.stats.triangles, model_entity.bounds.min, model_entity.bounds.max
    );

    let subspace_bytes = match &subspace_path {
        Some(path) => std::fs::read(path).expect("read subspace cbor"),
        None => {
            let (bmin, bmax) = (model_entity.bounds.min, model_entity.bounds.max);
            let bounds = [
                bmin.0 as f64,
                bmax.0 as f64,
                bmin.1 as f64,
                bmax.1 as f64,
                bmin.2 as f64,
                bmax.2 as f64,
            ];
            use BoundSelector::{Min, Span};
            let plane = Subspace::from_bounds(&bounds, &[Span, Span, Min])
                .expect("bottom-face plane construction failed");
            encode_subspace(&plane)
        }
    };
    let subspace_entity = build_preview_scene(&request(
        "subspace",
        subspace_bytes,
        Some(AssetTypeHint::Subspace),
        PreviewPlan::Subspace,
    ))
    .expect("subspace preview build failed");
    println!("subspace: {:?}", subspace_entity.stats.detail);
    let subspace = subspace_entity
        .subspace
        .clone()
        .expect("subspace entity must carry the decoded value");

    // Union bounds, exactly like the viewport frames the scene.
    let bounds = model_entity.bounds.union(subspace_entity.bounds);
    let center = glam::Vec3::new(
        (bounds.min.0 + bounds.max.0) * 0.5,
        (bounds.min.1 + bounds.max.1) * 0.5,
        (bounds.min.2 + bounds.max.2) * 0.5,
    );
    let extent = (bounds.max.0 - bounds.min.0)
        .max(bounds.max.1 - bounds.min.1)
        .max(bounds.max.2 - bounds.min.2)
        .max(0.01);

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

        for (mesh, transform, material) in &model_entity.scene.meshes {
            renderer.submit_mesh(mesh, *transform, *material);
        }
        submit_subspace_gizmo(&mut renderer, &subspace, bounds);

        // Default orbit angle, pulled back to frame the union bounds.
        let camera = Camera::new(center, extent * 2.2);

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
        let data = slice.get_mapped_range().expect("map readback buffer");
        let mut img = image::RgbaImage::new(w, h);
        img.copy_from_slice(&data);
        img.save(positional[1]).expect("save png");
        println!("wrote {}", positional[1]);
    });
}

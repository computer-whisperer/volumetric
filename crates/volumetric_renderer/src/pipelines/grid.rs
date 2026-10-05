//! The grid pass: the ground grid and world axis lines, drawn per pixel
//! by `grid.wgsl` over the lit scene and depth-tested against it.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::{CameraView, GridSettings, GridSpacing};

/// The smallest a minor cell is drawn, in pixels, where the view is
/// looking; a decade finer would be smaller, so the grid steps up.
const MIN_CELL_PX: f32 = 8.0;

/// The grid's three levels of lines for one frame: minor lines `minor`
/// apart, major lines every ten of them, and super-major lines every
/// hundred, each with the weight its lines are drawn at.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct GridLevels {
    pub minor: f32,
    pub weights: [f32; 3],
}

impl GridLevels {
    /// `pixel_size` is the world size of a pixel where the view is
    /// looking.
    ///
    /// Automatic spacing keeps the minor cell between [`MIN_CELL_PX`] and
    /// ten times that. A level's weight is [`weight`] of how many decades
    /// its cell has grown past the smallest minor cell; when the view
    /// zooms out past a decade each level takes over the weight the next
    /// one had, so nothing pops.
    pub fn choose(spacing: GridSpacing, pixel_size: f32) -> Self {
        match spacing {
            GridSpacing::Fixed(minor) => Self {
                minor: minor.max(f32::MIN_POSITIVE),
                weights: [weight(1.0), 1.0, 1.0],
            },
            GridSpacing::Auto { .. } => {
                let smallest = (pixel_size * MIN_CELL_PX).max(f32::MIN_POSITIVE).log10();
                let decade = smallest.ceil();
                // How far into its decade the minor cell has grown, 0..1.
                let grown = decade - smallest;
                Self {
                    minor: 10f32.powf(decade),
                    weights: [weight(grown), weight(1.0 + grown), 1.0],
                }
            }
        }
    }
}

/// The weight of a level whose cell is `decades` decades larger than the
/// smallest minor cell: nothing at 0, full from 2 on, rising fastest at
/// first so a level is legible soon after it appears.
fn weight(decades: f32) -> f32 {
    let left = 1.0 - (decades * 0.5).clamp(0.0, 1.0);
    1.0 - left * left
}

/// Uniforms of `grid.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GridUniforms {
    inv_view_proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    axis_u: [f32; 4],
    axis_v: [f32; 4],
    axis_n: [f32; 4],
    line_color: [f32; 4],
    color_u: [f32; 4],
    color_v: [f32; 4],
    color_n: [f32; 4],
    levels: [f32; 4],
    viewport: [f32; 4],
}

pub(crate) struct GridPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl GridPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("grid_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/grid.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grid_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("grid_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grid_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_grid"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_grid"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::GBuffer::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_uniforms"),
            size: std::mem::size_of::<GridUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("grid_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        Self {
            pipeline,
            uniform_buffer,
            bind_group,
        }
    }

    /// Writes the frame's uniforms and returns the levels chosen, the
    /// minor spacing among them.
    pub fn prepare(
        &self,
        queue: &wgpu::Queue,
        settings: &GridSettings,
        view: &CameraView,
        size: (u32, u32),
    ) -> GridLevels {
        let pixel_size = match settings.spacing {
            GridSpacing::Auto { focus_depth } => view.pixel_size(focus_depth, size.1),
            GridSpacing::Fixed(_) => 0.0,
        };
        let levels = GridLevels::choose(settings.spacing, pixel_size);

        let [u, v, n] = settings.plane.axes();
        let axis = |i: usize| Vec3::AXES[i].extend(0.0).to_array();
        let axis_color = |i: usize| {
            let [r, g, b] = settings.axis_colors[i];
            [r, g, b, settings.axes as u32 as f32]
        };
        let [r, g, b] = settings.line_color;
        // The inverse is taken in double precision: with the far plane
        // tens of thousands of near planes away, a single-precision
        // inverse puts visible wobble into the lines.
        let view_proj = view.view_projection();
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&GridUniforms {
                inv_view_proj: view_proj.as_dmat4().inverse().as_mat4().to_cols_array_2d(),
                view_proj: view_proj.to_cols_array_2d(),
                axis_u: axis(u),
                axis_v: axis(v),
                axis_n: axis(n),
                line_color: [r, g, b, settings.opacity.clamp(0.0, 1.0)],
                color_u: axis_color(u),
                color_v: axis_color(v),
                color_n: axis_color(n),
                levels: [
                    levels.weights[0],
                    levels.weights[1],
                    levels.weights[2],
                    levels.minor,
                ],
                viewport: [size.0 as f32, size.1 as f32, 0.0, 0.0],
            }),
        );
        levels
    }

    /// Draws the plane, then the normal axis when axes are drawn.
    pub fn render(&self, pass: &mut wgpu::RenderPass<'_>, axes: bool) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..if axes { 2 } else { 1 });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Automatic spacing is a power of ten whose cells are between the
    /// minimum size and ten times it, and zooming across a decade
    /// boundary changes no line's weight.
    #[test]
    fn automatic_levels_are_decades_and_continuous() {
        let auto = GridSpacing::Auto { focus_depth: 1.0 };
        for i in 0..400 {
            let pixel_size = 10f32.powf(-6.0 + i as f32 * 0.02);
            let levels = GridLevels::choose(auto, pixel_size);
            let cell_px = levels.minor / pixel_size;
            assert!(
                (MIN_CELL_PX * 0.999..MIN_CELL_PX * 10.001).contains(&cell_px),
                "{cell_px} px at {pixel_size}"
            );
            let decade = levels.minor.log10();
            assert!((decade - decade.round()).abs() < 1e-4, "{}", levels.minor);
        }

        // Just either side of the step from 1 mm to 1 cm minor cells.
        let step = 0.001 / MIN_CELL_PX;
        let finer = GridLevels::choose(auto, step * 0.999);
        let coarser = GridLevels::choose(auto, step * 1.001);
        assert!((finer.minor - 0.001).abs() < 1e-7);
        assert!((coarser.minor - 0.01).abs() < 1e-6);
        // The vanished level had no weight; each line keeps its own.
        assert!(finer.weights[0] < 0.01);
        assert!((finer.weights[1] - coarser.weights[0]).abs() < 0.01);
        assert!((finer.weights[2] - coarser.weights[1]).abs() < 0.01);
    }

    #[test]
    fn fixed_spacing_is_taken_as_given() {
        let levels = GridLevels::choose(GridSpacing::Fixed(0.25), 123.0);
        assert_eq!(levels.minor, 0.25);
    }
}

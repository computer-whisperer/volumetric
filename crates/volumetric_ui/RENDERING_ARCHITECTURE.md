# Volumetric UI Rendering Engine Architecture

## Overview

This document outlines the architecture for a revised rendering engine supporting high-quality simultaneous mesh, line, and point rendering on both native (Vulkan/Metal/DX12) and WebGPU backends.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Line rendering | Vertex shader quad expansion | WebGPU lacks geometry shaders; matches point cloud pattern |
| Web backend | WebGPU preferred, WebGL fallback | Broadest compatibility |
| Transparency | Alpha blending supported | Lines/points benefit from smooth edges |
| Depth handling | Correct by default | Optional "overlay" mode for annotation primitives |
| Resolution | Full display | No downscaling |

### Note on Geometry Shaders

WebGPU/wgpu intentionally excludes geometry shaders - they are considered legacy with performance issues. The alternative approaches are:
1. **Mesh shaders** (wgpu v28+) - Not yet broadly supported in browser WebGPU
2. **Vertex shader expansion** (chosen) - Portable, matches existing point cloud technique
3. **Compute shader preprocessing** - More complex, overkill for line rendering

We use vertex shader expansion: store line segments in an instance buffer, expand each to a screen-aligned quad using `vertex_index` to select quad corners.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Renderer                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Scene Data                          GPU Resources                  │
│  ┌─────────────────────┐            ┌─────────────────────┐        │
│  │ Camera (target,     │            │ SharedDepthBuffer   │        │
│  │   orbit, pan, zoom) │            │ GBufferTextures     │        │
│  │ MeshBatch           │            │ AoTexture           │        │
│  │ LineBatch           │            │ Pipelines           │        │
│  │ PointBatch          │            └─────────────────────┘        │
│  │ GridSettings        │                                           │
│  │ AxisIndicator       │                                           │
│  └─────────────────────┘                                           │
│                                                                     │
│  Render Passes (ordered)                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 1. MeshGBufferPass    → writes: color, normal, depth        │   │
│  │ 2. SsaoPass           → reads: normal, depth → writes: ao   │   │
│  │ 3. CompositePass      → mesh color * ao → final target      │   │
│  │ 4. GridPass           → depth-tested grid lines             │   │
│  │ 5. LinePass           → depth-tested scene lines            │   │
│  │ 6. PointPass          → depth-tested scene points           │   │
│  │ 7. OverlayLinePass    → overlay lines (no depth test)       │   │
│  │ 8. OverlayPointPass   → overlay points (no depth test)      │   │
│  │ 9. AxisIndicatorPass  → mini-viewport in corner             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Camera Input Flow:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ egui Input   │────▶│ Input Handler│────▶│   Camera     │
│ (drag/scroll)│     │ (placeholder)│     │ orbit/pan/   │
└──────────────┘     └──────────────┘     │ zoom/focus   │
                                          └──────────────┘
```

## Core Types

### Render Primitives

```rust
/// A batch of triangle mesh data
pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Option<Vec<u32>>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// A batch of line segments
pub struct LineData {
    pub segments: Vec<LineSegment>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineSegment {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub color: [f32; 4],  // RGBA with alpha
}

/// A batch of points
pub struct PointData {
    pub points: Vec<PointInstance>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointInstance {
    pub position: [f32; 3],
    pub color: [f32; 4],  // RGBA with alpha
}
```

### Render Styles

```rust
/// Line rendering style
pub struct LineStyle {
    pub width: f32,
    pub width_mode: WidthMode,
    pub pattern: LinePattern,
    pub depth_mode: DepthMode,
}

pub enum WidthMode {
    /// Width in screen pixels (constant regardless of distance)
    ScreenSpace,
    /// Width in world units (appears smaller at distance)
    WorldSpace,
}

pub enum LinePattern {
    Solid,
    Dashed { dash_length: f32, gap_length: f32 },
    Dotted { spacing: f32 },
}

pub enum DepthMode {
    /// Normal depth testing against scene
    Normal,
    /// Render on top of everything (for annotations/overlays)
    Overlay,
}

/// Point rendering style
pub struct PointStyle {
    pub size: f32,
    pub size_mode: WidthMode,
    pub shape: PointShape,
    pub depth_mode: DepthMode,
}

pub enum PointShape {
    Circle,
    Square,
    Diamond,
}
```

### Renderer Interface

```rust
/// Main renderer that manages all GPU resources and rendering
pub struct Renderer {
    // GPU resources
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Shared resources
    depth_buffer: DepthBuffer,
    gbuffer: GBuffer,
    ao_texture: Option<wgpu::Texture>,

    // Pipelines
    mesh_pipeline: MeshPipeline,
    line_pipeline: LinePipeline,
    point_pipeline: PointPipeline,
    ssao_pipeline: SsaoPipeline,
    composite_pipeline: CompositePipeline,

    // Frame state
    frame_meshes: Vec<(MeshData, Mat4, MaterialId)>,
    frame_lines: Vec<(LineData, Mat4, LineStyle)>,
    frame_points: Vec<(PointData, Mat4, PointStyle)>,
}

impl Renderer {
    /// Submit mesh geometry for this frame
    pub fn submit_mesh(&mut self, mesh: &MeshData, transform: Mat4, material: MaterialId);

    /// Submit line segments for this frame
    pub fn submit_lines(&mut self, lines: &LineData, transform: Mat4, style: LineStyle);

    /// Submit points for this frame
    pub fn submit_points(&mut self, points: &PointData, transform: Mat4, style: PointStyle);

    /// Execute all rendering for the frame
    pub fn render(&mut self, camera: &Camera, settings: &RenderSettings, target: &wgpu::TextureView);

    /// Clear frame state for next frame
    pub fn end_frame(&mut self);
}
```

## Pipeline Details

### 1. Mesh G-Buffer Pipeline

**Purpose:** Render opaque mesh geometry to deferred buffers.

**Inputs:**
- Vertex buffer: `MeshVertex[]`
- Index buffer: `u32[]` (optional)
- Uniforms: view_proj, light_dir, base_color

**Outputs (G-Buffer):**
- Color attachment (RGBA8): Lit diffuse color
- Normal attachment (RGBA8): World-space normal encoded to [0,1]
- Depth attachment (R32Float): Linear depth for SSAO

**Depth:** Write enabled, test LessEqual

### 2. SSAO Pipeline

**Purpose:** Compute screen-space ambient occlusion.

**Inputs:**
- G-Buffer normal texture
- G-Buffer depth texture
- Uniforms: inv_view_proj, radius, bias, strength

**Outputs:**
- AO texture (R8): Occlusion factor [0,1]

**Configuration:**
- Native: 16 samples (high quality)
- WebGPU: 8 samples (reduced for performance)

### 3. Line Pipeline (NEW)

**Purpose:** GPU-accelerated wide line rendering with proper depth.

**Technique:** Vertex shader quad expansion (instanced rendering)

Each line segment is stored as an instance. The vertex shader receives a quad vertex (4 vertices forming 2 triangles) plus the line segment data per-instance, and expands the line to a screen-aligned quad.

**Instance Data (per line segment):**
```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineInstance {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub color: [f32; 4],
    pub width: f32,
    pub _pad: [f32; 3],
}
```

**Vertex Data (quad template, 4 vertices):**
```rust
// corner.x: 0 = start, 1 = end (position along line)
// corner.y: -1 = left edge, +1 = right edge (perpendicular offset)
const QUAD_VERTICES: [[f32; 2]; 4] = [
    [0.0, -1.0],  // start, left
    [0.0,  1.0],  // start, right
    [1.0, -1.0],  // end, left
    [1.0,  1.0],  // end, right
];
const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 1, 3];
```

**Vertex Shader Expansion Logic:**
1. Select endpoint based on `corner.x` (0=start, 1=end)
2. Project both endpoints to clip space
3. Compute screen-space line direction and perpendicular
4. Offset position by `corner.y * half_width` along perpendicular
5. Output clip position with proper depth

**Fragment Shader:**
- SDF-based anti-aliasing at edges using `edge_coord` interpolant
- Alpha output for smooth blending
- Optional dash pattern via `line_coord` (distance along line)

**Depth:**
- Normal mode: Read/write depth buffer
- Overlay mode: Depth test disabled

**Blending:** Premultiplied alpha (SrcAlpha, OneMinusSrcAlpha)

### 4. Point Pipeline (Enhanced)

**Purpose:** Render points as screen-aligned quads.

**Current technique preserved:** Instanced quad rendering with per-instance position/color.

**Enhancements:**
- Add alpha channel support to color
- Add shape uniform (circle/square/diamond selection)
- Add overlay depth mode

**Fragment Shader (enhanced):**
```wgsl
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv * 2.0 - 1.0;  // Map to [-1, 1]

    var alpha: f32;
    switch uniforms.shape {
        case 0u: {  // Circle
            let dist = length(uv);
            alpha = 1.0 - smoothstep(0.9, 1.0, dist);
        }
        case 1u: {  // Square
            alpha = 1.0;
        }
        case 2u: {  // Diamond
            let dist = abs(uv.x) + abs(uv.y);
            alpha = 1.0 - smoothstep(0.9, 1.0, dist);
        }
        default: {
            alpha = 1.0;
        }
    }

    if alpha < 0.01 { discard; }
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
```

### 5. Composite Pipeline

**Purpose:** Combine G-buffer color with AO and overlay forward-rendered content.

**Process:**
1. Sample G-buffer color
2. Sample AO texture (or use 1.0 if SSAO disabled)
3. Multiply: `final_color = gbuffer_color * ao`
4. Forward-rendered lines/points are already in the target, so composite writes behind them

**Note:** Lines and points render directly to the final target with blending, so composite only needs to handle the mesh G-buffer.

## Render Pass Execution Order

```
Frame Start
│
├─► Clear depth buffer
├─► Clear G-buffer attachments
│
├─► [MESH G-BUFFER PASS]
│   └─► Render all meshes to G-buffer
│       - Writes: color, normal, depth (G-buffer)
│       - Writes: shared depth buffer
│
├─► [SSAO PASS] (if enabled)
│   └─► Fullscreen compute AO from G-buffer
│       - Reads: G-normal, G-depth
│       - Writes: AO texture
│
├─► [COMPOSITE PASS]
│   └─► Combine G-buffer color with AO
│       - Reads: G-color, AO
│       - Writes: final target (opaque)
│
├─► [LINE PASS - DEPTH TESTED]
│   └─► Render lines with DepthMode::Normal
│       - Reads: shared depth buffer
│       - Writes: shared depth buffer, final target (blended)
│
├─► [POINT PASS - DEPTH TESTED]
│   └─► Render points with DepthMode::Normal
│       - Reads: shared depth buffer
│       - Writes: shared depth buffer, final target (blended)
│
├─► [LINE PASS - OVERLAY]
│   └─► Render lines with DepthMode::Overlay
│       - No depth testing
│       - Writes: final target (blended)
│
├─► [POINT PASS - OVERLAY]
│   └─► Render points with DepthMode::Overlay
│       - No depth testing
│       - Writes: final target (blended)
│
Frame End
```

## GPU Resource Management

### Buffer Strategy

```rust
struct DynamicBuffer<T> {
    buffer: wgpu::Buffer,
    capacity: usize,
    len: usize,
}

impl<T: bytemuck::Pod> DynamicBuffer<T> {
    /// Ensure buffer can hold at least `required` elements.
    /// Reallocates with 2x growth if needed.
    fn ensure_capacity(&mut self, device: &wgpu::Device, required: usize);

    /// Upload data to GPU. Reallocates if needed.
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[T]);
}
```

### Texture Management

G-buffer and depth textures are recreated on viewport resize:

```rust
struct GBuffer {
    color: wgpu::Texture,
    normal: wgpu::Texture,
    depth: wgpu::Texture,
    size: (u32, u32),
}

impl GBuffer {
    fn resize_if_needed(&mut self, device: &wgpu::Device, new_size: (u32, u32));
}
```

## Shader Files

### New/Modified Shaders

| File | Purpose |
|------|---------|
| `shaders/line.wgsl` | **NEW** - Line geometry expansion + fragment AA |
| `shaders/point_cloud.wgsl` | **MODIFIED** - Add alpha, shape selection |
| `shaders/marching_cubes.wgsl` | Unchanged |
| `shaders/mesh_ssao.wgsl` | Minor: configurable sample count |
| `shaders/composite.wgsl` | **NEW** - Extracted from mesh_ssao.wgsl |

### Line Shader Design

```wgsl
// line.wgsl

struct Uniforms {
    view_proj: mat4x4<f32>,
    screen_size: vec2<f32>,
    width_mode: u32,       // 0 = screen pixels, 1 = world units
    default_width: f32,    // Used if instance width is 0
    dash_length: f32,
    gap_length: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// Per-vertex quad corner
struct QuadVertex {
    @location(0) corner: vec2<f32>,  // x: 0=start, 1=end; y: -1=left, +1=right
}

// Per-instance line segment
struct LineInstance {
    @location(1) start: vec3<f32>,
    @location(2) end: vec3<f32>,
    @location(3) color: vec4<f32>,
    @location(4) width: f32,
}

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) edge_coord: f32,     // -1 to +1 across line width
    @location(2) line_coord: f32,     // Distance along line (world units)
}

@vertex
fn vs_main(quad: QuadVertex, line: LineInstance) -> VsOut {
    var out: VsOut;

    // Select position along line (0 = start, 1 = end)
    let t = quad.corner.x;
    let world_pos = mix(line.start, line.end, t);

    // Project both endpoints to clip space
    let clip_start = uniforms.view_proj * vec4(line.start, 1.0);
    let clip_end = uniforms.view_proj * vec4(line.end, 1.0);

    // Convert to NDC
    let ndc_start = clip_start.xy / clip_start.w;
    let ndc_end = clip_end.xy / clip_end.w;

    // Screen-space direction and perpendicular
    let screen_start = (ndc_start * 0.5 + 0.5) * uniforms.screen_size;
    let screen_end = (ndc_end * 0.5 + 0.5) * uniforms.screen_size;
    let screen_dir = screen_end - screen_start;
    let screen_len = length(screen_dir);

    // Perpendicular (rotate 90 degrees)
    let perp = vec2(-screen_dir.y, screen_dir.x) / max(screen_len, 0.001);

    // Determine width
    let width = select(uniforms.default_width, line.width, line.width > 0.0);
    var half_width_px: f32;
    if uniforms.width_mode == 0u {
        // Screen-space: width is in pixels
        half_width_px = width * 0.5;
    } else {
        // World-space: approximate pixel width from clip.w
        let clip_w = mix(clip_start.w, clip_end.w, t);
        half_width_px = (width / clip_w) * uniforms.screen_size.y * 0.5;
    }

    // Offset in screen space
    let offset_px = perp * half_width_px * quad.corner.y;

    // Convert offset back to NDC
    let offset_ndc = offset_px * 2.0 / uniforms.screen_size;

    // Current clip position
    let clip_pos = mix(clip_start, clip_end, t);
    out.position = vec4(clip_pos.xy + offset_ndc * clip_pos.w, clip_pos.zw);

    out.color = line.color;
    out.edge_coord = quad.corner.y;  // -1 to +1
    out.line_coord = t * length(line.end - line.start);

    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Anti-aliased edges using smoothstep
    let edge_dist = abs(in.edge_coord);
    let edge_aa = 1.0 - smoothstep(0.85, 1.0, edge_dist);

    // Dash pattern (if enabled)
    var pattern_alpha = 1.0;
    if uniforms.dash_length > 0.0 {
        let cycle = uniforms.dash_length + uniforms.gap_length;
        let pos_in_cycle = in.line_coord % cycle;
        // Soft edges on dash transitions
        let dash_edge = smoothstep(0.0, 0.1, pos_in_cycle)
                      * (1.0 - smoothstep(uniforms.dash_length - 0.1, uniforms.dash_length, pos_in_cycle));
        pattern_alpha = dash_edge;
    }

    let final_alpha = in.color.a * edge_aa * pattern_alpha;
    if final_alpha < 0.01 { discard; }

    // Premultiplied alpha output
    return vec4(in.color.rgb * final_alpha, final_alpha);
}
```

## Integration with egui

The renderer integrates with egui via custom paint callbacks:

```rust
pub struct SceneCallback {
    renderer: Arc<Mutex<Renderer>>,
    scene: RenderScene,
    camera: Camera,
    settings: RenderSettings,
}

impl egui_wgpu::CallbackTrait for SceneCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut renderer = self.renderer.lock().unwrap();

        // Submit all scene data
        for (mesh, transform, material) in &self.scene.meshes {
            renderer.submit_mesh(mesh, *transform, *material);
        }
        for (lines, transform, style) in &self.scene.lines {
            renderer.submit_lines(lines, *transform, style.clone());
        }
        for (points, transform, style) in &self.scene.points {
            renderer.submit_points(points, *transform, style.clone());
        }

        // Render to offscreen target, return command buffers
        renderer.render_offscreen(&self.camera, &self.settings)
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let renderer = self.renderer.lock().unwrap();
        renderer.blit_to_screen(render_pass);
    }
}
```

## Platform Considerations

### Native vs WebGPU Differences

| Feature | Native | WebGPU |
|---------|--------|--------|
| SSAO samples | 16 | 8 |
| Geometry shaders | Full support | Full support |
| Texture formats | All | All standard |
| Max texture size | 16384+ | 8192 typical |

### Feature Detection

```rust
pub struct RendererCapabilities {
    pub max_texture_size: u32,
    pub supports_geometry_shaders: bool,
    pub recommended_ssao_samples: u32,
}

impl RendererCapabilities {
    pub fn detect(adapter: &wgpu::Adapter) -> Self {
        let limits = adapter.limits();
        let is_web = cfg!(target_arch = "wasm32");

        Self {
            max_texture_size: limits.max_texture_dimension_2d,
            supports_geometry_shaders: true,  // WebGPU supports this
            recommended_ssao_samples: if is_web { 8 } else { 16 },
        }
    }
}
```

## Migration Plan

### Phase 1: Renderer Scaffold
- [ ] Create `renderer/mod.rs` module structure
- [ ] Define core types (MeshData, LineData, PointData, styles)
- [ ] Implement DynamicBuffer utility
- [ ] Create Renderer struct with placeholder methods
- [ ] Implement Camera struct with orbit/pan/zoom

### Phase 2: Mesh Pipeline Migration
- [ ] Move existing mesh G-buffer code into MeshPipeline
- [ ] Move SSAO code into SsaoPipeline
- [ ] Extract composite shader to separate file
- [ ] Verify mesh rendering still works

### Phase 3: Line Pipeline Implementation
- [ ] Implement line.wgsl shader with vertex expansion
- [ ] Create LinePipeline struct
- [ ] Add instanced quad rendering for lines
- [ ] Add depth-tested line rendering
- [ ] Add overlay line rendering (DepthMode::Overlay)
- [ ] Implement anti-aliased edges

### Phase 4: Point Pipeline Enhancement
- [ ] Add alpha support to point shader
- [ ] Add shape selection (circle/square/diamond)
- [ ] Add overlay depth mode
- [ ] Verify point rendering

### Phase 5: Scene Environment
- [ ] Implement GridSettings and grid line generation
- [ ] Add grid rendering to frame (XZ ground plane default)
- [ ] Implement axis indicator with mini-viewport
- [ ] Add axis labels via egui text overlay

### Phase 6: Camera System
- [ ] Implement enhanced Camera with target point
- [ ] Add orbit(), pan(), zoom(), focus_on() methods
- [ ] Implement placeholder input handling (left=orbit, middle=pan, scroll=zoom)
- [ ] Hook up to egui input events

### Phase 7: Unified Rendering
- [ ] Implement frame submission API
- [ ] Implement render pass ordering
- [ ] Ensure proper depth sharing between all passes
- [ ] Test simultaneous mesh + line + point + grid rendering

### Phase 8: egui Integration
- [ ] Create unified SceneCallback
- [ ] Replace existing PointCloudCallback usage
- [ ] Replace existing MarchingCubesCallback usage
- [ ] Integrate axis indicator rendering
- [ ] Verify UI integration

### Phase 9: Polish
- [ ] Add line dash patterns
- [ ] Add line width modes (screen vs world)
- [ ] Grid distance fade/falloff
- [ ] Performance optimization
- [ ] WebGPU testing and validation

## Scene Environment

### Grid Planes

The scene should display reference grid planes at x=0, y=0, and z=0 to provide spatial context (replacing the empty void backdrop).

**Grid Configuration:**
```rust
pub struct GridSettings {
    /// Which planes to display
    pub planes: GridPlanes,
    /// Spacing between grid lines (world units)
    pub spacing: f32,
    /// Extent of grid from origin (world units)
    pub extent: f32,
    /// Primary line color (every N lines)
    pub major_color: [f32; 4],
    /// Secondary line color
    pub minor_color: [f32; 4],
    /// Lines between major lines
    pub subdivisions: u32,
}

bitflags! {
    pub struct GridPlanes: u8 {
        const XY = 0b001;  // z = 0 plane
        const XZ = 0b010;  // y = 0 plane (ground plane)
        const YZ = 0b100;  // x = 0 plane
    }
}
```

**Default Settings:**
- `planes`: XZ only (ground plane at y=0)
- `spacing`: 1.0 world units
- `extent`: 10.0 world units (grid spans -10 to +10)
- `major_color`: `[0.5, 0.5, 0.5, 0.8]` (gray, every 5 lines)
- `minor_color`: `[0.3, 0.3, 0.3, 0.4]` (darker gray)
- `subdivisions`: 5

**Implementation:**
- Grid lines are generated each frame (or cached if camera hasn't moved significantly)
- Uses the line rendering pipeline with `DepthMode::Normal`
- Major axis lines (x=0, z=0 on ground plane) can be colored (red for X, blue for Z)
- Grid fades with distance or has a circular falloff to avoid harsh edges

**Grid Line Generation:**
```rust
fn generate_grid_lines(settings: &GridSettings) -> Vec<LineSegment> {
    let mut lines = Vec::new();
    let n = (settings.extent / settings.spacing) as i32;

    if settings.planes.contains(GridPlanes::XZ) {
        // Ground plane (y = 0)
        for i in -n..=n {
            let pos = i as f32 * settings.spacing;
            let is_major = i % settings.subdivisions as i32 == 0;
            let is_axis = i == 0;

            let color = if is_axis {
                if i == 0 { /* X axis: red, Z axis: blue */ }
            } else if is_major {
                settings.major_color
            } else {
                settings.minor_color
            };

            // Line parallel to X axis
            lines.push(LineSegment {
                start: [-settings.extent, 0.0, pos],
                end: [settings.extent, 0.0, pos],
                color,
            });
            // Line parallel to Z axis
            lines.push(LineSegment {
                start: [pos, 0.0, -settings.extent],
                end: [pos, 0.0, settings.extent],
                color,
            });
        }
    }
    // Similar for XY and YZ planes...
    lines
}
```

### Axis Indicator (View Gizmo)

A small 3D axis indicator in the bottom-right corner shows camera orientation.

**Behavior:**
- Fixed screen position (bottom-right corner with padding)
- Fixed screen size (e.g., 60x60 pixels)
- Rotates with camera (shows which way X/Y/Z point in view space)
- Always rendered on top (overlay mode)
- Semi-transparent background circle/sphere optional

**Implementation Approach:**
Render as a mini-viewport with its own projection:

```rust
pub struct AxisIndicator {
    /// Screen position (normalized, 0-1, from bottom-left)
    pub position: [f32; 2],  // e.g., [0.92, 0.08]
    /// Size in pixels
    pub size: f32,           // e.g., 60.0
    /// Axis colors
    pub x_color: [f32; 4],   // Red
    pub y_color: [f32; 4],   // Green
    pub z_color: [f32; 4],   // Blue
    /// Show axis labels (X, Y, Z)
    pub show_labels: bool,
}
```

**Rendering:**
1. Compute axis directions in view space from camera rotation
2. Project to 2D using orthographic projection
3. Render as overlay lines with `DepthMode::Overlay`
4. Optionally render axis labels using egui text

```rust
fn render_axis_indicator(camera: &Camera, indicator: &AxisIndicator) -> Vec<LineSegment> {
    let view_rotation = camera.view_matrix().to_scale_rotation_translation().1;
    let inv_rotation = view_rotation.inverse();

    // Transform world axes to view space
    let x_dir = inv_rotation * Vec3::X;
    let y_dir = inv_rotation * Vec3::Y;
    let z_dir = inv_rotation * Vec3::Z;

    // Create lines from origin along each axis
    let origin = Vec3::ZERO;
    let len = 1.0;  // Normalized, will be scaled by viewport

    vec![
        LineSegment {
            start: origin.into(),
            end: (origin + x_dir * len).into(),
            color: indicator.x_color,
        },
        LineSegment {
            start: origin.into(),
            end: (origin + y_dir * len).into(),
            color: indicator.y_color,
        },
        LineSegment {
            start: origin.into(),
            end: (origin + z_dir * len).into(),
            color: indicator.z_color,
        },
    ]
}
```

**Viewport Setup:**
The axis indicator uses a separate orthographic projection and viewport:
```rust
// Viewport in bottom-right corner
let viewport = wgpu::Viewport {
    x: screen_width - indicator.size - padding,
    y: screen_height - indicator.size - padding,
    width: indicator.size,
    height: indicator.size,
    min_depth: 0.0,
    max_depth: 1.0,
};

// Orthographic projection for the mini-view
let ortho = Mat4::orthographic_rh(-1.5, 1.5, -1.5, 1.5, -10.0, 10.0);
```

## Camera System

### Enhanced Camera Model

The camera supports both **orbit** (rotation around target) and **pan** (translation of target).

```rust
pub struct Camera {
    /// Point the camera orbits around / looks at
    pub target: Vec3,

    /// Spherical coordinates relative to target
    pub radius: f32,      // Distance from target
    pub theta: f32,       // Azimuth angle (rotation around Y axis)
    pub phi: f32,         // Elevation angle (from Y axis)

    /// Projection parameters
    pub fov_y: f32,       // Vertical field of view (radians)
    pub near: f32,        // Near clip plane
    pub far: f32,         // Far clip plane
}

impl Camera {
    /// Compute eye position from spherical coordinates
    pub fn eye_position(&self) -> Vec3 {
        let x = self.radius * self.phi.sin() * self.theta.sin();
        let y = self.radius * self.phi.cos();
        let z = self.radius * self.phi.sin() * self.theta.cos();
        self.target + Vec3::new(x, y, z)
    }

    /// Compute view matrix
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye_position(), self.target, Vec3::Y)
    }

    /// Compute projection matrix
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
    }

    /// Orbit: rotate around target
    pub fn orbit(&mut self, delta_theta: f32, delta_phi: f32) {
        self.theta += delta_theta;
        self.phi = (self.phi + delta_phi).clamp(0.01, std::f32::consts::PI - 0.01);
    }

    /// Pan: translate target in view plane
    pub fn pan(&mut self, delta_screen: Vec2, viewport_size: Vec2) {
        // Convert screen delta to world delta in the view plane
        let view = self.view_matrix();
        let right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x);
        let up = Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y);

        // Scale by distance for consistent feel
        let scale = self.radius * 0.002;
        self.target += right * (-delta_screen.x * scale);
        self.target += up * (delta_screen.y * scale);
    }

    /// Zoom: adjust distance from target
    pub fn zoom(&mut self, delta: f32) {
        self.radius = (self.radius * (1.0 - delta * 0.1)).clamp(0.1, 1000.0);
    }

    /// Focus on a bounding box
    pub fn focus_on(&mut self, bounds: &Bounds3) {
        self.target = bounds.center();
        self.radius = bounds.diagonal_length() * 1.5;
    }
}
```

### Camera Input (Deferred)

Specific input bindings (Blender-style, CAD-style, etc.) are deferred to a later project. The camera exposes these operations that input handlers will map to:

| Operation | Description |
|-----------|-------------|
| `orbit(delta_theta, delta_phi)` | Rotate view around target |
| `pan(delta_screen, viewport_size)` | Translate target in view plane |
| `zoom(delta)` | Move closer/further from target |
| `focus_on(bounds)` | Frame object in view |

**Placeholder Input (Current):**
For now, a simple default mapping:
- **Left drag**: Orbit
- **Middle drag** or **Shift+Left drag**: Pan
- **Scroll**: Zoom
- **F key** or **Home**: Focus on scene bounds

## Open Questions

1. **Transparency sorting:** Current design assumes lines/points are mostly opaque or don't overlap significantly. Full OIT (order-independent transparency) would add complexity - is it needed?

2. **Wireframe overlay:** Should meshes support a wireframe overlay mode? Could use barycentric coordinates in fragment shader (requires mesh preprocessing to add barycentric attributes).

3. **Line caps and joins:** Current design renders line segments independently without proper caps/joins. For polylines, should we add:
   - Round/square/butt caps at endpoints?
   - Miter/round/bevel joins between connected segments?

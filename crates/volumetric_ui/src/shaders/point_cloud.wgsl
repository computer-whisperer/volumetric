struct Uniforms {
    view_proj: mat4x4<f32>,
    point_size_px: f32,
    _pad0: f32,
    screen_size_px: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VsIn {
    // Quad vertex (per-vertex)
    @location(0) corner: vec2<f32>, // -1..+1
    @location(1) uv: vec2<f32>,     //  0..+1

    // Per-instance point data
    @location(2) position: vec3<f32>,
    @location(3) color: vec3<f32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let clip = uniforms.view_proj * vec4<f32>(in.position, 1.0);

    // Convert a pixel offset into clip space so point size stays constant in screen-space.
    // in.corner is -1..+1, so it represents half-extents.
    let half_size_px = uniforms.point_size_px * 0.5;
    let offset_ndc = in.corner * (half_size_px * 2.0 / uniforms.screen_size_px);
    out.position = vec4<f32>(clip.xy + offset_ndc * clip.w, clip.zw);

    out.color = in.color;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Make points round by discarding pixels outside radius.
    let pc = in.uv * 2.0 - vec2<f32>(1.0, 1.0);
    if dot(pc, pc) > 1.0 {
        discard;
    }

    return vec4<f32>(in.color, 1.0);
}

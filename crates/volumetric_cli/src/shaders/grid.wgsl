// Simple grid line shader with per-vertex color and depth

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    light_dir: vec3<f32>,
    _pad1: f32,
    base_color: vec3<f32>,
    rim_strength: f32,
    sky_color: vec3<f32>,
    fog_density: f32,
    ground_color: vec3<f32>,
    fog_start: f32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_position: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    out.world_position = in.position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply depth fog to grid lines for depth cue
    let dist = length(uniforms.camera_pos - in.world_position);
    let fog_factor = 1.0 - exp(-uniforms.fog_density * max(dist - uniforms.fog_start, 0.0));
    let fog_color = mix(uniforms.sky_color, uniforms.ground_color, 0.5);
    let color = mix(in.color, fog_color, fog_factor * 0.5);

    return vec4<f32>(color, 1.0);
}

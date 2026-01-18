// Headless mesh rendering shader for information-dense visualization
// Features: hemisphere ambient, directional diffuse with wrap lighting,
// rim lighting for silhouette enhancement, and depth fog

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
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.world_position = in.position;
    out.world_normal = in.normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let view_dir = normalize(uniforms.camera_pos - in.world_position);
    let light_dir = normalize(uniforms.light_dir);

    // Hemisphere ambient: blend between ground and sky based on normal.y
    let hemisphere_factor = normal.y * 0.5 + 0.5;
    let ambient = mix(uniforms.ground_color, uniforms.sky_color, hemisphere_factor);

    // Directional diffuse with wrap lighting (softer shadows)
    let wrap = 0.3;
    let ndotl = dot(normal, light_dir);
    let diffuse_factor = max((ndotl + wrap) / (1.0 + wrap), 0.0);
    let diffuse = uniforms.base_color * diffuse_factor;

    // Rim lighting: Fresnel-based edge highlight
    let ndotv = max(dot(normal, view_dir), 0.0);
    let fresnel = pow(1.0 - ndotv, 3.0);
    let rim = uniforms.rim_strength * fresnel * uniforms.sky_color;

    // Combine lighting
    var color = ambient * 0.3 + diffuse * 0.7 + rim;

    // Depth fog: subtle distance fade for depth cue
    let dist = length(uniforms.camera_pos - in.world_position);
    let fog_factor = 1.0 - exp(-uniforms.fog_density * max(dist - uniforms.fog_start, 0.0));
    let fog_color = mix(uniforms.sky_color, uniforms.ground_color, 0.5);
    color = mix(color, fog_color, fog_factor * 0.3);

    return vec4<f32>(color, 1.0);
}

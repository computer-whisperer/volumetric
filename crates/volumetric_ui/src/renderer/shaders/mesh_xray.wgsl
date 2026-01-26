// XRay mesh rendering shader
//
// Renders semi-transparent meshes for X-Ray mode visualization.
// - Depth test enabled (reads from depth buffer)
// - Depth write disabled (doesn't occlude other objects)
// - Alpha blending for transparency
// - Shows both front and back faces

struct Uniforms {
    view_proj: mat4x4<f32>,
    light_dir_world: vec3<f32>,
    opacity: f32,
    base_color: vec3<f32>,
    _pad0: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal_world: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.normal_world = in.normal;
    return out;
}

struct FsIn {
    @builtin(position) position: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) normal_world: vec3<f32>,
};

@fragment
fn fs_main(in: FsIn) -> @location(0) vec4<f32> {
    // Flip normal for back faces
    var n = normalize(in.normal_world);
    if (!in.front_facing) {
        n = -n;
    }

    let l = normalize(uniforms.light_dir_world);

    // Basic diffuse lighting
    let ndotl = max(dot(n, l), 0.0);
    let ambient = 0.3;
    let diffuse = 0.7 * ndotl;
    let color = uniforms.base_color * (ambient + diffuse);

    // Slight edge highlighting for back faces to show internal structure
    var alpha = uniforms.opacity;
    if (!in.front_facing) {
        // Back faces are slightly more transparent
        alpha *= 0.7;
    }

    // Output premultiplied alpha for correct blending
    return vec4<f32>(color * alpha, alpha);
}

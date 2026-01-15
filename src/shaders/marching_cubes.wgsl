struct Uniforms {
    view_proj: mat4x4<f32>,
    light_dir_world: vec3<f32>,
    _pad0: f32,
    base_color: vec3<f32>,
    _pad1: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(uniforms.light_dir_world);

    let ndotl = abs(dot(n, l));
    let ambient = 0.25;
    let diffuse = 0.75 * ndotl;
    let color = uniforms.base_color * (ambient + diffuse);

    return vec4<f32>(color, 1.0);
}

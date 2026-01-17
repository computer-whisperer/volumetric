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
    @location(0) normal_world: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.normal_world = in.normal;
    return out;
}

struct FsOut {
    @location(0) color: vec4<f32>,
    // Encoded normal in 0..1 for sampling
    @location(1) normal_enc: vec4<f32>,
    // Store fragment depth (0..1, wgpu NDC Z range) for SSAO sampling
    @location(2) depth: f32,
};

@fragment
fn fs_gbuffer(in: VsOut) -> FsOut {
    let n = normalize(in.normal_world);
    let l = normalize(uniforms.light_dir_world);

    // Manifold meshes: use one-sided lighting and rely on back-face culling.
    let ndotl = max(dot(n, l), 0.0);
    let ambient = 0.22;
    let diffuse = 0.78 * ndotl;
    let color = uniforms.base_color * (ambient + diffuse);

    var out: FsOut;
    out.color = vec4<f32>(color, 1.0);
    out.normal_enc = vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
    // `in.position` is clip-space; convert to wgpu NDC depth in [0,1].
    out.depth = in.position.z / in.position.w;
    return out;
}

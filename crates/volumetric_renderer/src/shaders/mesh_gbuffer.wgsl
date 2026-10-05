// Mesh geometry source: fills the G-buffer. No lighting here.

struct Uniforms {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    // Per draw, one instance each: the model matrix (rigid or
    // uniform-scale only: normals take its rotation), then the object id
    // and material index.
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(7) ids: vec2<u32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal_world: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) ids: vec2<u32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    var out: VsOut;
    out.position = uniforms.view_proj * model * vec4<f32>(in.position, 1.0);
    out.normal_world = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz) * in.normal;
    out.color = in.color;
    out.ids = in.ids;
    return out;
}

struct FsOut {
    // rgb: base colour, square-root encoded so 8 bits hold the darks;
    // a: material index / 255.
    @location(0) albedo: vec4<f32>,
    // rgb: world normal mapped to 0..1; a: 1 = a normal was supplied.
    @location(1) normal: vec4<f32>,
    // r: object id; g: the bits of this fragment's depth.
    @location(2) surface: vec2<u32>,
};

@fragment
fn fs_gbuffer(in: VsOut) -> FsOut {
    // A degenerate interpolated normal must not reach a division: NaN in
    // a unorm target clamps to white on many drivers.
    let n_len = length(in.normal_world);
    let n = select(vec3<f32>(0.0, 0.0, 1.0), in.normal_world / n_len, n_len > 1e-8);

    var out: FsOut;
    out.albedo = vec4<f32>(
        sqrt(clamp(in.color.rgb, vec3<f32>(0.0), vec3<f32>(1.0))),
        f32(in.ids.y) / 255.0,
    );
    out.normal = vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
    out.surface = vec2<u32>(in.ids.x, bitcast<u32>(in.position.z));
    return out;
}

// Mesh G-buffer rendering shader
//
// Renders triangle meshes to the G-buffer with:
// - Color: Lit diffuse color
// - Normal: World-space normal encoded to [0,1]
// - Depth: NDC depth for SSAO sampling
//
// Supports multiple render modes:
// - Shaded: Standard lit shading
// - BackFaceDebug: Front faces normal color, back faces red

struct Uniforms {
    view_proj: mat4x4<f32>,
    light_dir_world: vec3<f32>,
    _pad0: f32,
    base_color: vec3<f32>,
    // Render mode: 0 = Shaded, 1 = BackFaceDebug
    render_mode: u32,
    back_face_color: vec3<f32>,
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

struct FsIn {
    @builtin(position) position: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) normal_world: vec3<f32>,
};

struct FsOut {
    @location(0) color: vec4<f32>,
    // Encoded normal in 0..1 for sampling
    @location(1) normal_enc: vec4<f32>,
    // Store fragment depth (0..1, wgpu NDC Z range) for SSAO sampling
    // Output as vec4 for compatibility with both R32Float and Rgba16Float formats
    @location(2) depth: vec4<f32>,
};

@fragment
fn fs_gbuffer(in: FsIn) -> FsOut {
    // Flip normal for back faces to ensure correct lighting
    var n = normalize(in.normal_world);
    if (!in.front_facing) {
        n = -n;
    }

    let l = normalize(uniforms.light_dir_world);

    // Manifold meshes: use one-sided lighting and rely on back-face culling.
    let ndotl = max(dot(n, l), 0.0);
    let ambient = 0.22;
    let diffuse = 0.78 * ndotl;

    // Select color based on render mode and face orientation
    var base_color = uniforms.base_color;
    if (uniforms.render_mode == 1u && !in.front_facing) {
        // BackFaceDebug mode: use back_face_color for back faces
        base_color = uniforms.back_face_color;
    }

    let color = base_color * (ambient + diffuse);

    var out: FsOut;
    out.color = vec4<f32>(color, 1.0);
    out.normal_enc = vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
    // `in.position` is clip-space; z/w gives NDC depth in [0,1].
    let depth_value = in.position.z / in.position.w;
    out.depth = vec4<f32>(depth_value, 0.0, 0.0, 1.0);
    return out;
}

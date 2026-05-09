// Composite shader
//
// Combines G-buffer color with ambient occlusion for final output.

@group(0) @binding(0)
var g_color: texture_2d<f32>;

@group(0) @binding(1)
var g_ao: texture_2d<f32>;

@group(0) @binding(2)
var g_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    // Fullscreen triangle (covers entire viewport with 3 vertices)
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VsOut;
    out.pos = vec4<f32>(p[vi], 0.0, 1.0);
    // Map NDC -> UV with Y flip (UV origin is top-left)
    out.uv = vec2<f32>(
        0.5 * (out.pos.x + 1.0),
        1.0 - 0.5 * (out.pos.y + 1.0)
    );
    return out;
}

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let color = textureSample(g_color, g_sampler, uv);
    let ao = textureSample(g_ao, g_sampler, uv).r;
    return vec4<f32>(color.rgb * ao, color.a);
}

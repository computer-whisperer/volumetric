// Pick: one texel of the G-buffer's surface target (object id, depth
// bits), copied to a 1x1 target for read-back. Follows `fullscreen_vs.wgsl`.

struct PickUniforms {
    pixel: vec2<i32>,
    _pad0: vec2<i32>,
};

@group(0) @binding(0)
var<uniform> uniforms: PickUniforms;

@group(0) @binding(1)
var g_surface: texture_2d<u32>;

@fragment
fn fs_pick(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<u32> {
    let texel = textureLoad(g_surface, uniforms.pixel, 0);
    return vec4<u32>(texel.r, texel.g, 0u, 0u);
}

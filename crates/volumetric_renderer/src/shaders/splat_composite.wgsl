// Lays the splat layer over the scene. Splats blend among themselves in
// the value space their trainer used (sRGB values treated as numbers),
// so the layer holds premultiplied sRGB; here it is linearised for the
// sRGB target, which encodes on write.

@group(0) @binding(0)
var splat_layer: texture_2d<f32>;

@group(0) @binding(1)
var layer_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VsOut;
    out.pos = vec4<f32>(p[vi], 0.0, 1.0);
    out.uv = vec2<f32>(0.5 * (out.pos.x + 1.0), 1.0 - 0.5 * (out.pos.y + 1.0));
    return out;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let layer = textureSample(splat_layer, layer_sampler, in.uv);
    let a = layer.a;
    if a <= 0.0 {
        discard;
    }
    let c = clamp(layer.rgb / a, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(srgb_to_linear(c) * a, a);
}

// Lays the gathered layers of the transparent meshes over the lit opaque
// scene: their colour is premultiplied by their coverage, and the pass
// blends it over. Follows `fullscreen_vs.wgsl`.

@group(0) @binding(1)
var gathered: texture_2d<f32>;

@fragment
fn fs_layer_composite(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(gathered, vec2<i32>(frag.xy), 0);
}

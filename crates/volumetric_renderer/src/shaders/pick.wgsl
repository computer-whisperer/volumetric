// Pick: what one pixel of the last frame holds, copied to a row of
// texels for read-back: the texel of the G-buffer's surface target
// (object id, depth bits) first, then the texel of each transparent
// layer's surface target, nearest layer first. Which of them the pick
// lands on is settled where the row is read (`Picker::poll`), which
// knows how opaque each object was drawn. Follows `fullscreen_vs.wgsl`.

struct PickUniforms {
    pixel: vec2<i32>,
    _pad0: vec2<i32>,
};

@group(0) @binding(0)
var<uniform> uniforms: PickUniforms;

@group(0) @binding(1)
var g_surface: texture_2d<u32>;

// The layers' surface targets, nearest first. The opaque scene's stands
// in for the ones the frame did not peel, which are not read back.
@group(0) @binding(2)
var g_layer_0: texture_2d<u32>;
@group(0) @binding(3)
var g_layer_1: texture_2d<u32>;
@group(0) @binding(4)
var g_layer_2: texture_2d<u32>;
@group(0) @binding(5)
var g_layer_3: texture_2d<u32>;
@group(0) @binding(6)
var g_layer_4: texture_2d<u32>;
@group(0) @binding(7)
var g_layer_5: texture_2d<u32>;
@group(0) @binding(8)
var g_layer_6: texture_2d<u32>;
@group(0) @binding(9)
var g_layer_7: texture_2d<u32>;

@fragment
fn fs_pick(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<u32> {
    let at = uniforms.pixel;
    var texel: vec4<u32>;
    // The target is one texel for the opaque scene and one a layer.
    switch i32(frag.x) {
        case 0 { texel = textureLoad(g_surface, at, 0); }
        case 1 { texel = textureLoad(g_layer_0, at, 0); }
        case 2 { texel = textureLoad(g_layer_1, at, 0); }
        case 3 { texel = textureLoad(g_layer_2, at, 0); }
        case 4 { texel = textureLoad(g_layer_3, at, 0); }
        case 5 { texel = textureLoad(g_layer_4, at, 0); }
        case 6 { texel = textureLoad(g_layer_5, at, 0); }
        case 7 { texel = textureLoad(g_layer_6, at, 0); }
        default { texel = textureLoad(g_layer_7, at, 0); }
    }
    return vec4<u32>(texel.r, texel.g, 0u, 0u);
}

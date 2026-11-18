// Resolve for one layer of the transparent meshes: lit and edged as the
// opaque scene is by `resolve.wgsl`, without ambient occlusion (which is
// gathered from the opaque scene alone), for blending under the layers
// nearer the eye. Follows `resolve_common.wgsl`.
//
// A layer holds, at each pixel, the transparent surface that is so many
// surfaces deep there. So a surface is not all in one layer: it moves a
// layer farther wherever something transparent in front of it begins,
// and a layer nearer wherever that ends. The edge test follows it there,
// which is why every layer's surface target is filled before any layer
// is lit.

// How opaque the layer's surface is (see `opacity_of`).
@group(0) @binding(4)
var g_opacity: texture_2d<f32>;

// The opaque scene's surface target: what is beside a surface where its
// layer is empty.
@group(0) @binding(5)
var g_opaque_surface: texture_2d<u32>;

// The surface targets of the other layers, nearer and farther, in any
// order. The layer's own target stands in for the ones there are not.
@group(0) @binding(6)
var g_other_0: texture_2d<u32>;
@group(0) @binding(7)
var g_other_1: texture_2d<u32>;
@group(0) @binding(8)
var g_other_2: texture_2d<u32>;
@group(0) @binding(9)
var g_other_3: texture_2d<u32>;
@group(0) @binding(10)
var g_other_4: texture_2d<u32>;
@group(0) @binding(11)
var g_other_5: texture_2d<u32>;
@group(0) @binding(12)
var g_other_6: texture_2d<u32>;

const OTHER_LAYERS: i32 = 7;

fn other_layer(i: i32, at: vec2<i32>) -> vec4<u32> {
    switch i {
        case 0 { return textureLoad(g_other_0, at, 0); }
        case 1 { return textureLoad(g_other_1, at, 0); }
        case 2 { return textureLoad(g_other_2, at, 0); }
        case 3 { return textureLoad(g_other_3, at, 0); }
        case 4 { return textureLoad(g_other_4, at, 0); }
        case 5 { return textureLoad(g_other_5, at, 0); }
        default { return textureLoad(g_other_6, at, 0); }
    }
}

// Whether the surface of `object` is at `at` in another layer: the same
// object there, `expected` from the eye give or take `tolerance`.
fn in_another_layer(at: vec2<i32>, object: u32, expected: f32, tolerance: f32) -> bool {
    let k = uniforms.depth_to_distance;
    for (var i = 0; i < OTHER_LAYERS; i = i + 1) {
        let texel = other_layer(i, at);
        let depth = surface_depth(texel);
        if depth < 1.0 && texel.r == object && abs(eye_distance(depth, k) - expected) <= tolerance {
            return true;
        }
    }
    return false;
}

// What a layer's surface has beside it at a pixel.
struct Beside {
    // The surface target's texel: depth 1 where there is nothing.
    surface: vec4<u32>,
    // Whether it is this layer's own, so that its normal and opacity are
    // in the layer's targets.
    own: bool,
};

// The layer's own surface at `at`, or the opaque scene's where the layer
// drew nothing: that is behind every layer, so it is what shows there.
fn beside(at: vec2<i32>) -> Beside {
    let own = textureLoad(g_surface, at, 0);
    if surface_depth(own) < 1.0 {
        return Beside(own, true);
    }
    return Beside(textureLoad(g_opaque_surface, at, 0), false);
}

// The depth of the surface of `object` at `at` in another layer, where
// it can be taken for one plane with the surface of depth `depth` beside
// it; 1 where there is none.
fn depth_in_another_layer(at: vec2<i32>, object: u32, depth: f32) -> f32 {
    for (var i = 0; i < OTHER_LAYERS; i = i + 1) {
        let texel = other_layer(i, at);
        if texel.r == object && carries_plane(depth, surface_depth(texel)) {
            return surface_depth(texel);
        }
    }
    return 1.0;
}

// The mark a surface takes toward its neighbour at `at`, which it would
// be `expected` from the eye at: what `edge` of `resolve.wgsl` marks for
// the opaque scene (the near side of a step in depth, a crease, a change
// of object, the last two `forward` only: toward the right and down, so
// the line is one pixel wide). It is judged against what is beside the
// surface and not against the layer's next pixel, which may hold another
// surface altogether, and where the neighbour is a transparent object
// the mark follows how opaque it is:
//
// - What is beside it is farther, or nothing: the surface has ended (1),
//   unless it is to be found there in another layer, which is then a
//   nearer one: what was in front of it has ended, and it has not (0).
// - Another object is beside it, as near or nearer, and the surface is
//   to be found there in another layer: it passes behind that object or
//   through it. The mark is that object's outline falling on this
//   surface, or the line where the two cross, and is as strong as that
//   object is opaque.
// - Another object is beside it and the surface is nowhere there: it
//   ends against that object. Forward it is marked (1). The other way,
//   the opaque scene leaves the mark to the other object's pixel, so
//   here it is the mark this surface would have with that object gone
//   (1), as far as that object is not opaque.
fn mark_toward(
    at: vec2<i32>,
    forward: bool,
    n: vec3<f32>,
    object: u32,
    expected: f32,
    tolerance: f32,
) -> f32 {
    let k = uniforms.depth_to_distance;
    let other = beside(at);
    let other_depth = surface_depth(other.surface);

    // How much farther than the surface would be there.
    var farther = 0.0;
    if other_depth < 1.0 {
        farther = eye_distance(other_depth, k) - expected;
    }
    let stepped = other_depth >= 1.0 || farther > tolerance;
    let another = !stepped && other.surface.r != object;

    if stepped || another {
        let elsewhere = in_another_layer(at, object, expected, tolerance);
        var other_opacity = 1.0;
        if other.own {
            other_opacity = opacity_of(textureLoad(g_opacity, at, 0));
        }
        if stepped {
            return select(1.0, 0.0, elsewhere);
        }
        if elsewhere {
            return select(0.0, other_opacity, forward);
        }
        return select(1.0 - other_opacity, 1.0, forward);
    }
    if forward && other.own && dot(n, normal_at(at)) < uniforms.switches.x {
        return 1.0;
    }
    return 0.0;
}

// How much of an edge line covers `px`: the strongest of the marks
// toward its four neighbours.
//
// A mark is found by extrapolating the surface from the pixel a step the
// other way, and the surface may be in another layer there. If it is in
// a nearer one, that is where it is extrapolated from. If it is in a
// farther one, behind a transparent surface of this layer, there are two
// answers: the one an opaque surface there would leave (no
// extrapolation, as in the opaque scene) and the one with that surface
// gone (extrapolated from this surface's own depth there). The mark is
// the blend of the two by that surface's opacity.
//
// So the marks arrive at the all-opaque frame's as opacities reach 1, a
// mesh fading out takes its marks on other meshes with it, and a mesh
// keeps its own outline whatever fades in beside it.
fn edge_in_layer(px: vec2<i32>, depth: f32, n: vec3<f32>, object: u32) -> f32 {
    let k = uniforms.depth_to_distance;
    let limit = vec2<i32>(textureDimensions(g_surface)) - vec2<i32>(1);
    let width = max(i32(uniforms.switches.z), 1);
    let tolerance = eye_distance(depth, k) * DEPTH_STEP;
    var strength = 0.0;
    // Right, down, left, up.
    for (var i = 0; i < 4; i = i + 1) {
        let along = select(vec2<i32>(0, width), vec2<i32>(width, 0), i % 2 == 0);
        let step = select(along, -along, i >= 2);
        let at = clamp(px + step, vec2<i32>(0), limit);
        let back = clamp(px - step, vec2<i32>(0), limit);
        let forward = i < 2;

        let behind = beside(back);
        let behind_depth = surface_depth(behind.surface);
        let expected = expected_distance(depth, behind_depth);
        var mark = mark_toward(at, forward, n, object, expected, tolerance);
        if !(behind.own && behind.surface.r == object) {
            let own = depth_in_another_layer(back, object, depth);
            if own < 1.0 {
                let through =
                    mark_toward(at, forward, n, object, expected_distance(depth, own), tolerance);
                if behind.own && behind_depth < own {
                    mark = mix(through, mark, opacity_of(textureLoad(g_opacity, back, 0)));
                } else {
                    mark = through;
                }
            }
        }
        strength = max(strength, mark);
    }
    return strength;
}

// The colour is premultiplied by the surface's opacity: the pass blends
// it under the layers already gathered.
@fragment
fn fs_resolve_layer(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(frag.xy);
    let surface = textureLoad(g_surface, px, 0);
    let depth = surface_depth(surface);
    if depth >= 1.0 {
        discard;
    }
    let n = normal_at(px);
    var color = shade(frag, n, 1.0);

    if uniforms.edge_color.a > 0.0 {
        color = mix(color, uniforms.edge_color.rgb, uniforms.edge_color.a * edge_in_layer(px, depth, n, surface.r));
    }
    let opacity = opacity_of(textureLoad(g_opacity, px, 0));
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)) * opacity, opacity);
}

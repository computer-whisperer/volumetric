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

// The surface targets of the frame's other layers, nearer and farther,
// in the first `other_layers()` of these. The layer's own target fills
// the rest, which are not read.
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

// How many other layers the frame peels.
fn other_layers() -> i32 {
    return clamp(i32(uniforms.switches.w) - 1, 0, OTHER_LAYERS);
}

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

// Whether `object` has another surface at `px`, in another layer, that
// is nearer than this one (`distance` from the eye) to being `there`
// from the eye.
fn another_of_its_surfaces_is_nearer(px: vec2<i32>, object: u32, distance: f32, there: f32) -> bool {
    let k = uniforms.depth_to_distance;
    let off = abs(distance - there);
    let others = other_layers();
    for (var i = 0; i < others; i = i + 1) {
        let texel = other_layer(i, px);
        let depth = surface_depth(texel);
        if depth < 1.0 && texel.r == object && abs(eye_distance(depth, k) - there) < off {
            return true;
        }
    }
    return false;
}

// The depth at `at`, in another layer, of the surface of `object` that
// is `distance` from the eye at `px`: the same object there, `expected`
// from the eye give or take `tolerance`. 1 where there is none.
//
// An object can have two surfaces that close together (the walls of a
// thin shell), and the one found at `at` may be the other one, going on
// where this one has ended. So a surface found there is taken for this
// one only if no other surface of the object at `px` fits it better:
// carried back to `px` alongside this surface, it must be nearer this
// one than any of them.
fn in_another_layer(
    px: vec2<i32>,
    at: vec2<i32>,
    object: u32,
    distance: f32,
    expected: f32,
    tolerance: f32,
) -> f32 {
    let k = uniforms.depth_to_distance;
    let others = other_layers();
    for (var i = 0; i < others; i = i + 1) {
        let texel = other_layer(i, at);
        let depth = surface_depth(texel);
        if depth >= 1.0 || texel.r != object {
            continue;
        }
        let found = eye_distance(depth, k);
        if abs(found - expected) <= tolerance
            && !another_of_its_surfaces_is_nearer(px, object, distance, found - (expected - distance)) {
            return depth;
        }
    }
    return 1.0;
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
    let others = other_layers();
    for (var i = 0; i < others; i = i + 1) {
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
//   to be found there in another layer, behind that object: it passes
//   behind it or through it. The mark is that object's outline falling
//   on this surface, or the line where the two cross, and is as strong
//   as that object is opaque.
// - The same, but the surface is found there in front of that object
//   (which is then close behind it, by less than the tolerances of the
//   two tests): the surface has come out from behind something, as in
//   the first case, and the other object only lies close behind it there
//   or crosses it. No mark (0): the line where two surfaces cross is
//   marked once, as in the opaque scene, on the pixel of the pair that
//   is to the left or above, by the surface in front there. This takes
//   it that nothing new is in front of the surface at `at`; where
//   something is (two outlines in one place), that object's outline is
//   owed and not made.
// - Another object is beside it and the surface is nowhere there: it
//   ends against that object. Forward it is marked (1). The other way,
//   the opaque scene leaves the mark to the other object's pixel, so
//   here it is the mark this surface would have with that object gone
//   (1), as far as that object is not opaque.
fn mark_toward(
    px: vec2<i32>,
    at: vec2<i32>,
    forward: bool,
    n: vec3<f32>,
    object: u32,
    distance: f32,
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
        let found = in_another_layer(px, at, object, distance, expected, tolerance);
        let elsewhere = found < 1.0;
        var other_opacity = 1.0;
        if other.own {
            other_opacity = opacity_of(textureLoad(g_opacity, at, 0));
        }
        if stepped {
            return select(1.0, 0.0, elsewhere);
        }
        if elsewhere {
            // In front of the other object there, as the peel orders
            // them: the nearer, or at one depth the higher object.
            if found < other_depth || (found == other_depth && object > other.surface.r) {
                return 0.0;
            }
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
    let distance = eye_distance(depth, k);
    let tolerance = distance * DEPTH_STEP;
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
        var mark = mark_toward(px, at, forward, n, object, distance, expected, tolerance);
        if !(behind.own && behind.surface.r == object) {
            let own = depth_in_another_layer(back, object, depth);
            if own < 1.0 {
                let through = mark_toward(
                    px,
                    at,
                    forward,
                    n,
                    object,
                    distance,
                    expected_distance(depth, own),
                    tolerance,
                );
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

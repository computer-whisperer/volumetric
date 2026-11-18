// Resolve: lights the G-buffer into display-referred colour and draws
// edge lines where the G-buffer is discontinuous. Pixels no geometry
// source wrote are left to the pass's clear (the background). Follows
// `resolve_common.wgsl`.

@group(0) @binding(4)
var g_ao: texture_2d<f32>;

// In a frame with transparent meshes (`switches.w`): the surface target
// of their nearest layer, and how opaque that layer's surfaces are. Not
// read otherwise.
@group(0) @binding(5)
var g_nearest_layer: texture_2d<u32>;

@group(0) @binding(6)
var g_nearest_opacity: texture_2d<f32>;

// Whether the surface steps away from the eye between `px` and its
// neighbour at `px + step`: the neighbour is farther than the plane
// through `px` and the neighbour on the other side would put it.
fn steps_away(px: vec2<i32>, step: vec2<i32>, depth: f32, limit: vec2<i32>) -> bool {
    let k = uniforms.depth_to_distance;
    let neighbour = surface_depth(textureLoad(g_surface, clamp(px + step, vec2<i32>(0), limit), 0));
    if neighbour >= 1.0 {
        return true;
    }
    let behind = surface_depth(textureLoad(g_surface, clamp(px - step, vec2<i32>(0), limit), 0));
    return eye_distance(neighbour, k) - expected_distance(depth, behind)
        > eye_distance(depth, k) * DEPTH_STEP;
}

// How much of an edge line covers `px`: 1 on the near side of a step in
// depth, at a crease sharper than the crease angle, or where the object
// changes.
fn edge(px: vec2<i32>, depth: f32, n: vec3<f32>, object: u32) -> f32 {
    let limit = vec2<i32>(textureDimensions(g_surface)) - vec2<i32>(1);
    let width = max(i32(uniforms.switches.z), 1);
    let right = vec2<i32>(width, 0);
    let down = vec2<i32>(0, width);
    if steps_away(px, right, depth, limit) || steps_away(px, -right, depth, limit)
        || steps_away(px, down, depth, limit) || steps_away(px, -down, depth, limit) {
        return 1.0;
    }
    // Creases and object boundaries are marked from one side only, so
    // the line is one pixel wide.
    for (var i = 0; i < 2; i = i + 1) {
        let at = min(px + select(down, right, i == 0), limit);
        let texel = textureLoad(g_surface, at, 0);
        if surface_depth(texel) >= 1.0 {
            continue;
        }
        if texel.r != object || dot(n, normal_at(at)) < uniforms.switches.x {
            return 1.0;
        }
    }
    return 0.0;
}

// The mark `edge` makes toward one neighbour, for any neighbour: the
// surface target's texel `neighbour` at `at`, with `behind` the depth one
// step the other way. `forward` is toward the right or down, where
// creases and object boundaries are marked; a crease needs the
// neighbour's normal, which the G-buffer has for its own surfaces
// (`in_g_buffer`).
fn mark_against(
    neighbour: vec4<u32>,
    in_g_buffer: bool,
    at: vec2<i32>,
    forward: bool,
    depth: f32,
    n: vec3<f32>,
    object: u32,
    behind: f32,
) -> f32 {
    let k = uniforms.depth_to_distance;
    let neighbour_depth = surface_depth(neighbour);
    if neighbour_depth >= 1.0 {
        return 1.0;
    }
    if eye_distance(neighbour_depth, k) - expected_distance(depth, behind)
        > eye_distance(depth, k) * DEPTH_STEP {
        return 1.0;
    }
    if forward
        && (neighbour.r != object || (in_g_buffer && dot(n, normal_at(at)) < uniforms.switches.x)) {
        return 1.0;
    }
    return 0.0;
}

// `edge` in a frame with transparent meshes. The G-buffer does not hold
// them, but the nearest of them stand beside the opaque scene's surfaces
// as neighbours all the same, each as much as it is opaque: where a
// neighbouring pixel has one, the mark is a blend, by its opacity, of the
// mark the G-buffer alone gives and the mark that surface would give
// were it opaque and in the G-buffer. That goes for the neighbour a mark
// is made toward and for the one a step the other way, which the test
// extrapolates the surface from. So the edges of the opaque scene arrive
// at the all-opaque frame's as a mesh's opacity reaches 1, and at the
// frame's without the mesh as it reaches 0.
fn edge_beside_transparent(px: vec2<i32>, depth: f32, n: vec3<f32>, object: u32) -> f32 {
    // Under a transparent surface the opaque scene keeps its own edges.
    if surface_depth(textureLoad(g_nearest_layer, px, 0)) < 1.0 {
        return edge(px, depth, n, object);
    }
    let limit = vec2<i32>(textureDimensions(g_surface)) - vec2<i32>(1);
    let width = max(i32(uniforms.switches.z), 1);
    var strength = 0.0;
    // Right, down, left, up.
    for (var i = 0; i < 4; i = i + 1) {
        let along = select(vec2<i32>(0, width), vec2<i32>(width, 0), i % 2 == 0);
        let step = select(along, -along, i >= 2);
        let at = clamp(px + step, vec2<i32>(0), limit);
        let back = clamp(px - step, vec2<i32>(0), limit);
        let forward = i < 2;

        let neighbour = textureLoad(g_surface, at, 0);
        let behind = surface_depth(textureLoad(g_surface, back, 0));
        let layer_at = textureLoad(g_nearest_layer, at, 0);
        let layer_back = textureLoad(g_nearest_layer, back, 0);
        let has_at = surface_depth(layer_at) < 1.0;
        var opacity_at = 0.0;
        if has_at {
            opacity_at = opacity_of(textureLoad(g_nearest_opacity, at, 0));
        }

        var mark = mark_against(neighbour, true, at, forward, depth, n, object, behind);
        if has_at {
            let beside = mark_against(layer_at, false, at, forward, depth, n, object, behind);
            mark = mix(mark, beside, opacity_at);
        }
        if surface_depth(layer_back) < 1.0 {
            // The same again, extrapolated from the transparent surface
            // a step back: for an opaque one there, that is the test.
            let stand_in = surface_depth(layer_back);
            var other = mark_against(neighbour, true, at, forward, depth, n, object, stand_in);
            if has_at {
                let beside = mark_against(layer_at, false, at, forward, depth, n, object, stand_in);
                other = mix(other, beside, opacity_at);
            }
            mark = mix(mark, other, opacity_of(textureLoad(g_nearest_opacity, back, 0)));
        }
        strength = max(strength, mark);
    }
    return strength;
}

@fragment
fn fs_resolve(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(frag.xy);
    let surface = textureLoad(g_surface, px, 0);
    let depth = surface_depth(surface);
    if depth >= 1.0 {
        discard;
    }
    let n = normal_at(px);

    var ao = 1.0;
    if uniforms.switches.y > 0.5 {
        ao = textureLoad(g_ao, px, 0).r;
    }
    var color = shade(frag, n, ao);

    if uniforms.edge_color.a > 0.0 {
        var strength = 0.0;
        if uniforms.switches.w > 0.5 {
            strength = edge_beside_transparent(px, depth, n, surface.r);
        } else {
            strength = edge(px, depth, n, surface.r);
        }
        color = mix(color, uniforms.edge_color.rgb, uniforms.edge_color.a * strength);
    }
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

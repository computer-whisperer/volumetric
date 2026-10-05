// Ambient occlusion from the G-buffer's depth and normals. Around each
// surface point, points of the hemisphere above it are projected back to
// the screen; a point that lies behind what the G-buffer holds there is
// occluded. Distances are compared in world units in front of the eye,
// so the result is the same at any distance and clip range.
//
// The sample pattern turns with the pixel's place in a 4x4 tile, which
// the blur pass (`ao_blur.wgsl`) averages away. Follows
// `fullscreen_vs.wgsl`.

struct AoUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    depth_to_distance: vec4<f32>,
    // World radius occluders are looked for within.
    radius: f32,
    // How many of the 16 samples are taken.
    samples: u32,
    // The most pixels the radius may cover; nearer surfaces shrink it.
    max_radius_px: f32,
    // The target's height in pixels over the frame's height in world
    // units at distance 1 (0 for an orthographic frame, whose
    // `pixels_per_unit` then holds the whole scale).
    pixels_per_unit_at_1: f32,
    pixels_per_unit: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: AoUniforms;

@group(0) @binding(1)
var g_normal: texture_2d<f32>;

@group(0) @binding(2)
var g_surface: texture_2d<u32>;

fn world_at(px: vec2<i32>, depth01: f32, dim: vec2<f32>) -> vec3<f32> {
    // Pixel centres; texture rows run down while NDC +Y runs up.
    let uv = (vec2<f32>(px) + vec2<f32>(0.5)) / dim;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth01, 1.0);
    let world_h = uniforms.inv_view_proj * ndc;
    return world_h.xyz / world_h.w;
}

// A direction in the hemisphere around +Z, sample `i` of 16: a spiral
// that covers the hemisphere evenly, turned by `turn` radians.
fn hemisphere(i: f32, turn: f32) -> vec3<f32> {
    let phi = i * 2.399963 + turn;
    let cos_theta = 1.0 - (i + 0.5) / 16.0;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

fn tangent_frame(n: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.99);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

@fragment
fn fs_ao(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(frag.xy);
    let dim_i = vec2<i32>(textureDimensions(g_surface));
    let dim = vec2<f32>(dim_i);
    let depth01 = surface_depth(textureLoad(g_surface, px, 0));
    if depth01 >= 1.0 {
        return vec4<f32>(1.0);
    }

    let n = normalize(textureLoad(g_normal, px, 0).rgb * 2.0 - vec3<f32>(1.0));
    let p_world = world_at(px, depth01, dim);
    let p_distance = eye_distance(depth01, uniforms.depth_to_distance);

    // The radius, held to a pixel budget so a close surface does not
    // scatter its samples across the frame.
    let px_per_unit = uniforms.pixels_per_unit + uniforms.pixels_per_unit_at_1 / p_distance;
    let radius = min(uniforms.radius, uniforms.max_radius_px / px_per_unit);

    let frame = tangent_frame(n);
    let tile = vec2<i32>(px.x & 3, px.y & 3);
    let turn = f32(tile.x + tile.y * 4) * (6.2831853 / 16.0);

    var occlusion = 0.0;
    var taken = 0.0;
    for (var s = 0u; s < uniforms.samples; s = s + 1u) {
        // Spread the samples taken over all 16 of the pattern, at
        // lengths that fill the hemisphere's volume.
        let i = f32(s) * 16.0 / f32(uniforms.samples);
        let reach = mix(0.15, 1.0, (i + 1.0) / 16.0);
        let sample_world = p_world + frame * hemisphere(i, turn) * (radius * reach);
        let clip = uniforms.view_proj * vec4<f32>(sample_world, 1.0);
        if clip.w <= 0.0 {
            continue;
        }
        let ndc = clip.xyz / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0 {
            continue;
        }
        taken = taken + 1.0;
        let scene_depth = surface_depth(textureLoad(g_surface, vec2<i32>(uv * dim), 0));
        if scene_depth >= 1.0 {
            continue;
        }
        let scene_distance = eye_distance(scene_depth, uniforms.depth_to_distance);
        let sample_distance = eye_distance(ndc.z, uniforms.depth_to_distance);
        // Occluded when the scene is in front of the sample by more than
        // a sliver; an occluder far in front of this surface belongs to
        // something else and counts for less.
        if scene_distance < sample_distance - radius * 0.03 {
            occlusion = occlusion + smoothstep(0.0, 1.0, radius / abs(p_distance - scene_distance));
        }
    }

    let ao = 1.0 - occlusion / max(taken, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
}

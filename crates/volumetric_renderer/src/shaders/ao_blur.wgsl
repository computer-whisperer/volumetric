// Averages the raw occlusion over the 4x4 tile its sample pattern turns
// through, leaving out neighbours on a different surface (by distance
// from the eye and by normal) so occlusion does not bleed across
// silhouettes. Applies the strength. Follows `fullscreen_vs.wgsl`.

struct AoBlurUniforms {
    depth_to_distance: vec4<f32>,
    strength: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: AoBlurUniforms;

@group(0) @binding(1)
var ao_raw: texture_2d<f32>;

@group(0) @binding(2)
var g_normal: texture_2d<f32>;

@group(0) @binding(3)
var g_surface: texture_2d<u32>;

@fragment
fn fs_ao_blur(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(frag.xy);
    let limit = vec2<i32>(textureDimensions(g_surface)) - vec2<i32>(1);
    let depth01 = surface_depth(textureLoad(g_surface, px, 0));
    if depth01 >= 1.0 {
        return vec4<f32>(1.0);
    }
    let distance = eye_distance(depth01, uniforms.depth_to_distance);
    let n = textureLoad(g_normal, px, 0).rgb * 2.0 - vec3<f32>(1.0);

    var sum = 0.0;
    var weight = 0.0;
    for (var y = -2; y < 2; y = y + 1) {
        for (var x = -2; x < 2; x = x + 1) {
            let at = clamp(px + vec2<i32>(x, y), vec2<i32>(0), limit);
            let other_depth = surface_depth(textureLoad(g_surface, at, 0));
            let other = eye_distance(other_depth, uniforms.depth_to_distance);
            let other_n = textureLoad(g_normal, at, 0).rgb * 2.0 - vec3<f32>(1.0);
            let same = f32(other_depth < 1.0)
                * (1.0 - smoothstep(0.01, 0.03, abs(other - distance) / distance))
                * smoothstep(0.6, 0.9, dot(n, other_n));
            sum = sum + textureLoad(ao_raw, at, 0).r * same;
            weight = weight + same;
        }
    }
    // The pixel itself always counts fully, so the weight is never zero.
    let ao = pow(clamp(sum / weight, 0.0, 1.0), uniforms.strength);
    return vec4<f32>(ao, ao, ao, 1.0);
}

// Screen-space ambient occlusion from the G-buffer's depth and normals.
// Follows `fullscreen_vs.wgsl` in its module.

struct AoUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    radius: f32,
    bias: f32,
    strength: f32,
    _pad0: f32,
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

fn hash12(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn rand_dir(i: f32, seed: f32) -> vec3<f32> {
    // Two randoms to a direction on the hemisphere around +Z.
    let u1 = fract(sin((i + 1.0) * 12.9898 + seed) * 43758.5453);
    let u2 = fract(sin((i + 1.0) * 78.233 + seed * 1.37) * 43758.5453);
    let phi = 6.28318530718 * u1;
    let cos_theta = u2;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

fn build_tbn(n: vec3<f32>) -> mat3x3<f32> {
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
    if depth01 >= 0.999999 {
        return vec4<f32>(1.0);
    }

    let n = normalize(textureLoad(g_normal, px, 0).rgb * 2.0 - vec3<f32>(1.0));
    let p_world = world_at(px, depth01, dim);
    let tbn = build_tbn(n);
    let seed = hash12(frag.xy);

    var occ: f32 = 0.0;
    let sample_count: i32 = 16;
    for (var s: i32 = 0; s < sample_count; s = s + 1) {
        let i = f32(s);
        let scale = (i + 1.0) / f32(sample_count);
        let sample_vec = (tbn * rand_dir(i, seed)) * (uniforms.radius * scale);
        let clip = uniforms.view_proj * vec4<f32>(p_world + sample_vec, 1.0);
        let ndc = clip.xyz / clip.w;
        let sample_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            continue;
        }
        let sample_px = clamp(vec2<i32>(sample_uv * dim), vec2<i32>(0), dim_i - vec2<i32>(1));
        let scene_depth01 = surface_depth(textureLoad(g_surface, sample_px, 0));
        if scene_depth01 <= ndc.z - uniforms.bias {
            // Weight by distance to reduce haloing.
            occ = occ + 1.0 - clamp(length(sample_vec) / uniforms.radius, 0.0, 1.0);
        }
    }

    let ao = pow(clamp(1.0 - occ / f32(sample_count), 0.0, 1.0), uniforms.strength);
    return vec4<f32>(ao, ao, ao, 1.0);
}

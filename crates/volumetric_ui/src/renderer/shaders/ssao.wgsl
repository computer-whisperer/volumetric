// Screen-space ambient occlusion (SSAO) shader
//
// Computes ambient occlusion by sampling depth around each pixel
// and testing for occlusion based on surface normal orientation.

struct SsaoUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    screen_size_px: vec2<f32>,
    radius: f32,
    bias: f32,
    strength: f32,
    _pad0: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: SsaoUniforms;

@group(0) @binding(1)
var g_normal: texture_2d<f32>;

@group(0) @binding(2)
var g_depth: texture_2d<f32>;

@group(0) @binding(3)
var g_sampler: sampler;

fn depth01_at_uv(uv: vec2<f32>) -> f32 {
    // `g_depth` is R32Float, which is not filterable.
    // Use textureLoad with integer pixel coords.
    let dim = vec2<i32>(textureDimensions(g_depth));
    let px = clamp(vec2<i32>(uv * vec2<f32>(dim)), vec2<i32>(0, 0), dim - vec2<i32>(1, 1));
    return textureLoad(g_depth, px, 0).r;
}

fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn decode_normal(enc: vec3<f32>) -> vec3<f32> {
    return normalize(enc * 2.0 - vec3<f32>(1.0));
}

fn ndc_from_uv_depth(uv: vec2<f32>, depth01: f32) -> vec4<f32> {
    // wgpu NDC: x/y in [-1,1], z in [0,1]
    // WebGPU texture UV origin is top-left, while NDC +Y points up.
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = (1.0 - uv.y) * 2.0 - 1.0;
    return vec4<f32>(ndc_x, ndc_y, depth01, 1.0);
}

fn world_from_uv_depth(uv: vec2<f32>, depth01: f32) -> vec3<f32> {
    let clip = ndc_from_uv_depth(uv, depth01);
    let world_h = uniforms.inv_view_proj * clip;
    return world_h.xyz / world_h.w;
}

fn hash12(p: vec2<f32>) -> f32 {
    // Cheap interleaved gradient noise
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn rand_dir(i: f32, seed: f32) -> vec3<f32> {
    // Map 2 randoms to a direction on hemisphere around +Z
    let u1 = fract(sin((i + 1.0) * 12.9898 + seed) * 43758.5453);
    let u2 = fract(sin((i + 1.0) * 78.233 + seed * 1.37) * 43758.5453);
    let phi = 6.28318530718 * u1;
    let cos_theta = u2;
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    return vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

fn build_tbn(n: vec3<f32>) -> mat3x3<f32> {
    // Pick an arbitrary tangent
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.99);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    // Fullscreen triangle (covers entire viewport with 3 vertices)
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VsOut;
    out.pos = vec4<f32>(p[vi], 0.0, 1.0);
    // Map NDC -> UV with Y flip (UV origin is top-left)
    out.uv = vec2<f32>(
        0.5 * (out.pos.x + 1.0),
        1.0 - 0.5 * (out.pos.y + 1.0)
    );
    return out;
}

@fragment
fn fs_ssao(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let depth01 = depth01_at_uv(uv);

    // Background check
    if depth01 >= 0.999999 {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    let n_enc = textureSample(g_normal, g_sampler, uv).rgb;
    let n = decode_normal(n_enc);
    let p_world = world_from_uv_depth(uv, depth01);

    let tbn = build_tbn(n);
    let seed = hash12(uv * uniforms.screen_size_px);

    var occ: f32 = 0.0;
    let sample_count: i32 = 16;

    for (var s: i32 = 0; s < sample_count; s = s + 1) {
        let i = f32(s);
        // Spiral-ish radius distribution
        let scale = (i + 1.0) / f32(sample_count);
        let dir_h = rand_dir(i, seed);
        let sample_vec = (tbn * dir_h) * (uniforms.radius * scale);
        let sample_p = p_world + sample_vec;

        // Project sample point to screen
        let clip = uniforms.view_proj * vec4<f32>(sample_p, 1.0);
        let ndc = clip.xyz / clip.w;
        // Convert to UV; ndc.z is already 0..1 for wgpu
        let sample_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

        // Skip samples that fall off screen
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            continue;
        }

        let sample_depth01 = depth01_at_uv(sample_uv);
        // If sampled depth is closer than expected depth, it occludes
        let sample_expected_depth01 = ndc.z;
        let occluding = sample_depth01 <= (sample_expected_depth01 - uniforms.bias);

        if occluding {
            // Weight by distance to reduce haloing
            let range = uniforms.radius;
            let dist = length(sample_vec);
            let w = 1.0 - saturate(dist / range);
            occ = occ + w;
        }
    }

    let ao = 1.0 - (occ / f32(sample_count));
    // Strength curve
    let ao_final = pow(saturate(ao), uniforms.strength);
    return vec4<f32>(ao_final, ao_final, ao_final, 1.0);
}

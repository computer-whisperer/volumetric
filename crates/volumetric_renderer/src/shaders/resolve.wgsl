// Resolve: lights the G-buffer into display-referred colour. Pixels no
// geometry source wrote are left to the pass's clear (the background).
// Follows `fullscreen_vs.wgsl` in its module.

struct ResolveUniforms {
    light_dir_world: vec3<f32>,
    ao_enabled: u32,
    base_tint: vec3<f32>,
    _pad0: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: ResolveUniforms;

@group(0) @binding(1)
var g_albedo: texture_2d<f32>;

@group(0) @binding(2)
var g_normal: texture_2d<f32>;

@group(0) @binding(3)
var g_surface: texture_2d<u32>;

@group(0) @binding(4)
var g_ao: texture_2d<f32>;

@fragment
fn fs_resolve(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(frag.xy);
    if surface_depth(textureLoad(g_surface, px, 0)) >= 1.0 {
        discard;
    }

    let albedo_enc = textureLoad(g_albedo, px, 0).rgb;
    let albedo = albedo_enc * albedo_enc * uniforms.base_tint;
    let n = normalize(textureLoad(g_normal, px, 0).rgb * 2.0 - vec3<f32>(1.0));
    let l = normalize(uniforms.light_dir_world);

    let ambient = 0.22;
    let diffuse = 0.78 * max(dot(n, l), 0.0);
    var color = albedo * (ambient + diffuse);
    if uniforms.ao_enabled != 0u {
        color = color * textureLoad(g_ao, px, 0).r;
    }
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

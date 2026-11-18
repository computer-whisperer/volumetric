// What the resolve passes share: the lighting model and the plane a
// surface is extrapolated along to find a step in depth. `resolve.wgsl`
// (the opaque scene) or `resolve_layer.wgsl` (one layer of the
// transparent meshes) follows this in the same module, and this follows
// `fullscreen_vs.wgsl`.

struct Light {
    // xyz: the world direction the light shines from (unit).
    direction: vec4<f32>,
    color: vec4<f32>,
};

struct Material {
    // rgb: base tint; a: unused.
    tint: vec4<f32>,
    // x: the highlight's exponent; y: its strength.
    highlight: vec4<f32>,
};

struct ResolveUniforms {
    inv_view_proj: mat4x4<f32>,
    depth_to_distance: vec4<f32>,
    lights: array<Light, 3>,
    sky: vec4<f32>,
    ground: vec4<f32>,
    // rgb: edge line colour; a: opacity, 0 when edges are off.
    edge_color: vec4<f32>,
    // x: cosine of the crease angle; y: 1 when ambient occlusion is on;
    // z: how many pixels wide an edge line is; w: how many layers of
    // transparent meshes the frame peels, 0 when it has none.
    switches: vec4<f32>,
    materials: array<Material, 16>,
};

@group(0) @binding(0)
var<uniform> uniforms: ResolveUniforms;

// The targets being lit: the G-buffer's, or a layer's.
@group(0) @binding(1)
var g_albedo: texture_2d<f32>;

@group(0) @binding(2)
var g_normal: texture_2d<f32>;

@group(0) @binding(3)
var g_surface: texture_2d<u32>;

// A neighbour this much farther than the surface would put it, as a
// fraction of the surface's own distance, is a step in depth.
const DEPTH_STEP: f32 = 0.012;

fn unproject(ndc: vec2<f32>, z: f32) -> vec3<f32> {
    let h = uniforms.inv_view_proj * vec4<f32>(ndc, z, 1.0);
    return h.xyz / h.w;
}

fn normal_at(px: vec2<i32>) -> vec3<f32> {
    return normalize(textureLoad(g_normal, px, 0).rgb * 2.0 - vec3<f32>(1.0));
}

// Whether a surface of depth `depth` and one of depth `behind`, a pixel
// apart, can be taken for one plane: there is one, and no jump between.
fn carries_plane(depth: f32, behind: f32) -> bool {
    let k = uniforms.depth_to_distance;
    let distance = eye_distance(depth, k);
    return behind < 1.0 && abs(eye_distance(behind, k) - distance) < distance * 0.02;
}

// How far from the eye a surface of depth `depth` would be one step on,
// if it carried on as a plane: the plane through it and `behind`, the
// depth one step back. Depth is linear across a plane on screen, so the
// plane is extrapolated in depth and the answer given in world distance.
fn expected_distance(depth: f32, behind: f32) -> f32 {
    let k = uniforms.depth_to_distance;
    // With no surface behind, or a jump there, take this pixel's depth.
    var expected = eye_distance(depth, k);
    if carries_plane(depth, behind) {
        expected = eye_distance(clamp(2.0 * depth - behind, 0.0, 1.0), k);
    }
    return expected;
}

// How opaque a transparent surface is, from a layer's opacity target,
// which stores the square root.
fn opacity_of(stored: vec4<f32>) -> f32 {
    return stored.r * stored.r;
}

// Leaves colours below the knee alone and rolls the rest off toward 1,
// so a highlight saturates smoothly instead of clipping.
fn shoulder(color: vec3<f32>) -> vec3<f32> {
    let knee = 0.8;
    let over = max(color - vec3<f32>(knee), vec3<f32>(0.0)) / (1.0 - knee);
    return min(color, vec3<f32>(knee)) + (1.0 - knee) * (vec3<f32>(1.0) - exp(-over));
}

// The lit colour of the surface at `frag` whose normal is `n`, under
// occlusion `ao` (1 is none); no edge line yet.
fn shade(frag: vec4<f32>, n: vec3<f32>, ao: f32) -> vec3<f32> {
    let px = vec2<i32>(frag.xy);
    let albedo_texel = textureLoad(g_albedo, px, 0);
    let material = uniforms.materials[min(u32(albedo_texel.a * 255.0 + 0.5), 15u)];
    let albedo = albedo_texel.rgb * albedo_texel.rgb * material.tint.rgb;

    // Toward the eye, along this pixel's ray.
    let dim = vec2<f32>(textureDimensions(g_surface));
    let ndc = vec2<f32>(frag.x / dim.x * 2.0 - 1.0, 1.0 - frag.y / dim.y * 2.0);
    let v = normalize(unproject(ndc, 0.0) - unproject(ndc, 1.0));

    let ambient = mix(uniforms.ground.rgb, uniforms.sky.rgb, n.z * 0.5 + 0.5);
    var diffuse = vec3<f32>(0.0);
    var highlight = vec3<f32>(0.0);
    for (var i = 0; i < 3; i = i + 1) {
        let l = uniforms.lights[i].direction.xyz;
        let facing = max(dot(n, l), 0.0);
        diffuse = diffuse + uniforms.lights[i].color.rgb * facing;
        let h = normalize(l + v);
        highlight = highlight
            + uniforms.lights[i].color.rgb
                * (pow(max(dot(n, h), 0.0), material.highlight.x) * step(0.0, dot(n, l)));
    }
    // Occlusion takes the ambient light whole and direct light in part.
    let color = albedo * (ambient * ao + diffuse * mix(1.0, ao, 0.5))
        + highlight * (material.highlight.y * ao);
    return shoulder(color);
}

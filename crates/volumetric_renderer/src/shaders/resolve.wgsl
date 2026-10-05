// Resolve: lights the G-buffer into display-referred colour and draws
// edge lines where the G-buffer is discontinuous. Pixels no geometry
// source wrote are left to the pass's clear (the background). Follows
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
    // x: cosine of the crease angle; y: 1 when ambient occlusion is on.
    switches: vec4<f32>,
    materials: array<Material, 16>,
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

fn unproject(ndc: vec2<f32>, z: f32) -> vec3<f32> {
    let h = uniforms.inv_view_proj * vec4<f32>(ndc, z, 1.0);
    return h.xyz / h.w;
}

fn normal_at(px: vec2<i32>) -> vec3<f32> {
    return normalize(textureLoad(g_normal, px, 0).rgb * 2.0 - vec3<f32>(1.0));
}

// Whether the surface steps away from the eye between `px` and its
// neighbour at `px + step`: the neighbour is farther than the plane
// through `px` and the neighbour on the other side would put it. Depth
// is linear across a plane on screen, so the plane is extrapolated in
// depth and the comparison made in world distance.
fn steps_away(px: vec2<i32>, step: vec2<i32>, depth: f32, limit: vec2<i32>) -> bool {
    let k = uniforms.depth_to_distance;
    let neighbour = surface_depth(textureLoad(g_surface, clamp(px + step, vec2<i32>(0), limit), 0));
    if neighbour >= 1.0 {
        return true;
    }
    let behind = surface_depth(textureLoad(g_surface, clamp(px - step, vec2<i32>(0), limit), 0));
    let distance = eye_distance(depth, k);
    // With no surface behind, or a jump there, take this pixel's depth.
    var expected = distance;
    if behind < 1.0 && abs(eye_distance(behind, k) - distance) < distance * 0.02 {
        expected = eye_distance(clamp(2.0 * depth - behind, 0.0, 1.0), k);
    }
    return eye_distance(neighbour, k) - expected > distance * 0.012;
}

// How much of an edge line covers `px`: 1 on the near side of a step in
// depth, at a crease sharper than the crease angle, or where the object
// changes.
fn edge(px: vec2<i32>, depth: f32, n: vec3<f32>, object: u32) -> f32 {
    let limit = vec2<i32>(textureDimensions(g_surface)) - vec2<i32>(1);
    let right = vec2<i32>(1, 0);
    let down = vec2<i32>(0, 1);
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

// Leaves colours below the knee alone and rolls the rest off toward 1,
// so a highlight saturates smoothly instead of clipping.
fn shoulder(color: vec3<f32>) -> vec3<f32> {
    let knee = 0.8;
    let over = max(color - vec3<f32>(knee), vec3<f32>(0.0)) / (1.0 - knee);
    return min(color, vec3<f32>(knee)) + (1.0 - knee) * (vec3<f32>(1.0) - exp(-over));
}

@fragment
fn fs_resolve(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<i32>(frag.xy);
    let surface = textureLoad(g_surface, px, 0);
    let depth = surface_depth(surface);
    if depth >= 1.0 {
        discard;
    }

    let albedo_texel = textureLoad(g_albedo, px, 0);
    let material = uniforms.materials[min(u32(albedo_texel.a * 255.0 + 0.5), 15u)];
    let albedo = albedo_texel.rgb * albedo_texel.rgb * material.tint.rgb;
    let n = normal_at(px);

    // Toward the eye, along this pixel's ray.
    let dim = vec2<f32>(textureDimensions(g_surface));
    let ndc = vec2<f32>(frag.x / dim.x * 2.0 - 1.0, 1.0 - frag.y / dim.y * 2.0);
    let v = normalize(unproject(ndc, 0.0) - unproject(ndc, 1.0));

    var ao = 1.0;
    if uniforms.switches.y > 0.5 {
        ao = textureLoad(g_ao, px, 0).r;
    }

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
    var color = albedo * (ambient * ao + diffuse * mix(1.0, ao, 0.5))
        + highlight * (material.highlight.y * ao);
    color = shoulder(color);

    if uniforms.edge_color.a > 0.0 {
        color = mix(color, uniforms.edge_color.rgb, uniforms.edge_color.a * edge(px, depth, n, surface.r));
    }
    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

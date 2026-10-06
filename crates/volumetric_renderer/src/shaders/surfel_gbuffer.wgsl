// Surfel geometry source: each surface point is drawn as an opaque disc in
// its tangent plane, filling the G-buffer exactly as a mesh does. The disc
// is a quad per instance, cut round in the fragment shader; its depth is
// the plane's, so discs of a surface meet one another without gaps.

struct Uniforms {
    view_proj: mat4x4<f32>,
    // The eye, w = 1; or for an orthographic frame the view direction,
    // w = 0.
    eye: vec4<f32>,
};

// Per draw: the model matrix (rigid or uniform-scale only), then the
// object id and material index.
struct Draw {
    model: mat4x4<f32>,
    ids: vec2<u32>,
    _pad: vec2<u32>,
    // Linear RGBA for every disc of the draw.
    color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;
@group(1) @binding(0)
var<uniform> draw: Draw;

struct VsIn {
    @builtin(vertex_index) vertex: u32,
    @location(0) position: vec3<f32>,
    @location(1) radius: f32,
    @location(2) normal: vec3<f32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal_world: vec3<f32>,
    // Position in the disc: unit radius at the rim.
    @location(1) disc: vec2<f32>,
};

// Two triangles over the unit square, wound anticlockwise seen from the
// normal's side, so back-face culling hides discs facing away.
const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
);

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let rotation = mat3x3<f32>(draw.model[0].xyz, draw.model[1].xyz, draw.model[2].xyz);
    let n = normalize(rotation * in.normal);
    // A tangent frame: (t, b, n) right-handed.
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
    let t = normalize(cross(n, helper));
    let b = cross(n, t);
    let scale = length(draw.model[0].xyz);
    let corner = CORNERS[in.vertex];
    let centre = (draw.model * vec4<f32>(in.position, 1.0)).xyz;
    var offset = (t * corner.x + b * corner.y) * (in.radius * scale);

    // Seen at a slant, a disc narrows on screen and discs a pixel apart
    // leave pixels between them uncovered. Stretch it within its plane
    // along the slant, by up to five times, so its outline stays about
    // round on screen; the depth stays the plane's.
    let towards = select(normalize(centre - uniforms.eye.xyz), uniforms.eye.xyz, uniforms.eye.w == 0.0);
    let along = towards - n * dot(n, towards);
    let slant = length(along);
    if slant > 1e-4 {
        let u = along / slant;
        let cos_view = max(sqrt(max(1.0 - slant * slant, 0.0)), 0.2);
        offset += u * dot(offset, u) * (1.0 / cos_view - 1.0);
    }
    let world = centre + offset;

    var out: VsOut;
    out.position = uniforms.view_proj * vec4<f32>(world, 1.0);
    out.normal_world = n;
    out.disc = corner;
    return out;
}

struct FsOut {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) surface: vec2<u32>,
};

@fragment
fn fs_gbuffer(in: VsOut) -> FsOut {
    if dot(in.disc, in.disc) > 1.0 {
        discard;
    }
    let n = normalize(in.normal_world);
    var out: FsOut;
    // The draw's colour, stored as the mesh pass stores a vertex's.
    out.albedo = vec4<f32>(
        sqrt(clamp(draw.color.rgb, vec3<f32>(0.0), vec3<f32>(1.0))),
        f32(draw.ids.y) / 255.0,
    );
    out.normal = vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
    out.surface = vec2<u32>(draw.ids.x, bitcast<u32>(in.position.z));
    return out;
}

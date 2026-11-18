// Mesh geometry source: fills the G-buffer (`fs_gbuffer`), or one layer
// of the transparent meshes (`fs_layer_surface`, `fs_layer`). No lighting
// here.

struct Uniforms {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// The mesh's base-colour map (sRGB-encoded storage, so a sample is
// linear) and its repeating trilinear sampler; a 1x1 white texel for an
// untextured mesh.
@group(1) @binding(0)
var base_color: texture_2d<f32>;
@group(1) @binding(1)
var base_sampler: sampler;

// What a layer pass peels against (the layer entry points only): the opaque scene's
// surface target, and the surface target of the layer peeled before this
// one.
@group(2) @binding(0)
var opaque_surface: texture_2d<u32>;
@group(2) @binding(1)
var previous_surface: texture_2d<u32>;

struct Peel {
    // x: 1 when a layer was peeled before this one, 0 for the nearest.
    previous: vec4<u32>,
};

@group(2) @binding(2)
var<uniform> peel: Peel;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    // Per draw, one instance each: the model matrix (rigid or
    // uniform-scale only: normals take its rotation), then the object id
    // and material index.
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(7) ids: vec2<u32>,
    // Image-convention texture coordinate, (0, 0) the top-left texel, in
    // two halves: they fill the vertex's position and normal out to 16
    // bytes each.
    @location(8) u: f32,
    @location(9) v: f32,
    // Per draw: how opaque the mesh is. Only a layer pass reads it.
    @location(10) opacity: f32,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal_world: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) ids: vec2<u32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) opacity: f32,
};

fn vertex(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    var out: VsOut;
    out.position = uniforms.view_proj * model * vec4<f32>(in.position, 1.0);
    out.normal_world = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz) * in.normal;
    out.color = in.color;
    out.ids = in.ids;
    out.uv = vec2<f32>(in.u, in.v);
    out.opacity = in.opacity;
    return out;
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    return vertex(in);
}

// `VsOut` with an invariant position, for the layer passes. A layer pass
// compares a fragment's depth with the depth an earlier pass stored for
// the same fragment, and two pipelines draw the layers (`fs_layer_surface`
// and `fs_layer`): the attribute is what promises that both compute a
// vertex's position to the same bits. The G-buffer's fill does without
// it: there it moved pixels on the GLES adapter.
struct VsLayerOut {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) normal_world: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) ids: vec2<u32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) opacity: f32,
};

@vertex
fn vs_layer(in: VsIn) -> VsLayerOut {
    let v = vertex(in);
    var out: VsLayerOut;
    out.position = v.position;
    out.normal_world = v.normal_world;
    out.color = v.color;
    out.ids = v.ids;
    out.uv = v.uv;
    out.opacity = v.opacity;
    return out;
}

struct FsOut {
    // rgb: base colour, square-root encoded so 8 bits hold the darks;
    // a: material index / 255.
    @location(0) albedo: vec4<f32>,
    // rgb: world normal mapped to 0..1; a: 1 = a normal was supplied.
    @location(1) normal: vec4<f32>,
    // r: object id; g: the bits of this fragment's depth.
    @location(2) surface: vec2<u32>,
};

// What the fragment is, as the G-buffer stores it.
fn surface_of(in: VsOut) -> FsOut {
    // A degenerate interpolated normal must not reach a division: NaN in
    // a unorm target clamps to white on many drivers.
    let n_len = length(in.normal_world);
    let n = select(vec3<f32>(0.0, 0.0, 1.0), in.normal_world / n_len, n_len > 1e-8);

    // Vertex colour times the texture, both linear.
    let base = in.color.rgb * textureSample(base_color, base_sampler, in.uv).rgb;

    var out: FsOut;
    out.albedo = vec4<f32>(
        sqrt(clamp(base, vec3<f32>(0.0), vec3<f32>(1.0))),
        f32(in.ids.y) / 255.0,
    );
    out.normal = vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
    out.surface = vec2<u32>(in.ids.x, bitcast<u32>(in.position.z));
    return out;
}

@fragment
fn fs_gbuffer(in: VsOut) -> FsOut {
    return surface_of(in);
}

// The transparent meshes are drawn a layer at a time: of the surfaces in
// front of the opaque scene and behind the layer peeled before this one,
// the depth test keeps the nearest. Each layer is drawn twice: its
// surface target alone (`fs_layer_surface`), so that every layer's is
// known before any is lit, then the rest of what a G-buffer holds
// (`fs_layer`), just before it is lit.
//
// "Behind" compares a fragment's depth with the depth the layer before
// stored: for the fragment that layer kept, the very number this shader
// computed for it then. Surfaces at exactly one depth are taken in the
// order of their object ids, the highest in front: the draws come in
// rising order of id, so the depth test (less or equal) leaves the
// highest id still to be peeled on top, and a surface as deep as the
// opaque scene's is in front of it unless its id is the lower. (Among
// opaque meshes the last drawn is in front, which is the same thing for
// a host that numbers its draws in order.)
fn peeled_away(in: VsOut) -> bool {
    let px = vec2<i32>(in.position.xy);
    let depth = in.position.z;
    let opaque = textureLoad(opaque_surface, px, 0);
    let opaque_depth = bitcast<f32>(opaque.g);
    if depth > opaque_depth || (depth == opaque_depth && in.ids.x < opaque.r) {
        return true;
    }
    if peel.previous.x != 0u {
        let previous = textureLoad(previous_surface, px, 0);
        let previous_depth = bitcast<f32>(previous.g);
        if depth < previous_depth || (depth == previous_depth && in.ids.x >= previous.r) {
            return true;
        }
    }
    return false;
}

// A layer's surface target: r: object id; g: the bits of the depth.
@fragment
fn fs_layer_surface(in: VsOut) -> @location(0) vec2<u32> {
    if peeled_away(in) {
        discard;
    }
    return vec2<u32>(in.ids.x, bitcast<u32>(in.position.z));
}

struct LayerOut {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    // r: how opaque the surface is, square-root encoded so that 8 bits
    // hold a fade's faint end.
    @location(2) opacity: vec4<f32>,
};

@fragment
fn fs_layer(in: VsOut) -> LayerOut {
    // The texture is sampled before any discard: a derivative wants every
    // fragment of the quad.
    let filled = surface_of(in);
    if peeled_away(in) {
        discard;
    }
    var out: LayerOut;
    out.albedo = filled.albedo;
    out.normal = filled.normal;
    out.opacity = vec4<f32>(sqrt(clamp(in.opacity, 0.0, 1.0)), 0.0, 0.0, 1.0);
    return out;
}

// The ground grid and the world axis lines, drawn analytically: each
// pixel's ray is intersected with the grid plane (instance 0) or brought
// to its closest approach to the plane's normal axis (instance 1). There
// is no line geometry and no extent.
//
// Both write the depth of the point they found, so the depth test hides
// them behind geometry; neither writes it to the depth buffer.

struct GridUniforms {
    inv_view_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    // The plane's two axes and its normal (unit, world).
    axis_u: vec4<f32>,
    axis_v: vec4<f32>,
    axis_n: vec4<f32>,
    // rgb: the grid lines' colour; a: opacity of everything drawn.
    line_color: vec4<f32>,
    // rgb: the colours of the lines along u, v and n; a: 1 when drawn.
    color_u: vec4<f32>,
    color_v: vec4<f32>,
    color_n: vec4<f32>,
    // The weights of the minor, major and super-major lines, then the
    // minor spacing in world units. Each level is ten times the last.
    levels: vec4<f32>,
    // xy: the target's size in pixels.
    viewport: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: GridUniforms;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
    @location(1) @interpolate(flat) part: u32,
};

struct FsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@vertex
fn vs_grid(@builtin(vertex_index) vi: u32, @builtin(instance_index) part: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VsOut;
    out.position = vec4<f32>(p[vi], 0.0, 1.0);
    out.ndc = p[vi];
    out.part = part;
    return out;
}

fn unproject(ndc: vec2<f32>, z: f32) -> vec3<f32> {
    let h = uniforms.inv_view_proj * vec4<f32>(ndc, z, 1.0);
    return h.xyz / h.w;
}

// Coverage of a pixel by the lines at whole numbers of `coord`, per
// direction. `per_pixel` is how much `coord` changes across a pixel; a
// direction whose lines have drawn closer than a few pixels fades out.
fn lines(coord: vec2<f32>, per_pixel: vec2<f32>) -> f32 {
    let pixels = abs(fract(coord - 0.5) - 0.5) / per_pixel;
    let coverage = vec2<f32>(1.0) - min(pixels, vec2<f32>(1.0));
    let sparse = vec2<f32>(1.0) - smoothstep(vec2<f32>(0.03), vec2<f32>(0.25), per_pixel);
    let both = coverage * sparse;
    return max(both.x, both.y);
}

// Coverage by the single line at `coord` = 0, `half_width` pixels wide.
fn line_at_zero(coord: f32, per_pixel: f32, half_width: f32) -> f32 {
    return 1.0 - smoothstep(half_width - 0.5, half_width + 0.5, abs(coord) / per_pixel);
}

@fragment
fn fs_grid(in: VsOut) -> FsOut {
    let near = unproject(in.ndc, 0.0);
    let far = unproject(in.ndc, 1.0);
    let ray = far - near;
    let n = uniforms.axis_n.xyz;
    let opacity = uniforms.line_color.a;

    var out: FsOut;
    if in.part == 0u {
        // ---- The plane. Everything is computed for every pixel so the
        // derivatives are defined; pixels that miss are dropped last.
        let t = -dot(near, n) / dot(ray, n);
        let hit = near + ray * t;
        let coord = vec2<f32>(dot(hit, uniforms.axis_u.xyz), dot(hit, uniforms.axis_v.xyz));
        let coord_per_pixel = max(fwidth(coord), vec2<f32>(1e-12));

        let minor = uniforms.levels.w;
        let w = uniforms.levels.xyz;
        var alpha = 0.0;
        alpha = max(alpha, w.x * lines(coord / minor, coord_per_pixel / minor));
        alpha = max(alpha, w.y * lines(coord / (minor * 10.0), coord_per_pixel / (minor * 10.0)));
        alpha = max(alpha, w.z * lines(coord / (minor * 100.0), coord_per_pixel / (minor * 100.0)));
        var color = uniforms.line_color.rgb;
        alpha = alpha * 0.6;

        // The line along u is where v = 0, and the reverse.
        let along_u = uniforms.color_u.a * line_at_zero(coord.y, coord_per_pixel.y, 0.75);
        let along_v = uniforms.color_v.a * line_at_zero(coord.x, coord_per_pixel.x, 0.75);
        color = mix(color, uniforms.color_v.rgb, along_v);
        color = mix(color, uniforms.color_u.rgb, along_u);
        alpha = max(alpha, max(along_u, along_v));

        // Dimmer from the far side of the plane.
        if dot(near, n) < 0.0 {
            alpha = alpha * 0.45;
        }
        alpha = alpha * opacity;

        // A pixel's width further along the ray, so a surface lying in
        // the plane wins the depth test instead of flickering.
        let pushed = hit + normalize(ray) * 1.5 * length(fwidth(hit));
        let clip = uniforms.view_proj * vec4<f32>(pushed, 1.0);
        let depth = clip.z / clip.w;
        if !(t > 0.0 && t <= 1.0 && alpha > 0.002 && depth >= 0.0 && depth <= 1.0) {
            discard;
        }
        out.color = vec4<f32>(color * alpha, alpha);
        out.depth = depth;
        return out;
    }

    // ---- The normal axis: the point of the line through the origin
    // along n that this pixel's ray passes closest to.
    let a = dot(ray, ray);
    let b = dot(ray, n);
    let d = dot(ray, near);
    let e = dot(n, near);
    let along = (a * e - b * d) / (a - b * b);
    let closest = n * along;
    let clip = uniforms.view_proj * vec4<f32>(closest, 1.0);
    let ndc = clip.xyz / clip.w;
    let off = (ndc.xy - in.ndc) * 0.5 * uniforms.viewport.xy;
    let alpha = uniforms.color_n.a * opacity * (1.0 - smoothstep(0.25, 1.25, length(off)));
    if !(clip.w > 0.0 && ndc.z >= 0.0 && ndc.z <= 1.0 && alpha > 0.002) {
        discard;
    }
    out.color = vec4<f32>(uniforms.color_n.rgb * alpha, alpha);
    out.depth = ndc.z;
    return out;
}

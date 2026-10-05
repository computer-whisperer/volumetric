// The view gizmo: the three world axes as seen by the camera, drawn into
// a square of the target. Each axis has a labelled disc at its positive
// end, joined to the centre by an arm, and a smaller ringed disc at its
// negative end. Everything is a distance function of the pixel, painted
// back to front; `gizmo.rs` lays the ends out.

struct GizmoEnd {
    // xy: centre in target pixels (y down); z: radius; w: 0 for a
    // negative end, else 1 + the axis whose letter the disc carries.
    disc: vec4<f32>,
    // rgb: colour; a: 1 when the pointer is over this end.
    color: vec4<f32>,
};

struct GizmoUniforms {
    // xy: centre in target pixels; z: radius; w: 1 when the pointer is
    // over the gizmo.
    placement: vec4<f32>,
    // xy: the target's size in pixels.
    viewport: vec4<f32>,
    // Back to front.
    ends: array<GizmoEnd, 6>,
};

@group(0) @binding(0)
var<uniform> uniforms: GizmoUniforms;

@vertex
fn vs_gizmo(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // A triangle strip over the gizmo's square, a pixel of margin around.
    let corner = vec2<f32>(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - vec2<f32>(1.0);
    let px = uniforms.placement.xy + corner * (uniforms.placement.z + 1.0);
    let ndc = px / uniforms.viewport.xy * 2.0 - vec2<f32>(1.0);
    return vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
}

fn segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    return length(p - a - ab * t);
}

// Distance to the strokes of an axis letter in a box x -0.8..0.8,
// y -1..1 (y up).
fn letter(axis: u32, q: vec2<f32>) -> f32 {
    let tl = vec2<f32>(-0.8, 1.0);
    let tr = vec2<f32>(0.8, 1.0);
    let bl = vec2<f32>(-0.8, -1.0);
    let br = vec2<f32>(0.8, -1.0);
    let mid = vec2<f32>(0.0);
    if axis == 0u {
        return min(segment(q, bl, tr), segment(q, tl, br));
    } else if axis == 1u {
        return min(min(segment(q, tl, mid), segment(q, tr, mid)), segment(q, mid, vec2<f32>(0.0, -1.0)));
    }
    return min(min(segment(q, tl, tr), segment(q, tr, bl)), segment(q, bl, br));
}

// `src` at `alpha` over premultiplied `dst`.
fn over(dst: vec4<f32>, src: vec3<f32>, alpha: f32) -> vec4<f32> {
    return vec4<f32>(src * alpha, alpha) + dst * (1.0 - alpha);
}

// Coverage of a pixel `distance` pixels outside a shape's edge.
fn cover(distance: f32) -> f32 {
    return 1.0 - smoothstep(-0.5, 0.5, distance);
}

@fragment
fn fs_gizmo(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = frag.xy;
    let centre = uniforms.placement.xy;
    let radius = uniforms.placement.z;
    let stroke = radius * 0.03;

    var color = vec4<f32>(0.0);
    // A backdrop while the pointer is over the gizmo.
    color = over(color, vec3<f32>(1.0), 0.12 * uniforms.placement.w * cover(length(p - centre) - radius));

    for (var i = 0u; i < 6u; i = i + 1u) {
        let end = uniforms.ends[i];
        let at = end.disc.xy;
        let r = end.disc.z;
        let positive = end.disc.w > 0.5;
        let lit = mix(end.color.rgb, vec3<f32>(1.0), 0.4 * end.color.a);
        let edge = length(p - at) - r;
        if positive {
            color = over(color, end.color.rgb, cover(segment(p, centre, at) - stroke));
            color = over(color, lit, cover(edge));
            let q = (p - at) / (r * 0.42);
            let strokes = letter(u32(end.disc.w - 0.5), vec2<f32>(q.x, -q.y)) * r * 0.42;
            color = over(color, vec3<f32>(0.04), 0.9 * cover(strokes - stroke));
        } else {
            color = over(color, lit, mix(0.3, 0.7, end.color.a) * cover(edge));
            color = over(color, lit, cover(abs(edge + stroke) - stroke));
        }
    }
    return color;
}

// Line rendering shader with vertex shader quad expansion
//
// Each line segment is stored as an instance. The vertex shader receives a quad
// vertex (4 vertices forming 2 triangles) plus the line segment data per-instance,
// and expands the line to a screen-aligned quad.

struct Uniforms {
    view_proj: mat4x4<f32>,
    screen_size: vec2<f32>,
    width_mode: u32,       // 0 = screen pixels, 1 = world units
    default_width: f32,    // Used if instance width is 0
    dash_length: f32,
    gap_length: f32,
    _pad: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// Per-vertex quad corner
struct QuadVertex {
    @location(0) corner: vec2<f32>,  // x: 0=start, 1=end; y: -1=left, +1=right
};

// Per-instance line segment
struct LineInstance {
    @location(1) start: vec3<f32>,
    @location(2) end: vec3<f32>,
    @location(3) color: vec4<f32>,
    @location(4) width: f32,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) @interpolate(linear) edge_coord: f32,     // -1 to +1 across line width (for AA) - linear interpolation to avoid perspective distortion
    @location(2) @interpolate(linear) line_coord: f32,     // Distance along line (world units, for dashing)
};

@vertex
fn vs_main(quad: QuadVertex, line: LineInstance) -> VsOut {
    var out: VsOut;

    // Select position along line (0 = start, 1 = end)
    let t = quad.corner.x;

    // Project both endpoints to clip space
    let clip_start = uniforms.view_proj * vec4(line.start, 1.0);
    let clip_end = uniforms.view_proj * vec4(line.end, 1.0);

    // Current clip position for this vertex
    let clip_pos = mix(clip_start, clip_end, t);

    // Safe W values
    let w_min = 0.0001;
    let w_safe = max(clip_pos.w, w_min);

    // Clip-space line direction
    let clip_dir = clip_end - clip_start;

    // Compute the screen-space tangent direction at the current vertex position.
    // The derivative of screen position with respect to line parameter is:
    // d(screen)/dt = [(clip_dir.xy * w - clip.xy * clip_dir.w) / w²] * screen_size * 0.5
    // This gives the local screen-space direction of the line at this vertex.
    let screen_tangent = (clip_dir.xy * clip_pos.w - clip_pos.xy * clip_dir.w) / (w_safe * w_safe);

    // Convert to actual screen pixels (NDC to screen scale)
    let screen_tangent_px = screen_tangent * 0.5 * uniforms.screen_size;
    let tangent_len = length(screen_tangent_px);

    // Perpendicular in screen space (rotate 90 degrees)
    let screen_perp = vec2(-screen_tangent_px.y, screen_tangent_px.x) / max(tangent_len, 0.001);

    // Determine width
    let width = select(uniforms.default_width, line.width, line.width > 0.0);
    var half_width_px: f32;
    if uniforms.width_mode == 0u {
        // Screen-space: width is in pixels
        half_width_px = width * 0.5;
    } else {
        // World-space: approximate pixel width from clip.w
        half_width_px = (width / w_safe) * uniforms.screen_size.y * 0.5;
    }

    // Offset in screen pixels
    let offset_screen_px = screen_perp * half_width_px * quad.corner.y;

    // Convert screen pixel offset back to NDC
    let offset_ndc = offset_screen_px * 2.0 / uniforms.screen_size;

    // Apply offset in clip space (multiply by w to maintain correct perspective)
    out.position = vec4(clip_pos.xy + offset_ndc * clip_pos.w, clip_pos.zw);

    out.color = line.color;
    out.edge_coord = quad.corner.y;  // -1 to +1
    out.line_coord = t * length(line.end - line.start);

    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Anti-aliased edges using smoothstep
    let edge_dist = abs(in.edge_coord);
    let edge_aa = 1.0 - smoothstep(0.85, 1.0, edge_dist);

    // Dash pattern (if enabled)
    var pattern_alpha = 1.0;
    if uniforms.dash_length > 0.0 {
        let cycle = uniforms.dash_length + uniforms.gap_length;
        let pos_in_cycle = in.line_coord % cycle;
        // Soft edges on dash transitions
        let dash_edge = smoothstep(0.0, 0.1, pos_in_cycle)
                      * (1.0 - smoothstep(uniforms.dash_length - 0.1, uniforms.dash_length, pos_in_cycle));
        pattern_alpha = dash_edge;
    }

    let final_alpha = in.color.a * edge_aa * pattern_alpha;
    if final_alpha < 0.01 {
        discard;
    }

    // Premultiplied alpha output
    return vec4(in.color.rgb * final_alpha, final_alpha);
}

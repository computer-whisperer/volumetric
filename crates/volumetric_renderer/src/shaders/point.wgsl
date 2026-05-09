// Enhanced point rendering shader
//
// Renders points as screen-aligned quads with:
// - Alpha channel support for transparency
// - Shape selection (circle, square, diamond)
// - Anti-aliased edges

struct Uniforms {
    view_proj: mat4x4<f32>,
    screen_size_px: vec2<f32>,
    point_size_px: f32,
    size_mode: u32,        // 0 = screen pixels, 1 = world units
    shape: u32,            // 0 = circle, 1 = square, 2 = diamond
    _pad: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VsIn {
    // Quad vertex (per-vertex)
    @location(0) corner: vec2<f32>, // -1..+1 (maps to quad corners)
    @location(1) uv: vec2<f32>,     //  0..+1 (texture coordinates)

    // Per-instance point data
    @location(2) position: vec3<f32>,
    @location(3) color: vec4<f32>,  // RGBA with alpha
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let clip = uniforms.view_proj * vec4<f32>(in.position, 1.0);

    // Determine point size in pixels
    var half_size_px: f32;
    if uniforms.size_mode == 0u {
        // Screen-space: size is in pixels (constant regardless of distance)
        half_size_px = uniforms.point_size_px * 0.5;
    } else {
        // World-space: size shrinks with distance
        half_size_px = (uniforms.point_size_px / clip.w) * uniforms.screen_size_px.y * 0.5;
    }

    // Convert pixel offset to clip space
    let offset_ndc = in.corner * (half_size_px * 2.0 / uniforms.screen_size_px);
    out.position = vec4<f32>(clip.xy + offset_ndc * clip.w, clip.zw);

    out.color = in.color;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Map UV to [-1, 1] centered coordinates
    let uv = in.uv * 2.0 - vec2<f32>(1.0, 1.0);

    var alpha: f32;

    // Shape-based alpha calculation with anti-aliasing
    switch uniforms.shape {
        case 0u: {
            // Circle
            let dist = length(uv);
            alpha = 1.0 - smoothstep(0.9, 1.0, dist);
        }
        case 1u: {
            // Square
            let max_coord = max(abs(uv.x), abs(uv.y));
            alpha = 1.0 - smoothstep(0.9, 1.0, max_coord);
        }
        case 2u: {
            // Diamond (rotated square / L1 norm)
            let dist = abs(uv.x) + abs(uv.y);
            alpha = 1.0 - smoothstep(0.9, 1.0, dist);
        }
        default: {
            // Fallback to circle
            let dist = length(uv);
            alpha = 1.0 - smoothstep(0.9, 1.0, dist);
        }
    }

    let final_alpha = in.color.a * alpha;
    if final_alpha < 0.01 {
        discard;
    }

    // Premultiplied alpha output
    return vec4<f32>(in.color.rgb * final_alpha, final_alpha);
}

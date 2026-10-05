// Anti-aliasing of the lit scene (FXAA 3.11, after Lottes): finds each
// pixel's edge from the contrast of its neighbours, walks along the edge
// to its ends, and resamples the pixel shifted toward the edge by how
// far along it sits. Runs before the grid, lines and gizmo, which
// anti-alias themselves. Follows `fullscreen_vs.wgsl`.

struct FxaaUniforms {
    // The size of one texel of the lit scene in texture coordinates.
    texel: vec2<f32>,
    _pad0: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: FxaaUniforms;

@group(0) @binding(1)
var lit: texture_2d<f32>;

@group(0) @binding(2)
var lit_sampler: sampler;

// Contrast below this fraction of the local maximum is not an edge.
const EDGE_RELATIVE: f32 = 0.125;
// Nor is contrast below this, however dark the surroundings.
const EDGE_ABSOLUTE: f32 = 0.0312;
// How strongly a pixel that stands out from all its neighbours is
// blended into them.
const SUBPIXEL: f32 = 0.5;
const SEARCH_STEPS: i32 = 12;

fn fetch(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(lit, lit_sampler, uv, 0.0);
}

// Perceptual brightness of a linear colour.
fn luma(color: vec4<f32>) -> f32 {
    return dot(sqrt(max(color.rgb, vec3<f32>(0.0))), vec3<f32>(0.299, 0.587, 0.114));
}

fn luma_at(uv: vec2<f32>, x: f32, y: f32) -> f32 {
    return luma(fetch(uv + vec2<f32>(x, y) * uniforms.texel));
}

// How far the edge search advances at each step, in texels.
fn search_step(i: i32) -> f32 {
    if i < 5 {
        return 1.0;
    } else if i < 6 {
        return 1.5;
    } else if i < 10 {
        return 2.0;
    } else if i < 11 {
        return 4.0;
    }
    return 8.0;
}

@fragment
fn fs_fxaa(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let texel = uniforms.texel;
    let uv = frag.xy * texel;
    let center = fetch(uv);
    let lc = luma(center);

    // Neighbours, named by the sign of their offset in x then y.
    let l_0m = luma_at(uv, 0.0, -1.0);
    let l_0p = luma_at(uv, 0.0, 1.0);
    let l_m0 = luma_at(uv, -1.0, 0.0);
    let l_p0 = luma_at(uv, 1.0, 0.0);
    let brightest = max(lc, max(max(l_0m, l_0p), max(l_m0, l_p0)));
    let darkest = min(lc, min(min(l_0m, l_0p), min(l_m0, l_p0)));
    let contrast = brightest - darkest;
    if contrast < max(EDGE_ABSOLUTE, brightest * EDGE_RELATIVE) {
        return center;
    }

    let l_mm = luma_at(uv, -1.0, -1.0);
    let l_pm = luma_at(uv, 1.0, -1.0);
    let l_mp = luma_at(uv, -1.0, 1.0);
    let l_pp = luma_at(uv, 1.0, 1.0);

    // An edge is horizontal when brightness changes more down the
    // columns than along the rows.
    let down_columns = abs(l_mm + l_mp - 2.0 * l_m0)
        + 2.0 * abs(l_0m + l_0p - 2.0 * lc)
        + abs(l_pm + l_pp - 2.0 * l_p0);
    let along_rows = abs(l_mm + l_pm - 2.0 * l_0m)
        + 2.0 * abs(l_m0 + l_p0 - 2.0 * lc)
        + abs(l_mp + l_pp - 2.0 * l_0p);
    let horizontal = down_columns >= along_rows;

    // The neighbours across the edge, and which side it is on.
    let l_neg = select(l_m0, l_0m, horizontal);
    let l_pos = select(l_p0, l_0p, horizontal);
    let on_negative = abs(l_neg - lc) >= abs(l_pos - lc);
    let gradient = 0.25 * max(abs(l_neg - lc), abs(l_pos - lc));
    var across = select(texel.x, texel.y, horizontal);
    var edge_luma = 0.5 * (l_pos + lc);
    if on_negative {
        across = -across;
        edge_luma = 0.5 * (l_neg + lc);
    }

    // Walk both ways along the edge, half a texel onto it, until the
    // brightness there stops matching the edge's.
    var on_edge = uv;
    if horizontal {
        on_edge.y = on_edge.y + across * 0.5;
    } else {
        on_edge.x = on_edge.x + across * 0.5;
    }
    let along = select(vec2<f32>(0.0, texel.y), vec2<f32>(texel.x, 0.0), horizontal);
    var uv_a = on_edge - along;
    var uv_b = on_edge + along;
    var end_a = luma(fetch(uv_a)) - edge_luma;
    var end_b = luma(fetch(uv_b)) - edge_luma;
    var done_a = abs(end_a) >= gradient;
    var done_b = abs(end_b) >= gradient;
    for (var i = 2; i < SEARCH_STEPS; i = i + 1) {
        if done_a && done_b {
            break;
        }
        if !done_a {
            uv_a = uv_a - along * search_step(i);
            end_a = luma(fetch(uv_a)) - edge_luma;
            done_a = abs(end_a) >= gradient;
        }
        if !done_b {
            uv_b = uv_b + along * search_step(i);
            end_b = luma(fetch(uv_b)) - edge_luma;
            done_b = abs(end_b) >= gradient;
        }
    }

    // The nearer end says how far along the edge's step this pixel is,
    // and so how much of the other side covers it.
    let to_a = select(uv.y - uv_a.y, uv.x - uv_a.x, horizontal);
    let to_b = select(uv_b.y - uv.y, uv_b.x - uv.x, horizontal);
    let nearer_end = select(end_b, end_a, to_a < to_b);
    var shift = 0.5 - min(to_a, to_b) / (to_a + to_b);
    // Only when the edge really ends there by turning toward this pixel.
    if (nearer_end < 0.0) == (lc < edge_luma) {
        shift = 0.0;
    }

    // A pixel unlike all eight neighbours is blended regardless.
    let around = (2.0 * (l_0m + l_0p + l_m0 + l_p0) + l_mm + l_pm + l_mp + l_pp) / 12.0;
    let lone = clamp(abs(around - lc) / contrast, 0.0, 1.0);
    let smooth_lone = (3.0 - 2.0 * lone) * lone * lone;
    shift = max(shift, smooth_lone * smooth_lone * SUBPIXEL);

    var resampled = uv;
    if horizontal {
        resampled.y = resampled.y + shift * across;
    } else {
        resampled.x = resampled.x + shift * across;
    }
    return fetch(resampled);
}

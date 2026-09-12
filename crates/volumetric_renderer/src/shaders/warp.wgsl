// Lens warp
//
// Writes the output frame by sampling the pinhole frame where each
// output pixel's ray lands. The landing positions come as a coarse grid
// spanning the output edge to edge; they are interpolated bilinearly
// here, by hand, since the grid is 32-bit float and filters nowhere
// portable. A negative position marks a pixel with no image: background.

struct WarpUniforms {
    output_size: vec2<f32>,
    grid_cells: vec2<f32>,
    source_size: vec2<f32>,
    _pad: vec2<f32>,
    background: vec4<f32>,
};

@group(0) @binding(0)
var frame: texture_2d<f32>;

@group(0) @binding(1)
var frame_sampler: sampler;

@group(0) @binding(2)
var grid: texture_2d<f32>;

@group(0) @binding(3)
var<uniform> u: WarpUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VsOut;
    out.pos = vec4<f32>(p[vi], 0.0, 1.0);
    return out;
}

@fragment
fn fs_warp(in: VsOut) -> @location(0) vec4<f32> {
    // The fragment's position is its pixel centre.
    let g = clamp(in.pos.xy / u.output_size * u.grid_cells, vec2<f32>(0.0), u.grid_cells);
    let g0 = floor(g);
    let f = g - g0;
    let i0 = vec2<i32>(g0);
    let i1 = min(i0 + vec2<i32>(1, 1), vec2<i32>(u.grid_cells));
    let s00 = textureLoad(grid, vec2<i32>(i0.x, i0.y), 0).xy;
    let s10 = textureLoad(grid, vec2<i32>(i1.x, i0.y), 0).xy;
    let s01 = textureLoad(grid, vec2<i32>(i0.x, i1.y), 0).xy;
    let s11 = textureLoad(grid, vec2<i32>(i1.x, i1.y), 0).xy;
    let missing = min(min(s00.x, s00.y), min(min(s10.x, s10.y), min(min(s01.x, s01.y), min(s11.x, s11.y)))) < 0.0;
    let src = mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
    let color = textureSample(frame, frame_sampler, src / u.source_size);
    return select(color, u.background, missing);
}

// One triangle covering the viewport. Fragment shaders that follow this
// in the same module address the G-buffer by pixel through
// `@builtin(position)` and `textureLoad`, so no pass here needs a sampler.

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    return vec4<f32>(p[vi], 0.0, 1.0);
}

// The depth (0 near, 1 far or background) of a G-buffer surface texel.
fn surface_depth(texel: vec4<u32>) -> f32 {
    return bitcast<f32>(texel.g);
}


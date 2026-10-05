// One triangle covering the viewport. Fragment shaders that follow this
// in the same module address the G-buffer by pixel through
// `@builtin(position)` and `textureLoad`.

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

// The distance in front of the eye, in world units, of a point at depth
// `z` (0..1). `k` is the frame's `depth_to_distance`: the terms of the
// inverse projection that turn a depth into a view-space z, which depend
// on nothing else for any projection the renderer draws with.
fn eye_distance(z: f32, k: vec4<f32>) -> f32 {
    return abs((k.x * z + k.y) / (k.z * z + k.w));
}
